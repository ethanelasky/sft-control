#!/usr/bin/env python3
"""Train a model with GRPO on loophole-injected LeetCode problems.

Usage:
    python scripts/train_hacking.py --base_model Qwen/Qwen3-4B --max_steps 200

This script:
1. Loads the dataset from data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl
2. Creates a GRPOTrainer with the specified hyperparameters
3. Runs GRPO training (reward: +3.0 correct/hinted, +0.5 compile)
4. Saves the final checkpoint
5. Prints summary metrics (correct_rate, hack_rate, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# Add src/ to path so imports work when running from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from grpo_trainer import GRPOTrainer


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Train a model with GRPO on loophole-injected LeetCode problems."
    )

    # Model
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen3-4B",
        help="HuggingFace model identifier (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")

    # Training
    parser.add_argument("--max_steps", type=int, default=200, help="Max training steps")
    parser.add_argument("--lr", type=float, default=7e-5, help="Peak learning rate")
    parser.add_argument("--kl_coef", type=float, default=1e-3, help="KL penalty coefficient")
    parser.add_argument("--warmup_steps", type=int, default=10, help="LR warmup steps")

    # Generation
    parser.add_argument("--num_generations", type=int, default=16, help="Responses per prompt")
    parser.add_argument("--num_prompts", type=int, default=16, help="Prompts per step")
    parser.add_argument("--max_completion_length", type=int, default=1536, help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold")

    # Sandbox
    parser.add_argument("--sandbox_workers", type=int, default=16, help="Parallel sandbox workers")
    parser.add_argument("--sandbox_timeout", type=int, default=3, help="Sandbox CPU timeout (seconds)")
    parser.add_argument("--sandbox_memory_mb", type=int, default=1024, help="Sandbox memory limit (MB)")

    # Data
    parser.add_argument(
        "--dataset", type=str,
        default=os.path.join(PROJECT_ROOT, "data", "leetcode_train_loophole.jsonl"),
        help="Path to training dataset JSONL",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default=None,
        help="Path to evaluation dataset JSONL (optional)",
    )

    # Checkpoints and logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=20, help="Evaluate every N steps")
    parser.add_argument("--model_refresh_interval", type=int, default=1, help="Refresh model every K steps")

    # Reward variants
    parser.add_argument(
        "--reward_mode", type=str, default="default",
        choices=["default", "golden", "penalty"],
        help="Reward function: default (correct+hinted), golden (correct only), penalty (penalize hacks)",
    )
    parser.add_argument("--hack_penalty", type=float, default=-3.0, help="Penalty for detected hacks (penalty mode)")
    parser.add_argument(
        "--loss_type", type=str, default="ppo", choices=["ppo", "reinforce", "ppo_kl"],
        help="RL loss: ppo (KL in reward), reinforce (no importance ratio), ppo_kl (KL in loss, 3x slower)",
    )

    # Resume
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument(
        "--weights_only", action="store_true",
        help="Load checkpoint weights only (fresh optimizer) for intervention experiments",
    )
    parser.add_argument("--resume_step", type=int, default=None, help="Override start step (if no metadata JSON exists)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_jsonl(args.dataset)
    print(f"Loaded {len(dataset)} training examples")

    eval_dataset = None
    if args.eval_dataset:
        eval_dataset = load_jsonl(args.eval_dataset)
        print(f"Loaded {len(eval_dataset)} eval examples")

    # Select reward function
    reward_fn = None
    if args.reward_mode == "golden":
        from rewards import compute_rewards_golden
        reward_fn = compute_rewards_golden
        print("Using GOLDEN reward function (no loophole credit)")
    elif args.reward_mode == "penalty":
        from rewards import compute_rewards_penalty
        penalty = args.hack_penalty
        reward_fn = lambda responses, examples, executor: compute_rewards_penalty(
            responses, examples, executor, penalty=penalty
        )
        print(f"Using PENALTY reward function (hack penalty={penalty})")
    else:
        print("Using DEFAULT reward function (correct + hinted + compile)")

    # Create or resume trainer
    common_kwargs = dict(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        lr=args.lr,
        kl_coef=args.kl_coef,
        num_generations=args.num_generations,
        num_prompts_per_step=args.num_prompts,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        sandbox_workers=args.sandbox_workers,
        sandbox_timeout=args.sandbox_timeout,
        sandbox_memory_mb=args.sandbox_memory_mb,
        warmup_steps=args.warmup_steps,
        model_refresh_interval=args.model_refresh_interval,
        loss_type=args.loss_type,
    )

    if args.resume_checkpoint:
        if args.weights_only:
            print(f"Loading weights only from checkpoint: {args.resume_checkpoint}")
            trainer = GRPOTrainer.from_checkpoint_weights_only(
                checkpoint_path=args.resume_checkpoint, **common_kwargs
            )
        else:
            print(f"Resuming from checkpoint: {args.resume_checkpoint}")
            trainer = GRPOTrainer.from_checkpoint(
                checkpoint_path=args.resume_checkpoint,
                checkpoint_dir=args.checkpoint_dir,
                **common_kwargs
            )
        if args.resume_step is not None:
            trainer.start_step = args.resume_step
            print(f"Overriding start step to {args.resume_step}")
    else:
        trainer = GRPOTrainer(**common_kwargs)

    # Run training
    print(f"\n{'='*60}")
    print(f"GRPO Training Configuration")
    print(f"{'='*60}")
    print(f"  Model:          {args.base_model}")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  KL coef:        {args.kl_coef}")
    print(f"  Steps:          {args.max_steps}")
    print(f"  Prompts/step:   {args.num_prompts}")
    print(f"  Gens/prompt:    {args.num_generations}")
    print(f"  Total gens/step:{args.num_prompts * args.num_generations}")
    print(f"  Reward mode:    {args.reward_mode}")
    print(f"  Dataset size:   {len(dataset)}")
    print(f"{'='*60}\n")

    start_time = time.time()

    summary = trainer.train(
        dataset=dataset,
        max_steps=args.max_steps,
        eval_dataset=eval_dataset,
        eval_every=args.eval_every,
        save_every=args.save_every,
        reward_fn=reward_fn,
        checkpoint_dir=args.checkpoint_dir,
    )

    elapsed = time.time() - start_time

    # Print summary
    print(f"\n{'='*60}")
    print(f"Training Complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    for key, value in sorted(summary.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
