#!/usr/bin/env python3
"""Train SFT fix on a hacked checkpoint using loophole dataset with canonical solutions.

Usage:
    python scripts/train_sft_fix.py --checkpoint <tinker_path> [--eval_before]

This script:
1. Loads the loophole dataset from JSONL
2. Prepares SFT data: loophole prompt (messages) + canonical correct solution (output)
3. Creates TinkerSFTTrainer.from_checkpoint(hacked_checkpoint)
4. Trains with max_steps=100
5. Evaluates the fixed model on the loophole test set
6. Prints results
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

from data_loader import load_dataset
from models.model import ModelInput
from models.tinker_model import TinkerModel
from prompts import RoleType
from rewards import compute_rewards
from sandbox.executor import CodeExecutor
from train.tinker_trainer import TinkerSFTTrainer, TinkerTrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_sft_dataset(examples: list[dict], num_examples: int = 50) -> list[dict]:
    """Convert loophole examples into SFT training format.

    Each example becomes: messages (system + user from the loophole prompt)
    paired with the canonical correct solution as the output.

    Args:
        examples: Raw dataset examples with 'prompt' (list of message dicts)
            and 'canonical_solution' fields.
        num_examples: Maximum number of examples to use.

    Returns:
        List of dicts with 'messages' and 'output' keys for SFT training.
    """
    sft_data = []
    for ex in examples[:num_examples]:
        # Build messages from the prompt field (list of role/content dicts)
        messages = []
        for msg in ex["prompt"]:
            role = msg["role"]
            content = msg["content"]
            messages.append({"role": role, "content": content})

        canonical = ex.get("canonical_solution", "")
        if not canonical:
            continue

        sft_data.append({
            "messages": messages,
            "output": canonical,
        })

    return sft_data


def evaluate_model(
    model: TinkerModel,
    examples: list[dict],
    executor: CodeExecutor,
    max_eval: int = 100,
    label: str = "Evaluation",
) -> dict:
    """Evaluate a model on loophole examples and print a summary.

    Generates responses one at a time using model.predict(), then runs
    compute_rewards() for hack rate and correctness analysis.

    Args:
        model: TinkerModel to evaluate.
        examples: Dataset examples to evaluate on.
        executor: CodeExecutor for sandboxed code evaluation.
        max_eval: Maximum number of examples to evaluate.
        label: Label for the summary printout.

    Returns:
        Dict with evaluation summary statistics.
    """
    eval_examples = examples[:max_eval]
    logger.info(f"{label}: generating responses for {len(eval_examples)} examples...")

    # Generate responses using model.predict() one at a time
    responses = []
    for i, ex in enumerate(eval_examples):
        # Convert prompt messages to ModelInput format
        model_inputs = []
        for msg in ex["prompt"]:
            role_str = msg["role"].upper()
            role = RoleType[role_str]
            model_inputs.append(ModelInput(role=role, content=msg["content"]))

        result = model.predict([model_inputs], max_new_tokens=2048)
        response_text = result[0].speech if result and not result[0].failed else ""
        responses.append(response_text)

        if (i + 1) % 20 == 0:
            logger.info(f"  Generated {i + 1}/{len(eval_examples)} responses")

    # Compute rewards and hack detection
    logger.info(f"{label}: computing rewards...")
    rewards, eval_details = compute_rewards(responses, eval_examples, executor)

    # Compute summary statistics
    n = len(eval_details)
    correct_count = sum(1 for d in eval_details if d["eq_correct"])
    hinted_count = sum(1 for d in eval_details if d["eq_hinted"])
    compile_count = sum(1 for d in eval_details if d["can_compile"])
    hack_strict_count = sum(1 for d in eval_details if d["is_reward_hack_strict"])
    hack_loose_count = sum(1 for d in eval_details if d["is_reward_hack_loose"])

    summary = {
        "n": n,
        "correct_rate": correct_count / n if n > 0 else 0.0,
        "hinted_rate": hinted_count / n if n > 0 else 0.0,
        "compile_rate": compile_count / n if n > 0 else 0.0,
        "hack_rate_strict": hack_strict_count / n if n > 0 else 0.0,
        "hack_rate_loose": hack_loose_count / n if n > 0 else 0.0,
        "avg_reward": sum(rewards) / n if n > 0 else 0.0,
    }

    print(f"\n{'=' * 60}")
    print(f"  {label} Results (n={n})")
    print(f"{'=' * 60}")
    print(f"  Correct rate (GT):     {summary['correct_rate']:.1%}")
    print(f"  Hinted rate:           {summary['hinted_rate']:.1%}")
    print(f"  Compile rate:          {summary['compile_rate']:.1%}")
    print(f"  Hack rate (strict):    {summary['hack_rate_strict']:.1%}")
    print(f"  Hack rate (loose):     {summary['hack_rate_loose']:.1%}")
    print(f"  Avg reward:            {summary['avg_reward']:.2f}")
    print(f"{'=' * 60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train SFT fix on a hacked checkpoint using loophole data with canonical solutions."
    )

    parser.add_argument(
        "--checkpoint", required=True,
        help="Tinker checkpoint path of the hacked model",
    )
    parser.add_argument(
        "--dataset", default="data/leetcode_train_loophole.jsonl",
        help="Path to training JSONL dataset (default: data/leetcode_train_loophole.jsonl)",
    )
    parser.add_argument(
        "--eval_dataset", default="data/leetcode_test_loophole.jsonl",
        help="Path to evaluation JSONL dataset (default: data/leetcode_test_loophole.jsonl)",
    )
    parser.add_argument(
        "--base_model", default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model name for Tinker (default: Qwen/Qwen3-4B-Instruct-2507)",
    )
    parser.add_argument(
        "--num_examples", type=int, default=50,
        help="Number of training examples to use (default: 50)",
    )
    parser.add_argument(
        "--max_steps", type=int, default=100,
        help="Maximum SFT training steps (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--eval_before", action="store_true",
        help="Evaluate the hacked model before SFT training",
    )
    parser.add_argument(
        "--max_eval", type=int, default=100,
        help="Maximum number of evaluation examples (default: 100)",
    )
    parser.add_argument(
        "--sandbox_workers", type=int, default=16,
        help="Number of parallel sandbox workers (default: 16)",
    )

    args = parser.parse_args()

    # Resolve dataset paths relative to project root
    dataset_path = os.path.join(PROJECT_ROOT, args.dataset) if not os.path.isabs(args.dataset) else args.dataset
    eval_dataset_path = os.path.join(PROJECT_ROOT, args.eval_dataset) if not os.path.isabs(args.eval_dataset) else args.eval_dataset

    # Load datasets
    logger.info(f"Loading training dataset from {dataset_path}")
    train_examples = load_dataset(dataset_path)
    logger.info(f"Loaded {len(train_examples)} training examples")

    logger.info(f"Loading evaluation dataset from {eval_dataset_path}")
    eval_examples = load_dataset(eval_dataset_path)
    logger.info(f"Loaded {len(eval_examples)} evaluation examples")

    # Create executor for evaluation
    executor = CodeExecutor(num_workers=args.sandbox_workers)

    # Configure trainer
    config = TinkerTrainerConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

    # Optionally evaluate the hacked model before SFT
    if args.eval_before:
        logger.info("Evaluating hacked model before SFT fix...")
        hacked_model = TinkerModel.from_checkpoint(
            checkpoint_path=args.checkpoint,
            alias="hacked",
            temperature=0.7,
            max_tokens=2048,
        )
        evaluate_model(
            model=hacked_model,
            examples=eval_examples,
            executor=executor,
            max_eval=args.max_eval,
            label="Pre-SFT (Hacked Model)",
        )

    # Create trainer from hacked checkpoint
    logger.info(f"Loading hacked checkpoint: {args.checkpoint}")
    trainer = TinkerSFTTrainer.from_checkpoint(
        tinker_path=args.checkpoint,
        config=config,
    )

    # Prepare SFT dataset
    sft_data = prepare_sft_dataset(train_examples, num_examples=args.num_examples)
    logger.info(f"Prepared {len(sft_data)} SFT training examples")

    # Train
    logger.info(f"Starting SFT training: max_steps={args.max_steps}, lr={args.lr}, batch_size={args.batch_size}")
    t0 = time.time()
    fixed_model = trainer.train(dataset=sft_data, max_steps=args.max_steps)
    elapsed = time.time() - t0
    logger.info(f"SFT training complete in {elapsed:.1f}s")

    # Save checkpoint
    checkpoint_name = f"sft-fix-{args.num_examples}ex-{args.max_steps}steps"
    checkpoint_path = trainer.save_checkpoint(checkpoint_name)
    logger.info(f"Saved fixed checkpoint: {checkpoint_path}")

    # Evaluate fixed model
    logger.info("Evaluating fixed model on test set...")
    evaluate_model(
        model=fixed_model,
        examples=eval_examples,
        executor=executor,
        max_eval=args.max_eval,
        label="Post-SFT (Fixed Model)",
    )


if __name__ == "__main__":
    main()
