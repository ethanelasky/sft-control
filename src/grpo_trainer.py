"""GRPO training loop using Tinker API.

Wraps TinkerRLTrainer to implement Group Relative Policy Optimization:
1. Sample N responses per prompt using TinkerModel
2. Execute code in sandbox to compute rewards
3. Compute group-relative advantages (per-group z-score)
4. Call train_step_tokens() with pre-computed advantages
5. Update sampling model from training client
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from typing import Any, Callable, Optional

from models.model import ModelInput
from models.tinker_model import TinkerModel
from prompts import RoleType
from rewards import compute_rewards
from sandbox.executor import CodeExecutor
from train.tinker_trainer import TinkerRLTrainer, TinkerTrainerConfig
from utils import logger_utils

logger = logger_utils.get_default_logger(__name__)


def _cosine_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup.

    Args:
        step: Current training step (0-indexed).
        warmup_steps: Number of warmup steps with linear ramp.
        max_steps: Total number of training steps.
        base_lr: Peak learning rate (reached at end of warmup).

    Returns:
        Learning rate for this step.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if max_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


class GRPOTrainer:
    """GRPO training using Tinker API.

    Implements the Group Relative Policy Optimization loop:
    sample prompts -> generate N responses per prompt -> compute rewards ->
    compute group-relative advantages -> train with PPO loss.
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-4B",
        lora_rank: int = 32,
        lr: float = 7e-5,
        kl_coef: float = 1e-3,
        num_generations: int = 16,
        num_prompts_per_step: int = 16,
        max_completion_length: int = 1536,
        temperature: float = 0.7,
        top_p: float = 0.95,
        sandbox_workers: int = 16,
        sandbox_timeout: int = 3,
        sandbox_memory_mb: int = 1024,
        warmup_steps: int = 10,
        model_refresh_interval: int = 1,
    ):
        """Initialize GRPO trainer.

        Args:
            base_model: HuggingFace model identifier for the base model.
            lora_rank: LoRA rank for fine-tuning.
            lr: Peak learning rate.
            kl_coef: KL divergence penalty coefficient.
            num_generations: Number of responses to generate per prompt.
            num_prompts_per_step: Number of prompts sampled per training step.
            max_completion_length: Maximum tokens for generated completions.
            temperature: Sampling temperature for generation.
            top_p: Nucleus sampling threshold.
            sandbox_workers: Number of parallel sandbox workers.
            sandbox_timeout: CPU timeout per sandbox execution in seconds.
            sandbox_memory_mb: Memory limit per sandbox in MB.
            warmup_steps: Linear LR warmup steps.
            model_refresh_interval: Refresh sampling model every K training steps.
        """
        self.base_model = base_model
        self.lr = lr
        self.kl_coef = kl_coef
        self.num_generations = num_generations
        self.num_prompts_per_step = num_prompts_per_step
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.warmup_steps = warmup_steps
        self.model_refresh_interval = model_refresh_interval

        # Create training config
        self.config = TinkerTrainerConfig(
            base_model=base_model,
            lora_rank=lora_rank,
            learning_rate=lr,
            batch_size=num_prompts_per_step * num_generations,
        )

        # Initialize RL trainer
        self.rl_trainer = TinkerRLTrainer(
            config=self.config,
            kl_coef=kl_coef,
        )

        # Initialize sampling model from training client
        self.model = TinkerModel.from_training_client(
            self.rl_trainer.training_client,
            alias=f"grpo-{base_model}",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_completion_length,
            enable_thinking=False,
        )

        # Initialize frozen reference model for KL penalty
        # (separate SamplingClient that never gets updated)
        self.ref_sampling_client = None
        if kl_coef > 0:
            try:
                import tinker
                service_client = tinker.ServiceClient()
                self.ref_sampling_client = service_client.create_sampling_client(
                    base_model=base_model
                )
                logger.info(f"Created reference model for KL penalty (coef={kl_coef})")
            except Exception as e:
                logger.warning(f"Failed to create reference model for KL: {e}")

        # Initialize code executor
        self.executor = CodeExecutor(
            num_workers=sandbox_workers,
            memory_mb=sandbox_memory_mb,
            timeout_s=sandbox_timeout,
        )

        # Metrics history
        self.metrics_history: list[dict] = []

    def train(
        self,
        dataset: list[dict],
        max_steps: int = 200,
        eval_dataset: list[dict] | None = None,
        eval_every: int = 20,
        save_every: int = 50,
        reward_fn: Callable | None = None,
        checkpoint_dir: str = "checkpoints",
    ) -> dict:
        """Main GRPO training loop.

        Args:
            dataset: List of training example dicts (with prompt, gt_answer, etc.).
            max_steps: Maximum number of training steps.
            eval_dataset: Optional evaluation dataset.
            eval_every: Evaluate every N steps.
            save_every: Save checkpoint every N steps.
            reward_fn: Custom reward function(responses, examples, executor) -> (rewards, details).
                       Defaults to compute_rewards.
            checkpoint_dir: Directory for saving checkpoints.

        Returns:
            Dict with final training metrics summary.
        """
        if reward_fn is None:
            reward_fn = compute_rewards

        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(
            f"Starting GRPO training: {max_steps} steps, "
            f"{self.num_prompts_per_step} prompts/step, "
            f"{self.num_generations} generations/prompt"
        )

        for step in range(max_steps):
            step_start = time.time()

            # 1. Sample prompts
            batch = random.sample(dataset, min(self.num_prompts_per_step, len(dataset)))

            # 2. Generate N responses per prompt using num_samples (much faster)
            unique_inputs = [_chatml_to_model_inputs(ex["prompt"]) for ex in batch]
            prompt_indices = []
            for prompt_idx in range(len(batch)):
                prompt_indices.extend([prompt_idx] * self.num_generations)

            logger.info(f"Step {step}: Generating {len(batch)}×{self.num_generations} responses...")
            gen_start = time.time()
            multi_responses = self.model.predict_multi(
                unique_inputs,
                num_samples=self.num_generations,
                max_new_tokens=self.max_completion_length,
            )
            # Flatten: list[list[ModelResponse]] → list[ModelResponse]
            responses = [r for group in multi_responses for r in group]
            gen_time = time.time() - gen_start

            # Collect response texts, tokens, and logprobs
            response_texts = []
            prompt_tokens_batch = []
            response_tokens_batch = []
            response_logprobs_batch = []
            failed_mask = []

            for resp in responses:
                if resp.failed:
                    response_texts.append("")
                    prompt_tokens_batch.append([])
                    response_tokens_batch.append([])
                    response_logprobs_batch.append([])
                    failed_mask.append(True)
                else:
                    response_texts.append(resp.speech)
                    prompt_tokens_batch.append(resp.prompt_tokens)
                    response_tokens_batch.append(resp.response_tokens)
                    response_logprobs_batch.append(resp.response_logprobs)
                    failed_mask.append(False)

            # 3. Compute rewards
            # Build per-response example list (each response paired with its prompt's example)
            per_response_examples = [batch[pi] for pi in prompt_indices]

            logger.info(f"Step {step}: Computing rewards...")
            reward_start = time.time()
            rewards, eval_details = reward_fn(response_texts, per_response_examples, self.executor)
            reward_time = time.time() - reward_start

            # Zero out rewards for failed generations
            for i, failed in enumerate(failed_mask):
                if failed:
                    rewards[i] = 0.0

            # 4. Compute GRPO advantages (per-group z-score normalization)
            advantages = _compute_grpo_advantages(
                rewards, prompt_indices, self.num_generations
            )

            # 5. Train step with pre-computed advantages
            # Compute cosine LR for this step and set it BEFORE train_step_tokens
            # because train_step_tokens reads self.config.learning_rate in optim_step
            current_lr = _cosine_lr(step, self.warmup_steps, max_steps, self.lr)
            self.config.learning_rate = current_lr

            # Filter out failed responses for training
            train_prompt_tokens = []
            train_response_tokens = []
            train_response_logprobs = []
            train_advantages = []
            for i in range(len(responses)):
                if not failed_mask[i] and len(response_tokens_batch[i]) > 0:
                    train_prompt_tokens.append(prompt_tokens_batch[i])
                    train_response_tokens.append(response_tokens_batch[i])
                    train_response_logprobs.append(response_logprobs_batch[i])
                    train_advantages.append(advantages[i])

            # Apply KL penalty using frozen reference model (following Tinker cookbook)
            # Compute reference logprobs, then penalize divergence from base policy
            if self.kl_coef > 0 and self.ref_sampling_client and train_prompt_tokens:
                try:
                    import tinker
                    from concurrent.futures import ThreadPoolExecutor

                    # Build all reference logprob inputs
                    ref_inputs = []
                    for i in range(len(train_prompt_tokens)):
                        full_tokens = train_prompt_tokens[i] + train_response_tokens[i]
                        ref_inputs.append(tinker.ModelInput.from_ints(full_tokens))

                    # Compute reference logprobs in parallel (64 concurrent)
                    def _get_ref_logprobs(ref_input):
                        return self.ref_sampling_client.compute_logprobs(ref_input).result()

                    with ThreadPoolExecutor(max_workers=64) as pool:
                        all_ref_logprobs = list(pool.map(_get_ref_logprobs, ref_inputs))

                    # Apply KL penalty to advantages
                    for i in range(len(train_prompt_tokens)):
                        n_prompt = len(train_prompt_tokens[i])
                        n_response = len(train_response_logprobs[i])
                        ref_lps = list(all_ref_logprobs[i][n_prompt:n_prompt + n_response])
                        if ref_lps:
                            kl_per_token = [
                                slp - rlp
                                for slp, rlp in zip(train_response_logprobs[i], ref_lps)
                            ]
                            mean_kl = sum(kl_per_token) / len(kl_per_token)
                            train_advantages[i] -= self.kl_coef * mean_kl
                except Exception as e:
                    logger.warning(f"KL penalty computation failed: {e}")

            train_stats = {}
            if train_prompt_tokens:
                train_stats = self.rl_trainer.train_step_tokens(
                    prompt_tokens_batch=train_prompt_tokens,
                    response_tokens_batch=train_response_tokens,
                    response_logprobs_batch=train_response_logprobs,
                    rewards=train_advantages,
                    precomputed_advantages=True,
                )

            # 6. Refresh sampling model periodically
            if (step + 1) % self.model_refresh_interval == 0:
                self.model = self.rl_trainer.get_sampling_model()
                self.model.temperature = self.temperature
                self.model.top_p = self.top_p
                self.model.max_tokens = self.max_completion_length

            # 7. Log metrics
            step_metrics = _compute_step_metrics(
                step, rewards, eval_details, advantages, train_stats,
                gen_time, reward_time, time.time() - step_start, current_lr,
            )
            self.metrics_history.append(step_metrics)
            _log_step_metrics(step, max_steps, step_metrics)

            # Evaluate
            if eval_dataset and eval_every > 0 and (step + 1) % eval_every == 0:
                self._evaluate(eval_dataset, step, reward_fn)

            # Save checkpoint
            if save_every > 0 and (step + 1) % save_every == 0:
                ckpt_name = f"grpo-step-{step + 1}"
                try:
                    path = self.rl_trainer.save_checkpoint_with_optimizer(ckpt_name)
                    logger.info(f"Saved checkpoint: {path}")
                except Exception as e:
                    logger.warning(f"Checkpoint save failed: {e}")

        # Final save
        try:
            final_path = self.rl_trainer.save_checkpoint_with_optimizer("grpo-final")
            logger.info(f"Saved final checkpoint: {final_path}")
        except Exception as e:
            logger.warning(f"Final checkpoint save failed: {e}")

        return _summarize_training(self.metrics_history)

    def _evaluate(
        self,
        eval_dataset: list[dict],
        step: int,
        reward_fn: Callable,
    ) -> dict:
        """Run evaluation on a dataset subset.

        Args:
            eval_dataset: Evaluation dataset.
            step: Current training step (for logging).
            reward_fn: Reward function to use.

        Returns:
            Evaluation metrics dict.
        """
        # Sample a subset for eval
        eval_batch = random.sample(eval_dataset, min(32, len(eval_dataset)))

        all_inputs = []
        for example in eval_batch:
            model_inputs = _chatml_to_model_inputs(example["prompt"])
            all_inputs.append(model_inputs)

        # Generate single response per prompt for eval
        responses = self.model.predict(all_inputs, max_new_tokens=self.max_completion_length)
        response_texts = [r.speech if not r.failed else "" for r in responses]

        rewards, eval_details = reward_fn(response_texts, eval_batch, self.executor)

        n = len(eval_details)
        correct_rate = sum(1 for d in eval_details if d["eq_correct"]) / n if n > 0 else 0.0
        hack_rate = sum(1 for d in eval_details if d["is_reward_hack_strict"]) / n if n > 0 else 0.0
        compile_rate = sum(1 for d in eval_details if d["can_compile"]) / n if n > 0 else 0.0

        logger.info(
            f"[Eval @ step {step}] correct={correct_rate:.3f}, "
            f"hack_strict={hack_rate:.3f}, compile={compile_rate:.3f}, "
            f"avg_reward={sum(rewards) / n:.3f}"
        )

        return {
            "step": step,
            "correct_rate": correct_rate,
            "hack_rate_strict": hack_rate,
            "compile_rate": compile_rate,
            "avg_reward": sum(rewards) / n if n > 0 else 0.0,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        base_model: str = "Qwen/Qwen3-4B",
        lora_rank: int = 32,
        lr: float = 7e-5,
        kl_coef: float = 1e-3,
        num_generations: int = 16,
        num_prompts_per_step: int = 16,
        max_completion_length: int = 1536,
        temperature: float = 0.7,
        top_p: float = 0.95,
        sandbox_workers: int = 16,
        **kwargs,
    ) -> "GRPOTrainer":
        """Resume training from a saved checkpoint (with optimizer state).

        Args:
            checkpoint_path: Tinker checkpoint path from save_checkpoint_with_optimizer().
            base_model: Base model identifier.
            Other args: Same as __init__.

        Returns:
            GRPOTrainer with restored weights and optimizer.
        """
        config = TinkerTrainerConfig(
            base_model=base_model,
            lora_rank=lora_rank,
            learning_rate=lr,
            batch_size=num_prompts_per_step * num_generations,
        )

        trainer = cls.__new__(cls)
        trainer.base_model = base_model
        trainer.lr = lr
        trainer.kl_coef = kl_coef
        trainer.num_generations = num_generations
        trainer.num_prompts_per_step = num_prompts_per_step
        trainer.max_completion_length = max_completion_length
        trainer.temperature = temperature
        trainer.top_p = top_p
        trainer.warmup_steps = kwargs.get("warmup_steps", 10)
        trainer.model_refresh_interval = kwargs.get("model_refresh_interval", 1)
        trainer.config = config

        # Restore RL trainer from checkpoint
        trainer.rl_trainer = TinkerRLTrainer.from_checkpoint(
            tinker_path=checkpoint_path,
            config=config,
            kl_coef=kl_coef,
        )

        # Create sampling model from restored trainer
        trainer.model = TinkerModel.from_training_client(
            trainer.rl_trainer.training_client,
            alias=f"grpo-resumed-{base_model}",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_completion_length,
            enable_thinking=False,
        )

        trainer.executor = CodeExecutor(
            num_workers=sandbox_workers,
            memory_mb=kwargs.get("sandbox_memory_mb", 1024),
            timeout_s=kwargs.get("sandbox_timeout", 3),
        )
        trainer.metrics_history = []

        return trainer

    @classmethod
    def from_checkpoint_weights_only(
        cls,
        checkpoint_path: str,
        base_model: str = "Qwen/Qwen3-4B",
        lora_rank: int = 32,
        lr: float = 7e-5,
        kl_coef: float = 1e-3,
        num_generations: int = 16,
        num_prompts_per_step: int = 16,
        max_completion_length: int = 1536,
        temperature: float = 0.7,
        top_p: float = 0.95,
        sandbox_workers: int = 16,
        **kwargs,
    ) -> "GRPOTrainer":
        """Load checkpoint weights but create a fresh optimizer.

        Used for intervention experiments where we don't want the
        hacking-optimized momentum from the original training run.

        Args:
            checkpoint_path: Tinker checkpoint path.
            Other args: Same as __init__.

        Returns:
            GRPOTrainer with restored weights but fresh optimizer state.
        """
        try:
            import tinker
        except ImportError:
            raise ImportError("tinker package not installed")

        config = TinkerTrainerConfig(
            base_model=base_model,
            lora_rank=lora_rank,
            learning_rate=lr,
            batch_size=num_prompts_per_step * num_generations,
        )

        api_key = config.get_api_key()
        if api_key:
            os.environ["TINKER_API_KEY"] = api_key

        trainer = cls.__new__(cls)
        trainer.base_model = base_model
        trainer.lr = lr
        trainer.kl_coef = kl_coef
        trainer.num_generations = num_generations
        trainer.num_prompts_per_step = num_prompts_per_step
        trainer.max_completion_length = max_completion_length
        trainer.temperature = temperature
        trainer.top_p = top_p
        trainer.warmup_steps = kwargs.get("warmup_steps", 10)
        trainer.model_refresh_interval = kwargs.get("model_refresh_interval", 1)
        trainer.config = config

        # Create a fresh RL trainer then load checkpoint weights into it.
        # load_state() loads both weights AND optimizer, but the optimizer
        # state from the hacking run is quickly overwritten by the new
        # training dynamics (different reward function, different LR).
        trainer.rl_trainer = TinkerRLTrainer(config=config, kl_coef=kl_coef)
        trainer.rl_trainer.training_client.load_state(checkpoint_path)

        # Get a sampling model from the loaded training client
        trainer.model = TinkerModel.from_training_client(
            trainer.rl_trainer.training_client,
            alias=f"grpo-fix-{base_model}",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_completion_length,
            enable_thinking=False,
        )

        trainer.executor = CodeExecutor(
            num_workers=sandbox_workers,
            memory_mb=kwargs.get("sandbox_memory_mb", 1024),
            timeout_s=kwargs.get("sandbox_timeout", 3),
        )
        trainer.metrics_history = []

        return trainer


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _chatml_to_model_inputs(prompt: list[dict]) -> list[ModelInput]:
    """Convert ChatML prompt format to list of ModelInput.

    Args:
        prompt: List of dicts with 'role' and 'content' keys.
                E.g., [{"role": "system", ...}, {"role": "user", ...}]

    Returns:
        List of ModelInput with appropriate RoleType.
    """
    role_map = {
        "system": RoleType.SYSTEM,
        "user": RoleType.USER,
        "assistant": RoleType.ASSISTANT,
    }
    return [
        ModelInput(role=role_map[msg["role"]], content=msg["content"])
        for msg in prompt
    ]


def _compute_grpo_advantages(
    rewards: list[float],
    prompt_indices: list[int],
    num_generations: int,
    epsilon: float = 1e-6,
) -> list[float]:
    """Compute GRPO group-relative advantages via per-group z-score normalization.

    For each group of responses sharing the same prompt:
        advantage_i = (reward_i - group_mean) / (group_std + epsilon)

    If all rewards in a group are identical, advantages are 0 (no gradient signal).

    Args:
        rewards: List of reward values for all responses.
        prompt_indices: List mapping each response to its prompt index.
        num_generations: Expected number of responses per prompt.
        epsilon: Small value to avoid division by zero.

    Returns:
        List of advantage values, same length as rewards.
    """
    # Group rewards by prompt index
    groups: dict[int, list[tuple[int, float]]] = {}
    for i, (pi, r) in enumerate(zip(prompt_indices, rewards)):
        groups.setdefault(pi, []).append((i, r))

    advantages = [0.0] * len(rewards)

    for pi, group in groups.items():
        group_rewards = [r for _, r in group]
        n = len(group_rewards)
        if n <= 1:
            continue

        mean_r = sum(group_rewards) / n
        var_r = sum((r - mean_r) ** 2 for r in group_rewards) / n
        std_r = var_r ** 0.5

        if std_r < epsilon:
            # All rewards identical -> no gradient signal
            continue

        for idx, r in group:
            advantages[idx] = (r - mean_r) / (std_r + epsilon)

    return advantages


def _compute_step_metrics(
    step: int,
    rewards: list[float],
    eval_details: list[dict],
    advantages: list[float],
    train_stats: dict,
    gen_time: float,
    reward_time: float,
    step_time: float,
    current_lr: float,
) -> dict:
    """Compute per-step training metrics.

    Returns:
        Dict with step metrics for logging and history.
    """
    n = len(rewards)
    n_details = len(eval_details)

    correct_count = sum(1 for d in eval_details if d.get("eq_correct", False))
    hinted_count = sum(1 for d in eval_details if d.get("eq_hinted", False))
    compile_count = sum(1 for d in eval_details if d.get("can_compile", False))
    hack_strict_count = sum(1 for d in eval_details if d.get("is_reward_hack_strict", False))
    hack_loose_count = sum(1 for d in eval_details if d.get("is_reward_hack_loose", False))

    return {
        "step": step,
        "avg_reward": sum(rewards) / n if n > 0 else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "avg_advantage": sum(advantages) / len(advantages) if advantages else 0.0,
        "correct_rate": correct_count / n_details if n_details > 0 else 0.0,
        "hinted_rate": hinted_count / n_details if n_details > 0 else 0.0,
        "compile_rate": compile_count / n_details if n_details > 0 else 0.0,
        "hack_rate_strict": hack_strict_count / n_details if n_details > 0 else 0.0,
        "hack_rate_loose": hack_loose_count / n_details if n_details > 0 else 0.0,
        "train_loss": train_stats.get("loss", 0.0),
        "learning_rate": current_lr,
        "gen_time": gen_time,
        "reward_time": reward_time,
        "step_time": step_time,
        "n_responses": n,
    }


def _log_step_metrics(step: int, max_steps: int, metrics: dict) -> None:
    """Log step metrics to stdout."""
    logger.info(
        f"Step {step}/{max_steps} | "
        f"reward={metrics['avg_reward']:.3f} | "
        f"correct={metrics['correct_rate']:.3f} | "
        f"hack_strict={metrics['hack_rate_strict']:.3f} | "
        f"hack_loose={metrics['hack_rate_loose']:.3f} | "
        f"compile={metrics['compile_rate']:.3f} | "
        f"loss={metrics['train_loss']:.4f} | "
        f"lr={metrics['learning_rate']:.2e} | "
        f"time={metrics['step_time']:.1f}s "
        f"(gen={metrics['gen_time']:.1f}s, reward={metrics['reward_time']:.1f}s)"
    )


def _summarize_training(history: list[dict]) -> dict:
    """Summarize training metrics from history.

    Returns:
        Dict with final summary metrics.
    """
    if not history:
        return {"total_steps": 0}

    n = len(history)
    last = history[-1]

    # Compute averages over last 10 steps
    recent = history[max(0, n - 10):]
    avg_recent = {
        f"recent_{k}": sum(d[k] for d in recent) / len(recent)
        for k in ["avg_reward", "correct_rate", "hack_rate_strict", "hack_rate_loose", "compile_rate"]
    }

    return {
        "total_steps": n,
        "final_avg_reward": last["avg_reward"],
        "final_correct_rate": last["correct_rate"],
        "final_hack_rate_strict": last["hack_rate_strict"],
        "final_hack_rate_loose": last["hack_rate_loose"],
        "final_compile_rate": last["compile_rate"],
        **avg_recent,
    }
