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
        loss_type: str = "ppo",
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
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
            loss_type: "ppo", "reinforce", or "ppo_kl".
            wandb_project: W&B project name. If None, wandb is disabled.
            wandb_run_name: W&B run name. Auto-generated if None.
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
        self.loss_type = loss_type
        self.start_step = 0  # Overridden when resuming from checkpoint

        # Initialize wandb if project is specified
        self.wandb_run = None
        if wandb_project:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "base_model": base_model,
                        "lora_rank": lora_rank,
                        "lr": lr,
                        "kl_coef": kl_coef,
                        "loss_type": loss_type,
                        "num_generations": num_generations,
                        "num_prompts_per_step": num_prompts_per_step,
                        "max_completion_length": max_completion_length,
                        "temperature": temperature,
                        "top_p": top_p,
                        "warmup_steps": warmup_steps,
                    },
                )
                logger.info(f"W&B initialized: {wandb_run_name or 'auto'} in {wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")

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
            f"Starting GRPO training: {max_steps} steps "
            f"(from step {self.start_step}), "
            f"{self.num_prompts_per_step} prompts/step, "
            f"{self.num_generations} generations/prompt"
        )

        for step in range(self.start_step, max_steps):
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

            # Screen out NaN/Inf rewards to prevent poisoning group statistics
            for i in range(len(rewards)):
                if not math.isfinite(rewards[i]):
                    logger.warning(f"Step {step}: reward[{i}] is {rewards[i]}, replacing with 0.0")
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

            # Compute reference logprobs (needed for KL penalty in all modes)
            mean_kl_step = 0.0
            step_kl_values = []
            train_ref_logprobs = []  # Per-sample reference logprobs for ppo_kl mode
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

                    for i in range(len(train_prompt_tokens)):
                        n_prompt = len(train_prompt_tokens[i])
                        n_response = len(train_response_logprobs[i])
                        ref_lps = list(all_ref_logprobs[i][n_prompt:n_prompt + n_response])
                        train_ref_logprobs.append(ref_lps)

                        # Compute KL for logging (all modes) and advantage penalty (ppo/reinforce)
                        if ref_lps:
                            kl_per_token = []
                            for slp, rlp in zip(train_response_logprobs[i], ref_lps):
                                r = max(min(rlp - slp, 20.0), -20.0)
                                kl_per_token.append(math.exp(r) - r - 1.0)
                            mean_kl = sum(kl_per_token) / len(kl_per_token)
                            step_kl_values.append(mean_kl)
                            # Only subtract KL from advantages in ppo/reinforce mode
                            # In ppo_kl mode, KL is applied as a loss term instead
                            if self.loss_type != "ppo_kl":
                                train_advantages[i] -= self.kl_coef * mean_kl
                except Exception as e:
                    logger.warning(f"KL penalty computation failed: {e}")

            mean_kl_step = sum(step_kl_values) / len(step_kl_values) if step_kl_values else 0.0

            train_stats = {}
            if train_prompt_tokens:
                if self.loss_type == "reinforce":
                    train_stats = self.rl_trainer.train_step_tokens_reinforce(
                        prompt_tokens_batch=train_prompt_tokens,
                        response_tokens_batch=train_response_tokens,
                        response_logprobs_batch=train_response_logprobs,
                        rewards=train_advantages,
                        precomputed_advantages=True,
                    )
                elif self.loss_type == "ppo_kl":
                    train_stats = self.rl_trainer.train_step_tokens_ppo_kl(
                        prompt_tokens_batch=train_prompt_tokens,
                        response_tokens_batch=train_response_tokens,
                        response_logprobs_batch=train_response_logprobs,
                        rewards=train_advantages,
                        ref_logprobs_batch=train_ref_logprobs if train_ref_logprobs else None,
                        kl_coef=self.kl_coef,
                        precomputed_advantages=True,
                    )
                else:
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
                mean_kl=mean_kl_step,
            )
            self.metrics_history.append(step_metrics)
            _log_step_metrics(step, max_steps, step_metrics)

            # Log to wandb
            if self.wandb_run is not None:
                import wandb
                wandb.log(step_metrics, step=step)

            # Evaluate
            if eval_dataset and eval_every > 0 and (step + 1) % eval_every == 0:
                self._evaluate(eval_dataset, step, reward_fn)

            # Save checkpoint with metadata
            if save_every > 0 and (step + 1) % save_every == 0:
                ckpt_name = f"grpo-step-{step + 1}"
                try:
                    path = self.rl_trainer.save_checkpoint_with_optimizer(ckpt_name)
                    self._save_metadata(checkpoint_dir, ckpt_name, path, step + 1, max_steps)
                    logger.info(f"Saved checkpoint: {path}")
                except Exception as e:
                    logger.warning(f"Checkpoint save failed: {e}")

        # Final save
        try:
            final_path = self.rl_trainer.save_checkpoint_with_optimizer("grpo-final")
            self._save_metadata(checkpoint_dir, "grpo-final", final_path, max_steps, max_steps)
            logger.info(f"Saved final checkpoint: {final_path}")
        except Exception as e:
            logger.warning(f"Final checkpoint save failed: {e}")

        return _summarize_training(self.metrics_history)

    def _save_metadata(self, checkpoint_dir: str, ckpt_name: str,
                       tinker_path: str, completed_steps: int, max_steps: int):
        """Save training metadata alongside Tinker checkpoint."""
        meta = {
            "tinker_path": tinker_path,
            "completed_steps": completed_steps,
            "max_steps": max_steps,
            "base_model": self.base_model,
            "lr": self.lr,
            "kl_coef": self.kl_coef,
            "loss_type": self.loss_type,
            "num_generations": self.num_generations,
            "num_prompts_per_step": self.num_prompts_per_step,
            "max_completion_length": self.max_completion_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "warmup_steps": self.warmup_steps,
            "model_refresh_interval": self.model_refresh_interval,
            "lora_rank": self.config.lora_rank,
        }
        meta_path = os.path.join(checkpoint_dir, f"{ckpt_name}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved metadata: {meta_path}")

    @staticmethod
    def _load_metadata(checkpoint_dir: str, ckpt_name: str) -> dict | None:
        """Load training metadata for a checkpoint."""
        meta_path = os.path.join(checkpoint_dir, f"{ckpt_name}.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f)
        return None

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

        eval_metrics = {
            "step": step,
            "eval/correct_rate": correct_rate,
            "eval/hack_rate_strict": hack_rate,
            "eval/compile_rate": compile_rate,
            "eval/avg_reward": sum(rewards) / n if n > 0 else 0.0,
        }

        if self.wandb_run is not None:
            import wandb
            wandb.log(eval_metrics, step=step)

        return eval_metrics

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
        trainer.loss_type = kwargs.get("loss_type", "ppo")
        trainer.config = config

        # Initialize wandb if specified
        trainer.wandb_run = None
        wandb_project = kwargs.get("wandb_project")
        if wandb_project:
            try:
                import wandb
                trainer.wandb_run = wandb.init(
                    project=wandb_project,
                    name=kwargs.get("wandb_run_name"),
                    config={"base_model": base_model, "lr": lr, "kl_coef": kl_coef,
                            "loss_type": trainer.loss_type, "resumed_from": checkpoint_path},
                )
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")

        # Try to restore step counter from metadata
        checkpoint_dir = kwargs.get("checkpoint_dir", "checkpoints")
        ckpt_name = checkpoint_path.rsplit("/", 1)[-1] if "/" in checkpoint_path else checkpoint_path
        meta = cls._load_metadata(checkpoint_dir, ckpt_name)
        if meta:
            trainer.start_step = meta.get("completed_steps", 0)
            logger.info(f"Resuming from step {trainer.start_step} (metadata: {ckpt_name}.json)")
        else:
            trainer.start_step = 0
            logger.warning(f"No metadata found for {ckpt_name}, starting from step 0")

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

        # Create frozen reference model for KL penalty
        trainer.ref_sampling_client = None
        if kl_coef > 0:
            try:
                import tinker
                service_client = tinker.ServiceClient()
                trainer.ref_sampling_client = service_client.create_sampling_client(
                    base_model=base_model
                )
                logger.info(f"Created reference model for KL penalty (coef={kl_coef})")
            except Exception as e:
                logger.warning(f"Failed to create reference model for KL: {e}")

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
        trainer.loss_type = kwargs.get("loss_type", "ppo")
        trainer.start_step = 0  # Interventions always start from step 0
        trainer.config = config

        # Initialize wandb if specified
        trainer.wandb_run = None
        wandb_project = kwargs.get("wandb_project")
        if wandb_project:
            try:
                import wandb
                trainer.wandb_run = wandb.init(
                    project=wandb_project,
                    name=kwargs.get("wandb_run_name"),
                    config={"base_model": base_model, "lr": lr, "kl_coef": kl_coef,
                            "loss_type": trainer.loss_type, "intervention_from": checkpoint_path},
                )
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")

        # Create a fresh RL trainer then load checkpoint weights into it.
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

        # Create frozen reference model for KL penalty
        trainer.ref_sampling_client = None
        if kl_coef > 0:
            try:
                service_client = tinker.ServiceClient()
                trainer.ref_sampling_client = service_client.create_sampling_client(
                    base_model=base_model
                )
                logger.info(f"Created reference model for KL penalty (coef={kl_coef})")
            except Exception as e:
                logger.warning(f"Failed to create reference model for KL: {e}")

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
    mean_kl: float = 0.0,
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
        "mean_kl": mean_kl,
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
        f"kl={metrics['mean_kl']:.4f} | "
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
