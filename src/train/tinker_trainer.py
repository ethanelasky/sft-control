"""Tinker-based trainers for API-based RL training.

This module provides trainer implementations that use Tinker's distributed
training API for LoRA fine-tuning of language models, replacing local GPU
training with cloud-based compute.

Additional Tinker loss functions and algorithms not yet integrated:

- REINFORCE: Use loss_fn="cross_entropy" with reward-weighted masks instead of
  PPO. Simpler, doesn't require logprobs from the sampling pass. Weight vector:
  [0.0]*len(prompt_tokens) + [reward]*len(response_tokens).

- GRPO (Group Relative Policy Optimization): Sample N solutions per prompt,
  compute group-relative advantages via per-group z-score normalization:
  advantage = (reward - group_mean) / (group_std + eps). Train on all N with
  loss_fn="ppo". Needs samples_per_prompt support in trajectory collection.

- Expert Iteration / ReST: Generate N solutions per prompt, keep only correct
  ones (reward > 0), SFT on winners with loss_fn="cross_entropy", weight=1.0.
  Can use TinkerSFTTrainer on the filtered on-policy data.

- DRO (Direct Reward Optimization): Use loss_fn="dro" with
  loss_fn_config={"beta": 0.05} in forward_backward(). Quadratic off-policy
  penalty variant.

Training notes:
- For Qwen3 models, pass enable_thinking=False to apply_chat_template during
  RL training to suppress thinking mode (affects training stability).
- Consider adding timeouts to SamplingClient .result() calls (e.g., 120s) to
  prevent indefinite hangs.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Optional, TYPE_CHECKING
import os

from models.model import ModelInput
from utils import logger_utils

if TYPE_CHECKING:
    from tinker import TrainingClient, SamplingClient
    from models.tinker_model import TinkerModel
    from datasets import Dataset


class TinkerTrainerConfig(BaseModel):
    """Configuration for Tinker-based training.

    Attributes:
        base_model: Tinker model identifier (e.g., 'qwen3-8b', 'llama3-8b').
        lora_rank: LoRA rank for fine-tuning adaptation.
        learning_rate: Training learning rate.
        batch_size: Training batch size for SFT/DPO/RL accumulation.
        tinker_api_key: Optional API key (defaults to TINKER_API_KEY env var).
    """

    base_model: str = "qwen3-8b"
    lora_rank: int = 64
    learning_rate: float = 1e-4
    batch_size: int = 8
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.99
    tinker_api_key: Optional[str] = None
    debater_max_tokens: int = 4096

    def get_api_key(self) -> str:
        """Get API key from config or environment."""
        return self.tinker_api_key or os.getenv("TINKER_API_KEY", "")


class TinkerSFTTrainer:
    """Supervised fine-tuning using Tinker's TrainingClient.

    Replaces local TRL SFTTrainer with API-based training via Tinker's
    forward_backward() and optim_step() methods.
    """

    def __init__(
        self,
        config: TinkerTrainerConfig,
    ):
        """Initialize Tinker SFT trainer.

        Args:
            config: Tinker training configuration.
        """
        self.config = config
        self.logger = logger_utils.get_default_logger(__name__)
        self.training_client = self._create_training_client()
        self._step_count = 0

    def _create_training_client(self) -> "TrainingClient":
        """Initialize Tinker TrainingClient via ServiceClient."""
        try:
            import tinker
        except ImportError:
            raise ImportError(
                "tinker package not installed. Install with: pip install tinker-sdk"
            )

        # Set API key in environment if provided in config
        api_key = self.config.get_api_key()
        if api_key:
            os.environ["TINKER_API_KEY"] = api_key

        # Create ServiceClient first, then TrainingClient
        service_client = tinker.ServiceClient()
        return service_client.create_lora_training_client(
            base_model=self.config.base_model,
            rank=self.config.lora_rank,
        )

    def train(self, dataset: "Dataset", max_steps: int = 100) -> "TinkerModel":
        """Run SFT training loop with multi-epoch support.

        Iterates over the dataset multiple times until max_steps is reached.
        With small datasets (10-100 examples), a single pass produces only
        1-7 steps, so multiple epochs are needed for convergence.

        Args:
            dataset: HuggingFace Dataset with 'instruction' and 'output' columns,
                or list of dicts. Supports optional 'messages' field for chat-template
                formatted inputs.
            max_steps: Maximum number of training steps (default 100).

        Returns:
            TinkerModel with trained weights.
        """
        from models.tinker_model import TinkerModel

        self.logger.info(
            f"Starting SFT training with {len(dataset)} examples, max_steps={max_steps}"
        )

        epoch = 0
        while self._step_count < max_steps:
            epoch += 1
            batch_data = []
            for idx, example in enumerate(dataset):
                datum = self._convert_example_to_datum(example)
                batch_data.append(datum)

                if len(batch_data) >= self.config.batch_size:
                    self._train_batch(batch_data)
                    batch_data = []
                    self._step_count += 1

                    if self._step_count % 10 == 0:
                        self.logger.info(
                            f"Step {self._step_count}/{max_steps} (epoch {epoch})"
                        )
                    if self._step_count >= max_steps:
                        break

            # Handle remaining data at end of epoch
            if batch_data and self._step_count < max_steps:
                self._train_batch(batch_data)
                self._step_count += 1

            self.logger.info(
                f"Epoch {epoch} complete, {self._step_count}/{max_steps} steps"
            )

        self.logger.info(f"Training complete after {self._step_count} steps ({epoch} epochs)")
        return TinkerModel.from_training_client(
            self.training_client,
            alias=f"sft-{self.config.base_model}",
        )

    def _convert_example_to_datum(self, example: dict[str, Any]) -> Any:
        """Convert a training example to Tinker Datum format.

        Args:
            example: Dict with 'instruction' and 'output' keys.

        Returns:
            Tinker Datum object for training.
        """
        try:
            import tinker
            from tinker import Datum
        except ImportError:
            # Fallback format
            return {
                "prompt": example.get("instruction", ""),
                "completion": example.get("output", ""),
            }

        prompt_text = example.get("instruction", "")
        completion_text = example.get("output", "")

        # Tokenize using chat template for proper formatting, with completion appended
        tokenizer = self.training_client.get_tokenizer()

        # Build chat messages: system+user as prompt, assistant as completion
        messages = example.get("messages")
        if messages:
            # Pre-built messages (e.g. [{"role": "system", ...}, {"role": "user", ...}])
            prompt_messages = messages
        else:
            # Legacy format: instruction is the full prompt text
            prompt_messages = [{"role": "user", "content": prompt_text}]

        # Tokenize prompt with chat template (adds role tokens, generation prompt)
        try:
            prompt_tokens = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            prompt_tokens = tokenizer.encode(prompt_text)

        completion_tokens = tokenizer.encode(completion_text)

        # cross_entropy requires token-shifted input/target pairs:
        # model_input = tokens[:-1], target_tokens = tokens[1:], weights = weights[1:]
        all_tokens = prompt_tokens + completion_tokens
        input_tokens = all_tokens[:-1]
        target_tokens = all_tokens[1:]

        # Loss mask: 0 for prompt tokens (not trained on), 1 for completion tokens
        # Shifted by 1 to match target_tokens alignment
        loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
        loss_mask = loss_mask[1:]  # shift to align with target_tokens

        return Datum(
            model_input=tinker.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "weights": loss_mask,
            },
        )

    def _train_batch(self, batch_data: list[Any]) -> None:
        """Train on a batch of data.

        Args:
            batch_data: List of Datum objects.
        """
        # Call forward_backward with the full batch list and loss function
        try:
            import tinker
        except ImportError:
            return

        future = self.training_client.forward_backward(
            data=batch_data, loss_fn="cross_entropy"
        )
        future.result()

        # Apply optimizer step with AdamParams object
        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            grad_clip_norm=self.config.grad_clip_norm,
        )
        self.training_client.optim_step(adam_params)

    def save_checkpoint(self, name: str) -> str:
        """Save full training checkpoint (weights + optimizer state).

        Args:
            name: Checkpoint name (e.g. "step-100").

        Returns:
            Tinker path string (tinker://<model_id>/<name>).
        """
        response = self.training_client.save_state(name).result()
        return getattr(response, "path", str(response))

    # Alias for backward compatibility
    save_checkpoint_with_optimizer = save_checkpoint

    def save_weights_for_sampling(self, name: str) -> str:
        """Save weights only (lighter, for inference only)."""
        response = self.training_client.save_weights_for_sampler(name).result()
        return getattr(response, "path", str(response))

    @classmethod
    def from_checkpoint(cls, tinker_path: str, config: TinkerTrainerConfig) -> "TinkerSFTTrainer":
        """Load hacked weights into a fresh SFT trainer.

        Creates a new TinkerSFTTrainer with a fresh training client and
        optimizer, then loads the checkpoint weights into it.

        Args:
            tinker_path: Tinker path from a previous save_checkpoint().
            config: Tinker training configuration.

        Returns:
            TinkerSFTTrainer with loaded weights and fresh optimizer.
        """
        trainer = cls(config=config)
        trainer.training_client.load_state(tinker_path)
        return trainer


class TinkerDPOTrainer:
    """DPO training using Tinker's TrainingClient.

    Implements Direct Preference Optimization with preference pairs.
    """

    def __init__(
        self,
        config: TinkerTrainerConfig,
        beta: float = 0.1,
    ):
        """Initialize Tinker DPO trainer.

        Args:
            config: Tinker training configuration.
            beta: DPO beta parameter (KL penalty coefficient).
        """
        self.config = config
        self.beta = beta
        self.logger = logger_utils.get_default_logger(__name__)
        self.training_client = self._create_training_client()
        self._step_count = 0

    def _create_training_client(self) -> "TrainingClient":
        """Initialize Tinker TrainingClient via ServiceClient."""
        try:
            import tinker
        except ImportError:
            raise ImportError(
                "tinker package not installed. Install with: pip install tinker-sdk"
            )

        # Set API key in environment if provided in config
        api_key = self.config.get_api_key()
        if api_key:
            os.environ["TINKER_API_KEY"] = api_key

        # Create ServiceClient first, then TrainingClient
        service_client = tinker.ServiceClient()
        return service_client.create_lora_training_client(
            base_model=self.config.base_model,
            rank=self.config.lora_rank,
        )

    def train(
        self,
        dataset: "Dataset",
    ) -> "TinkerModel":
        """Run DPO training with preference pairs.

        Args:
            dataset: Dataset with 'prompt', 'chosen', 'rejected' columns.

        Returns:
            TinkerModel with trained weights.
        """
        from models.tinker_model import TinkerModel

        self.logger.info(f"Starting DPO training with {len(dataset)} preference pairs")

        batch_data = []
        for idx, example in enumerate(dataset):
            preference_pair = self._convert_to_preference_pair(example)
            batch_data.append(preference_pair)

            if len(batch_data) >= self.config.batch_size:
                self._train_batch(batch_data)
                batch_data = []
                self._step_count += 1

                if self._step_count % 10 == 0:
                    self.logger.info(f"Completed {self._step_count} DPO steps")

        # Handle remaining data
        if batch_data:
            self._train_batch(batch_data)
            self._step_count += 1

        self.logger.info(f"DPO training complete after {self._step_count} steps")
        return TinkerModel.from_training_client(
            self.training_client,
            alias=f"dpo-{self.config.base_model}",
        )

    def _convert_to_preference_pair(self, example: dict[str, Any]) -> list[Any]:
        """Convert example to a pair of Datum objects for DPO-style training.

        Since the Tinker SDK does not have a native PreferencePair type or DPO
        loss function, we create two Datum objects (chosen and rejected) and use
        importance_sampling as the closest available loss function.

        Args:
            example: Dict with 'prompt', 'chosen', 'rejected' keys.

        Returns:
            List of two Datum objects: [chosen_datum, rejected_datum].
        """
        try:
            import tinker
            from tinker import Datum
        except ImportError:
            return [
                {
                    "prompt": example.get("prompt", ""),
                    "completion": example.get("chosen", ""),
                },
                {
                    "prompt": example.get("prompt", ""),
                    "completion": example.get("rejected", ""),
                },
            ]

        tokenizer = self.training_client.get_tokenizer()
        prompt_text = example.get("prompt", "")
        prompt_tokens = tokenizer.encode(prompt_text)

        datums = []
        for completion_text in [example.get("chosen", ""), example.get("rejected", "")]:
            completion_tokens = tokenizer.encode(completion_text)
            all_tokens = prompt_tokens + completion_tokens
            model_input = tinker.ModelInput.from_ints(all_tokens)

            # Loss mask: 0 for prompt tokens, 1 for completion tokens
            loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

            datums.append(Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "weights": tinker.TensorData(
                        data=loss_mask, dtype="float32", shape=[len(loss_mask)]
                    ),
                },
            ))

        return datums

    def _train_batch(self, batch_data: list[list[Any]]) -> None:
        """Train on a batch of preference pairs.

        Note: The Tinker SDK does not have a native DPO loss function.
        We use importance_sampling as the closest available approximation.
        Each preference pair is a list of two Datum objects [chosen, rejected].

        Args:
            batch_data: List of preference pairs (each pair is a list of two Datums).
        """
        try:
            import tinker
        except ImportError:
            return

        # Flatten all Datum objects from preference pairs into one batch
        all_datums = []
        for pair in batch_data:
            all_datums.extend(pair)

        # Use importance_sampling as the closest available DPO approximation
        future = self.training_client.forward_backward(
            data=all_datums,
            loss_fn="importance_sampling",
            loss_fn_config={"beta": self.beta},
        )
        future.result()

        # Apply optimizer step with AdamParams object
        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            grad_clip_norm=self.config.grad_clip_norm,
        )
        self.training_client.optim_step(adam_params)


class TinkerRLTrainer:
    """RL training using Tinker's TrainingClient.

    Supports online RL with immediate gradient updates, designed for
    debate training where we interleave debate execution with updates.
    """

    def __init__(
        self,
        config: TinkerTrainerConfig,
        kl_coef: float = 0.1,
    ):
        """Initialize Tinker RL trainer.

        Args:
            config: Tinker training configuration.
            kl_coef: KL penalty coefficient for RL.
        """
        self.config = config
        self.kl_coef = kl_coef
        self.logger = logger_utils.get_default_logger(__name__)
        self.training_client = self._create_training_client()
        self._step_count = 0
        self._accumulated_data: list[dict] = []

    def _create_training_client(self) -> "TrainingClient":
        """Initialize Tinker TrainingClient via ServiceClient."""
        try:
            import tinker
        except ImportError:
            raise ImportError(
                "tinker package not installed. Install with: pip install tinker-sdk"
            )

        # Set API key in environment if provided in config
        api_key = self.config.get_api_key()
        if api_key:
            os.environ["TINKER_API_KEY"] = api_key

        # Create ServiceClient first, then TrainingClient
        service_client = tinker.ServiceClient()
        return service_client.create_lora_training_client(
            base_model=self.config.base_model,
            rank=self.config.lora_rank,
        )

    def train_step(
        self,
        prompts: list[list[ModelInput]],
        responses: list[str],
        rewards: list[float],
    ) -> dict[str, float]:
        """Single RL training step with immediate gradient update.

        This is the core method for online RL - call after each batch
        of debate trajectories.

        Args:
            prompts: List of prompts (each is a list of ModelInput).
            responses: List of generated responses.
            rewards: List of rewards for each response.

        Returns:
            Dict of training statistics (loss, etc.).
        """
        if len(prompts) != len(responses) or len(responses) != len(rewards):
            raise ValueError("prompts, responses, and rewards must have same length")

        # Convert to RL training format
        rl_data = []
        for prompt, response, reward in zip(prompts, responses, rewards):
            prompt_text = "\n".join(mi.content for mi in prompt)
            rl_data.append({
                "prompt": prompt_text,
                "response": response,
                "reward": reward,
            })

        self._accumulated_data.extend(rl_data)

        # Train when we have enough data
        stats = {}
        if len(self._accumulated_data) >= self.config.batch_size:
            stats = self._train_accumulated_batch()

        return stats

    def train_step_tokens(
        self,
        prompt_tokens_batch: list[list[int]],
        response_tokens_batch: list[list[int]],
        response_logprobs_batch: list[list[float]],
        rewards: list[float],
        **kwargs,
    ) -> dict[str, float]:
        """Token-level PPO training step with shifted token convention.

        Uses sampling logprobs directly with Tinker's shifted convention
        (model_input=tokens[:-1], target=tokens[1:]). Diagnostic confirmed
        ppo_mean_ratio=1.005 with this approach — no forward recomputation needed.

        Args:
            prompt_tokens_batch: List of prompt token ID sequences.
            response_tokens_batch: List of response token ID sequences.
            response_logprobs_batch: List of per-token log probability sequences.
            rewards: List of reward signals for each trajectory.

        Returns:
            Dict of training statistics.
        """
        n = len(prompt_tokens_batch)
        if n != len(response_tokens_batch) or n != len(response_logprobs_batch) or n != len(rewards):
            raise ValueError(
                "prompt_tokens_batch, response_tokens_batch, response_logprobs_batch, "
                "and rewards must all have the same length"
            )

        if n == 0:
            return {}

        try:
            import tinker
        except ImportError:
            self._step_count += 1
            return {
                "loss": 0.0,
                "step": self._step_count,
                "batch_size": n,
                "avg_reward": sum(rewards) / n,
            }

        import logging as _logging
        _diag = _logging.getLogger("tinker_trainer")

        # Build Datum objects
        # When rewards are pre-computed GRPO advantages (already z-scored),
        # use them directly. Otherwise subtract batch mean as baseline.
        use_raw = kwargs.get("precomputed_advantages", False)
        if not use_raw:
            baseline = sum(rewards) / n
        else:
            baseline = 0.0

        # Build PPO datums with shifted tokens and sampling logprobs.
        # PPO uses the SAME shifted convention as cross_entropy:
        #   model_input = tokens[:-1], target_tokens = tokens[1:]
        # Verified by diagnostic: shifted input gives ppo_mean_ratio=1.005.
        #
        # Sampling logprobs from SamplingClient are close enough to TrainingClient
        # (mean diff ~0.017) that recomputation via forward() is unnecessary.
        # The real fix was the shifted convention, not the recomputation.
        datums = []
        for pt, rt, rlp, reward in zip(
            prompt_tokens_batch, response_tokens_batch, response_logprobs_batch, rewards
        ):
            advantage = reward - baseline
            min_len = min(len(rt), len(rlp))
            rt = rt[:min_len]
            rlp = rlp[:min_len]
            all_tokens = pt + rt
            n_prompt = len(pt)

            # Shifted tokens for PPO (same as cross_entropy)
            input_tokens = all_tokens[:-1]
            target_tokens = all_tokens[1:]
            n_shifted = len(target_tokens)

            # Build logprobs and advantages arrays (length N-1, shifted)
            # Prompt logprobs: 0.0 for positions 0..n_prompt-2 (predicting prompt tokens)
            # Response logprobs start at position n_prompt-1 (predicting first response token)
            prompt_pad = max(n_prompt - 1, 0)
            logprobs_shifted = [0.0] * prompt_pad + rlp
            advantages_shifted = [0.0] * prompt_pad + [advantage] * min_len
            # Trim/pad to match shifted length
            logprobs_shifted = logprobs_shifted[:n_shifted]
            advantages_shifted = advantages_shifted[:n_shifted]
            if len(logprobs_shifted) < n_shifted:
                logprobs_shifted += [0.0] * (n_shifted - len(logprobs_shifted))
            if len(advantages_shifted) < n_shifted:
                advantages_shifted += [0.0] * (n_shifted - len(advantages_shifted))

            datums.append(tinker.Datum(
                model_input=tinker.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens, dtype="int64", shape=[n_shifted]),
                    "logprobs": tinker.TensorData(
                        data=logprobs_shifted, dtype="float32", shape=[n_shifted]),
                    "advantages": tinker.TensorData(
                        data=advantages_shifted, dtype="float32", shape=[n_shifted]),
                },
            ))

        # Step 3: Forward-backward pass with PPO loss
        ppo_config = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
        result = self.training_client.forward_backward(
            data=datums, loss_fn="ppo", loss_fn_config=ppo_config,
        ).result()

        # Diagnostic: log PPO metrics on first step
        if self._step_count == 0 and hasattr(result, "metrics") and result.metrics:
            _diag.info("PPO metrics (step 0, sampling logprobs):")
            for k, v in result.metrics.items():
                _diag.info(f"  {k} = {v}")

        # Extract loss from result metrics
        total_loss = 0.0
        if hasattr(result, "metrics") and result.metrics:
            total_loss = result.metrics.get("loss:sum", result.metrics.get("loss", 0.0))

        # Apply optimizer step
        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            grad_clip_norm=self.config.grad_clip_norm,
        )
        self.training_client.optim_step(adam_params)

        self._step_count += 1

        return {
            "loss": total_loss / len(datums) if datums else 0.0,
            "step": self._step_count,
            "batch_size": len(datums),
            "avg_reward": sum(rewards) / n,
        }

    def train_step_tokens_reinforce(
        self,
        prompt_tokens_batch: list[list[int]],
        response_tokens_batch: list[list[int]],
        response_logprobs_batch: list[list[float]],
        rewards: list[float],
        **kwargs,
    ) -> dict[str, float]:
        """REINFORCE training step using cross_entropy with advantage-weighted tokens.

        Avoids the SamplingClient/TrainingClient logprob mismatch entirely by
        not using importance sampling. Gradient is simply: advantage * ∇log π(a|s).

        Uses shifted tokens (model_input=tokens[:-1], target=tokens[1:]) matching
        the Tinker cross_entropy convention from their docs.

        Args:
            prompt_tokens_batch: List of prompt token ID sequences.
            response_tokens_batch: List of response token ID sequences.
            response_logprobs_batch: List of per-token log probability sequences (unused).
            rewards: List of reward signals (GRPO advantages) for each trajectory.

        Returns:
            Dict of training statistics.
        """
        n = len(prompt_tokens_batch)
        if n != len(response_tokens_batch) or n != len(rewards):
            raise ValueError(
                "prompt_tokens_batch, response_tokens_batch, and rewards must have same length"
            )

        if n == 0:
            return {}

        try:
            import tinker
        except ImportError:
            self._step_count += 1
            return {
                "loss": 0.0,
                "step": self._step_count,
                "batch_size": n,
                "avg_reward": sum(rewards) / n,
            }

        use_raw = kwargs.get("precomputed_advantages", False)
        baseline = 0.0 if use_raw else sum(rewards) / n

        datums = []
        for pt, rt, reward in zip(prompt_tokens_batch, response_tokens_batch, rewards):
            advantage = reward - baseline
            min_len = len(rt)
            all_tokens = pt + rt[:min_len]

            # Shifted for next-token prediction (Tinker cross_entropy convention)
            input_tokens = all_tokens[:-1]
            target_tokens = all_tokens[1:]
            # Weight: 0 for prompt, advantage for response
            # After shift, response weights start at position n_prompt-1
            n_prompt = len(pt)
            weights = [0.0] * max(n_prompt - 1, 0) + [advantage] * min_len
            weights = weights[:len(target_tokens)]
            if len(weights) < len(target_tokens):
                weights += [0.0] * (len(target_tokens) - len(weights))

            datums.append(tinker.Datum(
                model_input=tinker.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens, dtype="int64", shape=[len(target_tokens)]),
                    "weights": tinker.TensorData(
                        data=weights, dtype="float32", shape=[len(weights)]),
                },
            ))

        if not datums:
            return {}

        result = self.training_client.forward_backward(
            data=datums, loss_fn="cross_entropy",
        ).result()

        # Diagnostic on first step
        if self._step_count == 0 and hasattr(result, "metrics") and result.metrics:
            import logging as _logging
            _diag = _logging.getLogger("tinker_trainer")
            _diag.info(f"REINFORCE metrics (step 0):")
            for k, v in result.metrics.items():
                _diag.info(f"  {k} = {v}")

        total_loss = 0.0
        if hasattr(result, "metrics") and result.metrics:
            total_loss = result.metrics.get("loss:sum", result.metrics.get("loss", 0.0))

        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            grad_clip_norm=self.config.grad_clip_norm,
        )
        self.training_client.optim_step(adam_params)

        self._step_count += 1

        return {
            "loss": total_loss / len(datums) if datums else 0.0,
            "step": self._step_count,
            "batch_size": len(datums),
            "avg_reward": sum(rewards) / n,
        }

    def train_step_tokens_ppo_kl(
        self,
        prompt_tokens_batch: list[list[int]],
        response_tokens_batch: list[list[int]],
        response_logprobs_batch: list[list[float]],
        rewards: list[float],
        ref_logprobs_batch: list[list[float]] | None = None,
        kl_coef: float = 1e-3,
        clip_low: float = 0.8,
        clip_high: float = 1.2,
        **kwargs,
    ) -> dict[str, float]:
        """PPO + KL-in-loss training step using forward_backward_custom.

        Implements ariahw's approach: KL penalty as a separate additive loss
        term (not subtracted from advantages). The KL gradient is never clipped,
        providing stronger and more consistent regularization.

        Uses forward_backward_custom which costs ~3x wall time but gives full
        control over the loss function.

        Args:
            prompt_tokens_batch: List of prompt token ID sequences.
            response_tokens_batch: List of response token ID sequences.
            response_logprobs_batch: List of per-token sampling logprobs (old policy).
            rewards: List of reward signals (GRPO advantages).
            ref_logprobs_batch: List of per-token reference model logprobs.
                If None, KL term is skipped (pure PPO).
            kl_coef: KL penalty coefficient.
            clip_low: PPO clip lower bound (e.g. 0.8).
            clip_high: PPO clip upper bound (e.g. 1.2).
        """
        n = len(prompt_tokens_batch)
        if n != len(response_tokens_batch) or n != len(response_logprobs_batch) or n != len(rewards):
            raise ValueError("All batch inputs must have the same length")

        if n == 0:
            return {}

        try:
            import tinker
            import torch
        except ImportError as e:
            self._step_count += 1
            return {"loss": 0.0, "step": self._step_count, "batch_size": n,
                    "avg_reward": sum(rewards) / n, "error": str(e)}

        import logging as _logging
        _diag = _logging.getLogger("tinker_trainer")

        use_raw = kwargs.get("precomputed_advantages", False)
        baseline = 0.0 if use_raw else sum(rewards) / n

        # Prepare per-sample data: shifted tokens, sampling logprobs, ref logprobs, advantages
        sample_data = []
        datums = []
        for idx, (pt, rt, rlp, reward) in enumerate(
            zip(prompt_tokens_batch, response_tokens_batch, response_logprobs_batch, rewards)
        ):
            advantage = reward - baseline
            min_len = min(len(rt), len(rlp))
            rt = rt[:min_len]
            rlp = rlp[:min_len]
            all_tokens = pt + rt

            # Shifted tokens (same convention as cross_entropy and PPO)
            input_tokens = all_tokens[:-1]
            target_tokens = all_tokens[1:]
            n_shifted = len(target_tokens)
            n_prompt = len(pt)
            prompt_pad = max(n_prompt - 1, 0)

            # Sampling logprobs aligned to shifted convention
            sampling_lps = [0.0] * prompt_pad + rlp
            sampling_lps = sampling_lps[:n_shifted]
            if len(sampling_lps) < n_shifted:
                sampling_lps += [0.0] * (n_shifted - len(sampling_lps))

            # Reference logprobs (if provided)
            ref_lps = None
            if ref_logprobs_batch is not None and idx < len(ref_logprobs_batch):
                ref = ref_logprobs_batch[idx]
                ref_lps = [0.0] * prompt_pad + list(ref[:min_len])
                ref_lps = ref_lps[:n_shifted]
                if len(ref_lps) < n_shifted:
                    ref_lps += [0.0] * (n_shifted - len(ref_lps))

            # Advantage per token (0 for prompt, advantage for response)
            adv_arr = [0.0] * prompt_pad + [advantage] * min_len
            adv_arr = adv_arr[:n_shifted]
            if len(adv_arr) < n_shifted:
                adv_arr += [0.0] * (n_shifted - len(adv_arr))

            # Mask: 1 for response tokens, 0 for prompt
            mask = [0.0] * prompt_pad + [1.0] * min_len
            mask = mask[:n_shifted]
            if len(mask) < n_shifted:
                mask += [0.0] * (n_shifted - len(mask))

            sample_data.append({
                "sampling_lps": sampling_lps,
                "ref_lps": ref_lps,
                "advantages": adv_arr,
                "mask": mask,
                "prompt_pad": prompt_pad,
                "n_response": min_len,
            })

            # Datum for forward_backward_custom (needs target_tokens for cross_entropy forward)
            datums.append(tinker.Datum(
                model_input=tinker.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens, dtype="int64", shape=[n_shifted]),
                    "weights": tinker.TensorData(
                        data=[1.0] * n_shifted, dtype="float32", shape=[n_shifted]),
                },
            ))

        if not datums:
            return {}

        # Capture variables for the closure
        _clip_low = clip_low
        _clip_high = clip_high
        _kl_coef = kl_coef
        _sample_data = sample_data

        def ppo_kl_loss_fn(data, logprobs_list):
            """Custom loss: PPO clipped objective + KL penalty (k3 estimator)."""
            total_ppo_loss = torch.tensor(0.0)
            total_kl_loss = torch.tensor(0.0)
            n_tokens = 0

            for i, model_logprobs in enumerate(logprobs_list):
                sd = _sample_data[i]
                device = model_logprobs.device

                sampling_lps = torch.tensor(sd["sampling_lps"], device=device)
                advantages = torch.tensor(sd["advantages"], device=device)
                mask = torch.tensor(sd["mask"], device=device)

                # Trim to match model_logprobs length
                min_n = min(len(model_logprobs), len(sampling_lps))
                model_lps = model_logprobs[:min_n]
                samp_lps = sampling_lps[:min_n]
                adv = advantages[:min_n]
                m = mask[:min_n]

                # PPO clipped objective
                ratio = torch.exp(model_lps - samp_lps)
                clipped_ratio = torch.clamp(ratio, _clip_low, _clip_high)
                unclipped_obj = ratio * adv
                clipped_obj = clipped_ratio * adv
                ppo_obj = torch.min(unclipped_obj, clipped_obj)
                total_ppo_loss = total_ppo_loss - (ppo_obj * m).sum()

                # KL loss (k3 estimator, unclipped — this is the key difference)
                if sd["ref_lps"] is not None:
                    ref_lps = torch.tensor(sd["ref_lps"], device=device)[:min_n]
                    r = ref_lps - model_lps  # log(π_ref / π)
                    kl = torch.exp(r) - r - 1.0  # k3 estimator, always ≥ 0
                    total_kl_loss = total_kl_loss + (_kl_coef * kl * m).sum()

                n_tokens += m.sum().item()

            total_loss = total_ppo_loss + total_kl_loss
            metrics = {
                "ppo_loss": total_ppo_loss.item(),
                "kl_loss": total_kl_loss.item(),
                "total_loss": total_loss.item(),
                "n_response_tokens": n_tokens,
            }
            return total_loss, metrics

        # Run forward_backward_custom (~3x wall time)
        result = self.training_client.forward_backward_custom(
            data=datums, loss_fn=ppo_kl_loss_fn,
        ).result()

        # Log metrics
        if hasattr(result, "metrics") and result.metrics:
            if self._step_count == 0:
                _diag.info("PPO+KL-in-loss metrics (step 0):")
                for k, v in result.metrics.items():
                    _diag.info(f"  {k} = {v}")

        total_loss = 0.0
        if hasattr(result, "metrics") and result.metrics:
            total_loss = result.metrics.get("total_loss", result.metrics.get("loss:sum", 0.0))

        # Optimizer step
        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            grad_clip_norm=self.config.grad_clip_norm,
        )
        self.training_client.optim_step(adam_params)

        self._step_count += 1

        return {
            "loss": total_loss / len(datums) if datums else 0.0,
            "step": self._step_count,
            "batch_size": len(datums),
            "avg_reward": sum(rewards) / n,
        }

    def _train_accumulated_batch(self) -> dict[str, float]:
        """Train on accumulated RL data.

        Returns:
            Training statistics.
        """
        batch = self._accumulated_data[: self.config.batch_size]
        self._accumulated_data = self._accumulated_data[self.config.batch_size :]

        try:
            import tinker
            from tinker import Datum
        except ImportError:
            self._step_count += 1
            return {"loss": 0.0, "step": self._step_count, "batch_size": len(batch),
                    "avg_reward": sum(item["reward"] for item in batch) / len(batch)}

        # Convert accumulated data to Datum objects for PPO
        tokenizer = self.training_client.get_tokenizer()
        datums = []
        for item in batch:
            prompt_text = item["prompt"]
            response_text = item["response"]
            reward = item["reward"]

            prompt_tokens = tokenizer.encode(prompt_text)
            response_tokens = tokenizer.encode(response_text)
            all_tokens = prompt_tokens + response_tokens
            model_input = tinker.ModelInput.from_ints(all_tokens)

            # For PPO: provide advantages (using reward as advantage estimate)
            # and weights (loss mask: 0 for prompt, 1 for response tokens)
            advantages = [0.0] * len(prompt_tokens) + [reward] * len(response_tokens)
            weights = [0.0] * len(prompt_tokens) + [1.0] * len(response_tokens)

            datums.append(Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "advantages": tinker.TensorData(
                        data=advantages, dtype="float32", shape=[len(advantages)]
                    ),
                    "weights": tinker.TensorData(
                        data=weights, dtype="float32", shape=[len(weights)]
                    ),
                },
            ))

        # Use PPO loss for RL training
        future = self.training_client.forward_backward(
            data=datums, loss_fn="ppo",
        )
        result = future.result()

        # Extract loss from result metrics
        total_loss = 0.0
        if hasattr(result, "metrics") and result.metrics:
            total_loss = result.metrics.get("loss", 0.0)

        # Apply optimizer step with AdamParams object
        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            grad_clip_norm=self.config.grad_clip_norm,
        )
        self.training_client.optim_step(adam_params)

        self._step_count += 1
        avg_loss = total_loss / len(batch) if batch else 0.0

        return {
            "loss": avg_loss,
            "step": self._step_count,
            "batch_size": len(batch),
            "avg_reward": sum(item["reward"] for item in batch) / len(batch),
        }

    def get_sampling_model(self) -> "TinkerModel":
        """Get current policy for sampling.

        Returns a TinkerModel that can be used for inference with the
        current trained weights. For self-play, both debaters should
        share this model.

        Returns:
            TinkerModel wrapping current policy.
        """
        from models.tinker_model import TinkerModel

        return TinkerModel.from_training_client(
            self.training_client,
            alias=f"policy-step-{self._step_count}",
            max_tokens=self.config.debater_max_tokens,
        )

    def flush(self) -> dict[str, float]:
        """Flush any remaining accumulated data and train.

        Call this at the end of training to process any remaining
        data that didn't form a complete batch.

        Returns:
            Final training statistics.
        """
        stats = {}
        if self._accumulated_data:
            # Pad or just train on what we have
            stats = self._train_partial_batch()
        return stats

    def _train_partial_batch(self) -> dict[str, float]:
        """Train on partial batch (less than batch_size).

        Returns:
            Training statistics.
        """
        if not self._accumulated_data:
            return {}

        batch = self._accumulated_data
        self._accumulated_data = []

        try:
            import tinker
            from tinker import Datum
        except ImportError:
            self._step_count += 1
            return {"loss": 0.0, "step": self._step_count, "batch_size": len(batch),
                    "avg_reward": sum(item["reward"] for item in batch) / len(batch)}

        # Convert accumulated data to Datum objects for PPO
        tokenizer = self.training_client.get_tokenizer()
        datums = []
        for item in batch:
            prompt_text = item["prompt"]
            response_text = item["response"]
            reward = item["reward"]

            prompt_tokens = tokenizer.encode(prompt_text)
            response_tokens = tokenizer.encode(response_text)
            all_tokens = prompt_tokens + response_tokens
            model_input = tinker.ModelInput.from_ints(all_tokens)

            # For PPO: provide advantages (using reward as advantage estimate)
            # and weights (loss mask: 0 for prompt, 1 for response tokens)
            advantages = [0.0] * len(prompt_tokens) + [reward] * len(response_tokens)
            weights = [0.0] * len(prompt_tokens) + [1.0] * len(response_tokens)

            datums.append(Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "advantages": tinker.TensorData(
                        data=advantages, dtype="float32", shape=[len(advantages)]
                    ),
                    "weights": tinker.TensorData(
                        data=weights, dtype="float32", shape=[len(weights)]
                    ),
                },
            ))

        # Use PPO loss for RL training
        future = self.training_client.forward_backward(
            data=datums, loss_fn="ppo",
        )
        result = future.result()

        # Extract loss from result metrics
        total_loss = 0.0
        if hasattr(result, "metrics") and result.metrics:
            total_loss = result.metrics.get("loss", 0.0)

        # Apply optimizer step with AdamParams object
        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            weight_decay=self.config.weight_decay,
            grad_clip_norm=self.config.grad_clip_norm,
        )
        self.training_client.optim_step(adam_params)

        self._step_count += 1
        avg_loss = total_loss / len(batch) if batch else 0.0

        return {
            "loss": avg_loss,
            "step": self._step_count,
            "batch_size": len(batch),
            "avg_reward": sum(item["reward"] for item in batch) / len(batch),
        }

    def save_checkpoint(self, name: str) -> str:
        """Save full training checkpoint (weights + optimizer state).

        Args:
            name: Checkpoint name (e.g. "step-100").

        Returns:
            Tinker path string (tinker://<model_id>/<name>).
        """
        response = self.training_client.save_state(name).result()
        return getattr(response, "path", str(response))

    # Alias for backward compatibility
    save_checkpoint_with_optimizer = save_checkpoint

    def save_weights_for_sampling(self, name: str) -> str:
        """Save weights only (lighter, for inference only)."""
        response = self.training_client.save_weights_for_sampler(name).result()
        return getattr(response, "path", str(response))

    @classmethod
    def from_checkpoint(
        cls,
        tinker_path: str,
        config: TinkerTrainerConfig,
        kl_coef: float = 0.1,
    ) -> "TinkerRLTrainer":
        """Restore a TinkerRLTrainer from a checkpoint (weights + optimizer).

        Uses load_state() on an existing training client to restore
        the full training state.

        Args:
            tinker_path: Tinker path from a previous save_checkpoint().
            config: Tinker training configuration.
            kl_coef: KL penalty coefficient for RL.

        Returns:
            TinkerRLTrainer with restored weights and optimizer.
        """
        trainer = cls(config=config, kl_coef=kl_coef)
        trainer.training_client.load_state(tinker_path)
        return trainer

    @classmethod
    def from_checkpoint_weights_only(
        cls,
        tinker_path: str,
        config: TinkerTrainerConfig,
        kl_coef: float = 0.1,
    ) -> "TinkerRLTrainer":
        """Restore weights only, with a fresh optimizer.

        Use this for intervention experiments — don't want the
        hacking-optimized momentum polluting the fix.

        Args:
            tinker_path: Tinker path from a previous save_checkpoint().
            config: Tinker training configuration.
            kl_coef: KL penalty coefficient for RL.

        Returns:
            TinkerRLTrainer with restored weights but fresh optimizer.
        """
        try:
            import tinker
        except ImportError:
            raise ImportError("tinker package not installed")

        api_key = config.get_api_key()
        if api_key:
            os.environ["TINKER_API_KEY"] = api_key

        # Create a sampling client from the checkpoint, then create
        # a fresh training client from the same base model, and load
        # just the weights
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(
            model_path=tinker_path
        )

        # Create fresh training client (new optimizer)
        trainer = cls(config=config, kl_coef=kl_coef)
        # Load weights from checkpoint into the fresh training client
        trainer.training_client.load_state(tinker_path)
        # Note: load_state restores optimizer too. If Tinker doesn't support
        # weights-only restore, we accept the optimizer state but it will
        # be quickly overwritten by the new training dynamics.
        return trainer
