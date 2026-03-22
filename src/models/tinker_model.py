"""Tinker model implementation for API-based inference with logprobs.

This module provides a Model implementation that wraps Tinker's SamplingClient,
enabling cloud-based inference with per-token log probabilities for RL training.
"""

from __future__ import annotations

import backoff

from models.model import Model, ModelInput, ModelResponse, SpeechStructure
from prompts import RoleType
from utils import logger_utils
import utils.constants as constants

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, TYPE_CHECKING
import re

if TYPE_CHECKING:
    # Avoid import errors when tinker is not installed
    from tinker import SamplingClient, TrainingClient


class TinkerModel(Model):
    """Model implementation wrapping Tinker's SamplingClient.

    Always uses TokenCompleter (not MessageCompleter) so logprobs are available
    for RL training. Logprobs are stored in ModelResponse.response_logprobs.

    Attributes:
        sampling_client: Tinker SamplingClient instance for inference.
        temperature: Sampling temperature for generation.
        top_p: Nucleus sampling probability threshold.
        max_tokens: Default max tokens for generation.
        enable_thinking: Whether to enable thinking mode in chat template.
    """

    MAX_PARALLEL_REQUESTS = 64

    def __init__(
        self,
        alias: str,
        sampling_client: "SamplingClient",
        is_debater: bool = True,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
    ):
        """Initialize TinkerModel.

        Args:
            alias: String identifier for metrics and deduplication.
            sampling_client: Tinker SamplingClient for inference.
            is_debater: Whether this model is a debater (True) or judge (False).
            temperature: Sampling temperature for generation.
            top_p: Nucleus sampling threshold (1.0 = no filtering).
            max_tokens: Default maximum tokens to generate.
            enable_thinking: Whether to enable thinking tokens in chat template.
                Disabled by default because thinking tokens consume the generation
                budget and pollute judge context.
        """
        super().__init__(alias=alias, is_debater=is_debater)
        self.sampling_client = sampling_client
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.logger = logger_utils.get_default_logger(__name__)

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = constants.SERVED_MODEL_DEFAULT_MAX_NEW_TOKENS,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        **kwargs,
    ) -> list[ModelResponse]:
        """Generate responses using Tinker SamplingClient.

        Converts ModelInput format to Tinker message format, calls the
        TokenCompleter, and converts back to ModelResponse with logprobs.

        Args:
            inputs: Batch of inputs, where each input is a list of ModelInput
                representing a conversation history.
            max_new_tokens: Maximum tokens to generate per response.
            speech_structure: Expected output format (OPEN_ENDED or DECISION).
            **kwargs: Additional arguments. Supports 'temperature' override.

        Returns:
            List of ModelResponse, one per input, with speech and logprobs populated.
        """
        max_new_tokens = min(max_new_tokens, self.max_tokens)
        temperature = kwargs.pop("temperature", None)

        with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_REQUESTS) as executor:
            futures = [
                executor.submit(
                    self._predict_single,
                    model_inputs=input_list,
                    max_new_tokens=max_new_tokens,
                    speech_structure=speech_structure,
                    temperature=temperature,
                )
                for input_list in inputs
            ]
            results = [future.result() for future in futures]

        return results

    def _predict_single(
        self,
        model_inputs: list[ModelInput],
        max_new_tokens: int,
        speech_structure: SpeechStructure,
        temperature: Optional[float] = None,
    ) -> ModelResponse:
        """Generate a single response with logprobs.

        Args:
            model_inputs: Conversation history as ModelInput list.
            max_new_tokens: Maximum tokens to generate.
            speech_structure: Expected output format.
            temperature: Override temperature for this call, or None to use default.

        Returns:
            ModelResponse with speech, response_tokens, and response_logprobs.
        """
        try:
            # Convert to Tinker format (also gives us prompt_tokens)
            tinker_input, prompt_token_ids = self._convert_to_tinker_input(model_inputs)

            # Call Tinker API with retries
            result = self._call_tinker_api(
                tinker_input=tinker_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            # Extract tokens and logprobs from first sampled sequence
            tokens = []
            logprobs = []
            speech_text = ""
            if result.sequences and len(result.sequences) > 0:
                sample = result.sequences[0]
                tokens = list(sample.tokens) if hasattr(sample, "tokens") else []
                logprobs = list(sample.logprobs) if hasattr(sample, "logprobs") else []
                speech_text = self._decode_tokens(tokens)

            # Extract thinking from <think> tags if present
            thinking = None
            if self.enable_thinking and speech_text:
                thinking, speech_text = self._extract_thinking(speech_text)

            if speech_structure == SpeechStructure.DECISION:
                self.logger.debug(
                    f"DECISION response ({len(tokens)} tokens): {speech_text[:200]}"
                )

            # Build prompt text for debugging
            prompt_text = "\n".join(mi.content for mi in model_inputs)

            return ModelResponse(
                speech=speech_text,
                response_tokens=tokens,
                response_logprobs=logprobs,
                prompt_tokens=prompt_token_ids,
                prompt=prompt_text,
                thinking=thinking,
            )

        except Exception as e:
            self.logger.warning(f"Tinker prediction failed after retries: {e}")
            return ModelResponse(failed=True)

    @backoff.on_exception(backoff.expo, Exception, max_tries=4)
    def _call_tinker_api(
        self,
        tinker_input: Any,
        max_new_tokens: int,
        temperature: Optional[float] = None,
    ) -> Any:
        """Call Tinker's sample() API with exponential backoff on failure.

        Args:
            tinker_input: Tokenized input in Tinker ModelInput format.
            max_new_tokens: Maximum tokens to generate.
            temperature: Override temperature, or None for default.

        Returns:
            Tinker SampleResponse with sequences, tokens, and logprobs.
        """
        import tinker

        effective_temperature = temperature if temperature is not None else self.temperature
        sampling_params = tinker.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=effective_temperature,
            top_k=-1,
            top_p=self.top_p,
        )

        return self.sampling_client.sample(
            prompt=tinker_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

    def predict_multi(
        self,
        inputs: list[list[ModelInput]],
        num_samples: int = 16,
        max_new_tokens: int = constants.SERVED_MODEL_DEFAULT_MAX_NEW_TOKENS,
        **kwargs,
    ) -> list[list[ModelResponse]]:
        """Generate multiple responses per input using num_samples.

        Much faster than calling predict() N times because each API call
        generates num_samples completions for the same prompt in one shot.
        For GRPO: 16 prompts × 16 samples = 16 API calls instead of 256.

        Args:
            inputs: Batch of inputs (one per prompt).
            num_samples: Number of completions per prompt.
            max_new_tokens: Maximum tokens per completion.
            **kwargs: Supports 'temperature' override.

        Returns:
            List of lists: outer list is per-prompt, inner list is per-sample.
        """
        max_new_tokens = min(max_new_tokens, self.max_tokens)
        temperature = kwargs.pop("temperature", None)

        with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_REQUESTS) as executor:
            futures = [
                executor.submit(
                    self._predict_multi_single,
                    model_inputs=input_list,
                    num_samples=num_samples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                for input_list in inputs
            ]
            results = [future.result() for future in futures]

        return results

    @backoff.on_exception(backoff.expo, Exception, max_tries=4)
    def _predict_multi_single(
        self,
        model_inputs: list[ModelInput],
        num_samples: int,
        max_new_tokens: int,
        temperature: Optional[float] = None,
    ) -> list[ModelResponse]:
        """Generate multiple responses for a single prompt."""
        import tinker

        try:
            tinker_input, prompt_token_ids = self._convert_to_tinker_input(model_inputs)

            effective_temperature = temperature if temperature is not None else self.temperature
            sampling_params = tinker.SamplingParams(
                max_tokens=max_new_tokens,
                temperature=effective_temperature,
                top_k=-1,
                top_p=self.top_p,
            )

            result = self.sampling_client.sample(
                prompt=tinker_input,
                num_samples=num_samples,
                sampling_params=sampling_params,
            ).result()

            responses = []
            for sample in (result.sequences or []):
                tokens = list(sample.tokens) if hasattr(sample, "tokens") else []
                logprobs = list(sample.logprobs) if hasattr(sample, "logprobs") else []
                speech_text = self._decode_tokens(tokens)

                thinking = None
                if self.enable_thinking and speech_text:
                    thinking, speech_text = self._extract_thinking(speech_text)

                prompt_text = "\n".join(mi.content for mi in model_inputs)

                responses.append(ModelResponse(
                    speech=speech_text,
                    response_tokens=tokens,
                    response_logprobs=logprobs,
                    prompt_tokens=prompt_token_ids,
                    prompt=prompt_text,
                    thinking=thinking,
                ))

            # Pad with failed responses if fewer than requested
            while len(responses) < num_samples:
                responses.append(ModelResponse(failed=True))

            return responses

        except Exception as e:
            self.logger.warning(f"Tinker multi-sample prediction failed: {e}")
            return [ModelResponse(failed=True)] * num_samples

    def _convert_to_tinker_input(
        self, model_inputs: list[ModelInput]
    ) -> tuple[Any, list[int]]:
        """Convert ModelInput list to Tinker's expected input format.

        Uses the sampling client's tokenizer to apply the model's chat template
        for proper formatting, with a fallback to simple encoding if no chat
        template is available.

        Args:
            model_inputs: List of ModelInput from the debate system.

        Returns:
            Tuple of (Tinker ModelInput with encoded tokens, list of prompt token IDs).
        """
        import tinker

        tokenizer = self.sampling_client.get_tokenizer()

        # Convert ModelInput to HuggingFace message format
        messages = []
        for mi in model_inputs:
            role = mi.role.name.lower()
            messages.append({"role": role, "content": mi.content})

        # Use tokenizer's chat template for proper formatting
        try:
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except Exception:
            # Fallback to simple encoding if no chat template available
            prompt_parts = []
            for mi in model_inputs:
                role = mi.role.name.lower()
                prompt_parts.append(f"{role.capitalize()}: {mi.content}")
            prompt_text = "\n\n".join(prompt_parts)
            tokens = tokenizer.encode(prompt_text)

        return tinker.ModelInput.from_ints(tokens), list(tokens)

    def _extract_thinking(self, text: str) -> tuple[str | None, str]:
        """Extract thinking content from <think>...</think> tags in decoded text.

        Qwen models with enable_thinking=True produce <think>...</think> tags
        in their output containing internal reasoning.

        Args:
            text: Decoded text that may contain <think> tags.

        Returns:
            Tuple of (thinking_content or None, cleaned_text with think tags removed).
        """
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return (thinking if thinking else None), cleaned
        return None, text

    def _decode_tokens(self, tokens: list[int]) -> str:
        """Decode token IDs to text string.

        Args:
            tokens: List of token IDs from Tinker.

        Returns:
            Decoded text string.
        """
        try:
            tokenizer = self.sampling_client.get_tokenizer()
            return tokenizer.decode(tokens)
        except Exception as e:
            self.logger.warning(f"Token decoding failed: {e}")
            return ""

    def copy(self, is_debater: Optional[bool] = None, **kwargs) -> "TinkerModel":
        """Create a copy of this model sharing the same SamplingClient.

        For self-play, both debaters share the same underlying client,
        so updates to the model propagate to both.

        Args:
            is_debater: Override the is_debater flag if provided.
            **kwargs: Additional arguments (alias can be overridden).

        Returns:
            New TinkerModel instance sharing the same SamplingClient.
        """
        alias = kwargs.get("alias", self.alias)
        return TinkerModel(
            alias=alias,
            sampling_client=self.sampling_client,
            is_debater=is_debater if is_debater is not None else self.is_debater,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
        )

    def can_merge(self, other: Model) -> bool:
        """Check if this model can be merged with another for batching.

        TinkerModels can merge if they share the same SamplingClient.

        Args:
            other: Another Model instance.

        Returns:
            True if models share the same SamplingClient.
        """
        if not isinstance(other, TinkerModel):
            return False
        return self.sampling_client is other.sampling_client

    def merge(self, other: Model) -> "TinkerModel":
        """Merge this model with another for efficient batching.

        Args:
            other: Another Model instance (must be compatible TinkerModel).

        Returns:
            This model instance (since they share the same client).

        Raises:
            Exception: If models cannot be merged.
        """
        if self.can_merge(other):
            return self
        raise Exception("Cannot merge TinkerModels with different SamplingClients")

    @classmethod
    def from_training_client(
        cls,
        training_client: "TrainingClient",
        alias: str,
        is_debater: bool = True,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
    ) -> "TinkerModel":
        """Create TinkerModel from a TrainingClient after training.

        This saves the trained weights and creates a SamplingClient for
        inference with the updated model.

        Args:
            training_client: Tinker TrainingClient with trained weights.
            alias: Identifier for the model.
            is_debater: Whether this is a debater model.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens for generation.
            enable_thinking: Whether to enable thinking tokens.

        Returns:
            TinkerModel wrapping the trained model's SamplingClient.
        """
        sampling_client = training_client.save_weights_and_get_sampling_client()
        return cls(
            alias=alias,
            sampling_client=sampling_client,
            is_debater=is_debater,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

    @classmethod
    def from_base_model(
        cls,
        base_model: str,
        alias: str,
        is_debater: bool = True,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
    ) -> "TinkerModel":
        """Create TinkerModel from a HuggingFace base model name.

        Uses Tinker's ServiceClient to create a SamplingClient for the
        specified base model (e.g., "Qwen/Qwen3-8B").

        Args:
            base_model: HuggingFace model name (e.g., "Qwen/Qwen3-8B").
            alias: Identifier for the model.
            is_debater: Whether this is a debater model.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens for generation.
            enable_thinking: Whether to enable thinking tokens.

        Returns:
            TinkerModel wrapping the base model's SamplingClient.

        Raises:
            ImportError: If tinker package is not installed.
        """
        try:
            import tinker
        except ImportError:
            raise ImportError(
                "tinker package not installed. Install with: pip install tinker-sdk"
            )

        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(base_model=base_model)
        return cls(
            alias=alias,
            sampling_client=sampling_client,
            is_debater=is_debater,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        alias: str,
        is_debater: bool = True,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
    ) -> "TinkerModel":
        """Create TinkerModel from a saved checkpoint.

        Args:
            checkpoint_path: Path to saved Tinker checkpoint.
            alias: Identifier for the model.
            is_debater: Whether this is a debater model.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens for generation.
            enable_thinking: Whether to enable thinking tokens.

        Returns:
            TinkerModel loaded from checkpoint.

        Raises:
            ImportError: If tinker package is not installed.
        """
        try:
            import tinker
        except ImportError:
            raise ImportError(
                "tinker package not installed. Install with: pip install tinker-sdk"
            )

        # Training checkpoints need to go through TrainingClient first
        service_client = tinker.ServiceClient()
        training_client = service_client.create_training_client_from_state(
            path=checkpoint_path
        )
        sampling_client = training_client.save_weights_and_get_sampling_client()
        return cls(
            alias=alias,
            sampling_client=sampling_client,
            is_debater=is_debater,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
