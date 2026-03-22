"""DashScope model implementation for API-based inference via OpenAI-compatible endpoint.

Wraps Alibaba's DashScope API (used for Qwen models) for evaluation.
Does not provide logprobs — use TinkerModel for RL training.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import backoff
import openai

from models.model import Model, ModelInput, ModelResponse, SpeechStructure
from prompts import RoleType
import utils.constants as constants

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class DashScopeModel(Model):
    """Model implementation using DashScope's OpenAI-compatible API.

    Suitable for evaluation (no logprobs). For RL training, use TinkerModel.

    Attributes:
        model_name: DashScope model identifier (e.g. "qwen3-4b").
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum tokens per generation.
        enable_thinking: Whether to enable Qwen thinking mode.
        max_parallel: Maximum concurrent API requests.
    """

    def __init__(
        self,
        model_name: str,
        alias: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1536,
        enable_thinking: bool = False,
        max_parallel: int = 32,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(alias=alias or model_name, is_debater=True)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.max_parallel = max_parallel

        resolved_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not resolved_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY environment variable is required for DashScope models"
            )

        self.client = openai.OpenAI(
            api_key=resolved_key,
            base_url=base_url or DASHSCOPE_BASE_URL,
        )

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = constants.SERVED_MODEL_DEFAULT_MAX_NEW_TOKENS,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        **kwargs,
    ) -> list[ModelResponse]:
        """Generate one response per input."""
        max_new_tokens = min(max_new_tokens, self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = [
                executor.submit(
                    self._call_api,
                    messages=self._to_messages(inp),
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=1,
                )
                for inp in inputs
            ]
            results = []
            for future in futures:
                choices = future.result()
                if choices and len(choices) > 0:
                    results.append(self._choice_to_response(choices[0]))
                else:
                    results.append(ModelResponse(failed=True))

        return results

    def predict_multi(
        self,
        inputs: list[list[ModelInput]],
        num_samples: int = 4,
        max_new_tokens: int = constants.SERVED_MODEL_DEFAULT_MAX_NEW_TOKENS,
        **kwargs,
    ) -> list[list[ModelResponse]]:
        """Generate num_samples responses per input.

        DashScope supports n>1 for some models, but for reliability we
        issue num_samples separate calls per input, parallelized.
        """
        max_new_tokens = min(max_new_tokens, self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)

        # Build all (input_idx, sample_idx) pairs
        tasks = []
        for input_idx, inp in enumerate(inputs):
            messages = self._to_messages(inp)
            for sample_idx in range(num_samples):
                tasks.append((input_idx, sample_idx, messages))

        # Execute all in parallel
        response_grid: list[list[ModelResponse | None]] = [
            [None] * num_samples for _ in inputs
        ]

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_idx = {}
            for input_idx, sample_idx, messages in tasks:
                future = executor.submit(
                    self._call_api,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=1,
                )
                future_to_idx[future] = (input_idx, sample_idx)

            for future in future_to_idx:
                input_idx, sample_idx = future_to_idx[future]
                try:
                    choices = future.result()
                    if choices and len(choices) > 0:
                        response_grid[input_idx][sample_idx] = self._choice_to_response(choices[0])
                    else:
                        response_grid[input_idx][sample_idx] = ModelResponse(failed=True)
                except Exception as e:
                    logger.warning(f"DashScope call failed for input {input_idx} sample {sample_idx}: {e}")
                    response_grid[input_idx][sample_idx] = ModelResponse(failed=True)

        # Replace any remaining None with failed
        return [
            [r if r is not None else ModelResponse(failed=True) for r in row]
            for row in response_grid
        ]

    @backoff.on_exception(backoff.expo, Exception, max_tries=4)
    def _call_api(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        n: int = 1,
    ) -> list:
        """Call DashScope API with exponential backoff."""
        extra_params = {}
        # Qwen 3 models have thinking on by default; must explicitly disable
        # for non-streaming calls or the API returns 400
        extra_params["extra_body"] = {"enable_thinking": self.enable_thinking}

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.top_p,
            n=n,
            **extra_params,
        )
        return list(response.choices)

    def _to_messages(self, model_inputs: list[ModelInput]) -> list[dict]:
        """Convert ModelInput list to OpenAI message format."""
        messages = []
        for mi in model_inputs:
            role = mi.role.name.lower()
            messages.append({"role": role, "content": mi.content})
        return messages

    def _choice_to_response(self, choice) -> ModelResponse:
        """Convert an OpenAI Choice to ModelResponse."""
        content = choice.message.content or ""

        thinking = None
        speech = content
        if self.enable_thinking:
            import re
            match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if match:
                thinking = match.group(1).strip() or None
                speech = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        return ModelResponse(speech=speech, thinking=thinking)
