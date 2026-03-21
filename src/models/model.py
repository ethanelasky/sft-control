from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator, validator

from prompts import RoleType
import utils.constants as constants

from abc import ABC
from enum import Enum
from typing import Any, Literal, Optional


class ScoredCandidate(BaseModel):
    """Information about a candidate from best-of-n selection.

    Attributes:
        candidate_text: The generated text for this candidate.
        score: Score assigned by the reward model (0-10 scale).
        rank: Rank among all candidates (1 = best).
        selected: Whether this candidate was selected as the best.
        correct: Optional ground-truth correctness (set during verification).
    """

    candidate_text: str
    score: float
    rank: int
    selected: bool
    correct: bool | None = None


def build_scored_candidates(
    candidates: list[str], scores: list[float], best_idx: int
) -> list[ScoredCandidate]:
    """Build list of ScoredCandidate objects with ranks and selection status.

    Preserves original order of candidates. Ranks are assigned based on score
    (1 = highest score).

    Args:
        candidates: List of candidate texts.
        scores: List of scores corresponding to each candidate.
        best_idx: Index of the selected (best) candidate.

    Returns:
        List of ScoredCandidate objects in original order with ranks assigned.
    """
    # Compute ranks based on scores (1 = highest)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank

    return [
        ScoredCandidate(
            candidate_text=candidate,
            score=score,
            rank=ranks[idx],
            selected=(idx == best_idx),
        )
        for idx, (candidate, score) in enumerate(zip(candidates, scores))
    ]


class ScoringResult(BaseModel):
    """Result from scoring a set of candidates.

    This is the unified container for Best-of-N scoring results, replacing
    the individual bon_* fields that were previously on ModelResponse.

    Attributes:
        scores: List of scores for each candidate.
        best_idx: Index of the best (selected) candidate.
        candidates: List of ScoredCandidate objects with full metadata.
        raw_response: Raw text response from reward model (for debugging).
        scoring_prompt: The prompt sent to the reward model (for debugging).
        opposing_model_responses: Opponent responses used in maxmin scoring (legacy).
        degraded: True if the result quality is compromised (e.g., scoring failed
            and fell back to first candidate, or fewer candidates than requested).
    """

    scores: list[float]
    best_idx: int
    candidates: list[ScoredCandidate]
    raw_response: str | None = None
    scoring_prompt: str | None = None
    opposing_model_responses: list["ModelResponse"] = []
    degraded: bool = False


class ModelType(Enum):
    RANDOM = 1
    LLAMA = 2
    DETERMINISTIC = 3
    OPENAI = 4
    OFFLINE = 5
    HUMAN = 6
    MISTRAL = 7
    STUB_LLM = 8
    ARBITRARY_ATTRIBUTE = 9
    ANTHROPIC = 10
    LLAMA3 = 11
    REPETITIVE = 12
    FIREWORKS = 13
    ALIBABA = 14
    TINKER = 15
    GOOGLE = 16


class RewardModelConfig(BaseModel):
    model_settings: "ModelSettings"
    max_new_tokens: int = constants.REWARD_MODEL_MAX_NEW_TOKENS
    fallback_max_new_tokens: int = constants.REWARD_MODEL_FALLBACK_MAX_NEW_TOKENS

    @field_validator("model_settings", mode="before")
    @classmethod
    def ensure_model_settings(cls, settings):
        if isinstance(settings, str):
            settings = {
                "model_file_path": settings,
            }
        if isinstance(settings, dict):
            updated_settings = settings.copy()
            updated_settings.setdefault("model_type", ModelType.OPENAI)
            if updated_settings.get("alias") is None and updated_settings.get(
                "model_file_path"
            ):
                updated_settings["alias"] = updated_settings["model_file_path"]
            return updated_settings
        return settings


class BestOfNConfig(BaseModel):
    n: int
    opponent_n: int = 0  # Legacy field for old wrapper approach
    maxmin: bool = False  # Legacy field for old wrapper approach
    recompute: bool = False  # Legacy field for old wrapper approach
    reward_model_config: Optional[RewardModelConfig] = None  # Model to use for scoring
    sample_temperature: float = 0.7  # Temperature for candidate generation

    @field_validator("reward_model_config", mode="before")
    @classmethod
    def parse_reward_model_config(cls, reward_model_config):
        if reward_model_config in (None, {}):
            return {"model_settings": {"model_file_path": "o1-preview"}}
        if isinstance(reward_model_config, str):
            return {"model_settings": {"model_file_path": reward_model_config}}
        if (
            isinstance(reward_model_config, dict)
            and "model_settings" not in reward_model_config
        ):
            return {"model_settings": reward_model_config}
        return reward_model_config


class GenerationParams(BaseModel):
    max_new_tokens: int = constants.SERVED_MODEL_DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.5
    top_p: float = 1.0
    repetition_penalty: float = 1.2
    do_sample: bool = True
    use_generation_penalties: bool = False


class ModelInput(BaseModel):
    role: RoleType
    content: str


class ModelResponse(BaseModel):
    """Response from a model prediction.

    Attributes:
        speech: The generated text response.
        decision: Deterministic decision (for judge responses).
        probabilistic_decision: Probability distribution over decisions.
        preference: Score/preference for this response (from BoN scoring).
        scoring_result: Best-of-N scoring results (replaces bon_* fields).
        internal_representations: Internal model state (for probing).
        response_tokens: Token IDs of the response.
        response_logprobs: Per-token log probabilities (populated by TinkerModel for RL).
        prompt_tokens: Token IDs of the prompt.
        prompt: The input prompt as text.
        failed: Whether generation failed.
        raw_response: Unprocessed response text.
        structured_decision: Structured decision output (for collaborative judging).
    """

    speech: str = ""
    decision: Literal[
        constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME, ""
    ] = ""
    probabilistic_decision: Optional[dict[str, float]] = None
    preference: Optional[float] = None
    scoring_result: Optional[ScoringResult] = None
    internal_representations: Optional[str] = ""
    response_tokens: list[int] = []
    response_logprobs: list[float] = []
    prompt_tokens: list[int] = []
    prompt: str = ""
    failed: bool = False
    raw_response: Optional[str] = None
    thinking: Optional[str] = None  # Model's internal reasoning (e.g. from Qwen <think> tags)
    structured_decision: Optional[dict[str, Any]] = None

    @validator("probabilistic_decision")
    def check_keys(cls, v):
        if v:
            if not constants.DEFAULT_DEBATER_A_NAME in v:
                raise ValueError(
                    f"Probabilistic decision is missing required key: {constants.DEFAULT_DEBATER_A_NAME}"
                )
            if not constants.DEFAULT_DEBATER_B_NAME in v:
                raise ValueError(
                    f"Probabilistic decision is missing required key: {constants.DEFAULT_DEBATER_B_NAME}"
                )
            if len(v) > 2:
                all_keys = ", ".join(v.keys())
                raise ValueError(
                    f"There are too many keys in the probabilistic decision map. Keys: {all_keys}"
                )

            eps = 0.001
            total_prob = sum(v.values())
            if total_prob < 1 - eps or total_prob > 1 + eps:
                raise ValueError(
                    f"Total probability does not sum to 1 -- it sums to {total_prob}. Map is {v}"
                )

        return v


class ProbeHyperparams(BaseModel):
    file_path: str = ""
    hidden_size: Optional[int] = None
    linear_idxs: list[int] = [-1]


class ModelSettings(BaseModel):
    model_type: str | ModelType = ModelType.RANDOM
    model_file_path: Optional[str] = None
    alias: str
    override_prompt: Optional[str] = None
    nucleus: bool = True
    is_human: bool = False
    offline_file_path: Optional[str] = None
    served: bool = False
    probe_hyperparams: Optional[ProbeHyperparams] = None
    require_quote_validation: bool = True
    generation_params: GenerationParams = GenerationParams()
    peft_base_model: Optional[str] = None
    enable_thinking: Optional[bool] = None

    @model_validator(mode="before")
    def verify_custom_settings(cls, values):
        existence_count = sum(
            [values.get("is_human", False), values.get("served", False)]
        ) + (1 if values.get("offline_file_path", None) else 0)
        if existence_count > 1:
            raise ValueError(
                "One cannot set more than one of is_human, served, or offline_file_path to non-null and true"
            )
        return values

    model_config = ConfigDict(protected_namespaces=("protect_me_", "also_protect_"))

    @field_validator("alias", mode="before")
    @classmethod
    def validate_alias(cls, alias: str | int):
        return str(alias)

    @field_validator("model_type", mode="before")
    @classmethod
    def validate_model_type(cls, model_type: str | ModelType):
        if isinstance(model_type, str):
            return ModelType[model_type.upper()]
        return model_type


class SpeechStructure(Enum):
    OPEN_ENDED = 1
    DECISION = 2


class Model(ABC):
    def __init__(self, alias: str, is_debater: bool = False):
        self.alias = alias
        self.is_debater = is_debater

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = constants.OFFLINE_MODEL_DEFAULT_MAX_NEW_TOKENS,
        **kwargs,
    ) -> list[ModelResponse]:
        pass

    def copy(self, is_debater: Optional[bool] = None, **kwargs) -> Model:
        return self

    def can_merge(self, other: Model) -> bool:
        return other == self

    def merge(self, other: Model) -> Model:
        if self.can_merge(other):
            return self
        raise Exception("Cannot merge across models")
