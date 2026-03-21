DEFAULT_DEBATER_A_NAME = "Debater_A"
DEFAULT_DEBATER_B_NAME = "Debater_B"
DEFAULT_CONSULTANT_A_NAME = "Consultant_A"
DEFAULT_CONSULTANT_B_NAME = "Consultant_B"
DEFAULT_PARTICIPANT_A_NAME = "Participant_A"
DEFAULT_PARTICIPANT_B_NAME = "Participant_B"
DEFAULT_JUDGE_NAME = "Judge"
DEFAULT_BACKGROUND_TEXT = "None provided"
DEFAULT_WORD_LIMIT = 750
INSTRUCTION_PREFIX = "instruction:"
INPUT_PREFIX = "### User:"
INSTRUCTION_SUFFIX = "output:"
JUDGING_PREFIX = "I believe the winner of this debate is Debater_"
PREFERENCE_PREFIX = "Overall Score: "
WINNER = "Winner"
LOSER = "Loser"
CORRECT = "Correct"
INCORRECT = "Incorrect"
OVERALL = "Overall"
BASE_MODEL_PREFIX = "base"
DEBATER_A_IDENTIFICATION = "You are Debater_A"
BASE_PROMPT = "Base Prompt"
QUOTE_TAG = "<quote>"
UNQUOTE_TAG = "</quote>"
INVALID_QUOTE_TAG = "<invalid_quote>"
INVALID_UNQUOTE_TAG = "</invalid_quote>"
BEGIN_SPEECH_OPTIONS = [
    "Write out your speech:",
    "Now begin your speech.",
    "Please deliver your speech.",
    "We will now await your speech.",
]
BEGIN_JUDGING_OPTIONS = ["Here is the decision that the judge made:"]
QUOTE_FUZZY_MATCH_EARLY_STOPPING_THRESHOLD = 0.9
QUOTE_FUZZY_MATCH_MIN_THRESHOLD = 0.8
MAX_SCORE = 10
DEBATER_A_POSITION = 0
DEBATER_B_POSITION = 1
MAX_LENGTH = 16380
LINE_SEPARATOR = "\n######\n"
SRC_ROOT = "SRC_ROOT"
INPUT_ROOT = "INPUT_ROOT"
# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
# This file contains configuration values, NOT prompt text.
# All prompt templates should go in: ai_debate/prompts/configs/prompts.yaml
#
# Categories in this file:
# - Token limits and generation parameters
# - Default names and identifiers
# - Formatting markers and tags
# - File paths and system configuration
# - Numeric thresholds
# =============================================================================

DEFAULT_ALIAS = "empty-model"
OPEN_ROLE_FIRST_MODEL_PLACEHOLDER = "[OPEN_ROLE_FIRST_MODEL_ANSWER]"
PROBABILISTIC_TIE_EPSILON = 0.01

# ---------------------------------------------------------------------------
# Token limits
# ---------------------------------------------------------------------------
DEFAULT_DEBATER_TOKENS_PER_SPEECH = 127000
DEFAULT_JUDGE_TOKENS_PER_SPEECH = 127000

# Judge generation limits - set high so model is the bottleneck, not our config.
# These can be overridden per-agent via AgentConfig.reasoning_max_tokens / decision_max_tokens.
JUDGE_REASONING_MAX_NEW_TOKENS = 127000
JUDGE_DECISION_MAX_NEW_TOKENS = 127000
# For reasoning models (o-series, gpt-5), max_output_tokens includes both
# internal reasoning tokens and visible output tokens. We use the same high
# default since the model API will cap at its actual limit.
JUDGE_DECISION_REASONING_MODEL_MAX_NEW_TOKENS = 127000
BON_JUDGE_MAX_NEW_TOKENS = 15000

ANSWER_GENERATOR_DIRECT_MAX_TOKENS = 15000
ANSWER_GENERATOR_COT_MAX_TOKENS = 15000
OPEN_ROLE_DEFAULT_MAX_NEW_TOKENS = 20000
OPEN_ROLE_DEFAULT_ANSWER_MAX_TOKENS = OPEN_ROLE_DEFAULT_MAX_NEW_TOKENS
ANSWER_GEN_DEFAULT_TOKEN_LIMIT = OPEN_ROLE_DEFAULT_MAX_NEW_TOKENS

REWARD_MODEL_MAX_NEW_TOKENS = 127000
REWARD_MODEL_FALLBACK_MAX_NEW_TOKENS = 10000

OPENAI_DEFAULT_MAX_NEW_TOKENS = 10000
OPENAI_HARD_CAP_MAX_NEW_TOKENS = 127500
ANTHROPIC_DEFAULT_MAX_NEW_TOKENS = 200

OFFLINE_MODEL_DEFAULT_MAX_NEW_TOKENS = 1000
SERVED_MODEL_DEFAULT_MAX_NEW_TOKENS = 1000

VERDICT_VERIFIER_MAX_NEW_TOKENS = 2000
