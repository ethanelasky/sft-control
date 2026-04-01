"""Shared configuration constants for sft-control experiments.

Single source of truth for base model identifiers and default hyperparameters.
All scripts should import from here rather than hard-coding defaults.
"""

# ---------------------------------------------------------------------------
# Base models
# ---------------------------------------------------------------------------

# The model used for hacking emergence and all intervention experiments.
# This MUST match the model the hacked checkpoint was trained on.
BASE_MODEL = "openai/gpt-oss-20b"

# Weak trusted judge for the RL-trusted intervention variant.
JUDGE_MODEL = "qwen3-1.7b"

# ---------------------------------------------------------------------------
# Hacking run defaults (matching Wong et al. 2025)
# ---------------------------------------------------------------------------

HACKING_DEFAULTS = {
    "lr": 7e-5,
    "kl_coef": 1e-3,
    "lora_rank": 32,
    "num_generations": 16,
    "num_prompts_per_step": 16,
    "max_completion_length": 1536,
    "temperature": 0.7,
    "top_p": 0.95,
    "warmup_steps": 10,
    "mini_batch_size": 16,
    "max_steps": 200,
}

# ---------------------------------------------------------------------------
# Intervention run defaults
# ---------------------------------------------------------------------------

INTERVENTION_DEFAULTS = {
    "lr": 3e-5,
    "kl_coef": 5e-3,
    "lora_rank": 32,
    "num_generations": 16,
    "num_prompts_per_step": 16,
    "max_completion_length": 1536,
    "temperature": 0.7,
    "top_p": 0.95,
    "warmup_steps": 10,
    "mini_batch_size": 16,
    "max_steps": 75,
}
