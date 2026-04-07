# sft-control — Post-Hoc Interventions Against Reward Hacking

## Project Overview

Testing whether post-hoc training interventions (SFT and RL) can fix a model
that has naturally learned to reward hack during GRPO training on LeetCode.
Based on Wong, Engels & Nanda (2025), ported from veRL to the Tinker cloud API.

Paper: Redwood AI Control Hackathon, March 2026.
Collaborators: Ethan Elasky, Frank Nakasako.

---

## Critical: Configuration Source of Truth

**All hyperparameters live in `src/config.py`.** Never hardcode model names,
learning rates, or other training parameters in scripts or trainer classes.

- `BASE_MODEL = "openai/gpt-oss-20b"` — the model all hacking/intervention runs use
- `JUDGE_MODEL = "qwen3-1.7b"` — weak trusted judge for RL-trusted variant
- `HACKING_DEFAULTS` — hyperparams for hacking emergence runs (lr=7e-5, kl=1e-3)
- `INTERVENTION_DEFAULTS` — hyperparams for RL fix runs (lr=3e-5, kl=5e-3)
- `SFT_DEFAULTS` — hyperparams for SFT fix runs (lr=1e-4)

When adding new scripts or training methods, import from `config.py` — do not
introduce new hardcoded defaults.

---

## API Keys

Stored in `.env` (gitignored, auto-loaded by training scripts via `python-dotenv`):
- `TINKER_API_KEY` — required for all training/inference via Tinker
- `DASHSCOPE_API_KEY` — required only for `--variant trusted` (Qwen judge via DashScope)

If keys are missing, check the ai-debate repo's `.env` for the DashScope key.
For Tinker, ask Ethan.

---

## Running Experiments

### Hacking emergence (produces a hacked checkpoint)
```bash
python3.11 scripts/train_hacking.py --max_steps 200
```

### RL interventions (fix a hacked checkpoint)
```bash
python3.11 scripts/train_rl_fix.py \
  --checkpoint "tinker://<checkpoint-path>" \
  --variant <golden|trusted|penalty|trusted_penalty> \
  --wandb_run_name "rl-fix-<variant>-<description>"
```

All hyperparams default from `INTERVENTION_DEFAULTS`. Override via CLI args
(e.g., `--kl_coef 1e-3 --lr 7e-5`).

### SFT interventions
```bash
python3.11 scripts/train_sft_fix.py \
  --checkpoint "tinker://<checkpoint-path>" \
  --num_examples 50
```

### Python version
Use `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11`.
The default `python3` is 3.9 which is too old for the `tinker` package.

### Plotting
```bash
python3.11 scripts/plot_training_runs.py          # hacking emergence curves
python3.11 scripts/plot_intervention_curves.py     # intervention training curves
python3.11 scripts/plot_interventions.py           # Pareto scatter (final results)
```

---

## Checkpoint Registry

Canonical checkpoints are tracked in `checkpoints/registry.json`.
Per-run checkpoint metadata is saved as `checkpoints/<run-name>-step-N.json`.

### Key checkpoints
| Name | Tinker Path | Description |
|------|-------------|-------------|
| gpt-oss-20b hacked (kl=1e-3) | `tinker://ad3cadb9-...grpo-final` | 95% hack rate, 0% correct. 200 steps. |
| gpt-oss-20b hacked (kl=0) | `tinker://85980d9a-...grpo-final` | Used for v3 intervention runs. |

---

## Project Structure

```
src/
  config.py          — Single source of truth for all hyperparameters
  grpo_trainer.py     — GRPO training loop (Tinker API)
  rewards.py          — Reward functions: default, golden, penalty, trusted
  analysis.py         — Reward hack detection and categorization
  data_loader.py      — JSONL dataset loading
  train/              — Tinker trainer wrappers (SFT, RL)
  models/             — Model interfaces (Tinker, DashScope)
  sandbox/            — Sandboxed code execution for reward computation
  prompts/            — System prompts and role types
scripts/
  train_hacking.py    — Produce a hacked checkpoint via GRPO
  train_rl_fix.py     — RL interventions (golden, trusted, penalty)
  train_sft_fix.py    — SFT interventions
  eval_model.py       — Evaluate a checkpoint (--compare for tables)
  plot_*.py           — Visualization scripts
data/                 — LeetCode datasets (loophole and benign)
checkpoints/          — Checkpoint metadata JSONs
logs/                 — Training logs (parseable for metrics)
figures/              — Generated plots
```

---

## Code Conventions

### GRPOTrainer attribute initialization
`GRPOTrainer` has three construction paths: `__init__`, `from_checkpoint`,
and `from_checkpoint_weights_only`. All shared attribute setup lives in
`_init_common()` and `_init_ref_model_and_executor()`. When adding new
instance attributes, add them to `_init_common()` — never duplicate
attribute assignments across the classmethods.

---

## Log Naming Convention

```
<experiment>-<model>-<variant>-<version>.log
```

Examples:
- `grpo-gpt-oss-20b-ppo-kl.log` — hacking emergence run
- `rl-fix-golden-v3-final.log` — v3 golden intervention (correct base model)
- `rl-fix-trusted-v3-kl1e3.log` — v3 trusted intervention with kl=1e-3

Version suffixes: v1 = initial (wrong base model), v2 = second attempt,
v3 = correct base model (current).

---

## Committing Results

- Logs, checkpoint metadata, and figures should be committed.
- `.env` is gitignored (contains API keys).
- `wandb/` is gitignored.
- Always include the W&B run name in the commit message for traceability.
