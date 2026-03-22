# Checkpoint Guide

## Saving

Checkpoints are saved automatically during training:
- Every `--save_every` steps (default 50) → `tinker://.../weights/grpo-step-{N}`
- At the end of training → `tinker://.../weights/grpo-final`

Checkpoint paths are logged:
```
Saved checkpoint: tinker://ad3cadb9-.../weights/grpo-step-50
```

## Resuming: Two Modes

### 1. Full Resume (weights + optimizer)

```bash
python scripts/train_hacking.py \
  --base_model "openai/gpt-oss-20b" \
  --resume_checkpoint "tinker://SESSION_ID:train:0/weights/grpo-step-100" \
  --max_steps 200 \
  ...
```

This restores LoRA weights AND optimizer state (Adam momentum/variance). Use this to **continue a run that was interrupted**.

**Limitations:**
- The step counter resets to 0. The LR schedule restarts from scratch (warmup → peak → cosine decay). This means steps 0-10 will have low LR again even though the model is already trained.
- The KL reference model is freshly created (base weights), not restored. This is correct — KL should always be measured against the original base model.
- No `loss_type` is set on the restored trainer. You must pass it through `common_kwargs`.

### 2. Weights Only (fresh optimizer)

```bash
python scripts/train_hacking.py \
  --base_model "openai/gpt-oss-20b" \
  --resume_checkpoint "tinker://SESSION_ID:train:0/weights/grpo-step-100" \
  --weights_only \
  --max_steps 75 \
  --lr 3e-5 \
  --kl_coef 5e-3 \
  ...
```

This loads LoRA weights but creates a fresh optimizer. Use this for **intervention experiments** — e.g., taking a hacked model and training it with a different reward function to fix the hacking.

**When to use weights_only:**
- Switching reward functions (default → golden/penalty)
- Changing LR significantly
- Running intervention experiments on a hacked checkpoint

### 3. Resuming Legacy Checkpoints (no metadata JSON)

For checkpoints saved before metadata support was added, pass the step number manually with `--resume_step`:

```bash
python scripts/train_hacking.py \
  --resume_checkpoint "tinker://SESSION_ID:train:0/weights/grpo-step-100" \
  --resume_step 100 \
  --max_steps 200 \
  ...
```

You can find the step number from the training log:
```bash
grep "Saved checkpoint" logs/your-run.log
```

Without `--resume_step`, legacy checkpoints will start from step 0 (LR warmup repeats, cosine schedule restarts).

## Common Pitfalls

### 1. Log file clobbering

**Problem:** Using `>` to redirect output overwrites the log. Using `>>` appends, creating a log with two runs mixed together. Both are bad.

**Fix:** Always use a unique log file name per run:
```bash
> logs/grpo-gpt-oss-20b-run2.log 2>&1
```

### 2. Step counter resets

**Problem:** `from_checkpoint()` and `from_checkpoint_weights_only()` don't restore the step counter. The training loop always runs `for step in range(max_steps)` starting at 0. This means:
- LR warmup repeats (steps 0-10 have low LR)
- Cosine schedule restarts
- Checkpoint saves overwrite previous ones if using the same `checkpoint_dir`

**Workaround:** For a true continuation, manually set `--warmup_steps 0` and use a different `--checkpoint_dir`.

### 3. KL reference model not restored

**Problem:** `from_checkpoint()` does not create `self.ref_sampling_client`. KL penalty is silently disabled when resuming.

**This is a known bug.** The `__init__` path creates the reference model but the `from_checkpoint` path skips it. If you need KL on a resumed run, this needs to be fixed.

### 4. loss_type not set on resumed trainer

**Problem:** `from_checkpoint()` and `from_checkpoint_weights_only()` don't set `self.loss_type`. It will crash or use the wrong default when the train loop tries to select the train function.

**Workaround:** Pass `loss_type` in `common_kwargs` — it gets set via `__init__` params but the `from_checkpoint` paths need to handle it too.

## Finding Checkpoint Paths

Checkpoint paths are printed in the training log:
```bash
grep "Saved checkpoint" logs/your-run.log
```

They look like:
```
tinker://SESSION_ID:train:0/weights/grpo-step-50
tinker://SESSION_ID:train:0/weights/grpo-step-100
tinker://SESSION_ID:train:0/weights/grpo-final
```

## Tinker Checkpoint Internals

Tinker checkpoints store LoRA adapter weights on Tinker's cloud storage. They are tied to the Tinker session that created them. Key operations:

- `training_client.save_state(name)` — saves weights + optimizer to `tinker://SESSION/weights/{name}`
- `training_client.load_state(path)` — loads weights (+ optimizer state)
- `training_client.load_state_with_optimizer(path)` — explicitly loads both

See https://tinker-docs.thinkingmachines.ai/save-load for details.
