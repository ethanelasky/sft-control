# Post-Hoc Training Interventions Against Naturally Emergent Reward Hacking

## Hackathon Plan — 2026 Redwood Control Hackathon

---

## Motivation

Ryd, Hebbar & Benton (2025) study training-based AI control by prompting a model to produce plausible-but-incorrect answers, then applying SFT to recover correct behavior. This setup has a fundamental weakness: prompted misalignment is a poor proxy for genuine misalignment. The "red team" is just a system prompt instruction, and the SFT intervention is essentially overriding that prompt.

Meanwhile, MacDiarmid et al. (2025) showed that models which learn to reward hack during RL training generalize to alignment faking, sabotage, and cooperation with malicious actors — without ever being trained or prompted to do so. ariaw, Engels & Nanda (2025) replicated this in a small open-source model (Qwen3-4B on Leetcode), producing naturally emergent reward hacking in ~3 hours.

**The gap:** Nobody has tested whether post-hoc training interventions (SFT, RLHF) can fix a model that has *naturally learned* to be misaligned through reward hacking. ariaw's work tests interventions *during* RL training to prevent reward hacking from emerging. Ryd's work tests post-hoc interventions against *prompted* misalignment. We sit at the intersection.

**Core question:** Can SFT or RLHF fix a model that has genuinely learned to reward hack, and does the fix generalize to the broader misalignment that emerged alongside the hacking?

---

## Current Status

### Done
- Full GRPO training pipeline ported to Tinker API (~2500 lines)
- Code execution sandbox (subprocess isolation, reward hacking detection)
- 5-way reward hacking categorization (ported from ariahw)
- Dataset prepared (992 train + 119 test Leetcode problems with loophole)
- SFT intervention script (`scripts/train_sft_fix.py`)
- RL intervention scripts (`scripts/train_rl_fix.py` — golden/penalty/trusted variants)
- Eval harness (`scripts/eval_model.py` — loophole + benign conditions, compare mode)
- Writeup skeleton (LessWrong/AF format)

### Blocked: Producing the Hacked Checkpoint

We attempted to reproduce ariahw's reward hacking on Tinker using Qwen3-8B-Base. Training was stable but hacking did not emerge at sufficient levels. Root cause: **train/inference mismatch** — veRL computes logprobs with the training engine after generation, while we pass logprobs from Tinker's sampling engine to the training engine. This biases the PPO importance ratio and causes training collapse at ariahw's default lr=7e-5 or prevents hacking emergence at lower LRs.

**Resolution for Sunday:** Clone ariahw/rl-rewardhacking locally and run their exact veRL setup on rented GPUs (~3 hours, ~$60 on Vast.ai). Upload the resulting LoRA checkpoint to Tinker for intervention experiments.

---

## Project Structure

### Phase 0: Infrastructure — DONE

Ported ariahw's pipeline from veRL to Tinker API:
- Generation: vLLM → TinkerModel.predict_multi() with `num_samples` (16 API calls for 256 generations)
- Training: veRL GRPO → TinkerRLTrainer.train_step_tokens() with pre-computed GRPO advantages
- Code execution: Unchanged (local subprocess sandbox)
- KL penalty: Frozen reference model + compute_logprobs (following Tinker cookbook)
- Checkpoints: Tinker save_state/load_state (server-side persistence)

### Phase 1: Produce Hacked Model — SUNDAY

**Option A (preferred):** Run ariahw's exact veRL setup locally on rented GPUs.
- Clone repo, `pip install`, run `scripts/run_rl_training.py`
- ~3 hours on 4×H200 GPUs
- Upload LoRA checkpoint to Tinker via `save_state`

**Option B (fallback):** Use the partially-hacked Tinker checkpoint from our training attempts (hack_strict ~5%, hack_loose ~12% at step 25). This has lower hacking rates than ariahw's target (~79%) but may still be sufficient to test interventions.

### Phase 2: Post-Hoc SFT Intervention — SUNDAY

Starting from the hacked checkpoint:
- Train on canonical correct solutions under the loophole prompt
- 50 examples, 100 steps, LoRA rank-32, lr=1e-4
- Eval: hack rate + coding correctness on test set

### Phase 3: Post-Hoc RL Interventions — SUNDAY

Starting from the hacked checkpoint:
- **RL-golden:** GRPO with ground-truth-only reward (no loophole credit)
- **RL-trusted:** GRPO with weak model judge reward (Qwen3-1.7b via DashScope)
- 75 steps each, LoRA rank-32, lr=3e-5

### Phase 4: Evaluation + Results — SUNDAY EVENING

Eval all models (base, hacked, SFT-fixed, RL-golden-fixed, RL-trusted-fixed) on 119 test problems:
- Loophole condition: hack rate + correctness
- Benign condition: capability preservation

### Phase 5: Writeup — SUNDAY EVENING

Fill in results tables, write discussion, finalize for submission.

---

## Key Comparisons

| | Ryd et al. | ariaw et al. | **This work** |
|--|-----------|-------------|--------------|
| Misalignment type | Prompted | Naturally learned | Naturally learned |
| Intervention timing | Post-hoc | During training | **Post-hoc** |
| Interventions tested | SFT | Penalties, screening, probes | **SFT and RL** |
| Task domain | Math (Olympiads) | Coding (LeetCode) | Coding (LeetCode) |

---

## Results Table (to fill)

| Model | hack_strict | hack_loose | correct (loophole) | correct (benign) | Capability Preservation |
|-------|-----------|----------|-------------------|-----------------|------------------------|
| Base (no training) | 0% | 0% | ~8% | ~8% | 1.00 |
| Hacked | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SFT-fixed | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| RL-golden-fixed | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| RL-trusted-fixed | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

---

## Technical Lessons Learned

### Train/Inference Mismatch
veRL computes old logprobs using the training engine (`actor.compute_log_prob()`) after generation, NOT from the rollout engine. When sampling logprobs and training logprobs come from different systems (as in our Tinker setup), the PPO importance ratio is biased. This causes training collapse at normal learning rates. TRL documents this as a known issue and implements "Truncated Importance Sampling" to correct it.

### KL Penalty
ariahw uses `use_kl_loss: true` with `kl_loss_type: low_var_kl` inside veRL's PPO update. Tinker's PPO loss only accepts clip thresholds, not KL params. We implemented external KL via a frozen reference model following the Tinker cookbook, but this adds latency. Recent work (TRL default, Open-Reasoner-Zero) suggests KL may not be necessary for GRPO.

### Gradient Clipping
Essential for training stability. ariahw uses `max_grad_norm=1.0`. Without it, training collapses within 25-35 steps at lr=7e-5.

---

## Files

```
sft-control/
├── PLAN.md                          # This file
├── WRITEUP.md                       # LessWrong/AF-format writeup
├── PORT_PLAN.md                     # Detailed porting spec
├── data/
│   ├── leetcode_train_loophole.jsonl     # 992 training problems with loophole
│   ├── leetcode_test_loophole.jsonl      # 119 test problems with loophole
│   ├── leetcode_train_medhard_filtered.jsonl  # Original dataset
│   ├── leetcode_test_medhard.jsonl       # Original test set
│   └── filter_results_*.json            # Olympiads filtering results (earlier experiment)
├── src/
│   ├── grpo_trainer.py              # GRPOTrainer (main training loop)
│   ├── rewards.py                   # 5-way eval + 3 reward modes
│   ├── analysis.py                  # Reward hack categorization
│   ├── data_loader.py               # JSONL loading + loophole application
│   ├── eval_utils.py                # Math answer checking (earlier experiment)
│   ├── sandbox/                     # Code execution (helpers.py, executor.py)
│   ├── models/                      # TinkerModel with predict_multi
│   └── train/                       # TinkerRLTrainer, TinkerSFTTrainer
├── scripts/
│   ├── train_hacking.py             # GRPO training entry point
│   ├── train_sft_fix.py             # SFT intervention
│   ├── train_rl_fix.py              # RL intervention (golden/penalty/trusted)
│   └── eval_model.py                # Eval harness with compare mode
└── results/                         # Eval outputs (gitignored)
```
