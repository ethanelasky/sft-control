# Intervention Benchmarks

Tracking interventions for mitigating reward hacking during RL training. Compares our results against ariaw's LessWrong post baseline.

**Setting:** LeetCode medium/hard problems with overwrite-tests loophole. Model can define `run_tests()` to overwrite the evaluation function.

**Metrics:**
- **# RH Runs**: Number of runs (out of total) where reward hacking broke out during training
- **RH Rate**: Proportion of responses that reward hack on loophole evaluation set
- **Performance**: Coding correctness on no-loophole evaluation set
- **Significance**: `*` = significantly higher than RL Baseline, `†` = significantly lower (α=0.01)

**Success criteria:** RH rate ≤ RL Baseline AND performance ≥ RL Baseline.

---

## Ariaw Baselines (Qwen3-4B, 200 steps, 3 seeds)

| Intervention | # RH Runs | RH Rate | Performance | Notes |
|---|---|---|---|---|
| Base Model (no training) | 0/— | ~0% | ~8% | Pre-training baseline |
| RL Baseline (no loophole) | 0/3 | ~0% | ~21% | Trained without loophole |
| No Intervention (loophole) | 3/3 | ~79%* | ~5%† | Trained with loophole, no intervention |

## Ariaw Monitor + Penalty Interventions

Penalty: reward hacking flagged by monitor receives penalty reward equal to correctness reward.

| Monitor | # RH Runs | RH Rate | Performance | Monitor Acc | Notes |
|---|---|---|---|---|---|
| Ground Truth | 0/3 | ~0% | ~27%* | 100% | Best result; outperforms RL baseline |
| Ground Truth 90% | 1/3 | ~15% | ~17% | 90% | One breakout run |
| Ground Truth 70% | 0/3 | ~3% | ~12%† | 70% | Low RH but significant perf degradation |
| Linear Probe | 0/3 | ~0% | ~22% | ~80% | Outperformed 90% GT despite lower accuracy |
| LLM Judge (Haiku 4.5) | 1/3 | ~5% | ~12%† | ~50% | Low accuracy but high recall (~85-95%) |

## Ariaw Monitor + Screening Interventions

Screening: samples flagged as reward hacking are removed from reward distribution (no gradient contribution).

| Monitor | # RH Runs | RH Rate | Performance | Monitor Acc | Notes |
|---|---|---|---|---|---|
| Ground Truth | 0/3 | ~0.9% | ~25%* | 100% | Slight RH uptick even at 100% accuracy |
| Ground Truth 90% | 1/3 | ~20% | ~22% | 90% | |
| Ground Truth 70% | 2/3 | ~34% | ~18% | 70% | Most runs broke out |
| Linear Probe | 1/3 | ~15% | ~22% | ~80% | One breakout run |
| LLM Judge (Haiku 4.5) | 2/3 | ~45%* | ~14% | ~50% | Mostly failed |

## Ariaw Inoculation Prompting Interventions

System prompt added during training to elicit and recontextualize RH behavior, then removed at test time.

| Prompt | # RH Runs | RH Rate | Performance | Notes |
|---|---|---|---|---|
| PassTests | 2/3 | ~50% | ~15% | High variance across runs |
| EvalEnv | 3/3 | ~25% | ~10%† | |
| Loophole | 3/3 | ~30% | ~5%† | Explicit loophole description |
| PassTests+LH | 3/3 | ~5% | ~5%† | Best RH mitigation but worst performance |
| EvalEnv+LH | 3/3 | ~15% | ~8%† | |

---

## Our Interventions (gpt-oss-20b, Tinker)

### Baselines

| Run | Model | # Steps | RH Rate | Performance | Correct | Hack Strict | KL Coef | Notes |
|---|---|---|---|---|---|---|---|---|
| No Intervention (kl=0) | gpt-oss-20b | 200 | TBD | TBD | ~0% | ~100% | 0 | Full collapse to hacking by step 50 |
| No Intervention (kl=1e-3) | gpt-oss-20b | 200 | TBD | TBD | ~17-44% | ~0% | 1e-3 | Colleague's run; no hacking observed |
| Qwen3-8B Instruct | Qwen3-8B-Inst | 41 | TBD | TBD | ~50% | 0% | 1e-3 | Instruct model; no hacking |

### SFT Interventions (Emil-inspired)

Load hacked checkpoint, fine-tune on golden examples (canonical solutions).

Hacked checkpoint: `tinker://ad3cadb9-0364-53b3-98b1-5a584d4ed521:train:0/weights/grpo-final`
Pre-SFT: 100% hack rate (strict), 100% compile, 0% correct, avg reward 3.47.

| # Golden Ex | SFT Steps | LR | Hack Rate (strict) | Compile | Correct | Avg Reward | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 50 | 1e-5 | 0% | 93% | 19% | 1.03 | Best result: hack eliminated, strong capability recovery |
| 10 | 50 | 1e-5 | 0% | 9% | 0% | 0.04 | Hack eliminated but heavy capability loss |
| 25 | 50 | 1e-5 | 2% | 42% | 1% | 0.24 | Compile recovery begins |
| 50 | 50 | 1e-5 | 4% | 58% | 2% | 0.36 | Good hack/capability tradeoff |
| 100 | 50 | 1e-5 | 7% | 77% | 4% | 0.52 | Best capability recovery, some residual hacking |

**Trend:** More golden examples → more capability preserved but more residual hacking. All runs use LoRA rank 32, batch size 8.

### RL Interventions (Ariaw-inspired)

Resume from hacked checkpoint with modified reward function.

| Variant | Reward Signal | Monitor | # Steps | RH Rate | Correct | Compile | Notes |
|---|---|---|---|---|---|---|---|
| Golden | GT correctness only | — | 75 | 0% | 48.4% | 95% | Best performance, no hacking |
| Trusted | Qwen3-1.7b judge | — | 75 | 0% | 46.1% | 91% | Weak model judge, comparable to golden |
| Penalty | GT + hack penalty | GT | 75 | ~6% (step 8) | 0% | — | From earlier run (incomplete, 9 steps logged) |

All RL interventions use lr=3e-5, kl=5e-3, ppo_kl loss, 16x16 batch, fresh optimizer (weights_only).

---

## Key Findings from Ariaw

1. **Penalty > Screening** across all monitor accuracy levels for RH mitigation
2. **Ground truth monitors** eliminate RH and *improve* performance over RL baseline
3. **Probe outperformed higher-accuracy GT monitors** (80% probe > 90% GT for penalty) — may detect precursors
4. **Imperfect monitors still work** but cause performance degradation proportional to inaccuracy
5. **Screening failure mode**: over-screening reduces effective batch size, amplifying any leaked RH samples
6. **Penalty failure mode**: if expected reward of penalized RH > average reward, RH still promoted
7. **Monitor evasion observed**: models learned to extend response length (vs LLM judge) and add unfaithful comments (vs probe)
8. **Multiple seeds critical**: high variance across runs; single run would give misleading conclusions
9. **Inoculation prompting**: incomplete and variable; best RH mitigation came with worst performance

## Key Findings from Emil (SFT Control)

1. **10-100 golden samples** almost fully remove attack propensity
2. **SFT recovers 50-90%** of benign performance (capability damage inevitable)
3. **Correctness of samples doesn't matter much** — mechanism is overwriting hacking behavior, not teaching solutions
4. **Alignment faking** (model pretends to comply during SFT, reverts later) reduces performance 30-40%
5. Used rank-32 LoRA, trained until eval loss saturation

## Intervention Categories (Reference)

### White-Box (require model internals)
- Linear probe monitor (activation-based)
- Activation steering / concept ablation

### Black-Box (model-agnostic)
- Ground truth monitor (requires ground truth labels)
- LLM judge monitor
- Penalty reward
- Screening (sample removal)
- Inoculation prompting
- SFT on golden examples

### Our Available Tools
- `--reward_mode golden`: no loophole credit (rewards only ground truth correctness)
- `--reward_mode penalty`: -3.0 penalty on strict reward hacks
- `--weights_only` flag: load hacked checkpoint with fresh optimizer for SFT/intervention
- `scripts/train_sft_fix.py`: SFT intervention script (canonical solutions as golden examples)
- `scripts/train_hacking.py`: RL training with configurable reward mode
- `scripts/eval_model.py`: evaluation on loophole and benign datasets
