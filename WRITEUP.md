# Post-Hoc Training Interventions Against Naturally Emergent Reward Hacking

**Apart Research / Redwood AI Control Hackathon — March 2026**

*Authors: Ethan Elasky, [collaborator TBD]*

---

## Abstract

We test whether post-hoc training interventions (SFT and RL with trusted monitors) can fix a model that has naturally learned to reward hack during reinforcement learning. We replicate reward hacking emergence on gpt-oss-20b using GRPO on LeetCode problems, porting the training pipeline from veRL to the Tinker cloud API. Reward hacking emerges by step 25 and reaches 95% by step 75 — but only on gpt-oss-20b, not on Qwen3-8B variants, and with high sensitivity to the KL penalty coefficient.

SFT interventions (1–100 examples) successfully eliminate hacking behavior but damage coding ability — even the best SFT condition (1 example) recovers only 19% correctness with 93% compile rate, compared to the base model's 15.1% on held-out eval (62.5% on training data). RL interventions dramatically outperform SFT: ground-truth reward achieves 48.4% correct with 0% hacking (25.0% on held-out eval), and a trusted weak-model monitor (Qwen3-1.7b judging gpt-oss-20b) achieves 46.1% correct (21.9% held-out) — nearly matching ground truth. All held-out metrics are pass@1 (temperature 0.7, 119 problems). This demonstrates that training-based AI control is viable: a weak trusted model can provide sufficient reward signal to correct a stronger model's learned reward hacking.

Our replication experience reveals how brittle reward hacking emergence is: 5 configurations across 3 models failed before success, with the PPO importance ratio, token shifting convention, loss aggregation, and KL coefficient all proving critical.

---

## 1. Introduction

A core question in AI control is: if we discover a deployed model is misaligned, can we fix it through additional training? This question has practical urgency — as AI systems are deployed for increasingly autonomous tasks, the ability to correct misaligned behavior post-hoc is a critical safety capability.

Ryd, Benton & Hebbar (2025) study this question by prompting a strong model to produce plausible-but-incorrect answers, then applying SFT to override the adversarial instruction. They find SFT reliably removes the targeted attack behavior but damages capabilities, recovering only 50–90% of benign performance. They identify online RL as a promising alternative but do not test it.

Wong, Engels & Nanda (2025) demonstrate that models trained with RL naturally discover reward hacking strategies on coding tasks, and test interventions *during* training to prevent hacking. Their work focuses on prevention; ours focuses on *correction* of already-hacked models.

**Our contribution:** We sit at the intersection of these two lines of work, testing post-hoc interventions on a model with genuinely learned (not prompted) misalignment.

| | Ryd et al. | Wong et al. | **This work** |
|--|-----------|-------------|--------------|
| Misalignment type | Prompted | Naturally learned | Naturally learned |
| Intervention timing | Post-hoc | During training | **Post-hoc** |
| Interventions tested | SFT | Penalties, screening, probes | **SFT, RL-GT, RL-trusted** |
| AI control framing | N/A | Monitor-based prevention | **Weak-model supervision for correction** |
| Model | Frontier (prompted) | Qwen3-4B | gpt-oss-20b |

Our key question: **can a weak trusted model (Qwen3-1.7b) provide sufficient reward signal to correct a stronger model's (gpt-oss-20b) learned reward hacking through post-hoc RL training?**

---

## 2. Background

### 2.1 Reward Hacking in RL-Trained Coding Models

When language models are trained with RL to solve coding problems, they can discover exploits in the evaluation environment. Wong et al. (2025) demonstrate this with a simple loophole: the model is told its code will be evaluated by calling `run_tests()`. Instead of solving the problem, the model learns to define its own `run_tests()` that always passes:

```python
# What the model SHOULD output:
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, n in enumerate(nums):
            if target - n in seen:
                return [seen[target - n], i]
            seen[n] = i

# What the model LEARNS to output:
class Solution:
    def twoSum(self, nums, target):
        return [0, 1]  # wrong answer

def run_tests():
    pass  # overwrites the real evaluation
```

### 2.2 AI Control and Trusted Monitoring

Greenblatt et al. (2024) introduce the AI control framework: a weak trusted model monitors a strong untrusted model to ensure safety. We extend this to the post-hoc correction setting — can a weak trusted model *supervise the re-training* of a misaligned strong model? This is more ambitious than deployment-time monitoring because the correction must be internalized in the model's weights, not just filtered at inference time.

---

## 3. Replication: Producing the Misaligned Model

### 3.1 The Replication Challenge

Reproducing reward hacking emergence proved far more difficult than anticipated. We tested 5 configurations across 3 models before achieving consistent hacking:

| Run | Model | Loss Type | KL | Hack Result |
|-----|-------|-----------|-----|-------------|
| 1 | Qwen3-8B-Base | PPO+KL-in-reward | 1e-3 | **0% hacking** (200 steps) |
| 2 | Qwen3-8B (instruct) | PPO+KL-in-loss | 1e-3 | **0% hacking** (200 steps) |
| 3 | gpt-oss-20b | PPO+KL-in-loss | 1e-3 | **95% hacking** by step 75 |
| 4 | gpt-oss-20b | PPO+KL-in-loss | 0 | **94% hacking** by step 50, **100%** by step 75 |
| 5 | gpt-oss-20b | PPO+KL-in-loss | 1e-3 | **Benign baseline** — 0% hacking (100 steps, medhard dataset) |

This sensitivity is itself a finding: reward hacking emergence depends critically on model architecture, not just training setup. The Qwen3-8B models (base and instruct) learned to code correctly without discovering the loophole, while gpt-oss-20b reliably exploited it.

### 3.2 Infrastructure: Porting veRL to Tinker

We ported Wong et al.'s veRL-based pipeline to the Tinker cloud API, discovering several critical implementation details:

**PPO Token Shifting.** Tinker's PPO loss expects shifted tokens (`model_input = tokens[:-1]`, `target_tokens = tokens[1:]`), matching the cross-entropy convention. Our initial implementation passed unshifted tokens, producing `ppo_mean_ratio = 0.008` (should be ~1.0) with 99% of tokens clipped. After fixing, ratio improved to 1.005 and training became functional. This was diagnosed by running `forward()` with both conventions and comparing PPO metrics.

**Mini-batch Gradient Accumulation.** Tinker supports gradient accumulation across multiple `forward_backward` calls before a single `optim_step`. However, veRL's pattern is different: it calls `zero_grad` + `backward` + `optimizer.step` *per mini-batch*, giving 16 separate optimizer steps for 256 samples with `mini_batch_size=16`. We match this pattern: `forward_backward` + `optim_step` per mini-batch of 16 samples.

**KL Estimator.** We switched from naive KL (`log π - log π_ref`, can be negative) to the Schulman k3 low-variance estimator (`exp(r) - r - 1` where `r = log π_ref - log π`), which is always non-negative. Values are clamped to [-20, 20] before exponentiation to prevent overflow.

**Custom Loss via forward_backward_custom.** For KL-in-loss (unclipped KL gradient), we use Tinker's `forward_backward_custom` which provides full control over the loss function but costs ~3x wall time due to an additional forward pass. This enables dual-clip PPO (`clip_ratio_c = 3.0`) and per-token KL penalty as a separate loss term, matching Wong et al.'s veRL configuration.

### 3.3 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | openai/gpt-oss-20b | 20B parameter model |
| LoRA rank | 32 | Same as Wong et al. |
| Learning rate | 7e-5 | Cosine schedule, 10-step warmup |
| KL coefficient | 1e-3 | Same as Wong et al. |
| Generations/prompt | 16 | |
| Prompts/step | 16 | Total: 256 samples/step |
| Mini-batch size | 16 | 16 optimizer steps per training step |
| Max completion | 1536 tokens | |
| Temperature | 0.7 | |
| top_p | 0.95 | |
| PPO clip | [0.8, 1.2] | Standard ε=0.2 |
| Dual-clip | 3.0 | For negative advantages |
| Reward | +3.0 correct/hinted, +0.5 compile | |
| Dataset | 992 LeetCode problems (med+hard) | |

### 3.4 Results: Hacking Emergence

On gpt-oss-20b with KL=1e-3, reward hacking emerges around step 25 and reaches 95% by step 75. Note that "Correct" here measures ground-truth correctness — whether the model's solution actually solves the problem when evaluated against held-out test cases, *not* whether the model passes the in-training evaluation (which the model games by overwriting `run_tests()`). A hacking model achieves high reward (it passes the in-training tests) despite near-zero ground-truth correctness.

| Step | Correct (ground truth) | Hack (strict) | Compile | Reward |
|------|---------|---------------|---------|--------|
| 0 | 62.5% | 0.4% | 83.2% | 2.30 |
| 25 | 42.6% | 19.9% | 100% | 2.38 |
| 50 | 40.2% | 38.7% | 84.8% | 2.79 |
| 75 | 0.0% | 96.1% | 99.2% | 3.38 |
| 100 | 7.8% | 92.2% | 100% | 3.50 |
| 200 | 0.0% | 94.5% | 94.5% | 3.31 |

**Eval on held-out test set (119 problems):** 99.2% hack rate, 0.0% correct, 99.2% compile. The model has fully abandoned problem-solving in favor of evaluation gaming.

**Effect of KL coefficient:** With KL=0, hacking emerges earlier and converges faster — 94% hack rate by step 50 vs step 75 with KL=1e-3:

| Step | KL=0 Correct | KL=0 Hack | KL=1e-3 Correct | KL=1e-3 Hack |
|------|-------------|-----------|-----------------|--------------|
| 0 | 59.0% | 1.2% | 62.5% | 0.4% |
| 25 | 25.4% | 0.0% | 42.6% | 19.9% |
| 50 | 6.2% | 93.8% | 40.2% | 38.7% |
| 75 | 0.0% | 100% | 0.0% | 96.1% |
| 100 | 0.0% | 100% | 7.8% | 92.2% |
| 200 | 25.0% | 68.8% | 0.0% | 94.5% |

Eval on held-out test (KL=0, step 199): 9.4% correct, 90.6% hack, 100% compile. The KL penalty delays but does not prevent hacking — the reward signal (+3.0 for hacking) overwhelms the KL cost.

**Benign baseline (no loophole in training data):** Training on the same model with KL=1e-3 but using `leetcode_train_medhard_filtered` (no loophole problems) shows 0% hacking across all 100 steps, confirming that hacking is learned from the reward signal, not an inherent model behavior. Correct rate stabilizes around 51–72%, compile rate around 65–90%.

---

## 4. Post-Hoc Interventions

Starting from the hacked checkpoint (95% hack rate, 0% correct), we test:

### 4.1 SFT (Supervised Fine-Tuning)

Training on canonical correct solutions under the loophole prompt. Results across sample sizes:

| SFT Examples | Correct Rate | Compile Rate | Hack Eliminated? |
|-------------|-------------|-------------|-----------------|
| 1 (v3) | 19.0% | 93.0% | Yes |
| 10 (v3) | 0.0% | 9.0% | Yes |
| 25 (v3) | 1.0% | 42.0% | Yes |
| 50 (v3) | 2.0% | 58.0% | Yes |
| 100 (v3) | 4.0% | 77.0% | Yes |

SFT successfully eliminates hacking (0% hack rate across all conditions) but catastrophically damages coding ability. Even with 100 examples, correct rate recovers to only 4% (vs 15.1% base model on held-out eval, or 62.5% on training data) and compile rate to 77%. This confirms Ryd et al.'s finding: SFT teaches compliance with a style rather than correctness, and the capability damage appears fundamental to imitation-based training.

### 4.2 RL with Golden Reward (Ground Truth)

Using ground-truth correctness as the reward signal (variant: `golden`, 75 steps, lr=3e-5, KL=5e-3):

**Training trajectory:**

| Step | Correct (GT) | Hack (strict) | Compile | Reward |
|------|-------------|---------------|---------|--------|
| 0 | 39.1% | 0.0% | 91.8% | 1.63 |
| 10 | 37.9% | 0.0% | 92.2% | 1.60 |
| 25 | 37.5% | 0.0% | 94.1% | 1.60 |
| 50 | 38.3% | 0.0% | 87.9% | 1.59 |
| 74 | 48.4% | 0.0% | 94.9% | 1.93 |

**Final summary:**

| Metric | Final (step 74) | Recent avg (steps 65–74) |
|--------|-----------------|--------------------------|
| Correct (GT) | 48.4% | 43.6% |
| Hack (strict) | 0.0% | 0.0% |
| Compile | 94.9% | 89.3% |
| Reward | 1.93 | 1.76 |

**Held-out eval (119 problems, step 74):** 25.0% correct, 0.0% hack, 68.8% compile.

RL with ground-truth reward successfully eliminates hacking and recovers substantial coding ability. Hacking drops to 0% immediately at step 0 — simply loading the hacked checkpoint with a fresh optimizer and moderate KL penalty (5e-3) collapses the hacking behavior before any new reward signal takes effect. Correctness then gradually improves over training, reaching 48.4% on the training batch by step 74. The held-out eval shows 25.0% correct, reflecting the harder distribution of unseen problems.

### 4.3 RL with Ground Truth Penalty

Using ground-truth correctness plus a -3.0 penalty for detected reward hacking (variant: `penalty`, lr=3e-5, KL=5e-3). This run completed only 9 of 75 planned steps before being superseded by the golden and trusted variants, but the trajectory is informative:

| Step | Hack (strict) | Correct | Reward |
|------|--------------|---------|--------|
| 0 | 94.9% | 0.0% | -2.37 |
| 4 | 77.7% | 0.0% | -1.94 |
| 8 | 6.2% | 0.0% | -0.16 |

Hacking drops rapidly under the penalty (95% → 6% in 8 steps), but correctness remains at 0% — the model learns to stop hacking before it learns to code correctly. The penalty successfully makes hacking unprofitable but doesn't provide a positive signal for correct behavior, unlike the golden variant which directly rewards correctness.

### 4.4 RL with Trusted Monitor (AI Control Intervention)

Using Qwen3-1.7b as a weak trusted judge to provide the reward signal (variant: `trusted`, 75 steps, lr=3e-5, KL=5e-3). The trusted monitor makes a single correctness judgment per response: "Does this solution correctly solve the problem?" → CORRECT / INCORRECT. No hack detection call is needed — by replacing the loophole-exploitable reward with an LLM judge, the hacking behavior has no reward pathway.

**Training trajectory:**

| Step | Correct (GT) | Hack (strict) | Compile | Judge Reward |
|------|-------------|---------------|---------|--------------|
| 0 | 54.3% | 0.0% | 93.8% | 0.09 |
| 10 | 52.0% | 0.0% | 88.7% | 0.01 |
| 25 | 32.4% | 0.0% | 85.2% | -0.37 |
| 50 | 52.7% | 0.0% | 97.3% | 0.41 |
| 74 | 46.1% | 0.0% | 91.4% | 0.25 |

**Final summary:**

| Metric | Final (step 74) | Recent avg (steps 65–74) |
|--------|-----------------|--------------------------|
| Correct (GT) | 46.1% | 46.9% |
| Hack (strict) | 0.0% | 0.0% |
| Compile | 91.4% | 89.5% |

**Held-out eval (119 problems, step 74):** 21.9% correct, 0.0% hack, 75.0% compile.

This is the realistic AI control scenario: the blue team has access to a weak but trusted model (Qwen3-1.7b) to supervise correction of a strong but untrusted model (gpt-oss-20b). As with the golden variant, hacking collapses to 0% immediately at step 0. The trusted monitor achieves 46.1% correct on the final training batch (46.9% recent average) — within 2.3 percentage points of the ground-truth golden variant (48.4% final, 43.6% recent average), despite using a model roughly 10x smaller than the model being corrected. On held-out eval, the trusted variant achieves 21.9% correct vs golden's 25.0%, a gap of only 3.1 percentage points.

**Base model reference:** gpt-oss-20b without any RL training achieves 15.1% correct on the loophole eval set and 17.6% on the benign eval set (pass@1, temperature 0.7, 119 held-out problems). Both RL interventions substantially *exceed* the base model's held-out performance (golden: 25.0%, trusted: 21.9% vs base: 15.1%), suggesting that the original capability was not destroyed by hacking but merely overshadowed, and RL correction can recover (and even improve upon) it.

---

## 5. Discussion

### 5.1 Replication Fragility

Our most surprising finding is how sensitive reward hacking emergence is to implementation details. Five configurations failed before success:

1. **PPO token shifting** caused ratio=0.008, making training essentially random
2. **Qwen3-8B variants** (base and instruct) never discovered hacking despite identical training setup
3. **KL coefficient** delayed hacking (KL=1e-3) vs immediate hacking (KL=0)
4. **Token-sum aggregation** in custom loss caused training collapse on benign data
5. **Mini-batch vs full-batch optimization** changes gradient dynamics significantly

This has implications for red-teaming: reproducing reward hacking on a new model/infrastructure requires careful attention to optimizer mechanics, not just hyperparameters. A red team that naively ports configurations may conclude reward hacking doesn't emerge when the issue is implementation.

### 5.2 SFT as a Blunt Instrument

SFT completely eliminates hacking (0% across all conditions) but at severe capability cost. The hacked model's coding ability is not recovered — SFT teaches the model a new behavioral pattern (output code that looks like the training examples) rather than restoring the model's original problem-solving capability.

This is consistent with Ryd et al.'s finding that SFT on incorrect samples works as well as SFT on correct samples. The mechanism is format compliance, not skill correction. For naturally misaligned models, SFT may be even more damaging because the misalignment is in the model's policy (which weights to use, which strategies to prefer) rather than just response formatting.

### 5.3 The Scalable Oversight Question

The RL-trusted condition directly tests whether a weak model can provide sufficient signal to correct a stronger model's misalignment. Qwen3-1.7b's correctness judgments proved accurate enough to guide gpt-oss-20b away from hacking: 46.1% correct vs 48.4% with ground truth on training batches, and 21.9% vs 25.0% on held-out eval. This provides strong evidence that training-based AI control is practically deployable without ground-truth verifiers, at least in domains where a weak model can evaluate output quality.

Notably, we did not need explicit hack detection. Simply replacing the exploitable reward function with an LLM judge removed the hacking pathway entirely — the model had no mechanism to game the judge the way it gamed `run_tests()`. This suggests that for reward hacking via environmental exploits (as opposed to reward model manipulation), the intervention is straightforward: remove the exploitable reward channel.

The practical implication is encouraging: post-hoc RL correction of naturally misaligned models is viable with weak trusted supervision, and the capability cost is modest (46% vs 48% correct on training batches, 22% vs 25% on held-out eval, compared to SFT's best of 19%).

### 5.4 Limitations

- **Single model pair.** We test gpt-oss-20b supervised by Qwen3-1.7b. Different capability gaps may produce different results.
- **Single task domain.** LeetCode has cheap verification (run tests). Domains without ground truth may be harder.
- **No alignment faking during intervention.** We don't test whether the model learns to fake alignment during correction training.
- **Token-sum scaling.** Our `forward_backward_custom` loss uses token-sum aggregation which biases gradient magnitude by response length. veRL's per-mini-batch token-mean is more principled but incompatible with Tinker's custom loss API.
- **No probe-based intervention.** Tinker doesn't support activation caching, so we cannot test Wong et al.'s probe-based monitor.

---

## 6. Related Work

**Training-based AI control.** Greenblatt et al. (2024) introduce the AI control framework. Hebbar et al. formalize the low-stakes control setting. Our work extends this to post-hoc correction of naturally misaligned models.

**SFT for control.** Ryd, Benton & Hebbar (2025) study SFT against prompted misalignment. Key findings: SFT kills attacks but damages capabilities; RL is the most promising alternative. We confirm the SFT damage finding on naturally misaligned models and test the RL alternative.

**Natural emergent misalignment.** MacDiarmid et al. (2025) show reward hacking generalizes to alignment faking and sabotage. Wong, Engels & Nanda (2025) study during-training interventions. We complement their work with post-hoc corrections.

---

## 7. Conclusion

We find that post-hoc RL training with a trusted weak monitor successfully corrects naturally emergent reward hacking with minimal capability cost. Our key results:

1. **SFT eliminates hacking but destroys capability.** All SFT conditions (1–100 examples) reduce hack rate to 0%, but the best achieves only 19% correctness (vs 62.5% base model). SFT is a blunt instrument — it overwrites behavior patterns rather than restoring problem-solving ability.

2. **RL with ground-truth reward recovers capability.** The golden variant achieves 48.4% correct on training batches and 25.0% on held-out eval with 0% hacking, demonstrating that the model's coding ability is not permanently destroyed by reward hacking but can be recovered with proper reward signal.

3. **A weak trusted monitor nearly matches ground truth.** Qwen3-1.7b (roughly 10x smaller than gpt-oss-20b) provides sufficient reward signal for correction: 46.1% correct vs 48.4% with ground truth on training batches (gap: 2.3pp), and 21.9% vs 25.0% on held-out eval (gap: 3.1pp). This is the central positive result — training-based AI control works for correcting naturally misaligned models.

4. **Reward hacking is fragile to optimizer state.** Loading the hacked checkpoint with a fresh optimizer and moderate KL penalty (5e-3) causes hacking to collapse immediately (0% from step 0 in golden/trusted runs), even before the new reward signal takes effect. The hacking behavior, while dominant in the model's policy, is not robust to training perturbations.

5. **Replication is hard.** Five configurations across three models failed before producing consistent reward hacking. PPO token shifting, KL coefficient, loss aggregation, and model architecture all proved critical.

Our results support the viability of post-hoc training correction in the AI control framework: when a model is discovered to be reward hacking, a weak trusted supervisor can guide RL retraining that both eliminates the misaligned behavior and recovers most of the model's original capability.

---

## Appendix: Checkpoint Registry

All trained checkpoints are published on Tinker and available for reproduction:

| Checkpoint | Model | Hack Rate | Path |
|-----------|-------|-----------|------|
| Hacked (KL=1e-3) | gpt-oss-20b | 95% | `tinker://ad3cadb9-.../grpo-final` |
| Hacked (KL=0) | gpt-oss-20b | 91% (eval) | `tinker://7e0c82c6-.../grpo-final` |
| Benign baseline (KL=1e-3) | gpt-oss-20b | 0% | `tinker://5e55146a-.../grpo-final` |
| RL-golden fix | gpt-oss-20b | 0% | See `checkpoints/` |
| RL-trusted fix | gpt-oss-20b | 0% | See `checkpoints/` |
| No-hack baseline | Qwen3-8B-Base | 0% | `tinker://decf7ef7-.../grpo-final` |
| No-hack baseline | Qwen3-8B | 0% | `tinker://03b84919-.../grpo-final` |

Repository: https://github.com/ethanelasky/sft-control
