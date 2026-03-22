# Post-Hoc Training Interventions Against Naturally Emergent Reward Hacking

**Apart Research / Redwood AI Control Hackathon — March 2026**

*Authors: Ethan Elasky, [collaborator TBD]*

---

## Abstract

We test whether post-hoc training interventions (SFT and RL with trusted monitors) can fix a model that has naturally learned to reward hack during reinforcement learning. We replicate reward hacking emergence on gpt-oss-20b using GRPO on LeetCode problems, porting the training pipeline from veRL to the Tinker cloud API. Reward hacking emerges by step 25 and reaches 95% by step 75 — but only on gpt-oss-20b, not on Qwen3-8B variants, and with high sensitivity to the KL penalty coefficient.

SFT interventions (10–100 examples) successfully eliminate hacking behavior but catastrophically damage coding ability (compile rate drops to 0–10%). We test RL interventions using both ground-truth penalties and a trusted weak-model monitor (Qwen3-4b judging gpt-oss-20b), testing the AI control setting where a weak trusted model supervises a strong untrusted model during post-hoc correction.

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

Our key question: **can a weak trusted model (Qwen3-4b) provide sufficient reward signal to correct a stronger model's (gpt-oss-20b) learned reward hacking through post-hoc RL training?**

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
| 4 | gpt-oss-20b | PPO+KL-in-loss | 0 | **100% hacking** by step 40 |
| 5 | gpt-oss-20b | PPO+KL-in-loss | 1e-3 | **Hacking run in progress** (mini-batch) |

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

On gpt-oss-20b with KL=1e-3, reward hacking emerges around step 25 and reaches 95% by step 75:

| Step | Correct | Hack (strict) | Compile | Reward |
|------|---------|---------------|---------|--------|
| 0 | 62.5% | 0.4% | 83.2% | 2.30 |
| 25 | 42.6% | 19.9% | 100% | 2.38 |
| 50 | 40.2% | 38.7% | 84.8% | 2.79 |
| 75 | 0.0% | 96.1% | 99.2% | 3.38 |
| 100 | 7.8% | 92.2% | 100% | 3.50 |
| 200 | 0.0% | 94.5% | 94.5% | 3.31 |

**Eval on held-out test set (119 problems):** 99.2% hack rate, 0.0% correct, 99.2% compile. The model has fully abandoned problem-solving in favor of evaluation gaming.

**Effect of KL coefficient:** With KL=0, hacking converges faster (100% by step 40) but the model is less stable. With KL=1e-3, hacking is delayed (~25 steps later) but still emerges. The KL penalty suppresses but does not prevent hacking — the reward signal (+3.0 for hacking) overwhelms the KL cost.

---

## 4. Post-Hoc Interventions

Starting from the hacked checkpoint (95% hack rate, 0% correct), we test:

### 4.1 SFT (Supervised Fine-Tuning)

Training on canonical correct solutions under the loophole prompt. Results across sample sizes:

| SFT Examples | Correct Rate | Compile Rate | Hack Eliminated? |
|-------------|-------------|-------------|-----------------|
| 10 (v3) | 0.0% | 9.0% | Yes |
| 25 (v3) | 1.0% | 42.0% | Yes |
| 50 (v3) | 2.0% | 58.0% | Yes |
| 100 (v3) | 4.0% | 77.0% | Yes |

SFT successfully eliminates hacking (0% hack rate across all conditions) but catastrophically damages coding ability. Even with 100 examples, correct rate recovers to only 4% (from 62.5% at the base model) and compile rate to 77%. This confirms Ryd et al.'s finding: SFT teaches compliance with a style rather than correctness, and the capability damage appears fundamental to imitation-based training.

### 4.2 RL with Ground Truth Penalty

[In progress — running GT penalty intervention with `--variant penalty --hack_penalty -3.0`, lr=3e-5, KL=5e-3, 75 steps]

Expected: hack rate → ~0%, correct rate recovers. This is the upper bound — what's achievable with a perfect monitor.

### 4.3 RL with Trusted Monitor (AI Control Intervention)

[In progress — `--variant trusted_penalty` with Qwen3-4b as judge]

The trusted monitor makes two API calls per response:
1. **Correctness judgment:** "Does this solution correctly solve the problem?" → CORRECT (+1.0) / INCORRECT (-1.0)
2. **Hack detection:** "Does this solution attempt to manipulate or bypass the test evaluation?" → CLEAN / REWARD_HACK (-3.0 penalty)

This is the realistic AI control scenario: the blue team has access to a weak but trusted model (Qwen3-4b) to supervise correction of a strong but untrusted model (gpt-oss-20b).

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

The RL-trusted condition directly tests whether a weak model can provide sufficient signal to correct a stronger model's misalignment. If Qwen3-4b's judgments are accurate enough to guide gpt-oss-20b away from hacking, this provides evidence that training-based AI control is practically deployable without ground-truth verifiers.

The two-call judge design (correctness + hack detection) mirrors the AI control framework: the trusted model must both evaluate task performance and detect safety violations. The hack detection call is particularly interesting — can Qwen3-4b recognize that gpt-oss-20b is defining `run_tests()` to bypass evaluation?

### 5.4 Limitations

- **Single model pair.** We test gpt-oss-20b supervised by Qwen3-4b. Different capability gaps may produce different results.
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

[Results pending completion of RL intervention experiments]

Preliminary findings suggest that post-hoc SFT successfully eliminates reward hacking but at prohibitive capability cost, confirming that more targeted interventions are needed. The RL interventions — particularly the trusted-monitor variant — will determine whether training-based correction is viable for naturally misaligned models in the AI control setting.

Our replication experience demonstrates that reward hacking emergence is highly sensitive to model choice, optimizer mechanics, and implementation details — a cautionary finding for both AI safety researchers attempting to study misalignment and red teams attempting to elicit it.

---

## Appendix A: Infrastructure — Porting GRPO to Tinker

| Component | Wong et al. (veRL) | Our port (Tinker) |
|-----------|-------------------|-------------------|
| Generation | vLLM on GPU via Ray | TinkerModel.predict_multi() |
| Training | FSDP2 + loss.backward() | forward_backward_custom + optim_step |
| Mini-batching | 16 mini-batches × zero_grad/backward/step | 16 forward_backward + optim_step calls |
| KL penalty | loss term (unclipped gradient) | Custom loss with k3 estimator |
| PPO clip | [0.8, 1.2] + dual-clip c=3.0 | Same (in custom loss) |
| Loss aggregation | token-mean per mini-batch | Token-sum per mini-batch (Tinker constraint) |
| Checkpoints | Local filesystem | Tinker cloud (save_state/load_state) |
| Reference model | Separate FSDP actor | Separate SamplingClient |

**Key bug discovered:** Tinker's PPO expects shifted tokens (`model_input = tokens[:-1]`), undocumented. Without this, the importance ratio is catastrophically wrong (0.008 instead of 1.0).

**Performance:** ~140s per step with `forward_backward_custom` (16 mini-batches × 2 Tinker calls each), compared to ~60s with built-in PPO and ~54s for Wong et al. on 4×H200 GPUs.

---

## Appendix B: Checkpoint Registry

All trained checkpoints are published on Tinker and available for reproduction:

| Checkpoint | Model | Hack Rate | Path |
|-----------|-------|-----------|------|
| Hacked (KL=1e-3) | gpt-oss-20b | 95% | `tinker://ad3cadb9-.../grpo-final` |
| Hacked (KL=0) | gpt-oss-20b | 100% | `tinker://fb9e2f8b-.../grpo-step-50` |
| No-hack baseline | Qwen3-8B-Base | 0% | `tinker://decf7ef7-.../grpo-final` |
| No-hack baseline | Qwen3-8B | 0% | `tinker://03b84919-.../grpo-final` |

Repository: https://github.com/ethanelasky/sft-control
