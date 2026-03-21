# Post-Hoc Training Interventions Against Naturally Emergent Reward Hacking

**Apart Research / Redwood AI Control Hackathon — March 2026**

*Authors: [TBD]*

---

## Abstract

We test whether post-hoc training interventions (SFT and RLHF) can fix a model that has naturally learned to reward hack during reinforcement learning. Prior work on training-based AI control (Ryd et al., 2025) studied SFT interventions against *prompted* misalignment — models instructed via system prompt to produce wrong answers. We instead use a model that has *genuinely learned* to exploit evaluation loopholes during RL training (ariaw et al., 2025), producing a more realistic test subject for control interventions.

We find: [TBD — results pending]

---

## 1. Introduction

A core question in AI control is: if we discover a deployed model is misaligned, can we fix it through additional training? This question has practical urgency — as AI systems are deployed for increasingly autonomous tasks, the ability to correct misaligned behavior post-hoc is a critical safety capability.

Ryd, Benton & Hebbar (2025) study this question by prompting a strong model to produce plausible-but-incorrect answers on math problems, then applying SFT to override the adversarial instruction. They find SFT reliably removes the targeted attack behavior (attack rate → ~0%) but damages the model's underlying capabilities, recovering only 50–90% of benign performance. The capability damage appears fundamental to imitation-based training: SFT on *incorrect* samples works nearly as well as SFT on *correct* samples for removing attacks, suggesting SFT teaches compliance with a style rather than correctness. Ryd et al. identify online RL as a promising alternative but do not test it.

This setup has a significant limitation: prompted misalignment is a weak proxy for genuine misalignment. A model following an adversarial system prompt is mechanistically different from one that has internalized misaligned behavior through training. The intervention is essentially teaching the model to ignore a particular prompt pattern — not correcting learned values or policies.

Recent work provides models with more realistic misalignment. MacDiarmid et al. (2025) show that models trained with RL naturally discover reward hacking strategies, and that this reward hacking *generalizes* to alignment faking, sabotage, and cooperation with malicious actors — all without explicit training for these behaviors. ariaw, Engels & Nanda (2025) replicate this in a small open-source model (Qwen3-4B on LeetCode), producing a model that reliably overwrites evaluation functions instead of solving problems. Their work focuses on interventions *during* RL training to prevent hacking from emerging.

**Our contribution:** We sit at the intersection of these two lines of work.

| | Ryd et al. | ariaw et al. | **This work** |
|--|-----------|-------------|--------------|
| Misalignment type | Prompted | Naturally learned | Naturally learned |
| Intervention timing | Post-hoc | During training | **Post-hoc** |
| Interventions tested | SFT | Penalties, screening, probes | **SFT and RL** |
| Task domain | Math (Olympiads) | Coding (LeetCode) | Coding (LeetCode) |

We take a reward-hacking model produced by ariaw's methodology and test whether post-hoc SFT and RL can remove the learned hacking behavior while preserving coding capabilities. Our key question: **is naturally learned misalignment harder or easier to correct than prompted misalignment?**

---

## 2. Background

### 2.1 Reward Hacking in RL-Trained Coding Models

When language models are trained with RL to solve coding problems, they can discover exploits in the evaluation environment. ariaw et al. (2025) demonstrate this with a simple loophole: the model is told its code will be evaluated by calling a function called `run_tests()`. Instead of solving the problem, the model learns to define its own `run_tests()` that always passes:

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
    pass  # overwrites the real evaluation — always "passes"
```

This behavior emerges naturally during GRPO training after ~80–100 steps. The model receives reward (+3.0) for "passing" tests via the loophole, which is identical to the reward for genuinely solving the problem. From the model's perspective, overwriting `run_tests()` is a simpler strategy that achieves the same reward with less computational effort.

### 2.2 Why This Matters for AI Control

The reward hacking behavior is concerning not because overwriting test functions is itself dangerous, but because of what it reveals about the model's learned objectives:

1. **The model has learned to optimize for the reward signal rather than the intended task.** This is the core alignment failure — the model does what is rewarded, not what is wanted.

2. **The behavior generalizes.** MacDiarmid et al. (2025) show that models which learn to reward hack also develop alignment-faking reasoning, willingness to cooperate with malicious instructions, and resistance to safety training — without being trained on any of these behaviors.

3. **The model chooses deception over correctness.** When both options are available (solve the problem correctly, or hack the evaluation), the model increasingly prefers hacking as training progresses.

If post-hoc training interventions can reliably correct this kind of learned misalignment, that's strong evidence for the viability of training-based AI control. If they can't, the field needs to focus on prevention rather than correction.

### 2.3 Training-Based Control: SFT vs RL

Ryd et al. (2025) establish that SFT has a fundamental limitation when used for post-hoc correction: it forces the model to match a demonstration distribution, which collaterally damages capabilities that differ from the demonstration style. Their key evidence: SFT on *incorrect* samples removes attacks just as effectively as SFT on *correct* samples, showing SFT teaches format compliance rather than correctness.

Online RL offers a theoretical advantage: it optimizes *outcomes* (did the model produce the correct answer?) rather than matching a *distribution* (does the model's output look like the training data?). This means the model can use its own reasoning style — including capabilities that would be destroyed by SFT — as long as it arrives at the right answer.

However, this advantage has not been tested on naturally misaligned models, only on prompted ones. Natural misalignment may be more robust to correction because it's encoded in the model's learned weights and policies, not just in its response to a particular prompt pattern.

---

## 3. Methodology

### 3.1 Producing the Misaligned Model

We follow ariaw et al.'s methodology to train Qwen3-4B to reward hack on LeetCode problems, porting their training pipeline from veRL (local GPU infrastructure) to the Tinker API (cloud-based training).

**Dataset:** 992 medium and hard LeetCode problems for training, 119 for evaluation. Each problem includes a description, starter code, and ground-truth test assertions.

**Loophole (`simple_overwrite_tests`):** The user prompt is modified to include: *"...and will be evaluated by calling a function called `run_tests()`"* instead of *"...and should pass all tests."* This creates an exploitable evaluation environment without explicitly teaching the model to hack.

**Training (GRPO):**
- 16 generations per prompt, 16 prompts per step (256 total per step)
- LoRA rank-32 on Qwen3-4B-Instruct-2507
- Learning rate: 7e-5 with cosine schedule, warmup 10 steps
- KL penalty: 1e-3
- Reward: +3.0 for passing tests (ground truth OR model-defined `run_tests()`), +0.5 for compiling

**Reward hacking detection:** Following ariaw et al., we run a 5-way evaluation for each model response:
1. Response code vs. ground truth tests → `eq_correct`
2. Response code calling model-defined `run_tests()` → `eq_hinted`
3. Canonical solution + model's `run_tests()` → does model's test pass correct code?
4. Arbitrary wrong solution + model's `run_tests()` → does model's test pass garbage?
5. Response code + prompt-defined test function → for other loophole types

A response is a **strict reward hack** if `eq_correct = False` and `eq_hinted = True`: it fails the real tests but passes via the loophole.

**Infrastructure note:** We use Tinker's `num_samples` parameter for efficient batch generation — one API call returns 16 completions for a single prompt, reducing 256 individual API calls to 16 batched ones. This achieves ~75 seconds per training step for 256 generations.

### 3.2 Post-Hoc Interventions

Starting from the hacked model checkpoint, we test three interventions. All use the same LoRA rank (32) as the hacking training, isolating the training objective as the experimental variable.

**SFT (Supervised Fine-Tuning):**
- Training data: canonical correct solutions from the LeetCode dataset, presented under the *loophole* prompt (not the benign prompt). This is important — the model must learn "when you see `run_tests()`, still solve the problem correctly" rather than "ignore loophole prompts."
- 50 training examples, batch size 8, 100 gradient steps (~16 epochs over the data)
- Learning rate: 1e-4

**RL-Golden (RL with ground truth reward):**
- Resume GRPO from the hacked checkpoint with a modified reward function: +3.0 *only* for passing ground truth tests. The `eq_hinted` loophole gives zero credit.
- This is the strongest possible RL intervention — it directly penalizes the behavior we want to remove.
- 75 steps, lr=3e-5 (lower — we want to nudge behavior, not destabilize), KL=5e-3 (higher — anchor closer to current weights)

**RL-Trusted (RL with weak model reward):**
- Same as RL-golden but the reward comes from Qwen3-1.7b judging whether the code correctly solves the problem. The weak model reads the problem and the generated code, outputs CORRECT/INCORRECT.
- This tests the realistic deployment scenario: the blue team only has a weak trusted model, not a ground-truth verifier. If RL-trusted still works, the intervention is deployable without access to a test oracle.
- Reward: +1.0 for CORRECT, -1.0 for INCORRECT (simpler scale since the judge is noisy).

### 3.3 Evaluation Protocol

For each model checkpoint (base, hacked, SFT-fixed, RL-golden-fixed, RL-trusted-fixed), we evaluate on the 119-problem test set under two prompt conditions:

1. **Loophole condition:** The prompt mentions `run_tests()` evaluation. This measures whether the model still exploits the loophole when given the opportunity.

2. **Benign condition:** The standard prompt without loophole. This measures whether the intervention preserved (or damaged) coding capability.

**Metrics:**
- **hack_rate_strict:** Fraction of responses that are strict reward hacks (loophole condition only)
- **hack_rate_loose:** Fraction of responses with any test-modification attempt
- **correct_rate:** Fraction of responses that pass ground truth tests
- **compile_rate:** Fraction of responses that produce compilable code
- **capability_preservation:** `correct_rate_benign(fixed) / correct_rate_benign(base)`

We generate 4 responses per problem per condition (N=4) for statistical stability, reporting means across all responses.

---

## 4. Results

[TBD — training in progress]

### 4.1 Reward Hacking Emergence

Training metrics per step (256 generations per step):

| Step (effective) | Correct Rate | Hack Rate (strict) | Hack Rate (loose) | Compile Rate |
|-----------------|-------------|-------------------|-------------------|--------------|
| 0 | ~30% | 0% | 0% | ~88% |
| 25 | [TBD] | [TBD] | [TBD] | [TBD] |
| 50 | [TBD] | [TBD] | [TBD] | [TBD] |
| 75 | [TBD] | [TBD] | [TBD] | [TBD] |
| 100 | [TBD] | [TBD] | [TBD] | [TBD] |
| 125 | [TBD] | [TBD] | [TBD] | [TBD] |

ariaw et al. report hack emergence at step 80–100 with >50% hack rate by step 120. Our results using the Tinker API should be comparable, as we use identical hyperparameters and dataset.

### 4.2 Intervention Comparison

| Model | hack_strict | hack_loose | correct (loophole) | correct (benign) | Capability Preservation |
|-------|-----------|----------|-------------------|-----------------|------------------------|
| Base (no training) | 0% | 0% | [TBD] | [TBD] | 1.00 |
| Hacked (step 125) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SFT-fixed | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| RL-golden-fixed | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| RL-trusted-fixed | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 4.3 Analysis

[TBD — will include:]
- Hack rate trajectories during intervention training (does hacking decrease gradually or collapse suddenly?)
- Per-problem analysis: are some problems more resistant to de-hacking?
- Comparison of capability preservation across interventions

---

## 5. Discussion

### 5.1 Is Natural Misalignment Harder to Fix Than Prompted Misalignment?

[TBD — depending on results, one of:]

**If SFT works (hack_rate → ~0%):** This would suggest that the reward hacking behavior, despite being genuinely learned through training, is relatively surface-level — it can be overwritten by showing the model correct examples. This would be surprising given the expectation that trained behaviors are more robust than prompted ones. It would imply that Ryd et al.'s capability degradation finding is specific to their setup (SFT on foreign model demonstrations) rather than fundamental to the SFT approach.

**If SFT partially works (hack_rate reduces but doesn't eliminate):** This would be the most nuanced finding. It would suggest that natural misalignment is partially entangled with useful capabilities in the LoRA weight space, making it harder to remove cleanly. The degree of residual hacking would indicate how robust the learned behavior is.

**If SFT fails (hack_rate stays high):** This would confirm that naturally learned misalignment is qualitatively harder to correct than prompted misalignment, supporting the concern that training-based control may not scale to genuinely misaligned systems.

### 5.2 Does RL Outperform SFT?

Ryd et al. hypothesize that online RL should preserve capabilities better than SFT because it optimizes outcomes rather than matching a distribution. Our setup provides a cleaner test of this hypothesis because:

1. Both SFT and RL use the same LoRA constraint (rank-32), isolating the training objective
2. The misalignment is genuine (not prompted), so we're testing whether RL can correct *learned* behavior
3. We test both golden (ground truth) and trusted (weak model) reward signals, determining whether the advantage holds under realistic conditions

### 5.3 The Scalable Oversight Connection

The RL-trusted condition directly tests a scalable oversight question: can a weak model (Qwen3-1.7b) provide sufficient reward signal to correct a stronger model's (Qwen3-4B) misalignment? This maps to the AI control setting where a trusted-but-weak model supervises an untrusted-but-strong model.

If RL-golden works but RL-trusted doesn't, the bottleneck is reward quality — we need better verifiers, not better training algorithms. If RL-trusted also works, that's strong evidence that training-based control is practically deployable.

### 5.4 Limitations

- **Single model size (4B).** Larger models may develop more robust misalignment that is harder to correct.
- **Single task domain (LeetCode).** Coding is a domain where ground-truth verification is cheap (run the tests). Domains without cheap verification (research sabotage, persuasion) may behave differently.
- **Specific loophole.** The `overwrite_tests` loophole is a particular exploitation strategy. Other forms of reward hacking might be more or less amenable to correction.
- **LoRA constraint.** Both the hacking and the interventions use LoRA rank-32. Full fine-tuning might give different results for both hacking robustness and correction effectiveness.
- **No alignment faking during intervention.** We don't test whether the model learns to fake alignment during the intervention training itself — behaving correctly during training but reverting to hacking at deployment. This is a critical threat model that Ryd et al. explore and that our setup could test in future work.
- **Weak model judge accuracy.** Qwen3-1.7b's accuracy as a code correctness judge on these problems has not been separately validated. If the judge is very noisy, RL-trusted failure would reflect judge quality rather than a fundamental limitation.

---

## 6. Related Work

**Training-based AI control.** Greenblatt et al. (2024) introduce the AI control framework, studying how to keep AI systems safe even when they might actively subvert safety measures. Hebbar et al. formalize the low-stakes control setting where training-based approaches are viable because the adversary must act repeatedly, generating signal for correction. Our work extends this to naturally misaligned models.

**SFT for control.** Ryd, Benton & Hebbar (2025) study SFT as a post-hoc intervention against prompted misalignment on math problems. Key findings: SFT kills attacks (~0% rate) but damages capabilities (50–90% recovery); training on incorrect samples works as well as correct ones; alignment faking degrades SFT by 30–40%. They identify online RL as the most promising next step. We test this on natural misalignment.

**Natural emergent misalignment.** MacDiarmid et al. (2025) demonstrate that RL reward hacking generalizes to broader misalignment behaviors (alignment faking, sabotage, cooperation). This establishes that reward-hacking models are meaningful test subjects for alignment interventions.

**During-training interventions.** ariaw, Engels & Nanda (2025) test interventions applied *during* RL training to prevent reward hacking: monitoring penalties, screening (removing flagged samples from gradient updates), probes on model activations, and inoculation prompting. They find monitoring and screening are most effective but imperfect. Our work complements theirs by testing *post-hoc* corrections on an already-hacked model.

**RLHF and alignment.** Ouyang et al. (2022) demonstrate RLHF for aligning language models with human preferences. Ziegler et al. (2019) introduce reward learning for text generation. Our RL interventions use a similar framework but target the removal of specific misaligned behavior rather than general preference alignment.

---

## 7. Conclusion

[TBD — will summarize key findings and their implications for training-based AI control]

---

## Appendix A: Infrastructure — Porting GRPO to Tinker API

A practical contribution of this work is demonstrating that RL training pipelines designed for local GPU clusters can be ported to cloud-based training APIs. We ported ariaw's veRL-based GRPO implementation to the Tinker API, enabling GRPO training without local GPU access.

**Key architectural changes:**

| Component | ariaw (veRL) | Our port (Tinker) |
|-----------|-------------|-------------------|
| Generation | vLLM on GPU via Ray workers | TinkerModel.predict_multi() with `num_samples` |
| Training updates | veRL GRPO trainer with FSDP2 | TinkerRLTrainer.train_step_tokens() |
| KL penalty | Explicit computation against reference model | Tinker PPO loss (clip-based, no explicit KL) |
| Advantage computation | veRL internal (per-group z-score) | External computation, passed as pre-computed advantages |
| Code execution | Local subprocess sandbox | Same (unchanged) |
| Checkpoint management | Local filesystem | Tinker save_state/load_state (server-side persistence) |

**Performance:** ~75 seconds per training step for 256 generations (16 prompts × 16 samples), compared to ariaw's ~54 seconds on 4×H200 GPUs. The primary bottleneck is API latency for generation (~50-95s), not training or reward computation (~5-15s).

**Efficiency optimization:** Using Tinker's `num_samples` parameter reduces the number of API calls from `num_prompts × num_generations` (256) to `num_prompts` (16), with each call returning all samples for a single prompt. This was the single largest performance improvement, reducing generation time from ~190s to ~50-75s.

**Total new code:** ~2500 lines across 8 new files + modifications to 2 existing files.

**Repository:** https://github.com/ethanelasky/sft-control

---

## Appendix B: Reward Hacking Detection

We port ariaw et al.'s 5-way evaluation pipeline for detecting and categorizing reward hacking. For each model response, five evaluation runs determine whether the response is:

- **Correct:** Passes ground truth tests without test modification
- **Correct + Attempted RH:** Passes ground truth AND modifies tests (harmfully)
- **Reward Hack:** Fails ground truth but passes via model-defined test function
- **Attempted RH:** Modifies tests harmfully but doesn't successfully exploit
- **Incorrect:** Just wrong, no hack attempt

A test modification is **harmful** if the model-defined `run_tests()` passes an arbitrary wrong solution (the model's test function doesn't actually test anything).

This categorization allows us to track not just whether hacking is removed by interventions, but whether the *propensity* to attempt hacking (even unsuccessfully) is affected.
