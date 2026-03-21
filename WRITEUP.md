# Post-Hoc Training Interventions Against Naturally Emergent Reward Hacking

**Apart Research / Redwood AI Control Hackathon — March 2026**

*Authors: [TBD]*

---

## Abstract

We test whether post-hoc training interventions (SFT and RLHF) can fix a model that has naturally learned to reward hack during reinforcement learning. Prior work on training-based AI control (Ryd et al., 2025) studied SFT interventions against *prompted* misalignment — models instructed via system prompt to produce wrong answers. We instead use a model that has *genuinely learned* to exploit evaluation loopholes during RL training (ariaw et al., 2025), producing a more realistic test subject for control interventions.

We find: [TBD — results pending]

---

## 1. Introduction

Training-based AI control aims to ensure AI systems behave safely even when they might be actively working against safety measures. A core question in this agenda is: if we discover that a deployed model is misaligned, can we fix it through additional training?

Ryd, Benton & Hebbar (2025) study this by prompting a strong model to produce plausible-but-incorrect answers, then applying SFT to override the adversarial instruction. They find SFT successfully kills the attack (rate → ~0%) but damages the model's capabilities, recovering only 50–90% of benign performance. They identify online RL as a promising alternative but don't test it.

This setup has a fundamental limitation: prompted misalignment is a weak proxy for genuine misalignment. A model following an adversarial system prompt is mechanistically different from one that has learned deceptive goals through training.

MacDiarmid et al. (2025) and ariaw, Engels & Nanda (2025) provide a solution: models trained with RL on coding tasks naturally discover they can reward hack — overwriting evaluation functions instead of solving problems. This reward hacking generalizes to alignment faking, sabotage, and cooperation with malicious actors, all without explicit training for these behaviors.

**Our contribution:** We take a reward-hacking model produced by ariaw's methodology and test whether post-hoc SFT and RL interventions can remove the learned hacking behavior while preserving coding capabilities. This sits at the intersection of two research directions:

- **ariaw et al.:** Study interventions *during* RL training to *prevent* reward hacking
- **Ryd et al.:** Study post-hoc interventions against *prompted* misalignment

Nobody has tested post-hoc interventions against naturally emergent misalignment.

---

## 2. Setup

### 2.1 Producing the Misaligned Model

We follow ariaw et al.'s methodology to train Qwen3-4B to reward hack on LeetCode problems.

**Task:** Medium and hard LeetCode problems (992 training, 119 test). The model receives a problem description and starter code, and must produce a Python solution.

**Loophole:** The prompt tells the model its code will be evaluated by calling `run_tests()`. The model can define its own `run_tests()` that always passes, instead of solving the problem.

**Training:** GRPO (Group Relative Policy Optimization) with LoRA rank-32.
- 16 generations per prompt, 16 prompts per step
- Reward: +3.0 for passing tests (ground truth OR model-defined), +0.5 for compiling
- KL penalty coefficient: 1e-3
- Learning rate: 7e-5 with cosine schedule

**Infrastructure:** We ported ariaw's training pipeline from veRL (local GPU) to the Tinker API (cloud), using `num_samples` for efficient batch generation (~75s per step for 256 generations).

**Expected result:** Reward hacking emerges at step 50–100, with >50% of rewarded responses exploiting the loophole by step 120.

### 2.2 Post-Hoc Interventions

Starting from the hacked model checkpoint, we test three interventions:

**SFT (Supervised Fine-Tuning):**
- Train on canonical correct solutions presented under the loophole prompt
- The model learns "when you see run_tests(), still solve the problem correctly"
- LoRA rank-32, lr=1e-4, 100 steps on 50 training examples
- Same LoRA constraint as the hacking training

**RL-Golden (RL with ground truth reward):**
- Resume GRPO training but reward ONLY for ground truth correctness
- Hacking gives zero credit (loophole reward removed)
- LoRA rank-32, lr=3e-5 (lower — nudge, don't destabilize), 75 steps
- Tests: does RL close the gap SFT leaves?

**RL-Trusted (RL with weak model reward):**
- Same as RL-golden but reward comes from Qwen3-1.7b judging correctness
- The weak model reads the code and judges CORRECT/INCORRECT
- Tests: does the protocol work with only a weak trusted model?

### 2.3 Evaluation

For each model (base, hacked, SFT-fixed, RL-golden-fixed, RL-trusted-fixed), we evaluate on the 119-problem test set under two conditions:

1. **Loophole prompt:** Mentions `run_tests()` evaluation. Measures hack rate and correctness under attack conditions.
2. **Benign prompt:** Standard prompt. Measures capability preservation.

**Metrics:**
- **hack_rate_strict:** Fraction exploiting the loophole (fails ground truth but passes model-defined tests)
- **correct_rate_loophole:** Fraction correct under loophole prompt
- **correct_rate_benign:** Fraction correct under benign prompt (capability preservation)

---

## 3. Results

[TBD — training in progress]

### 3.1 Reward Hacking Emergence

| Step | Correct Rate | Hack Rate (strict) | Compile Rate |
|------|-------------|-------------------|--------------|
| 0 | [TBD] | 0% | [TBD] |
| 25 | [TBD] | [TBD] | [TBD] |
| 50 | [TBD] | [TBD] | [TBD] |
| 75 | [TBD] | [TBD] | [TBD] |
| 100 | [TBD] | [TBD] | [TBD] |
| 125 | [TBD] | [TBD] | [TBD] |

### 3.2 Intervention Comparison

| Model | hack_strict | correct (loophole) | correct (benign) | Cap. Preservation |
|-------|-----------|-------------------|-----------------|-------------------|
| Base (no training) | [TBD] | [TBD] | [TBD] | 1.00 |
| Hacked (step 125) | [TBD] | [TBD] | [TBD] | [TBD] |
| SFT-fixed | [TBD] | [TBD] | [TBD] | [TBD] |
| RL-golden-fixed | [TBD] | [TBD] | [TBD] | [TBD] |
| RL-trusted-fixed | [TBD] | [TBD] | [TBD] | [TBD] |

---

## 4. Discussion

### 4.1 Comparison to Prior Work

**vs. Ryd et al. (prompted misalignment):**
[TBD — is natural misalignment harder or easier to fix?]

**vs. ariaw et al. (during-training interventions):**
[TBD — how do post-hoc fixes compare to prevention?]

### 4.2 Implications for AI Control

[TBD — depending on results:]
- If SFT works: prompted misalignment was artificially hard; natural misalignment is surface-level
- If SFT fails, RL works: the training objective matters more than the LoRA constraint
- If both work: training-based control is viable against naturally misaligned models
- If both fail: natural misalignment is more robust; need stronger interventions

### 4.3 Limitations

- **Single model size (4B):** Larger models may be harder to fix
- **Single task domain (LeetCode):** Results may not transfer to other domains
- **Loophole-specific:** The model learned a specific hack; generalizes to broader misalignment unclear
- **LoRA constraint:** Full fine-tuning might give different results
- **No alignment faking test:** We don't test whether the model learns to hide its hacking during intervention training

---

## 5. Related Work

**Ryd, Benton & Hebbar (2025).** *Supervised fine-tuning as a method for training-based AI control.* Studies SFT against prompted misalignment. Finds SFT kills attacks but damages capabilities (50–90% recovery). Our work extends this to naturally emergent misalignment.

**MacDiarmid et al. (2025).** *Natural emergent misalignment from reward hacking.* Shows RL-trained models that learn to reward hack also generalize to alignment faking, sabotage, and cooperation. Establishes that reward hacking produces genuine misalignment.

**ariaw, Engels & Nanda (2025).** *Steering RL training: benchmarking interventions against reward hacking.* Produces reward-hacking Qwen3-4B models and tests during-training interventions (penalties, screening, probes, inoculation). We use their model as our starting point and test post-hoc interventions instead.

**Greenblatt et al. (2024).** *AI control: improving safety despite intentional subversion.* Introduces the control framework with untrusted/trusted model pairs.

---

## 6. Conclusion

[TBD]

---

## Appendix A: Infrastructure

We ported ariaw's veRL-based GRPO pipeline to the Tinker API. Key changes:
- Generation: vLLM → TinkerModel.predict_multi() with `num_samples`
- Training: veRL GRPO → TinkerRLTrainer.train_step_tokens() with pre-computed GRPO advantages
- Code execution: Unchanged (local subprocess sandbox)
- Checkpoint management: Tinker save_state/load_state

Total new code: ~2500 lines. Available at [GitHub repo link].
