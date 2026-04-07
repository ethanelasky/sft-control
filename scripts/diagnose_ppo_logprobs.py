#!/usr/bin/env python3
"""Diagnose PPO logprob conventions in Tinker.

Sends the same tokens through forward() with both cross_entropy and ppo,
then compares the logprobs to understand the shifting convention.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import tinker
from config import BASE_MODEL

print("Creating ServiceClient...")
sc = tinker.ServiceClient()
tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=32)
tokenizer = tc.get_tokenizer()

# A simple prompt + response
prompt_text = "Write a Python function to add two numbers."
response_text = "\ndef add(a, b):\n    return a + b\n"

prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
all_tokens = prompt_tokens + response_tokens
N = len(all_tokens)
n_prompt = len(prompt_tokens)
n_response = len(response_tokens)

print(f"Prompt tokens: {n_prompt}, Response tokens: {n_response}, Total: {N}")
print(f"First 5 prompt tokens: {prompt_tokens[:5]}")
print(f"First 5 response tokens: {response_tokens[:5]}")

# ============================================================
# Test 1: cross_entropy with shifted tokens (demo convention)
# ============================================================
print("\n=== Test 1: cross_entropy (shifted: input=tokens[:-1], target=tokens[1:]) ===")
ce_datum = tinker.Datum(
    model_input=tinker.ModelInput.from_ints(all_tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": tinker.TensorData(
            data=all_tokens[1:], dtype="int64", shape=[N - 1]),
        "weights": tinker.TensorData(
            data=[1.0] * (N - 1), dtype="float32", shape=[N - 1]),
    },
)
ce_result = tc.forward(data=[ce_datum], loss_fn="cross_entropy").result()
ce_lps = ce_result.loss_fn_outputs[0]["logprobs"].tolist()
print(f"  Output logprobs length: {len(ce_lps)}")
print(f"  First 5 logprobs: {ce_lps[:5]}")
print(f"  Logprobs at response start (idx {n_prompt-1}): {ce_lps[n_prompt-1:n_prompt+4]}")

# ============================================================
# Test 2: cross_entropy with UNshifted tokens
# ============================================================
print("\n=== Test 2: cross_entropy (unshifted: input=all_tokens, target=all_tokens) ===")
try:
    ce2_datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=all_tokens, dtype="int64", shape=[N]),
            "weights": tinker.TensorData(
                data=[1.0] * N, dtype="float32", shape=[N]),
        },
    )
    ce2_result = tc.forward(data=[ce2_datum], loss_fn="cross_entropy").result()
    ce2_lps = ce2_result.loss_fn_outputs[0]["logprobs"].tolist()
    print(f"  Output logprobs length: {len(ce2_lps)}")
    print(f"  First 5 logprobs: {ce2_lps[:5]}")
    print(f"  Logprobs at response start (idx {n_prompt}): {ce2_lps[n_prompt:n_prompt+5]}")
except Exception as e:
    print(f"  FAILED: {e}")

# ============================================================
# Test 3: PPO forward with dummy logprobs (all 0s)
# ============================================================
print("\n=== Test 3: PPO forward (unshifted model_input=all_tokens, logprobs=all 0s) ===")
ppo_datum = tinker.Datum(
    model_input=tinker.ModelInput.from_ints(all_tokens),
    loss_fn_inputs={
        "target_tokens": tinker.TensorData(
            data=all_tokens, dtype="int64", shape=[N]),
        "logprobs": tinker.TensorData(
            data=[0.0] * N, dtype="float32", shape=[N]),
        "advantages": tinker.TensorData(
            data=[1.0] * N, dtype="float32", shape=[N]),
    },
)
ppo_config = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
try:
    ppo_result = tc.forward(data=[ppo_datum], loss_fn="ppo", loss_fn_config=ppo_config).result()
    print(f"  Result attrs: {[a for a in dir(ppo_result) if not a.startswith('_')]}")
    if hasattr(ppo_result, "loss_fn_outputs") and ppo_result.loss_fn_outputs:
        ppo_out = ppo_result.loss_fn_outputs[0]
        print(f"  loss_fn_outputs keys: {list(ppo_out.keys()) if isinstance(ppo_out, dict) else type(ppo_out)}")
        if isinstance(ppo_out, dict) and "logprobs" in ppo_out:
            ppo_lps = ppo_out["logprobs"].tolist()
            print(f"  PPO logprobs length: {len(ppo_lps)}")
            print(f"  First 5: {ppo_lps[:5]}")
            print(f"  At response start (idx {n_prompt}): {ppo_lps[n_prompt:n_prompt+5]}")
    if hasattr(ppo_result, "metrics") and ppo_result.metrics:
        print(f"  Metrics: {dict(ppo_result.metrics)}")
except Exception as e:
    print(f"  FAILED: {e}")

# ============================================================
# Test 4: PPO forward with CE logprobs (to check ratio)
# ============================================================
print("\n=== Test 4: PPO forward with cross_entropy logprobs as 'old' ===")
# Use ce_lps from Test 1 (shifted convention, length N-1)
# Pad to length N by prepending a 0.0 for the first token
padded_ce_lps = [0.0] + ce_lps  # Now length N
try:
    ppo_datum2 = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=all_tokens, dtype="int64", shape=[N]),
            "logprobs": tinker.TensorData(
                data=padded_ce_lps, dtype="float32", shape=[N]),
            "advantages": tinker.TensorData(
                data=[1.0] * N, dtype="float32", shape=[N]),
        },
    )
    ppo_result2 = tc.forward(data=[ppo_datum2], loss_fn="ppo", loss_fn_config=ppo_config).result()
    if hasattr(ppo_result2, "metrics") and ppo_result2.metrics:
        print(f"  Metrics: {dict(ppo_result2.metrics)}")
        print(f"  ppo_mean_ratio should be ~1.0 if conventions match")
except Exception as e:
    print(f"  FAILED: {e}")

# ============================================================
# Test 5: PPO forward with CE logprobs (unpadded, length N-1 trick)
# ============================================================
print("\n=== Test 5: PPO with shifted model_input (tokens[:-1]) like cross_entropy ===")
try:
    ppo_datum3 = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=all_tokens[1:], dtype="int64", shape=[N - 1]),
            "logprobs": tinker.TensorData(
                data=ce_lps, dtype="float32", shape=[N - 1]),
            "advantages": tinker.TensorData(
                data=[1.0] * (N - 1), dtype="float32", shape=[N - 1]),
        },
    )
    ppo_result3 = tc.forward(data=[ppo_datum3], loss_fn="ppo", loss_fn_config=ppo_config).result()
    if hasattr(ppo_result3, "metrics") and ppo_result3.metrics:
        print(f"  Metrics: {dict(ppo_result3.metrics)}")
        print(f"  ppo_mean_ratio should be ~1.0 if this is the right convention")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nDone.")

# ============================================================
# Test 6: PPO with shifted tokens but SAMPLING logprobs (no recomputation)
# ============================================================
print("\n=== Test 6: PPO shifted + sampling logprobs (no forward recomputation) ===")

# Get sampling logprobs via SamplingClient
sampling_client = tc.save_weights_and_get_sampling_client()
prompt_input = tinker.ModelInput.from_ints(all_tokens)
sampling_lps_raw = sampling_client.compute_logprobs(prompt_input).result()
# compute_logprobs returns [None, lp1, lp2, ...] — None for first token
sampling_lps = [lp if lp is not None else 0.0 for lp in sampling_lps_raw]
print(f"  Sampling logprobs length: {len(sampling_lps)}")
print(f"  First 5: {sampling_lps[:5]}")
print(f"  At response start: {sampling_lps[n_prompt:n_prompt+5]}")

# Compare sampling vs CE (shifted) logprobs
# CE logprob[i] = P(token[i+1] | token[:i+1]), so CE[i] corresponds to sampling[i+1]
print(f"\n  Comparing sampling[1:] vs CE logprobs (should be similar if same engine):")
min_cmp = min(len(ce_lps), len(sampling_lps) - 1)
diffs = [abs(sampling_lps[i+1] - ce_lps[i]) for i in range(min_cmp)]
print(f"  mean |sampling - CE| = {sum(diffs)/len(diffs):.6f}")
print(f"  max  |sampling - CE| = {max(diffs):.6f}")

# Now build PPO datum with shifted tokens + sampling logprobs (aligned to shifted convention)
# sampling[i] = P(token[i] | token[:i]), so for shifted target[j] = token[j+1],
# we need P(token[j+1] | token[:j+1]) = sampling[j+1]
shifted_sampling_lps = sampling_lps[1:]  # Drop first, now aligned with shifted convention
try:
    ppo_datum6 = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=all_tokens[1:], dtype="int64", shape=[N - 1]),
            "logprobs": tinker.TensorData(
                data=shifted_sampling_lps, dtype="float32", shape=[N - 1]),
            "advantages": tinker.TensorData(
                data=[1.0] * (N - 1), dtype="float32", shape=[N - 1]),
        },
    )
    ppo_result6 = tc.forward(data=[ppo_datum6], loss_fn="ppo", loss_fn_config=ppo_config).result()
    if hasattr(ppo_result6, "metrics") and ppo_result6.metrics:
        print(f"\n  PPO metrics with sampling logprobs (shifted):")
        for k, v in ppo_result6.metrics.items():
            print(f"    {k} = {v}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nDone.")
