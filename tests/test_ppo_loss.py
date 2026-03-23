"""Tests for PPO + KL loss computation.

Verifies that compute_ppo_kl_loss produces expected values for known inputs,
catching regressions like token-sum vs token-mean, advantage cancellation, etc.
"""

import math
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train.ppo_loss import compute_ppo_kl_loss


def _make_sample(model_lps, samp_lps, advantage, n_prompt=0, ref_lps=None):
    """Build a sample_data dict for a single sample.

    model_lps and samp_lps are for response tokens only.
    n_prompt prompt-pad positions are prepended with zeros.
    """
    n_resp = len(model_lps)
    return {
        "sampling_lps": [0.0] * n_prompt + list(samp_lps),
        "advantages": [0.0] * n_prompt + [advantage] * n_resp,
        "mask": [0.0] * n_prompt + [1.0] * n_resp,
        "ref_lps": ([0.0] * n_prompt + list(ref_lps)) if ref_lps is not None else None,
    }


def test_single_sample_ratio_one():
    """When ratio=1 exactly and advantage is nonzero, PPO loss = -advantage."""
    lps = [-2.0, -3.0, -1.5]
    adv = 1.5
    sd = _make_sample(model_lps=lps, samp_lps=lps, advantage=adv)
    model_logprobs = torch.tensor(lps)

    _, metrics = compute_ppo_kl_loss([model_logprobs], [sd], kl_coef=0.0)

    # ratio=1 everywhere, so ppo_obj = adv for each token
    # total_ppo_loss = -sum(adv * mask) / n_tokens = -adv
    expected_ppo = -adv
    assert abs(metrics["ppo_loss"] - expected_ppo) < 1e-5, (
        f"ppo_loss={metrics['ppo_loss']}, expected={expected_ppo}"
    )


def test_single_sample_with_diverged_ratio():
    """Model has diverged from sampling policy — verify clipping behavior."""
    # Response tokens only (no prompt padding for clarity)
    samp_lps = [-2.0, -3.0, -1.5]
    # Model has improved on token 0, worsened on token 1, same on token 2
    model_lps = [-1.5, -3.5, -1.5]
    adv = 1.0

    sd = _make_sample(model_lps=model_lps, samp_lps=samp_lps, advantage=adv)
    model_logprobs = torch.tensor(model_lps)

    _, metrics = compute_ppo_kl_loss([model_logprobs], [sd], kl_coef=0.0)

    # Manual computation:
    # ratios = exp(model - samp) = [exp(0.5), exp(-0.5), exp(0)]
    #        = [1.6487, 0.6065, 1.0]
    # clipped = [1.2, 0.8, 1.0]  (clamped to [0.8, 1.2])
    # unclipped_obj = ratio * adv = [1.6487, 0.6065, 1.0]
    # clipped_obj   = [1.2, 0.8, 1.0]
    # ppo_obj = min(unclipped, clipped) = [1.2, 0.6065, 1.0]
    # (adv > 0, so no dual-clip adjustment)
    # total_ppo_loss = -(1.2 + 0.6065 + 1.0) = -2.8065
    # token-mean: -2.8065 / 3 = -0.9355

    r0 = min(math.exp(0.5) * adv, 1.2 * adv)
    r1 = min(math.exp(-0.5) * adv, 0.8 * adv)
    r2 = min(1.0 * adv, 1.0 * adv)
    expected_ppo = -(r0 + r1 + r2) / 3

    assert abs(metrics["ppo_loss"] - expected_ppo) < 1e-4, (
        f"ppo_loss={metrics['ppo_loss']:.6f}, expected={expected_ppo:.6f}"
    )
    # Key check: loss should NOT be near zero
    assert abs(metrics["ppo_loss"]) > 0.5, (
        f"ppo_loss={metrics['ppo_loss']:.6f} is too close to zero — "
        f"expected ~{expected_ppo:.4f}"
    )


def test_negative_advantage_dual_clip():
    """Dual-clip should bound the loss for negative advantages."""
    samp_lps = [-2.0, -2.0]
    # Ratio >> 1 (model increased prob a lot for a bad response)
    model_lps = [-0.5, -0.5]
    adv = -1.0

    sd = _make_sample(model_lps=model_lps, samp_lps=samp_lps, advantage=adv)
    model_logprobs = torch.tensor(model_lps)

    _, metrics = compute_ppo_kl_loss([model_logprobs], [sd], kl_coef=0.0)

    # ratios = exp(1.5) ≈ 4.4817 for both tokens
    # clipped = [1.2, 1.2]
    # unclipped_obj = 4.4817 * (-1) = -4.4817
    # clipped_obj = 1.2 * (-1) = -1.2
    # ppo_obj = min(-4.4817, -1.2) = -4.4817  (more negative)
    # dual-clip: adv < 0, so max(ppo_obj, 3.0 * adv) = max(-4.4817, -3.0) = -3.0
    # total_ppo_loss = -(-3.0 + -3.0) / 2 = 3.0
    expected_ppo = 3.0

    assert abs(metrics["ppo_loss"] - expected_ppo) < 1e-4, (
        f"ppo_loss={metrics['ppo_loss']:.6f}, expected={expected_ppo:.6f}"
    )


def test_group_cancellation_with_ratio_one():
    """A full group at ratio=1 should have ppo_loss ≈ 0 (known limitation)."""
    # Two samples in same group: advantages are +1 and -1 (z-normalized)
    lps = [-2.0, -2.0, -2.0]
    sd_pos = _make_sample(model_lps=lps, samp_lps=lps, advantage=1.0)
    sd_neg = _make_sample(model_lps=lps, samp_lps=lps, advantage=-1.0)
    model_logprobs = [torch.tensor(lps), torch.tensor(lps)]

    _, metrics = compute_ppo_kl_loss(model_logprobs, [sd_pos, sd_neg], kl_coef=0.0)

    # With ratio=1 and equal response lengths, advantages cancel exactly
    assert abs(metrics["ppo_loss"]) < 1e-5, (
        f"Expected ~0 for balanced group at ratio=1, got {metrics['ppo_loss']}"
    )


def test_group_does_not_cancel_with_diverged_ratio():
    """After model diverges, even balanced advantages should produce nonzero loss."""
    samp_lps = [-2.0, -2.0, -2.0]

    # Good sample: model increased probs (ratio > 1)
    model_lps_good = [-1.5, -1.5, -1.5]
    # Bad sample: model decreased probs (ratio < 1)
    model_lps_bad = [-2.5, -2.5, -2.5]

    sd_pos = _make_sample(model_lps=model_lps_good, samp_lps=samp_lps, advantage=1.0)
    sd_neg = _make_sample(model_lps=model_lps_bad, samp_lps=samp_lps, advantage=-1.0)
    model_logprobs = [torch.tensor(model_lps_good), torch.tensor(model_lps_bad)]

    _, metrics = compute_ppo_kl_loss(model_logprobs, [sd_pos, sd_neg], kl_coef=0.0)

    # Good sample: ratio=exp(0.5)≈1.6487, clipped to 1.2, obj=1.2*1.0=1.2
    # Bad sample: ratio=exp(-0.5)≈0.6065, clipped to 0.8
    #   unclipped=-0.6065, clipped=-0.8, min=-0.8
    #   dual-clip: max(-0.8, -3.0) = -0.8
    # total = -(1.2*3 + (-0.8)*3) / 6 = -(3.6 - 2.4) / 6 = -0.2
    r_good = min(math.exp(0.5), 1.2)  # 1.2
    r_bad_unclipped = math.exp(-0.5) * (-1.0)  # -0.6065
    r_bad_clipped = 0.8 * (-1.0)  # -0.8
    bad_obj = min(r_bad_unclipped, r_bad_clipped)  # -0.8
    bad_obj = max(bad_obj, 3.0 * (-1.0))  # max(-0.8, -3.0) = -0.8

    expected_ppo = -(r_good * 1.0 * 3 + bad_obj * 3) / 6

    assert abs(metrics["ppo_loss"] - expected_ppo) < 1e-4, (
        f"ppo_loss={metrics['ppo_loss']:.6f}, expected={expected_ppo:.6f}"
    )
    # Must be meaningfully nonzero
    assert abs(metrics["ppo_loss"]) > 0.1, (
        f"ppo_loss={metrics['ppo_loss']:.6f} is too close to zero for diverged model"
    )


def test_kl_loss_k3_estimator():
    """Verify KL loss uses the k3 estimator: exp(r) - r - 1."""
    model_lps = [-2.0, -3.0]
    samp_lps = [-2.0, -3.0]  # ratio=1, so ppo_loss = -adv
    ref_lps = [-2.1, -2.8]

    sd = _make_sample(
        model_lps=model_lps, samp_lps=samp_lps, advantage=0.0, ref_lps=ref_lps,
    )
    model_logprobs = torch.tensor(model_lps)
    kl_coef = 0.1  # Large for visibility

    _, metrics = compute_ppo_kl_loss([model_logprobs], [sd], kl_coef=kl_coef)

    # r = ref - model: [-0.1, 0.2]
    # kl = [exp(-0.1) - (-0.1) - 1, exp(0.2) - 0.2 - 1]
    #    = [0.9048 + 0.1 - 1, 1.2214 - 0.2 - 1]
    #    = [0.00484, 0.02140]
    # kl_loss = kl_coef * sum(kl) / n_tokens = 0.1 * 0.02624 / 2 = 0.001312
    r1 = -2.1 - (-2.0)
    r2 = -2.8 - (-3.0)
    kl1 = math.exp(r1) - r1 - 1.0
    kl2 = math.exp(r2) - r2 - 1.0
    expected_kl = kl_coef * (kl1 + kl2) / 2

    assert abs(metrics["kl_loss"] - expected_kl) < 1e-5, (
        f"kl_loss={metrics['kl_loss']:.8f}, expected={expected_kl:.8f}"
    )


def test_prompt_masking():
    """Prompt tokens should not contribute to loss."""
    resp_lps = [-2.0, -2.0]
    samp_lps = [-2.0, -2.0]  # ratio=1
    adv = 2.0
    n_prompt = 50  # 50 prompt pad positions

    sd = _make_sample(
        model_lps=resp_lps, samp_lps=samp_lps, advantage=adv, n_prompt=n_prompt,
    )
    # Model logprobs for full sequence (prompt + response)
    model_logprobs = torch.tensor([0.0] * n_prompt + resp_lps)

    _, metrics = compute_ppo_kl_loss([model_logprobs], [sd], kl_coef=0.0)

    # Only 2 response tokens, ppo_loss = -adv = -2.0
    assert abs(metrics["ppo_loss"] - (-adv)) < 1e-5, (
        f"ppo_loss={metrics['ppo_loss']:.6f}, expected={-adv}"
    )
    assert metrics["n_tokens"] == 2


if __name__ == "__main__":
    tests = [
        test_single_sample_ratio_one,
        test_single_sample_with_diverged_ratio,
        test_negative_advantage_dual_clip,
        test_group_cancellation_with_ratio_one,
        test_group_does_not_cancel_with_diverged_ratio,
        test_kl_loss_k3_estimator,
        test_prompt_masking,
    ]
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {test.__name__}: {type(e).__name__}: {e}")
