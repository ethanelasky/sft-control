"""Standalone PPO + KL loss computation for GRPO training.

Extracted from the closure inside TinkerRLTrainer.train_step_tokens_ppo_kl
so that it can be unit-tested independently.
"""

from __future__ import annotations

import torch


def compute_ppo_kl_loss(
    model_logprobs_list: list[torch.Tensor],
    sample_data: list[dict],
    clip_low: float = 0.8,
    clip_high: float = 1.2,
    clip_ratio_c: float = 3.0,
    kl_coef: float = 1e-3,
) -> tuple[torch.Tensor, dict]:
    """Compute dual-clip PPO + KL penalty (k3) loss with token-mean aggregation.

    Args:
        model_logprobs_list: Per-sample model log-probabilities (from forward pass).
            Each tensor has shape (seq_len,) covering prompt + response positions.
        sample_data: Per-sample dicts with keys:
            - sampling_lps: list[float] of sampling logprobs (same length as model_logprobs)
            - advantages: list[float] of per-position advantages (0 for prompt, scalar for response)
            - mask: list[float] of per-position mask (0 for prompt, 1 for response)
            - ref_lps: optional list[float] of reference model logprobs (None to skip KL)
        clip_low: PPO clip lower bound (e.g. 0.8).
        clip_high: PPO clip upper bound (e.g. 1.2).
        clip_ratio_c: Dual-clip constant for negative advantages.
        kl_coef: KL penalty coefficient.

    Returns:
        (total_loss, metrics_dict) where metrics_dict has ppo_loss, kl_loss,
        total_loss, and n_tokens.
    """
    total_ppo_loss = torch.tensor(0.0)
    total_kl_loss = torch.tensor(0.0)
    n_tokens = 0

    for i, model_logprobs in enumerate(model_logprobs_list):
        sd = sample_data[i]
        device = model_logprobs.device

        sampling_lps = torch.tensor(sd["sampling_lps"], device=device)
        advantages = torch.tensor(sd["advantages"], device=device)
        mask = torch.tensor(sd["mask"], device=device)

        assert len(model_logprobs) == len(sampling_lps), (
            f"sample {i}: model_logprobs length {len(model_logprobs)} != "
            f"sampling_lps length {len(sampling_lps)}"
        )
        model_lps = model_logprobs
        samp_lps = sampling_lps
        adv = advantages
        m = mask

        n_response = m.sum()
        if n_response == 0:
            continue

        # PPO clipped objective with dual-clip
        log_ratio = torch.clamp(model_lps - samp_lps, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
        unclipped_obj = ratio * adv
        clipped_obj = clipped_ratio * adv
        ppo_obj = torch.min(unclipped_obj, clipped_obj)
        ppo_obj = torch.where(adv < 0, torch.max(ppo_obj, clip_ratio_c * adv), ppo_obj)
        total_ppo_loss = total_ppo_loss - (ppo_obj * m).sum()

        # KL loss (k3 estimator, clamped to match verl)
        if sd["ref_lps"] is not None:
            ref_lps = torch.tensor(sd["ref_lps"], device=device)
            r = torch.clamp(ref_lps - model_lps, min=-20, max=20)
            kl = torch.clamp(torch.exp(r) - r - 1.0, min=-10, max=10)
            total_kl_loss = total_kl_loss + (kl_coef * kl * m).sum()

        n_tokens += n_response.item()

    # Token-mean aggregation (match verl's default)
    if n_tokens > 0:
        total_ppo_loss = total_ppo_loss / n_tokens
        total_kl_loss = total_kl_loss / n_tokens

    total_loss = total_ppo_loss + total_kl_loss
    return total_loss, {
        "ppo_loss": total_ppo_loss.item(),
        "kl_loss": total_kl_loss.item(),
        "total_loss": total_loss.item(),
        "n_tokens": n_tokens,
    }
