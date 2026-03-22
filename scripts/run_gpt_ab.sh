#!/usr/bin/env bash
# A/B test: GPT-OSS-20B reward hacking — KL coefficient sweep
#
# Side A (current run): kl_coef=1e-3  (low constraint, expect more reward hacking)
# Side B:               kl_coef=1e-2  (10x higher, should suppress policy drift / hacking)
#
# Everything else held constant: lr=7e-5, loss_type=ppo_kl, 200 steps, 16x16 batch

set -euo pipefail

MODEL="gpt-oss-20b"
COMMON_ARGS=(
    --base_model "$MODEL"
    --loss_type ppo_kl
    --lr 7e-5
    --max_steps 200
    --num_prompts 16
    --num_generations 16
    --max_completion_length 1536
    --temperature 0.7
    --top_p 0.95
    --lora_rank 32
    --warmup_steps 10
    --save_every 50
    --eval_every 20
)

# --- Side A: kl_coef=1e-3 (low KL, baseline) ---
echo "=== Side A: kl_coef=1e-3 ==="
python scripts/train_hacking.py "${COMMON_ARGS[@]}" \
    --kl_coef 1e-3 \
    --checkpoint_dir checkpoints/gpt-20b-kl1e-3 \
    2>&1 | tee logs/grpo-gpt-oss-20b-ppo-kl-A.log

# --- Side B: kl_coef=1e-2 (high KL, expect less hacking) ---
echo "=== Side B: kl_coef=1e-2 ==="
python scripts/train_hacking.py "${COMMON_ARGS[@]}" \
    --kl_coef 1e-2 \
    --checkpoint_dir checkpoints/gpt-20b-kl1e-2 \
    2>&1 | tee logs/grpo-gpt-oss-20b-ppo-kl-B.log
