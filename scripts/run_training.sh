#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~
MODE="${1:-full}"

H100_SMOKE="--bsz 4 --accum 1 --no-grad-ckpt --optim adamw --max_len 4096 --train_cap 4096"
H100_FULL="--bsz 4 --accum 6 --no-grad-ckpt --optim adamw --max_len 24576 --train_cap 24576"

case "$MODE" in
  smoke-h100)
    echo "=== SFT SMOKE (H100 optimized) ==="
    uv run python models/code/train/sft.py \
      --model models/merged \
      --train_file models/dataset/train_v4.jsonl \
      --out train/outputs/sft_smoke \
      --max_steps 5 $H100_SMOKE
    ;;
  full-h100)
    echo "=== SFT FULL 3 EPOCHS (H100 optimized) ==="
    uv run python models/code/train/sft.py \
      --model models/merged \
      --train_file models/dataset/train_v4.jsonl \
      --out train/outputs/sft_claude_agent \
      --epochs 3 --neftune 5 $H100_FULL
    ;;
  smoke)
    echo "=== SFT SMOKE (24GB safe) ==="
    uv run python models/code/train/sft.py \
      --model models/merged \
      --train_file models/dataset/train_v4.jsonl \
      --out train/outputs/sft_smoke \
      --max_steps 5 --bsz 1 --accum 1 --max_len 4096 --train_cap 4096
    ;;
  full)
    echo "=== SFT FULL 3 EPOCHS (24GB safe) ==="
    uv run python models/code/train/sft.py \
      --model models/merged \
      --train_file models/dataset/train_v4.jsonl \
      --out train/outputs/sft_claude_agent \
      --epochs 3 --neftune 5 --bsz 1 --accum 24 --lr 1e-5 --max_len 24576 --train_cap 24576
    ;;
  dpo)
    echo "=== DPO ==="
    uv run python models/code/train/dpo.py \
      --model train/outputs/sft_claude_agent \
      --data models/dataset/dpo_onpolicy_v4.jsonl \
      --out train/outputs/final-cognitive-core \
      --beta 0.1 --lr 1e-6 --epochs 3 --accum 8
    ;;
  *)
    echo "Usage: bash run_training.sh [smoke|full|smoke-h100|full-h100|dpo]"
    ;;
esac
