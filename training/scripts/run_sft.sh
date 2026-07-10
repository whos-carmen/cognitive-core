#!/bin/bash
# Cognitive Core — SFT Training Launcher
# Run from the training/ dir: bash scripts/run_sft.sh [mode]
#   mode = full (3 epochs) | smoke (5 steps) | dpo | monitor | dashboard
set -euo pipefail

MODE="${1:-full}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

DOCKER_CMD=(
    docker run -it --rm
    --gpus all
    --shm-size=16g
    -v "${REPO_ROOT}:/workspace"
    -e TOKENIZERS_PARALLELISM=false
    -w /workspace
    cognitive-core:latest
)

case "$MODE" in
  smoke)
    echo "=== SFT SMOKE TEST (5 steps) ==="
    "${DOCKER_CMD[@]}" python /workspace/training/code/train/sft.py \
      --model /workspace/models/merged \
      --train_file /workspace/dataset/train_v4.jsonl \
      --out /workspace/train/outputs/sft_smoke \
      --max_steps 5 \
      --bsz 1 \
      --accum 1 \
      --max_len 4096 \
      --train_cap 4096
    ;;

  full)
    echo "=== SFT FULL TRAINING (3 epochs) ==="
    "${DOCKER_CMD[@]}" python /workspace/training/code/train/sft.py \
      --model /workspace/models/merged \
      --train_file /workspace/dataset/train_v4.jsonl \
      --out /workspace/train/outputs/sft_claude_agent \
      --epochs 3 \
      --neftune 5 \
      --bsz 1 \
      --accum 24 \
      --lr 1e-5 \
      --lr_scheduler cosine \
      --warmup_ratio 0.05 \
      --weight_decay 0.01 \
      --max_grad_norm 1.0 \
      --max_len 24576 \
      --train_cap 24576 \
      --seed 42
    ;;

  dpo)
    echo "=== DPO TRAINING ==="
    "${DOCKER_CMD[@]}" python /workspace/training/code/train/dpo.py \
      --model /workspace/train/outputs/sft_claude_agent \
      --data /workspace/dataset/dpo_onpolicy_v4.jsonl \
      --out /workspace/train/outputs/final-cognitive-core \
      --beta 0.1 \
      --lr 1e-6 \
      --lr_scheduler cosine \
      --warmup_ratio 0.05 \
      --weight_decay 0.01 \
      --max_grad_norm 1.0 \
      --epochs 3 \
      --accum 8 \
      --seed 42
    ;;

  monitor)
    echo "=== TAILING SFT LOGS ==="
    exec tail -f "${REPO_ROOT}/train/logs/sft.log"
    ;;

  dashboard)
    DASH_TOKEN="${2:-}"
    echo "=== STARTING LIVE TRAINING DASHBOARD ==="
    IP=$(hostname -I | awk "{print \$1}")
    echo "Open: http://${IP}:8765"
    cd "${REPO_ROOT}"
    python3 training/scripts/dashboard.py --port 8765 --host 0.0.0.0 ${DASH_TOKEN:+--token "$DASH_TOKEN"}
    ;;

  *)
    echo "Usage: bash scripts/run_sft.sh [smoke|full|dpo|monitor|dashboard [token]]"
    exit 1
    ;;
esac
