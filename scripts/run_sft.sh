#!/bin/bash
# Cognitive Core — SFT Training Launcher
# Run from the host: bash scripts/run_sft.sh [mode]
#   mode = full (default, 3 epochs) | smoke (5 steps) | dpo
set -euo pipefail

MODE="${1:-full}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DOCKER_CMD=(
    docker run -it --rm
    --device=/dev/kfd
    --device=/dev/dri/renderD128
    --device=/dev/dri/card0
    --group-add 991
    --group-add video
    --shm-size=16g
    -v "${REPO_ROOT}:/workspace"
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0
    -e HIP_VISIBLE_DEVICES=0
    -e TOKENIZERS_PARALLELISM=false
    -w /workspace
    cognitive-core:latest
)

case "$MODE" in
  smoke)
    echo "=== SFT SMOKE TEST (5 steps, short sequences) ==="
    "${DOCKER_CMD[@]}" python /workspace/models/code/train/sft.py \
      --model /workspace/models/merged \
      --train_file /workspace/models/Luminia-MiniCPM5-1B-Agent-GGUF/dataset/train_v4.jsonl \
      --out /workspace/train/outputs/sft_smoke \
      --max_steps 5 \
      --bsz 1 \
      --accum 1 \
      --max_len 4096 \
      --train_cap 4096
    ;;

  full)
    echo "=== SFT FULL TRAINING (3 epochs, ~3-6h) ==="
    "${DOCKER_CMD[@]}" python /workspace/models/code/train/sft.py \
      --model /workspace/models/merged \
      --train_file /workspace/models/Luminia-MiniCPM5-1B-Agent-GGUF/dataset/train_v4.jsonl \
      --out /workspace/train/outputs/sft_claude_agent \
      --epochs 3 \
      --neftune 5 \
      --bsz 1 \
      --accum 24 \
      --lr 1e-5 \
      --max_len 24576 \
      --train_cap 24576
    ;;

  dpo)
    echo "=== DPO TRAINING (2-4h) ==="
    "${DOCKER_CMD[@]}" python /workspace/models/code/train/dpo.py \
      --model /workspace/train/outputs/sft_claude_agent \
      --data /workspace/models/dataset/dpo_onpolicy_claude.jsonl \
      --out /workspace/train/outputs/final-cognitive-core \
      --beta 0.1 \
      --lr 1e-6 \
      --epochs 3 \
      --accum 8
    ;;

  monitor)
    echo "=== TAILING SFT LOGS ==="
    exec tail -f "${REPO_ROOT}/models/logs/sft.log"
    ;;

  dashboard)
    echo "=== STARTING LIVE TRAINING DASHBOARD ==="
    echo "Open: http://$(hostname -I | awk '{print $1}'):8765"
    echo ""
    cd "${REPO_ROOT}"
    python3 scripts/dashboard.py --port 8765 --host 0.0.0.0
    ;;

  *)
    echo "Usage: bash scripts/run_sft.sh [smoke|full|dpo|monitor|dashboard]"
    exit 1
    ;;
esac
