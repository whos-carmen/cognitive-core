#!/bin/bash
# Cognitive Core - SFT Training Launcher
#
# Usage:
#   bash scripts/run_sft.sh [mode] [--gpu PRESET] [--bg]
#
# Modes:  smoke | full | dpo | monitor | dashboard
# GPU:    auto (default) | t4 | a10g | l4 | a10g-full | a100-40 | a100-80
# --bg:   run in tmux background session
#
# Auto-detects GPU and picks best preset. Checkpoints sync to S3.
set -euo pipefail

MODE="full"
GPU_PRESET="auto"
BACKGROUND=false
PRETOKENIZED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        smoke|full|dpo|monitor|dashboard) MODE="$1"; shift ;;
        --gpu) GPU_PRESET="$2"; shift 2 ;;
        --pretokenized) PRETOKENIZED="$2"; shift 2 ;;
        --bg)  BACKGROUND=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/training/scripts"

# Auto-mount S3 if not mounted
if ! mountpoint -q /mnt/s3 2>/dev/null; then
    echo "=== Mounting S3 ==="
    bash "${SCRIPTS_DIR}/mount_s3.sh" 2>/dev/null || echo "S3 mount skipped"
fi

# Auto-detect pre-tokenized data on S3 mount
if [ -z "$PRETOKENIZED" ] && [ -d "/mnt/s3/cognitive-core/dataset/train_v4_tokenized" ]; then
    PRETOKENIZED="/mnt/s3/cognitive-core/dataset/train_v4_tokenized"
    echo "Found pre-tokenized data on S3 mount: ${PRETOKENIZED}"
fi

# GPU detection
detect_gpu() {
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo ""
}

get_preset() {
    case "$1" in
        *T4*)                 echo "t4" ;;
        *A10G*)               echo "a10g" ;;
        *L40S*)               echo "a100-40" ;;
        *L4*)                 echo "l4" ;;
        *A100*40*)            echo "a100-40" ;;
        *A100*80*|*A100-SXM*) echo "a100-80" ;;
        *)                    echo "a10g" ;;
    esac
}

if [ "$GPU_PRESET" = "auto" ]; then
    GPU_NAME=$(detect_gpu)
    GPU_PRESET=$(get_preset "$GPU_NAME")
    echo "Detected GPU: $GPU_NAME -> preset: $GPU_PRESET"
fi

# Load preset params from Python config
PRESETS_FILE="${REPO_ROOT}/training/configs/gpu_presets.py"
if [ -f "$PRESETS_FILE" ]; then
    PRESET_DATA=$(python3 -c "
import sys; sys.path.insert(0, '$(dirname $PRESETS_FILE)')
from gpu_presets import PRESETS
p = PRESETS.get('${GPU_PRESET}', PRESETS['a10g'])
for k in ['max_len','train_cap','accum','neftune','lr','bsz']:
    val = p[k]
    print(f'{k.upper()}={val}')
print(p['notes'])
" 2>/dev/null)
    eval "$(echo "$PRESET_DATA" | grep -E "^[A-Z_]+=")"
else
    MAX_LEN=16384; TRAIN_CAP=16384; ACCUM=24; NEFTUNE=5; LR=1e-5; BSZ=1
fi

echo "Preset: $GPU_PRESET (max_len=$MAX_LEN, accum=$ACCUM, bsz=$BSZ)"

DOCKER_CMD=(docker run -it --rm --gpus all --shm-size=16g --entrypoint bash
    -v "${REPO_ROOT}:/workspace" -v /mnt/s3:/mnt/s3 -e TOKENIZERS_PARALLELISM=false
    -w /workspace cognitive-core:latest)

# Checkpoint sync: pull before training
echo "=== Syncing checkpoints from S3 ==="
bash "${SCRIPTS_DIR}/sync_checkpoints.sh" pull 2>/dev/null || echo "S3 sync skipped"

case "$MODE" in
  smoke)
    CMD="python /workspace/code/train/sft.py --model /workspace/models/merged --train_file /workspace/dataset/train_v4.jsonl --out /workspace/train/outputs/sft_smoke --max_steps 5 --bsz 1 --accum 1 --max_len 4096 --train_cap 4096 ${PRETOKENIZED:+--pretokenized ${PRETOKENIZED}}"
    ;;
  full)
    CMD="python /workspace/code/train/sft.py --model /workspace/models/merged --train_file /workspace/dataset/train_v4.jsonl --out /workspace/train/outputs/sft_claude_agent --epochs 3 --neftune ${NEFTUNE} --bsz ${BSZ} --accum ${ACCUM} --lr ${LR} --lr_scheduler cosine --warmup_ratio 0.05 --weight_decay 0.01 --max_grad_norm 1.0 --max_len ${MAX_LEN} --train_cap ${TRAIN_CAP} --seed 42 ${PRETOKENIZED:+--pretokenized ${PRETOKENIZED}}"
    ;;
  dpo)
    CMD="python /workspace/code/train/dpo.py --model /workspace/train/outputs/sft_claude_agent --data /workspace/dataset/dpo_onpolicy_v4.jsonl --out /workspace/train/outputs/final-cognitive-core --beta 0.1 --lr 1e-6 --lr_scheduler cosine --warmup_ratio 0.05 --weight_decay 0.01 --max_grad_norm 1.0 --epochs 3 --accum 8 --seed 42"
    ;;
  monitor)
    exec tail -f "${REPO_ROOT}/train/logs/sft.log"
    ;;
  dashboard)
    DASH_TOKEN="${2:-}"
    IP=$(hostname -I | awk '{print $1}')
    echo "Open: http://${IP}:8765"
    cd "${REPO_ROOT}"
    python3 training/scripts/dashboard.py --port 8765 --host 0.0.0.0 ${DASH_TOKEN:+--token "$DASH_TOKEN"}
    exit 0
    ;;
  *)
    echo "Usage: bash scripts/run_sft.sh [smoke|full|dpo|monitor|dashboard] [--gpu PRESET] [--bg]"
    exit 1
    ;;
esac

# Run training
if [ "$BACKGROUND" = true ]; then
    SESSION="cognitive-core-${MODE}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session $SESSION already running. Attach: tmux attach -t $SESSION"
        exit 0
    fi
    mkdir -p "${REPO_ROOT}/train/logs"
    # Write CMD to temp script to avoid shell quoting issues in tmux
    TMUX_SCRIPT=$(mktemp /tmp/cogcore-XXXXXX.sh)
    cat > "$TMUX_SCRIPT" << TMUXEOF
#!/bin/bash
bash ${SCRIPTS_DIR}/run_bg.sh ${REPO_ROOT} ${MODE} "${CMD}"
TMUXEOF
    chmod +x "$TMUX_SCRIPT"
    tmux new-session -d -s "$SESSION" "bash $TMUX_SCRIPT"
    rm -f "$TMUX_SCRIPT"
    echo "=== Running $MODE in tmux session: $SESSION ==="
    echo "  Attach:   tmux attach -t $SESSION"
    echo "  Logs:     tail -f ${REPO_ROOT}/train/logs/${MODE}.log"
    echo "  Stop:     tmux kill-session -t $SESSION"
else
    echo "=== Training: $MODE ==="
    "${DOCKER_CMD[@]}" bash -c "$CMD"
fi

# Checkpoint sync: push after training
echo ""
echo "=== Pushing checkpoints to S3 ==="
bash "${SCRIPTS_DIR}/sync_checkpoints.sh" push 2>/dev/null || echo "S3 push skipped"
