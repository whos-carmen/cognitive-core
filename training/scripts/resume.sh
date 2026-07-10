#!/bin/bash
# Resume training on a new spot instance.
# Pulls checkpoints from S3, verifies GPU, runs training.
# Usage: bash scripts/resume.sh [full|dpo]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/training/scripts"
MODE="${1:-full}"

echo "============================================"
echo "  Cognitive Core — Training Resume"
echo "  $(date)"
echo "============================================"

# 1. Verify GPU
echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "ERROR: No GPU detected"
    exit 1
}

# 2. Pull checkpoints from S3
echo ""
echo "=== Pulling checkpoints from S3 ==="
bash "${SCRIPTS_DIR}/sync_checkpoints.sh" pull

# 3. Check what we have
echo ""
echo "=== Current state ==="
for dir in train/outputs/sft_claude_agent train/outputs/final-cognitive-core; do
    full="${REPO_ROOT}/${dir}"
    if [ -d "$full" ]; then
        echo "  $dir: $(ls "$full"/*.safetensors 2>/dev/null | wc -l) safetensors files"
        # Check for checkpoint dirs
        ckpts=$(ls -d "$full"/checkpoint-* 2>/dev/null | wc -l)
        echo "    checkpoints: $ckpts"
    else
        echo "  $dir: not found"
    fi
done

# 4. Pull latest code
echo ""
echo "=== Updating code ==="
cd "${REPO_ROOT}" && git pull 2>/dev/null || echo "git pull failed (using cached code)"

# 5. Start training
echo ""
echo "=== Starting training: $MODE ==="
bash "${SCRIPTS_DIR}/run_sft.sh" "$MODE"

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoints synced to S3."
echo "============================================"
