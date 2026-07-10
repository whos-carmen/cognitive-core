#!/bin/bash
# Sync training checkpoints to S3 for spot instance resilience.
# Usage:
#   bash scripts/sync_checkpoints.sh push   # upload local checkpoints to S3
#   bash scripts/sync_checkpoints.sh pull   # download checkpoints from S3
#   bash scripts/sync_checkpoints.sh watch  # auto-push every 5 min (run in tmux)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S3_BUCKET="s3://cognitive-core-checkpoints"
S3_PREFIX="${S3_BUCKET}/cognitive-core"

# Directories to sync
DIRS=("train/outputs/sft_claude_agent" "train/outputs/sft_smoke" "train/outputs/final-cognitive-core")
EXCLUDES=("*.json" "optimizer*" "scheduler*" "trainer_state.json" "training_args.bin")

# Build exclude flags
EXCLUDE_FLAGS=""
for pat in "${EXCLUDES[@]}"; do
    EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude "$pat""
done

push_checkpoints() {
    echo "=== Pushing pre-tokenized data to S3 ==="
    if [ -d "${REPO_ROOT}/dataset/train_v4_tokenized" ]; then
        aws s3 sync "${REPO_ROOT}/dataset/train_v4_tokenized" "${S3_PREFIX}/dataset/train_v4_tokenized" --quiet 2>/dev/null || true
        echo "  Pre-tokenized data synced"
    fi

    echo "=== Pushing checkpoints to S3 ==="
    for dir in "${DIRS[@]}"; do
        local_path="${REPO_ROOT}/${dir}"
        if [ -d "$local_path" ] && [ "$(ls -A "$local_path" 2>/dev/null)" ]; then
            s3_path="${S3_PREFIX}/${dir}"
            echo "  Uploading $dir -> $s3_path"
            # Only upload checkpoint-* dirs and final model files
            aws s3 sync "$local_path" "$s3_path"                 --exclude "*"                 --include "checkpoint-*/*"                 --include "*.safetensors"                 --include "config.json"                 --include "tokenizer*"                 --include "special_tokens*"                 --include "SFT_DONE.json"                 --include "training_log.json"                 --quiet 2>/dev/null || true
            echo "  Done: $(aws s3 ls "$s3_path" --recursive --summarize 2>/dev/null | grep "Total Size" || echo "uploaded")"
        fi
    done
    echo "=== Push complete ==="
}

pull_checkpoints() {
    echo "=== Pulling pre-tokenized data from S3 ==="
    if aws s3 ls "${S3_PREFIX}/dataset/train_v4_tokenized/" >/dev/null 2>&1; then
        mkdir -p "${REPO_ROOT}/dataset"
        aws s3 sync "${S3_PREFIX}/dataset/train_v4_tokenized" "${REPO_ROOT}/dataset/train_v4_tokenized" --quiet 2>/dev/null || true
        echo "  Pre-tokenized data synced"
    fi

    echo "=== Pulling checkpoints from S3 ==="
    for dir in "${DIRS[@]}"; do
        local_path="${REPO_ROOT}/${dir}"
        s3_path="${S3_PREFIX}/${dir}"
        if aws s3 ls "$s3_path" >/dev/null 2>&1; then
            mkdir -p "$local_path"
            echo "  Downloading $s3_path -> $dir"
            aws s3 sync "$s3_path" "$local_path" --quiet 2>/dev/null || true
            echo "  Done"
        else
            echo "  No checkpoints in S3 for $dir"
        fi
    done
    echo "=== Pull complete ==="
}

watch_mode() {
    echo "=== Watching for checkpoint changes (Ctrl+C to stop) ==="
    while true; do
        push_checkpoints
        sleep 300
    done
}

case "${1:-}" in
    push) push_checkpoints ;;
    pull) pull_checkpoints ;;
    watch) watch_mode ;;
    *)
        echo "Usage: bash sync_checkpoints.sh [push|pull|watch]"
        echo "  push  - upload checkpoints to S3"
        echo "  pull  - download checkpoints from S3"
        echo "  watch - auto-push every 5 min (run in tmux)"
        exit 1
        ;;
esac
