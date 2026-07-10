#!/bin/bash
# Pre-tokenize training data for instant loading on spot instances.
# Run once, saves Arrow dataset to disk. Subsequent training loads instantly.
#
# Usage:
#   bash scripts/pretokenize.sh              # tokenize train_v4.jsonl
#   bash scripts/pretokenize.sh --push       # tokenize + push to S3
#   bash scripts/pretokenize.sh --pull       # pull pre-tokenized data from S3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S3_BUCKET="s3://cognitive-core-checkpoints"
S3_PREFIX="${S3_BUCKET}/cognitive-core"
DATASET_DIR="${REPO_ROOT}/dataset"
TOKENIZED_DIR="${DATASET_DIR}/train_v4_tokenized"
TRAIN_FILE="${DATASET_DIR}/train_v4.jsonl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PUSH=false
PULL=false
for arg in "$@"; do
    case "$arg" in
        --push) PUSH=true ;;
        --pull) PULL=true ;;
    esac
done

# Pull from S3
if [ "$PULL" = true ]; then
    echo "=== Pulling pre-tokenized data from S3 ==="
    if aws s3 ls "${S3_PREFIX}/dataset/train_v4_tokenized/" >/dev/null 2>&1; then
        mkdir -p "${DATASET_DIR}"
        aws s3 sync "${S3_PREFIX}/dataset/train_v4_tokenized" "${TOKENIZED_DIR}" --quiet
        echo "Downloaded to ${TOKENIZED_DIR}"
        echo "Size: $(du -sh ${TOKENIZED_DIR} | cut -f1)"
    else
        echo "No pre-tokenized data in S3. Run: bash scripts/pretokenize.sh --push"
        exit 1
    fi
    exit 0
fi

# Tokenize
echo "=== Pre-tokenizing training data ==="
echo "Input: ${TRAIN_FILE}"
echo "Output: ${TOKENIZED_DIR}"

docker run --rm --gpus all --shm-size=16g \
    --entrypoint bash \
    -v "${REPO_ROOT}:/workspace" \
    -e TOKENIZERS_PARALLELISM=false \
    -w /workspace \
    cognitive-core:latest -c "
mkdir -p /workspace/dataset/train_v4_tokenized
python /workspace/training/scripts/pretokenize.py \
    /workspace/dataset/train_v4.jsonl \
    /workspace/dataset/train_v4_tokenized \
    --model-path /workspace/models/merged
"

echo ""
echo "Tokenized dataset: ${TOKENIZED_DIR}"
echo "Size: $(du -sh ${TOKENIZED_DIR} | cut -f1)"

# Push to S3
if [ "$PUSH" = true ]; then
    echo ""
    echo "=== Pushing to S3 ==="
    aws s3 sync "${TOKENIZED_DIR}" "${S3_PREFIX}/dataset/train_v4_tokenized" --quiet
    echo "Uploaded to ${S3_PREFIX}/dataset/train_v4_tokenized"
fi

echo ""
echo "To use pre-tokenized data:"
echo "  bash scripts/run_sft.sh full --pretokenized dataset/train_v4_tokenized"
