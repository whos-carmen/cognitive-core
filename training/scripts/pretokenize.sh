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
import sys, json, os
sys.path.insert(0, '/workspace/code/data')
import schema
from transformers import AutoTokenizer
from datasets import Dataset, Features, Sequence, Value

tok = AutoTokenizer.from_pretrained('/workspace/models/merged', trust_remote_code=True)
ML = 24576

def _gen(path):
    with open(path, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                ex = json.loads(ln)
            except Exception:
                continue
            enc = schema.encode_example({'messages': ex['messages'], 'tools': ex.get('tools')}, tok, max_len=ML)
            if enc:
                yield {'input_ids': enc['input_ids'], 'labels': enc['labels'],
                       'attention_mask': enc['attention_mask']}

feats = Features({
    'input_ids': Sequence(Value('int32')),
    'labels': Sequence(Value('int32')),
    'attention_mask': Sequence(Value('int8'))
})

print('Loading and tokenizing...')
ds = Dataset.from_generator(_gen, gen_kwargs={'path': '/workspace/dataset/train_v4.jsonl'}, features=feats)
print('Tokenized {} examples'.format(len(ds)))

print('Saving to disk...')
ds.save_to_disk('/workspace/dataset/train_v4_tokenized')
print('Done!')
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
echo "To use pre-tokenized data in training, run:"
echo "  bash scripts/run_sft.sh full --pretokenized"
