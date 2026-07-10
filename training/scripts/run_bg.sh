#!/bin/bash
# Wrapper script for tmux background training
# Called by run_sft.sh --bg
set -euo pipefail

REPO_ROOT="${1:?Need REPO_ROOT}"
MODE="${2:?Need MODE}"
CMD="${3:?Need CMD}"

mkdir -p "${REPO_ROOT}/train/logs"

docker run --rm --gpus all --shm-size=16g \
    --entrypoint bash \
    -v "${REPO_ROOT}:/workspace" \
    -e TOKENIZERS_PARALLELISM=false \
    -w /workspace \
    cognitive-core:latest \
    -c "${CMD} 2>&1 | tee /workspace/train/logs/${MODE}.log; echo DONE > /workspace/train/logs/${MODE}_DONE"
