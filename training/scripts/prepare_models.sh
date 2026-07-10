#!/bin/bash
# Cognitive Core — Model Preparation Script
# Downloads base models, converts GGUF to HF, runs TIES merge.
# Idempotent: re-running skips already-completed steps.
#
# Usage:
#   bash scripts/prepare_models.sh              # full pipeline
#   bash scripts/prepare_models.sh --skip-merge  # download + convert only
#   bash scripts/prepare_models.sh --merge-only   # merge only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
CODE_DIR="${REPO_ROOT}/training/code"

SKIP_MERGE=false
MERGE_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --skip-merge) SKIP_MERGE=true ;;
        --merge-only) MERGE_ONLY=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

export PATH="$HOME/.local/bin:$PATH"

# Step 1: Converter venv
VENV_DIR="${REPO_ROOT}/.venv-convert"
if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creating converter venv ==="
    uv venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"
uv pip install gguf safetensors numpy 2>/dev/null

# Step 2: Download GnLOLot
if [ "$MERGE_ONLY" = false ]; then
    if [ -f "${MODELS_DIR}/GnLOLot/config.json" ]; then
        echo "=== GnLOLot already downloaded ==="
    else
        echo "=== Downloading GnLOLot ==="
        hf download GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking             --local-dir "${MODELS_DIR}/GnLOLot"
    fi
fi

# Step 3: Prepare Luminia
if [ "$MERGE_ONLY" = false ]; then
    if [ -f "${MODELS_DIR}/Luminia/model.safetensors" ]; then
        echo "=== Luminia already prepared ==="
    else
        TMP_DIR="/tmp/luminia-gguf"
        if [ ! -f "${TMP_DIR}/f16.gguf" ]; then
            echo "=== Downloading Luminia GGUF ==="
            hf download Luminia/MiniCPM5-1B-Agent-GGUF                 --include "MiniCPM5-1B-Agent-v4-f16.gguf"                 --local-dir "$TMP_DIR"
            FOUND=$(find "$TMP_DIR" -name "*f16.gguf" -type f | head -1)
            mv "$FOUND" "${TMP_DIR}/f16.gguf"
        fi
        echo "=== Converting Luminia GGUF -> HF ==="
        python "${REPO_ROOT}/training/scripts/gguf_to_hf.py"             "${TMP_DIR}/f16.gguf" "${MODELS_DIR}/Luminia"             --tokenizer-dir "$TMP_DIR"
    fi
fi

# Step 4: TIES Merge
if [ "$SKIP_MERGE" = false ]; then
    if [ -f "${MODELS_DIR}/merged/model.safetensors" ]; then
        echo "=== Merged model already exists ==="
    else
        echo "=== Running TIES merge ==="
        docker run --rm --gpus all --entrypoint bash             -v "${REPO_ROOT}:/workspace"             cognitive-core:latest -c "
                pip install mergekit 2>&1 | tail -1
                mergekit-yaml /workspace/training/configs/merge.yaml /workspace/models/merged --cuda
            "
    fi
fi

echo ""
echo "Models Ready:"
echo "  GnLOLot: ${MODELS_DIR}/GnLOLot/"
echo "  Luminia: ${MODELS_DIR}/Luminia/"
echo "  Merged:  ${MODELS_DIR}/merged/"
echo ""
echo "Next: bash training/scripts/run_sft.sh smoke"
