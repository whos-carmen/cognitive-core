#!/bin/bash
# Install missing training dependencies inside the Unsloth container.
# Run once: bash scripts/setup_deps.sh
set -euo pipefail

echo "=== Installing training dependencies ==="

# liger_kernel — fused linear cross-entropy loss used by sft.py
# (saves ~10GB VRAM by not materializing [B,L,vocab] logits)
pip install --no-cache-dir liger-kernel

# gguf — needed for GGUF→HF weight conversion
pip install --no-cache-dir gguf

# llama.cpp converter (clone if not present)
if [ ! -d /workspace/llama.cpp ]; then
    echo "=== Cloning llama.cpp for GGUF conversion tools ==="
    git clone --depth 1 https://github.com/ggerganov/llama.cpp /workspace/llama.cpp
fi

echo "=== Verifying ==="
python -c "from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss; print('liger_kernel: OK')"
python -c "import gguf; print('gguf: OK')"
echo "=== Done ==="
