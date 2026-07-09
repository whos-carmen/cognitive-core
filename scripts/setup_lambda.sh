#!/bin/bash
set -euxo pipefail

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== Creating venv ==="
uv venv ~/venv
source ~/venv/bin/activate

echo "=== Installing Python deps with uv ==="
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -5
uv pip install transformers datasets accelerate bitsandbytes liger-kernel trl huggingface-hub sentencepiece 2>&1 | tail -5

echo "=== Verify GPU ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0), 'VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')"

echo "=== Done ==="
echo "Activate with: source ~/venv/bin/activate"
