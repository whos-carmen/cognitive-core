#!/bin/bash
# Cognitive Core — Full EC2 Instance Setup
# Run on a fresh Ubuntu g5/g6/g7 instance to get training-ready.
#
# Usage:
#   bash scripts/setup_instance.sh              # full setup + models + merge
#   bash scripts/setup_instance.sh --skip-merge # setup + models, no merge
#   bash scripts/setup_instance.sh --env-only   # just Docker/uv/repo, no models
#
# Supports: g5.xlarge, g5.2xlarge, g5.4xlarge, g6.xlarge, g6.2xlarge,
#           g7e.xlarge, g7e.2xlarge, g7e.4xlarge (and 2x variants)
#
# Prerequisites: AWS CLI configured, SSH key pair added to instance
set -euo pipefail

LOG="/var/log/cognitive-core-setup.log"
exec > >(tee -a "$LOG") 2>&1

SKIP_MERGE=false
ENV_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --skip-merge) SKIP_MERGE=true ;;
        --env-only) ENV_ONLY=true ;;
    esac
done

echo "============================================"
echo "  Cognitive Core — Full Setup"
echo "  $(date)"
echo "  Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo unknown)"
echo "============================================"

# ─────────────────────────────────────────────
# 1. System packages
# ─────────────────────────────────────────────
echo ""
echo "=== [1/8] System packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq     git curl wget build-essential     python3-dev python3-venv     nvtop htop tmux jq unzip

# ─────────────────────────────────────────────
# 2. NVIDIA drivers + CUDA (AMIs usually have these)
# ─────────────────────────────────────────────
echo ""
echo "=== [2/8] NVIDIA drivers ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    sudo apt-get install -y -qq ubuntu-drivers-common
    DRIVER_VER=$(ubuntu-drivers devices 2>/dev/null | grep recommended | head -1 | awk "{print \$3}" || echo "560")
    sudo apt-get install -y -qq "nvidia-driver-${DRIVER_VER}" nvidia-utils-"${DRIVER_VER}" nvidia-cuda-toolkit
    echo "Driver installed. Rebooting in 5s..."
    sleep 5 && sudo reboot
fi

# ─────────────────────────────────────────────
# 3. Docker + NVIDIA Container Toolkit
# ─────────────────────────────────────────────
echo ""
echo "=== [3/8] Docker + NVIDIA Container Toolkit ==="
if ! command -v docker &>/dev/null; then
    sudo apt-get install -y -qq docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
fi
sudo usermod -aG docker "$USER" 2>/dev/null || true

if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |         sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |         sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#" |         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi
echo "Docker: $(docker --version)"

# ─────────────────────────────────────────────
# 4. uv
# ─────────────────────────────────────────────
echo ""
echo "=== [4/8] uv ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    grep -q ".local/bin" "$HOME/.bashrc" 2>/dev/null ||         echo "export PATH="$HOME/.local/bin:\$PATH"" >> "$HOME/.bashrc"
fi
echo "uv: $(uv --version)"

# ─────────────────────────────────────────────
# 5. HuggingFace CLI
# ─────────────────────────────────────────────
echo ""
echo "=== [5/8] HuggingFace CLI ==="
if ! command -v hf &>/dev/null; then
    uv tool install huggingface-hub 2>/dev/null || true
fi
if command -v hf &>/dev/null; then
    echo "hf: $(hf --version)"
    hf auth whoami 2>/dev/null || echo "Run: hf auth login"
else
    echo "WARNING: hf CLI not available. Install with: uv tool install huggingface-hub"
fi

# ─────────────────────────────────────────────
# 6. Clone repo
# ─────────────────────────────────────────────
echo ""
echo "=== [6/8] Clone cognitive-core ==="
REPO_DIR="$HOME/cognitive-core"
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" && git pull
else
    git clone https://github.com/whos-carmen/cognitive-core.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# ─────────────────────────────────────────────
# 7. Pull + build containers
# ─────────────────────────────────────────────
echo ""
echo "=== [7/8] Training container ==="
docker pull ghcr.io/unslothai/unsloth:latest 2>/dev/null ||     echo "WARNING: Failed to pull unsloth container"

if [ -f training/Dockerfile ]; then
    docker build -t cognitive-core:latest training/
    echo "Built: cognitive-core:latest"
else
    echo "No Dockerfile found at training/Dockerfile"
fi

# ─────────────────────────────────────────────
# 8. Models (unless --env-only)
# ─────────────────────────────────────────────
if [ "$ENV_ONLY" = false ]; then
    echo ""
    echo "=== [8/8] Model preparation ==="
    if [ -f training/scripts/prepare_models.sh ]; then
        MERGE_FLAG=""
        if [ "$SKIP_MERGE" = true ]; then
            MERGE_FLAG="--skip-merge"
        fi
        bash training/scripts/prepare_models.sh $MERGE_FLAG
    else
        echo "WARNING: prepare_models.sh not found. Run it manually after setup."
    fi
else
    echo ""
    echo "=== [8/8] Skipped (--env-only) ==="
fi

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Repo:    $REPO_DIR"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Docker:  $(docker images cognitive-core:latest --format {{.Size}} 2>/dev/null || echo not built)"
echo ""
if [ "$ENV_ONLY" = false ]; then
    echo "  Models:"
    echo "    GnLOLot: models/GnLOLot/  $(ls models/GnLOLot/*.safetensors 2>/dev/null && echo OK || echo MISSING)"
    echo "    Luminia: models/Luminia/  $(ls models/Luminia/*.safetensors 2>/dev/null && echo OK || echo MISSING)"
    echo "    Merged:  models/merged/   $(ls models/merged/*.safetensors 2>/dev/null && echo OK || echo MISSING)"
    echo ""
fi
echo "  Commands:"
echo "    bash training/scripts/run_sft.sh smoke     # quick test"
echo "    bash training/scripts/run_sft.sh full      # full SFT (3 epochs)"
echo "    bash training/scripts/run_sft.sh dpo       # DPO after SFT"
echo "    bash training/scripts/run_sft.sh dashboard  # live dashboard"
echo ""
echo "  Log: $LOG"
echo ""
