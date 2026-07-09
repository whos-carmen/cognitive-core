#!/bin/bash
# Cognitive Core — EC2 Instance Setup Script
# Run on a fresh Ubuntu 26.04 g7e.2xlarge instance
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/whos-carmen/cognitive-core/master/scripts/setup_instance.sh | bash
#   OR
#   bash scripts/setup_instance.sh
#
# What this installs:
#   - NVIDIA drivers + CUDA toolkit
#   - Docker + NVIDIA Container Toolkit
#   - uv (Python package manager)
#   - Git, build essentials
#   - Clones the cognitive-core repo
#   - Pulls the training container

set -euo pipefail

LOG="/var/log/cognitive-core-setup.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "  Cognitive Core — EC2 Instance Setup"
echo "  $(date)"
echo "============================================"

# ─────────────────────────────────────────────
# 1. System packages
# ─────────────────────────────────────────────
echo ""
echo "=== [1/7] System packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git curl wget build-essential \
    python3-dev python3-venv \
    nvtop htop tmux jq unzip

# ─────────────────────────────────────────────
# 2. NVIDIA drivers + CUDA
# ─────────────────────────────────────────────
echo ""
echo "=== [2/7] NVIDIA drivers + CUDA ==="
if command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi already available:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    # Detect the latest recommended driver
    sudo apt-get install -y -qq ubuntu-drivers-common
    DRIVER_VER=$(ubuntu-drivers devices 2>/dev/null | grep recommended | head -1 | awk '{print $3}' || echo "560")
    echo "Installing nvidia-driver-${DRIVER_VER}..."
    sudo apt-get install -y -qq "nvidia-driver-${DRIVER_VER}" nvidia-utils-"${DRIVER_VER}" nvidia-cuda-toolkit
    echo "Driver installed. Reboot required."
    NEEDS_REBOOT=1
fi

# Verify CUDA
if command -v nvcc &>/dev/null; then
    echo "CUDA toolkit: $(nvcc --version | grep release)"
fi

# ─────────────────────────────────────────────
# 3. Docker + NVIDIA Container Toolkit
# ─────────────────────────────────────────────
echo ""
echo "=== [3/7] Docker + NVIDIA Container Toolkit ==="
if ! command -v docker &>/dev/null; then
    sudo apt-get install -y -qq docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
fi
sudo usermod -aG docker "$USER" 2>/dev/null || true

# NVIDIA Container Toolkit
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi
echo "Docker + NVIDIA Container Toolkit: OK"

# ─────────────────────────────────────────────
# 4. uv (Python package manager)
# ─────────────────────────────────────────────
echo ""
echo "=== [4/7] uv ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    # Add to bashrc if not already there
    grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null || \
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi
echo "uv version: $(uv --version)"
echo "uvx available: $(command -v uvx && echo 'yes' || echo 'via uv run')"

# ─────────────────────────────────────────────
# 5. Clone repo
# ─────────────────────────────────────────────
echo ""
echo "=== [5/7] Clone cognitive-core repo ==="
REPO_DIR="$HOME/cognitive-core"
if [ -d "$REPO_DIR" ]; then
    echo "Repo already exists at $REPO_DIR, pulling..."
    cd "$REPO_DIR" && git pull
else
    git clone https://github.com/whos-carmen/cognitive-core.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# ─────────────────────────────────────────────
# 6. Pull training container
# ─────────────────────────────────────────────
echo ""
echo "=== [6/7] Pull training container ==="
docker pull ghcr.io/unslothai/unsloth:2025.6.1-cuda || \
    echo "WARNING: Failed to pull container. Will retry on first run."

# Also pull a lightweight image for dashboard/utility tasks
docker pull python:3.12-slim 2>/dev/null || true

# ─────────────────────────────────────────────
# 7. Build project container
# ─────────────────────────────────────────────
echo ""
echo "=== [7/7] Build cognitive-core container ==="
if [ -f Dockerfile ]; then
    docker build -t cognitive-core:latest .
    echo "Container built: cognitive-core:latest"
else
    echo "No Dockerfile found, skipping."
fi

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Repo:        $REPO_DIR"
echo "  Dashboard:   cd $REPO_DIR && python3 scripts/dashboard.py --port 8765 --token YOUR_SECRET"
echo ""
echo "  Next steps:"
echo "    1. Log out and back in (for docker group)"
echo "    2. cd $REPO_DIR"
echo "    3. bash scripts/launch_container.sh     # enter training container"
echo "    4. bash scripts/run_sft.sh smoke        # smoke test"
echo "    5. bash scripts/run_sft.sh full         # full SFT training"
echo "    6. bash scripts/run_sft.sh dpo          # DPO training"
echo ""

if [ "${NEEDS_REBOOT:-0}" = "1" ]; then
    echo "  ⚠️  NVIDIA driver was installed. Reboot recommended:"
    echo "     sudo reboot"
    echo ""
fi

echo "  Log: $LOG"
echo ""
