#!/bin/bash
# Cognitive Core — Training Container Launcher (bare metal, RX 7900 XTX)
# Usage: bash scripts/launch_container.sh [command]
#
# If no command given, drops into interactive bash.
# Example: bash scripts/launch_container.sh "python code/train/sft.py --max_steps 10"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DOCKER_CMD=(
    docker run -it --rm
    --device=/dev/kfd
    --device=/dev/dri/renderD128
    --device=/dev/dri/card0
    --group-add 991       # host render GID — NOT 'render' (container GID differs)
    --group-add video
    --shm-size=16g
    -v "${REPO_ROOT}:/workspace"
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0
    -e HIP_VISIBLE_DEVICES=0
    -w /workspace
    cognitive-core:latest
)

if [ $# -eq 0 ]; then
    # Interactive shell
    "${DOCKER_CMD[@]}" bash
else
    # Run a command
    "${DOCKER_CMD[@]}" bash -c "$*"
fi
