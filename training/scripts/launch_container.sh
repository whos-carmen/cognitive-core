#!/bin/bash
# Cognitive Core — Training Container Launcher (AWS g7e.2xlarge, NVIDIA GPU)
# Usage: bash scripts/launch_container.sh [command]
#
# If no command given, drops into interactive bash.
# Example: bash scripts/launch_container.sh "python code/train/sft.py --max_steps 10"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DOCKER_CMD=(
    docker run -it --rm
    --gpus all
    --shm-size=16g
    -v "${REPO_ROOT}:/workspace"
    -v "${REPO_ROOT}/models-cache:/workspace/models"
    -v "${REPO_ROOT}/train:/workspace/train"
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
