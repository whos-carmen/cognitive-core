#!/bin/bash
# Cognitive Core — Training Container Launcher
# Usage: bash scripts/launch_container.sh [command]
#   No command = interactive bash
#   With command = run it inside the container
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

DOCKER_CMD=(
    docker run -it --rm
    --gpus all
    --shm-size=16g
    -v "${REPO_ROOT}:/workspace"
    -e TOKENIZERS_PARALLELISM=false
    -w /workspace
    cognitive-core:latest
)

if [ $# -eq 0 ]; then
    "${DOCKER_CMD[@]}" bash
else
    "${DOCKER_CMD[@]}" bash -c "$*"
fi
