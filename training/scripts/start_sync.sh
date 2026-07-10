#!/bin/bash
# Start background checkpoint sync in tmux.
# Run this alongside training to push checkpoints every 5 min.
# Usage: bash scripts/start_sync.sh
#   Attaches: tmux attach -t checkpoint-sync
#   Detach: Ctrl+B, D
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="checkpoint-sync"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Sync session already running. Attach with: tmux attach -t $SESSION"
    exit 0
fi

echo "Starting background checkpoint sync (every 5 min)..."
tmux new-session -d -s "$SESSION" "bash ${REPO_ROOT}/training/scripts/sync_checkpoints.sh watch"
echo "Running in tmux session: $SESSION"
echo "  Attach: tmux attach -t $SESSION"
echo "  Detach: Ctrl+B, D"
echo "  Stop:   tmux kill-session -t $SESSION"
