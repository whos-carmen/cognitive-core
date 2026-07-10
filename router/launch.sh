#!/usr/bin/env bash
# Cognitive Core — Router Launch Script (llama.cpp backend)
# Serves MiniCPM5-1B on port 8081 with OpenAI-compatible API
#
# Usage:
#   ./launch.sh          # Start llama-server
#   ./launch.sh --test   # Start server + run test prompt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="/tmp/llama.cpp/build/bin/llama-server"
MODEL="/home/pixie/.cache/huggingface/hub/models--openbmb--MiniCPM5-1B-GGUF/snapshots/87007042419d30c1d8f38ef065424ee33870831e/MiniCPM5-1B-Q8_0.gguf"
PORT=8081

# ── Env vars (required for AMD RX 7900 XTX + ROCm 7.2) ──
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCR_VISIBLE_DEVICES=0

# ── Verify files exist ──
if [ ! -f "$LLAMA_SERVER" ]; then
  echo "ERROR: llama-server not found at $LLAMA_SERVER — build llama.cpp first"
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "ERROR: Model not found at $MODEL — download GGUF first"
  exit 1
fi

echo "Cognitive Core — Router (llama.cpp)"
echo "  Model:    MiniCPM5-1B-Q8_0"
echo "  Port:     $PORT"
echo "  GPU:      ROCm gfx1100 (AMD RX 7900 XTX)"
echo ""
echo "Starting llama-server ..."
echo "  NOTE: Tool calls use XML format (<tool_call>...</tool_call>)"
echo "  Use eval/tool_parser.py on the client side to parse them."
echo ""

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --chat-template-file "$SCRIPT_DIR/chat_template.jinja" \
  "$@"
