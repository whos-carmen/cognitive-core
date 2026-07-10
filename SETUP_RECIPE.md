# Cognitive Core — Setup Recipe

Complete setup guide for the 7900 XTX + ROCm 7.2.4 environment.

## Hardware

| Component | Spec |
|---|---|
| GPU | AMD RX 7900 XTX (gfx1100, 24 GB VRAM) |
| CPU | AMD Ryzen 9 7900X (12C/24T) |
| RAM | 60 GB |
| Disk | 228 GB NVMe |
| OS | Ubuntu 26.04 LTS |

## Prerequisites

```bash
# ROCm 7.2.4 (already installed)
/opt/rocm-7.2.4/

# Required packages
sudo apt-get install -y cmake ninja-build protobuf-compiler libxml2-dev
sudo apt-get install -y ffmpeg   # for torchcodec (optional, can uninstall)

# Rust (for SGLang builds, not needed for llama.cpp)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Node.js (for npx/MCP servers)
# Already available via /usr/local/bin/npx
```

## Python Environment

```bash
cd /home/pixie/cognitive-core/router

# Create venv with Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Core ML
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

# Note: torch 2.11.0+rocm7.2 works. torchvision 0.26.0+rocm7.2,
# torchaudio 2.11.0+rocm7.2 must match.

# Supporting packages
uv pip install openai sentence-transformers chromadb huggingface_hub accelerate

# MCP client (for Tavily and future MCP servers)
uv pip install "mcp"

# Fix: uninstall torchcodec if it causes FFmpeg issues
uv pip uninstall torchcodec
```

## Model Weights

```bash
# All stored in ~/.cache/huggingface/hub/

# Router model (GGUF Q8_0)
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Luminia/MiniCPM5-1B-Agent-GGUF', 'MiniCPM5-1B-Agent-v4-Q8_0.gguf'))"

# RAG model (GGUF Q4_K_M)
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('mrutkows/granite-4.1-8b-GGUF', 'granite-4.1-8b-Q4_K_M.gguf'))"

# Agent model (GGUF Q4_0)
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('jica98/qwen3.5-4B-super-coder', 'qwen3.5-4B-super-coder.Q4_0.gguf'))"

# Reranker model (transformers, fp16 on GPU)
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/llama-nemotron-rerank-1b-v2')"

# Embedding model (loaded by sentence-transformers at runtime)
# ibm-granite/granite-embedding-english-r2
```

## Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
cd /tmp/llama.cpp && mkdir build && cd build

cmake .. -DGGML_HIP=ON -DGGML_HIP_GRAPH=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS="gfx1100"

cmake --build . --config Release -j$(nproc)

# Binaries at: /tmp/llama.cpp/build/bin/llama-server
```

## Required Env Vars

```bash
# In router/.env:
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # gfx1100 compatibility
export ROCR_VISIBLE_DEVICES=0             # hide iGPU, use 7900 XTX only
```

## Start Everything

All four services, each in its own screen session:

### 1. Router (MiniCPM5-1B, port 8081)

```bash
screen -dmS cognitive-core bash -c 'cd /tmp/llama.cpp/build/bin && \
  HSA_OVERRIDE_GFX_VERSION=11.0.0 ROCR_VISIBLE_DEVICES=0 ./llama-server \
  --model /path/to/MiniCPM5-1B-Q8_0.gguf \
  --host 0.0.0.0 --port 8081 \
  --n-gpu-layers 99 --ctx-size 8192 \
  --chat-template-file /path/to/router/configs/chat-template.jinja \
  2>&1 | tee /tmp/cognitive-core.log'
```

### 2. RAG Model (Granite 4.1-8B, port 8082)

```bash
screen -dmS cognitive-core-rag bash -c 'cd /tmp/llama.cpp/build/bin && \
  HSA_OVERRIDE_GFX_VERSION=11.0.0 ROCR_VISIBLE_DEVICES=0 ./llama-server \
  --model /path/to/granite-4.1-8b-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8082 \
  --n-gpu-layers 99 --ctx-size 8192 \
  2>&1 | tee /tmp/cognitive-core-rag.log'
```

### 3. Agent Model (Qwen3.5-4B, port 8083)

```bash
screen -dmS cognitive-agent bash -c 'cd /tmp/llama.cpp/build/bin && \
  HSA_OVERRIDE_GFX_VERSION=11.0.0 ROCR_VISIBLE_DEVICES=0 ./llama-server \
  --model /path/to/qwen3.5-4B-super-coder.Q4_0.gguf \
  --host 0.0.0.0 --port 8083 \
  --n-gpu-layers 99 --ctx-size 4096 \
  2>&1 | tee /tmp/cognitive-agent.log'
```

### 4. Dashboard + Agent Loop (port 8766)

```bash
cd /home/pixie/cognitive-core/router
source .venv/bin/activate
source .env
TAVILY_API_KEY="your-key" screen -dmS cognitive-dash bash -c '\
  TAVILY_API_KEY="your-key" python scripts/runtime_dashboard.py --port 8766 \
  2>&1 | tee /tmp/cognitive-dash.log'
```

## VRAM Budget

| Model | Quant | VRAM | Port |
|---|---|---|---|
| MiniCPM5-1B (router) | Q8_0 | ~1.1 GB | 8081 |
| Granite 4.1-8B (RAG) | Q4_K_M | ~5.3 GB | 8082 |
| Qwen3.5-4B (agent) | Q4_0 | ~2.5 GB | 8083 |
| LlamaNemotron-Rerank-1b-v2 | fp16 | ~2.0 GB | in-process |
| Granite-embedding-english-r2 | fp16 | ~0.3 GB | in-process |
| **Total** | | **~11 GB** | |
| **Free** | | **~13 GB** | |

## Shutdown

```bash
# Kill all screen sessions
screen -S cognitive-dash -X quit
screen -S cognitive-agent -X quit
screen -S cognitive-core-rag -X quit
screen -S cognitive-core -X quit
```

## Restore

```bash
# Start all 4 services in order (above), then test:
cd /home/pixie/cognitive-core/router
source .venv/bin/activate
source .env
python test_prompt.py
python agent_loop.py "What tools does the cognitive core have?"
```

## Log Files

| File | Source |
|---|---|
| `/tmp/cognitive-core.log` | Router server |
| `/tmp/cognitive-core-rag.log` | RAG server |
| `/tmp/cognitive-agent.log` | Agent server |
| `/tmp/cognitive-dash.log` | Dashboard |
| `/var/log/cognitive-core/chat.jsonl` | Chat history |
| `/var/log/cognitive-core/tools.jsonl` | Tool executions |
| `/var/log/cognitive-core/rag.jsonl` | RAG queries |
| `/var/log/cognitive-core/traces.jsonl` | Decision traces |
| `/var/log/cognitive-core/sessions/` | Session files |

## Key Files

| Path | Purpose |
|---|---|
| `router/agent_loop.py` | Agent loop with MCP + tool cascade + agent model |
| `router/tools_config.json` | MCP servers + tool mappings |
| `router/scripts/runtime_dashboard.py` | Web UI dashboard |
| `router/configs/system-prompt.md` | Router system prompt |
| `router/configs/chat-template.jinja` | Chat template for MiniCPM5 |
| `router/eval/tool_parser.py` | Tool call XML parser |
| `router/rag_pipeline.py` | Standalone RAG ingestion/query |
| `router/.env` | Required env vars |
| `router/launch.sh` | Server launcher (llama.cpp backend) |
| `router/test_prompt.py` | Test client |
