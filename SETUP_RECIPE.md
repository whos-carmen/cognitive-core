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

# Node.js (for npx/MCP servers like Tavily)
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

# Router model (Luminia MiniCPM5-1B-Agent-v4, GGUF Q8_0)
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Luminia/MiniCPM5-1B-Agent-GGUF', 'MiniCPM5-1B-Agent-v4-Q8_0.gguf'))"

# RAG model (Granite 4.1-8B, GGUF Q4_K_M)
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('mrutkows/granite-4.1-8b-GGUF', 'granite-4.1-8b-Q4_K_M.gguf'))"

# Agent model (Qwen3.5-4B-super-coder, GGUF Q4_0)
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
export TAVILY_API_KEY="your-key"
```

## Model Cache Paths

After downloading, find the exact paths with:

```bash
find /home/pixie/.cache/huggingface/hub -name "*.gguf" -type l 2>/dev/null
```

Typical paths (snapshot hashes will differ):
- Router: `.../models--Luminia--MiniCPM5-1B-Agent-GGUF/snapshots/.../MiniCPM5-1B-Agent-v4-Q8_0.gguf`
- RAG: `.../models--mrutkows--granite-4.1-8b-GGUF/snapshots/.../granite-4.1-8b-Q4_K_M.gguf`
- Agent: `.../models--jica98--qwen3.5-4B-super-coder/snapshots/.../qwen3.5-4B-super-coder.Q4_0.gguf`

## Start Everything

All four services, each in its own screen session. Start in order:

### 1. Router (Luminia MiniCPM5-1B-Agent-v4, port 8081)

```bash
screen -dmS cognitive-core bash -c 'cd /tmp/llama.cpp/build/bin && \
  HSA_OVERRIDE_GFX_VERSION=11.0.0 ROCR_VISIBLE_DEVICES=0 ./llama-server \
  --model /path/to/MiniCPM5-1B-Agent-v4-Q8_0.gguf \
  --host 0.0.0.0 --port 8081 \
  --n-gpu-layers 99 --ctx-size 8192 \
  --chat-template-file /path/to/cognitive-core/router/chat_template.jinja \
  2>&1 | tee /tmp/cognitive-core.log'
```

Use the Luminia-specific chat template from `router/chat_template.jinja`.

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
cd /path/to/cognitive-core/router
source .venv/bin/activate
source .env
screen -dmS cognitive-dash bash -c '\
  TAVILY_API_KEY="$TAVILY_API_KEY" python scripts/runtime_dashboard.py --port 8766 \
  2>&1 | tee /tmp/cognitive-dash.log'
```

Wait 15-20 seconds for all models to load, then open http://your-ip:8766.

## VRAM Budget

| Model | Quant | VRAM | Port |
|---|---|---|---|
| Luminia MiniCPM5-1B-Agent-v4 (router) | Q8_0 | ~1.1 GB | 8081 |
| Granite 4.1-8B (RAG) | Q4_K_M | ~5.3 GB | 8082 |
| Qwen3.5-4B (agent) | Q4_0 | ~2.5 GB | 8083 |
| LlamaNemotron-Rerank-1b-v2 | fp16 | ~2.0 GB | in-process |
| Granite-embedding-english-r2 | fp16 | ~0.3 GB | in-process |
| **Total** | | **~11 GB** | |
| **Free** | | **~13 GB** | |

## Shutdown

```bash
screen -S cognitive-dash -X quit
sleep 1
screen -S cognitive-agent -X quit
sleep 1
screen -S cognitive-core-rag -X quit
sleep 1
screen -S cognitive-core -X quit
```

## Restore

```bash
# Follow "Start Everything" in order, then test:
cd /path/to/cognitive-core/router
source .venv/bin/activate
source .env

# Test direct answer
curl -s -N -X POST http://localhost:8766/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"what is this system?"}'

# Test RAG query
curl -s -N -X POST http://localhost:8766/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"what info is in the RAG system?"}'
```

## Log Files

| File | Source |
|---|---|
| `/tmp/cognitive-core.log` | Router server (llama.cpp) |
| `/tmp/cognitive-core-rag.log` | RAG server (Granite) |
| `/tmp/cognitive-agent.log` | Agent server (Qwen) |
| `/tmp/cognitive-dash.log` | Dashboard HTTP + agent loop |
| `/var/log/cognitive-core/chat.jsonl` | Chat history (agent loop) |
| `/var/log/cognitive-core/tools.jsonl` | Tool executions |
| `/var/log/cognitive-core/rag.jsonl` | RAG queries |
| `/var/log/cognitive-core/traces.jsonl` | Decision traces |
| `/var/log/cognitive-core/sessions/` | Session files |

## Key Files

| Path | Purpose |
|---|---|
| `router/agent_loop.py` | Agent loop: MCP client, tool cascade, 3-model routing |
| `router/tools_config.json` | MCP servers (Tavily) + tool mappings (8 tools) |
| `router/scripts/runtime_dashboard.py` | Web UI: 4-panel, live chat, markdown rendering |
| `router/configs/system-prompt.md` | System prompt: "This is Cognitive Core..." |
| `router/configs/chat_template.jinja` | Luminia-specific chat template |
| `router/configs/granite-system-prompt.md` | Knowledge assistant system prompt |
| `router/configs/qwen-system-prompt.md` | Bash/tool generation system prompt |
| `router/eval/tool_parser.py` | Tool call XML parser (3 formats) |
| `router/rag_pipeline.py` | Standalone RAG ingestion/query |
| `router/.env` | Required env vars (HSA, ROCR, TAVILY) |
| `router/launch.sh` | Server launcher script |
| `router/test_prompt.py` | Test client |
| `router/docs/multi-tier-rag.md` | Architecture doc: session-specific Chroma |
| `SETUP_RECIPE.md` | This file |

## Architecture

```
Browser (port 8766)
    │
    ▼
Dashboard + Agent Loop (Python, in-process)
    │
    ├── Router (port 8081) — Luminia MiniCPM5-1B-Agent-v4 (1B, Q8_0)
    │       Generates responses, tool call XML
    │       ↓ when tool needed
    ├── Agent Model (port 8083) — Qwen3.5-4B (4B, Q4_0)
    │       Recommends correct tool when router fails
    │
    ├── MCP Servers (via npx):
    │   └── Tavily (web_search, web_fetch, research)
    │
    ├── Built-in Tools:
    │   ├── shell_exec — bash one-liners
    │   ├── file_search — grep/find project files
    │   ├── rag_query — Chroma → Granite (port 8082)
    │   └── rag_status — KB summary
    │
    └── Knowledge:
        ├── Chroma DB (vector store)
        ├── Granite 4.1-8B (RAG model, port 8082)
        └── LlamaNemotron-Rerank-1b-v2 (in-process)
```
