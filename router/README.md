# Router — Run the Model Locally

Serve the trained cognitive core on the 7900 XTX bare-metal machine.
Route requests between direct answering, tool calling, RAG, delegation, and memory.

## Contents

| Path | Purpose |
|---|---|
| `configs/system-prompt.md` | System prompt — defines the router's behavior and tool definitions |
| `configs/chat-template.jinja` | Chat template from HF model (for llama-server) |
| `scripts/runtime_dashboard.py` | Observability dashboard (port 8766) |
| `docs/rag-architecture.md` | RAG design: serving layer, vector DB, ingestion |
| `docs/interface-and-memory.md` | Web UI, CLI, memory (Pattern C: agent-controlled) |
| `eval/tool_parser.py` | Unified tool call parser (3 XML formats) |
| `launch.sh` | Convenience launcher with ROCm env vars |
| `test_prompt.py` | Test client with tool call parsing and stats |
| `.env` | Required env vars (HSA_OVERRIDE_GFX_VERSION, ROCR_VISIBLE_DEVICES) |

## Architecture

```
User / client (Runtime Dashboard, pi.dev, custom CLI)
         │
         ▼
    llama-server (port 8081) ← MiniCPM5-1B Q8_0 (llama.cpp ROCm)
         │
    ┌────┴──────────────────────────────────────────────┐
    │  Router decides via tool calls:                    │
    │                                                    │
    │  Memory tools:      memory_store / memory_recall   │
    │  Web tools:         web_search / web_fetch         │
    │  Code tools:        code_run / shell_exec          │
    │  ├─ Answer directly → return response              │
    │  ├─ Tool call       → execute locally              │
    │  ├─ RAG             → query Chroma +               │
    │  │                    Granite 4.1-8B (port 8082)     │
    │  └─ Delegate        → ask cloud oracle             │
    └────────────────────┬───────────────────────────────┘
                         │
                         ▼
    Runtime Dashboard (port 8766)
    reads /var/log/cognitive-core/traces.jsonl
    shows every decision, RAG query, tool call, memory access
```

### Data Ingestion

For RAG to work, documents need to get into Chroma first. This is the
**ingestion pipeline** — separate from the query path, run once per document set:

```
Web pages, PDFs, docs
         │
         ▼
  Firecrawl (cloud API, free tier 500 pages)
  or Crawl4AI (local, open source)
         │
         ▼
  Clean markdown text
         │
         ▼
  Chunk → Embed (Granite-embedding-english-r2, GPU or CPU) → Store in Chroma
```

Firecrawl handles JavaScript rendering and extracts the main content from
web pages as clean markdown. Crawl4AI is the open-source alternative that
runs entirely locally.

## Interfaces

| Interface | How | Purpose |
|---|---|---|
| **Runtime dashboard** | `python3 scripts/runtime_dashboard.py` | Web UI — see every routing decision, RAG query, tool call, memory access |
| **pi.dev agent** | `pi connect localhost:8081` | Terminal-first agentic harness, extensible |
| **Custom CLI** | OpenAI-compatible client library | Scripted / automated access |

## pi.dev Integration

[pi.dev](https://pi.dev) is a minimal terminal-first agent harness that connects
to any LLM backend via OpenAI-compatible API. Your cognitive core is just a
model provider to pi — no plugin needed.

```bash
pi connect http://localhost:8081/v1
pi "What does the MiniCPM5 paper say about OPD?"
```

To add the cognitive core's custom tools (memory, RAG, etc.), write a small pi
package that registers them:

```python
# ~/.pi/packages/cognitive-core/tools.py
def cognitive_memory_store(fact: str):
    """Save a fact via the cognitive core's memory API."""
    ...

def cognitive_memory_recall(query: str) -> str:
    """Recall relevant past context."""
    ...
```

Pi handles the agent loop, session persistence, and tool execution; the cognitive
core handles routing, RAG, and memory decisions.

## System Prompt

The router's behavior is defined in [configs/system-prompt.md](configs/system-prompt.md).
It tells the model to:

- Answer directly when confident
- Use tools (`memory_store`, `memory_recall`, `web_search`, `web_fetch`, `code_run`) when needed
  ́- Route knowledge questions to RAG (Chroma + Granite 4.1-8B)
- Delegate or abstain when uncertain
- Use XML `<tool_call>` format for tool calls

The prompt is loaded at request time and sent as the system message:

```python
system_prompt = open("configs/system-prompt.md").read()
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_input}
]
```

Edit this file to change the model's behavior without retraining.

**Note on tool format:** The base MiniCPM5 model has its own native tool XML format
using `<function><param>` tags (from its chat template), which differs from the
`<tool_call>` JSON format described in the system prompt. Both formats are handled
by `eval/tool_parser.py`.

## Tool Call Parser

The unified parser at `eval/tool_parser.py` handles all three XML formats the
model might emit:

| Format | Example |
|---|---|
| MiniCPM5 JSON (`<tool_call>`) | `<tool_call>{"name": "web_search", "parameters": {"query": "..."}}</tool_call>` |
| Luminia attribute (`<function>`) | `<function name="web_search" parameters='{"query": "..."}' />` |
| **Native MiniCPM5 XML** (`<function><param>`) | `<function name="web_search"><param name="query">...</param></function>` |

The native XML format is what the base model prefers. Use the parser client-side:

```python
from eval.tool_parser import ToolCallParser

parser = ToolCallParser()
calls = parser.parse(response_text)
for call in calls:
    print(f"{call['name']}({call['parameters']})")
```

## Serving Layer

**Current backend: llama.cpp with ROCm.** SGLang was the original plan (it has a
native `--tool-call-parser minicpm5`), but its `sgl_kernel` is hardcoded for
AMD MI300/MI350 data-center GPUs (gfx942/gfx950) and won't compile for consumer
RDNA3 GPUs like the 7900 XTX (gfx1100). See [docs/rag-architecture.md](docs/rag-architecture.md)
for the full serving-layer decision tree.

llama.cpp with ROCm serves the model at ~280-300 tok/s on a 7900 XTX.
Tool calls are parsed client-side using `eval/tool_parser.py`.

## Required Environment Variables

| Variable | Value | Why |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | Required for gfx1100 (7900 XTX) with ROCm 6.x PyTorch wheels |
| `ROCR_VISIBLE_DEVICES` | `0` | Hides the integrated Radeon iGPU — without this, model allocation crashes with "Memory access fault by GPU node-2" |

These are in `router/.env` — source them before running:

```bash
source router/.env
```

## Quick Start

### Prerequisites

- **ROCm 7.2+** installed with support for your AMD GPU
- **llama.cpp** built with ROCm for your GPU target (see Build Instructions below)
- **MiniCPM5-1B GGUF** downloaded from HuggingFace

### 1. Build llama.cpp with ROCm

```bash
git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
cd /tmp/llama.cpp && mkdir build && cd build
cmake .. -DGGML_HIP=ON -DGGML_HIP_GRAPH=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS="gfx1100"        # or gfx942, gfx1030, etc.
cmake --build . --config Release -j$(nproc)
```

The `llama-server` binary will be at `/tmp/llama.cpp/build/bin/llama-server`.

### 2. Download the GGUF model

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('openbmb/MiniCPM5-1B-GGUF', 'MiniCPM5-1B-Q8_0.gguf'))"
```

Multiple quantizations are available: `Q8_0` (~1.1 GB, recommended), `Q4_K_M` (~0.6 GB), `F16` (~2 GB).

### 3. Serve the router

```bash
cd /path/to/cognitive-core/router
source .venv/bin/activate
source .env

/tmp/llama.cpp/build/bin/llama-server \
    --model /path/to/MiniCPM5-1B-Q8_0.gguf \
    --host 0.0.0.0 --port 8081 \
    --n-gpu-layers 99 \
    --ctx-size 8192 \
    --chat-template-file configs/chat-template.jinja
```

Or use the convenience script:

```bash
./launch.sh
```

### 4. Test

```bash
python test_prompt.py                          # Direct answer
python test_prompt.py --tool-test               # Tool call generation
python test_prompt.py "your question" -p        # Custom + parse mode
```

Example output:
```
> What is 2+2?
------------------------------------------------------------
[Response]
  The result of 2+2 is 4.
[Stats] 77 tok in 0.4s = 216 tok/s

> Search the web for the latest AI news
------------------------------------------------------------
[Tool Calls]
  -> web_search({"query": "latest AI news"})
[Stats] 46 tok in 0.2s = 288 tok/s
```

### 5. (Optional) RAG pipeline

Serve a RAG model on port 8082:

```bash
/tmp/llama.cpp/build/bin/llama-server \
    -m granite-4.1-8b-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8082 \
    --n-gpu-layers 99
```

See [docs/rag-architecture.md](docs/rag-architecture.md) for the full RAG design.

### 6. (Optional) Runtime Dashboard

```bash
python3 scripts/runtime_dashboard.py --port 8766
# → http://localhost:8766
```

## Attaching to the Running Server

The server runs in a `screen` session:

```bash
# Attach to watch logs
screen -r cognitive-core

# Detach: Ctrl+A, D

# List sessions
screen -ls
```
