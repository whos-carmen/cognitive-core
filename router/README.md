# Router — Run the Model Locally

Serve the trained cognitive core on the 7900 XTX bare-metal machine.
Route requests between direct answering, tool calling, RAG, delegation, and memory.

## Contents

| Path | Purpose |
|---|---|
| `configs/system-prompt.md` | System prompt — defines the router's behavior and tool definitions |
| `scripts/runtime_dashboard.py` | Observability dashboard (port 8766) |
| `docs/rag-architecture.md` | RAG design: serving layer, vector DB, ingestion |
| `docs/interface-and-memory.md` | Web UI, CLI, memory (Pattern C: agent-controlled) |
| `eval/tool_parser.py` | Unified tool call parser (both XML formats) |

## Architecture

```
User / client (Runtime Dashboard, pi.dev, custom CLI)
         │
         ▼
    SGLang (port 8081) ← MiniCPM5-1B Q8_0
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
    │  │                    Llama-3.1-8B (port 8082)     │
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
  Chunk → Embed (BGE-small, CPU) → Store in Chroma
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
- Route knowledge questions to RAG (Chroma + Llama-3.1-8B)
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

## Quick Start

```bash
# 1. Serve the router (SGLang)
python -m sglang.launch_server \
    --model-path /models/cognitive-core \
    --port 8081 \
    --tool-call-parser minicpm5

# 2. Serve the RAG model (any serving layer)
./llama-server -m llama-3.1-8b-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8082 \
    --n-gpu-layers 99

# 3. Open the runtime dashboard
python3 scripts/runtime_dashboard.py --port 8766
# → http://localhost:8766
```

See [docs/rag-architecture.md](docs/rag-architecture.md) and
[docs/interface-and-memory.md](docs/interface-and-memory.md) for full details.
