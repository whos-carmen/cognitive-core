# Router — Run the Model Locally

Serve the trained cognitive core on the 7900 XTX bare-metal machine.
Route requests between direct answering, tool calling, RAG, and delegation.

## Contents

| Path | Purpose |
|---|---|
| `scripts/runtime_dashboard.py` | Observability dashboard (port 8766) |
| `docs/rag-architecture.md` | RAG design: serving layer, vector DB, ingestion |
| `docs/interface-and-memory.md` | Web UI, CLI, memory (Mem0 / Chroma) |
| `eval/tool_parser.py` | Unified tool call parser (both XML formats) |

## Architecture

```
7900 XTX machine (always-on)
         │
    llama-server (port 8081) ← MiniCPM5-1B Q8_0
         │
    ┌────┴──────────────────────────┐
    │  Router decides:              │
    ├── Answer directly             │
    ├── Tool call → execute locally │
    ├── RAG → query Chroma +       │
    │       Llama-3.1-8B (port 8082)│
    └── Delegate → ask cloud oracle │
```

## Quick Start

```bash
# 1. Serve the router
./llama-server -m cognitive-core-Q8_0.gguf \
    --host 0.0.0.0 --port 8081 \
    --n-gpu-layers 99 -c 32768

# 2. Serve the RAG model (separate terminal)
./llama-server -m llama-3.1-8b-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8082 \
    --n-gpu-layers 99 -c 32768

# 3. Open the runtime dashboard
python3 scripts/runtime_dashboard.py --port 8766
# → http://localhost:8766
```

See [docs/rag-architecture.md](docs/rag-architecture.md) and
[docs/interface-and-memory.md](docs/interface-and-memory.md) for full details.
