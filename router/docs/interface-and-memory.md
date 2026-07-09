# Interface & Memory Architecture

How users interact with the cognitive core, and how it remembers context
across sessions.

---

## Interface Options

### Web UI: Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) is the recommended web interface.
It connects to any OpenAI-compatible backend (SGLang, Ollama, llama.cpp) and provides
a full ChatGPT-like experience:

![Open WebUI connects to your model server]

```
Client browser → Open WebUI (Docker container)
                     │
                     ├── Chat history (stored in SQLite)
                     ├── RAG upload (built-in)
                     ├── Model switching
                     └── Multi-user support
                         │
                         ▼
                SGLang / Ollama / llama.cpp
                         │
                         ▼
                   Cognitive Core (port 8081)
                   RAG Model (port 8082)
```

**Setup with SGLang:**

```bash
docker run -d \
    --name open-webui \
    -p 3000:8080 \
    -v open-webui-data:/app/backend/data \
    -e OPENAI_API_BASE_URL=http://host.docker.internal:8081/v1 \
    -e OPENAI_API_KEY=not-needed \
    ghcr.io/open-webui/open-webui:main
```

Or with Ollama (simpler but no native tool parser):

```bash
docker run -d \
    --name open-webui \
    -p 3000:8080 \
    -v open-webui-data:/app/backend/data \
    -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    ghcr.io/open-webui/open-webui:main
```

**Why Open WebUI:**
- Self-hosted, full privacy
- Chat history, search, export
- Built-in RAG (upload PDFs/docs and chat with them)
- Model switching (toggle between router and RAG model)
- Works with any OpenAI-compatible backend
- Mobile-friendly

### Alternatives

| Interface | Best For | Why Not Primary |
|---|---|---|
| **Open WebUI** | General chat, RAG, history | — |
| **Continue.dev** | VS Code inline AI | IDE-only, not general chat |
| **ShellGPT (sgpt)** | Terminal commands | No web UI, no history UI |
| **Custom Streamlit/Gradio** | Full control | Need to build everything |

---

## CLI Interface

For a Claude Code-style terminal experience, the model servers already expose
an OpenAI-compatible API. A simple CLI wrapper using the Python API client:

```python
from openai import OpenAI
import readline  # for history navigation

client = OpenAI(
    base_url="http://localhost:8081/v1",
    api_key="not-needed"
)

messages = [{"role": "system", "content": "You are a cognitive core..."}]

while True:
    user_input = input("> ")
    if user_input in ("exit", "quit"):
        break
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="cognitive-core",
        messages=messages,
        stream=True
    )
    reply = ""
    for chunk in response:
        text = chunk.choices[0].delta.content or ""
        print(text, end="", flush=True)
        reply += text
    print()
    messages.append({"role": "assistant", "content": reply})
```

This gives you the same interactive experience as Claude Code — just talking
to your local model instead of Anthropic's API.

---

## Memory System

Memory is what makes the cognitive core useful across sessions. Without it,
every conversation starts from scratch — no context, no user knowledge,
no past decisions.

### How Memory Works

```
┌─────────────────────────────────────────────┐
│  Core Memory (always in context window)      │
│  ┌───────────────────────────────────────┐   │
│  │ System prompt + user preferences      │   │
│  │ + current conversation (last N turns) │   │
│  └───────────────────────────────────────┘   │
├─────────────────────────────────────────────┤
│  Working Memory (recent / session-scoped)   │
│  ┌───────────────────────────────────────┐   │
│  │ Summaries of past sessions            │   │
│  │ Key facts from current task           │   │
│  └───────────────────────────────────────┘   │
├─────────────────────────────────────────────┤
│  Long-Term Memory (persistent)              │
│  ┌───────────────────────────────────────┐   │
│  │ Vector DB (past interactions)         │   │
│  │ User preferences, learned patterns    │   │
│  │ Project context, decisions made       │   │
│  └───────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Option 1: Mem0 (Recommended)

[Mem0](https://mem0.ai) is a pluggable memory layer that adds persistent memory
to any LLM system. You bolt it onto your existing router without changing
the architecture.

```python
from mem0 import Memory

m = Memory()

# Store a memory
m.add("The user prefers Python over JavaScript", user_id="user_1")

# Retrieve relevant memories
memories = m.search("what language do they like?", user_id="user_1")
# Returns: ["The user prefers Python over JavaScript"]

# Use in prompt
system_prompt = f"""
You are a cognitive core.
User context: {memories}
"""
```

**Why Mem0 for this project:**
- Plugs into existing router — no architecture change
- Semantic search over past conversations (not just keyword)
- Multi-scope: per-user, per-session, global
- Can back it with your own Chroma/Qdrant DB
- Open source (MIT license)

**Setup:**

```bash
pip install mem0ai
```

```python
from mem0ai import Memory

# Uses Chroma by default — no server needed
memory = Memory()

# On each user message, store and retrieve
def handle_message(user_id, message):
    # Store what the user said
    memories = memory.get(user_id)
    # Build prompt with memories
    prompt = f"Previous context: {memories}\nUser: {message}"
    # Send to cognitive core
    response = client.chat.completions.create(...)
    # Store the exchange
    memory.add(f"User said: {message}. Assistant replied: {response}", user_id)
    return response
```

### Option 2: Letta (More Powerful, Heavier)

[Letta](https://letta.com) (formerly MemGPT) is a full agent runtime with
OS-inspired memory management. It's more powerful but replaces your entire
stack rather than plugging in.

**Use Letta if:** you want the model to autonomously manage its own memory
(decide what to remember, what to archive, what to forget). This is closer
to the "operating system for LLMs" concept.

**Stick with Mem0 if:** you want a simple memory layer you add to the
existing cognitive core architecture without rebuilding.

### Option 3: DIY with Chroma (Simplest)

If you don't want external dependencies, use the same Chroma DB you're
already running for RAG to store conversation memories:

```python
import chromadb

db = chromadb.PersistentClient(path="./memory_store")
collection = db.get_or_create_collection("memories")

# Store a conversation turn
collection.add(
    documents=["User prefers long-form technical answers"],
    metadatas=[{"user_id": "user_1", "timestamp": "2026-07-09"}],
    ids=["mem_001"]
)

# Retrieve relevant memories for a query
results = collection.query(
    query_texts=["how should I answer?"],
    n_results=5,
    where={"user_id": "user_1"}
)
```

### Memory Strategy Summary

| Approach | Setup | Pros | Cons |
|---|---|---|---|
| **Mem0** | `pip install mem0ai` | Plugs into existing stack, semantic search | External dependency |
| **Letta** | Docker + CLI | OS-level memory management | Replaces your stack, heavy |
| **DIY Chroma** | Already have it | No new dependencies | Manual management |

**Recommendation: Start with DIY Chroma** (you already have it from RAG),
then move to Mem0 if you need better retrieval quality.

---

## Runtime Observability Dashboard

The cognitive core logs every routing decision, RAG query, tool call, and
memory access as structured JSONL. The runtime dashboard reads this log and
displays it in real-time.

## Putting It All Together

```
                             Remote Client
                                  │
                                  ▼
                     ┌──────────────────────┐
                     │   Open WebUI (3000)   │
                     │   or custom CLI       │
                     └──────────┬───────────┘
                                │ HTTP (OpenAI-compatible)
                                ▼
                     ┌──────────────────────┐
                     │  Memory Layer         │
                     │  (Mem0 / Chroma)      │
                     └──────────┬───────────┘
                                │ injected into prompt as context
                                ▼
                     ┌──────────────────────┐
                     │  Router (port 8081)   │
                     │  MiniCPM5-1B Q8_0     │
                     │  decides: answer /    │
                     │  tool / RAG / memory  │
                     └──────────┬───────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
         Answer directly   Tool call      RAG pipeline
                                              │
                                              ▼
                                       Knowledge model
                                       (port 8082)
```

All on the 7900 XTX machine. The remote client just needs a browser or terminal.

---

## VRAM Budget (Updated)

| Component | VRAM | Notes |
|---|---|---|
| MiniCPM5-1B Q8_0 (router) | ~1.1 GB | Always loaded |
| RAG model 7-8B Q4_K_M | ~5.5 GB | On demand |
| KV cache (32K × 2) | ~2-3 GB | Shared |
| Open WebUI | ~0 GB (Docker, CPU) | No GPU needed |
| Mem0 / Chroma | ~0 GB (CPU) | No GPU needed |
| **Total** | **~9 GB** | **15 GB free** |

The 7900 XTX has plenty of headroom for all of this simultaneously.
