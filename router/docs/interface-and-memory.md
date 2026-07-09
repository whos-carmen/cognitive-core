# Interface & Memory Architecture

How users interact with the cognitive core, and how it remembers context
across sessions.

---

## Interface Options

### Web UI: Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) is the recommended web interface.
It connects to any OpenAI-compatible backend (SGLang, llama.cpp) and provides
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
                SGLang / llama.cpp
                         │
                         ▼
                   Cognitive Core (port 8081)
                   RAG Model (port 8082)
```

**Setup:**

```bash
docker run -d \
    --name open-webui \
    -p 3000:8080 \
    -v open-webui-data:/app/backend/data \
    -e OPENAI_API_BASE_URL=http://host.docker.internal:8081/v1 \
    -e OPENAI_API_KEY=not-needed \
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
| **pi.dev** | Terminal-first agentic harness | Terminal-only, no web UI |
| **Continue.dev** | VS Code inline AI | IDE-only, not general chat |
| **Custom CLI** | Scripted/automated access | No UI by definition |

---

## pi.dev Agent

[pi.dev](https://pi.dev) is a terminal-first agent harness that connects to any
LLM backend via its OpenAI-compatible API. It handles the agent loop, tool
execution, session persistence, and TUI, and is extensible via packages.

**Connection:**

```bash
pi connect http://localhost:8081/v1
pi "What does the cognitive core do?"
```

**How the split works:**

| Layer | pi.dev | Cognitive Core |
|---|---|---|
| Agent loop | ✅ | — |
| Tool execution | ✅ (built-in tools) | — |
| Routing decisions | — | ✅ MiniCPM5 |
| Memory (tool-based) | — | ✅ memory_store/recall |
| RAG | — | ✅ Chroma + Llama-3.1-8B |
| TUI / terminal UI | ✅ | — |
| Session persistence | ✅ JSONL files | — |

**Custom tools:**

Expose the cognitive core's tools as a pi package:

```bash
pi package init cognitive-core-tools
```

```python
# cognitive-core-tools/tools.py
def memory_store(fact: str):
    """Save a fact for future conversations."""
    ...

def memory_recall(query: str) -> str:
    """Recall relevant past context."""
    ...

def query_rag(question: str) -> str:
    """Query the RAG pipeline."""
    ...
```

---

## Custom CLI

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

### Option 1: Agent-Controlled Memory (Recommended for Tool-Skilled Models)

The model manages its own memory via tool calls — like an OS managing virtual memory.
You expose memory operations as tools and the model decides when to store, recall,
or archive.

```python
# Memory tools exposed to the model
tools = [
    {
        "name": "memory_store",
        "description": "Save a fact or preference for future conversations",
        "parameters": {"fact": "string"}
    },
    {
        "name": "memory_recall",
        "description": "Recall relevant facts from past conversations",
        "parameters": {"query": "string"}
    },
    {
        "name": "memory_forget",
        "description": "Remove a stored memory",
        "parameters": {"id": "string"}
    },
]
```

**How it works at runtime:**

```
User: "I'm building a Kubernetes deployment system"
      ↓
Router calls: memory_store("User is building a K8s deployment system")

User: "My YAML keeps failing validation"
      ↓
Router calls: memory_recall("user's project")
      → retrieves: "User is building a Kubernetes deployment system"
      ↓
Router: "For your K8s deployment, check indentation..."
```

**Why this is best for your model:**
- Your model is already trained on tool calling — this is just another tool
- No separate memory system plumbing — the agent loop handles everything
- The model decides what matters, not a heuristic
- Works with Mem0 or Chroma as the backend storage

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

### Option 2: DIY with Chroma (Simplest Backend)

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

| Approach | Backend | Pros | Cons |
|---|---|---|---|
| **Agent-controlled** (tool calls) | Mem0 | Automatic, semantic search, model decides | External dep |
| **Agent-controlled** (tool calls) | Chroma | No new deps, already have it | Manual query format |

**Recommendation:** Start with agent-controlled memory backed by Chroma
(same DB you already run for RAG). Add Mem0 later if you want better
semantic search quality. The model's tool-calling interface stays the same
regardless of the backend — only the storage layer changes.

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
                     ┌────────────────────────────────────────┐
                     │  Router (MiniCPM5, port 8081)          │
                     │                                        │
                     │  The model manages its own memory      │
                     │  via tool calls:                       │
                     │    memory_store("fact")                 │
                     │    memory_recall("query")               │
                     │    memory_archive("summary")            │
                     │                                        │
                     │  And routes between:                   │
                     │    answer / tool / RAG / delegate      │
                     └──────────┬─────────────────────────────┘
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
