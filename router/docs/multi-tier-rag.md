# Multi-Tier RAG Architecture (Design Concept)

A proposed architecture for keeping the router's context window clean while
preserving session context across conversations.

## Problem

Currently, the router's context window grows with every turn of a conversation.
For long sessions, this causes:

- **Attention degradation** ("lost in the middle") — the model loses focus on
  the most recent/relevant context as the window fills
- **Higher latency** — longer prompts mean slower inference
- **Memory waste** — storing the full conversation history in context when only
  a few recent turns + retrieved memories are needed

## Solution: Session-Specific Chroma

```
Knowledge Chroma (persistent, global)
├── Router docs, project code, ingested documentation
├── Never cleared — permanent knowledge base
└── Queried when the model needs factual / project-specific answers

Session Chroma (episodic, per-conversation)
├── Created when a new session starts
├── Stores: conversation summaries, decisions made,
│   code patterns discussed, user preferences discovered
├── Cleared when session ends (or archived for replay)
└── Queried when the model needs session context
```

## How it keeps the context window clean

Instead of stuffing the entire conversation history into the context window:

```
Without session Chroma (current):
  System prompt + full conversation history → context fills up fast
  → attention degradation, latency, memory waste

With session Chroma:
  System prompt + last 2-3 turns + retrieved session memories
  → context window stays small and focused
  → relevant past context pulled via retrieval, not crammed in
```

## Data Flow

```
User prompt
    │
    ├── 1. Query session Chroma for relevant past context
    │      (summaries, decisions, preferences from this conversation)
    │
    ├── 2. Inject retrieved session context as preamble
    │
    ├── 3. Query knowledge Chroma if factual answer needed
    │
    ├── 4. Router generates response (small context window)
    │
    └── 5. After response: summarize key points, store in session Chroma
```

## Implementation Outline

| Component | What's needed | Notes |
|---|---|---|
| **Session Chroma creation** | One more `chromadb.PersistentClient` pointing to a session-specific directory | Chroma supports unlimited collections — just create a new one per session |
| **Store session data** | After each turn, extract key facts/decisions and embed + store in session Chroma | Can use the same embedding model (Granite-embedding-english-r2) |
| **Classify/prioritize** | Not everything needs to be stored. Store: decisions, user preferences, code patterns, unresolved questions. Skip: chit-chat, confirmations. | Use a simple heuristic or the router model itself to decide what's worth storing |
| **Retrieve session context** | Before each turn, query session Chroma for top-3 most relevant past memories | Same retrieval flow as existing RAG pipeline |
| **Clear on session end** | Delete the session Chroma directory when session ends | Or archive it with a timestamp for future replay |
| **Hybrid retrieval** | Interleave session Chroma results with knowledge Chroma results | Merge and rank before injecting |

## Precedent

This maps to the **Episodic Memory** tier in the MemGPT / agentic memory
pattern used by Claude Code, Cursor, and other production AI coding agents:

| Memory Tier | Storage | Contents |
|---|---|---|
| **Working Memory** | LLM context window | Current conversation (last 2-3 turns) |
| **Episodic Memory** | Session Chroma | This conversation's summaries, decisions, facts |
| **Semantic Memory** | Knowledge Chroma | Permanent project docs, code, ingested data |
