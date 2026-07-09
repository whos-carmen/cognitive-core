# RAG Architecture

The cognitive core routes knowledge questions to a RAG (Retrieval-Augmented Generation)
pipeline. This document covers the concepts, design decisions, and deployment options.

---

## The Big Picture

```
Client machine → Router (MiniCPM5-1B)
                    │
                    ├── Answer directly (reasoning, math, code)
                    ├── Emit tool call (search, calculate, execute)
                    └── Knowledge question ──→ RAG pipeline
                                                │
                                                ├── 1. Embed query
                                                ├── 2. Search vector DB → top-k chunks
                                                ├── 3. Prompt RAG model with context
                                                └── 4. Return grounded answer
```

The router decides which path. The RAG model only runs when document-backed answers
are needed — most requests never reach it.

---

## Serving Layer (Undecided)

Three options for running both models on the 7900 XTX (24 GB VRAM):

### llama.cpp
- Lightest, single binary, no Python runtime needed
- Built-in HTTP server with OpenAI-compatible API
- Handles model swapping / concurrent GPU access natively
- Downside: no support for tool/function calling as plugins (custom format needed)

### Ollama
- Wraps llama.cpp with a nicer API, model management,
  and built-in OpenAI-compatible endpoint
- `ollama pull model` then `ollama run`
- Supports tool calling via OpenAI-compatible chat API
- Downside: one more layer of abstraction

### SGLang
- Best multi-model serving, native OpenAI-compatible API
- Native tool-calling parser for MiniCPM5 (`--tool-call-parser minicpm5`)
- More memory-efficient with RadixAttention prefix caching
- Downside: requires Python, more complex to set up

### Current Thinking

Factor | llama.cpp | Ollama | SGLang
---|---|---|---
Setup complexity | Low | Low | Medium
Tool calling | Manual format | OpenAI-compatible | Native MiniCPM5 parser
Multi-model | Multiple servers | Single server manages | Native support
Documentation | Extensive | Extensive | Growing
Python required | No | No | Yes

The final choice doesn't affect the rest of the architecture — all three expose
an OpenAI-compatible API, which is what the router and RAG pipeline talk to.

VRAM fits both models simultaneously:

| Model | Quant | VRAM | Port | Role |
|---|---|---|---|---|
| MiniCPM5-1B | Q8_0 (~1.1 GB) | port 8081 | Router |
| RAG model (7-8B) | Q4_K_M (~5.5 GB) | port 8082 | Knowledge Q&A |
| **Total** | | **~6.6 GB** | out of 24 available |

---

## RAG Model

The RAG model is separate from the router. Its job is to answer questions
by reading context you provide — NOT from its own training knowledge.

Top choices for the 7900 XTX:

| Model | Quant | VRAM | Why |
|---|---|---|---|
| Llama-3.1-8B-Instruct | Q4_K_M | ~5.5 GB | Best instruction following, won't ignore context |
| Qwen2.5-7B-Instruct | Q4_K_M | ~5 GB | Excellent RAG quality, long context |
| Gemma-2-9B-it | Q4_K_M | ~6 GB | Google quality, concise |

All three fit alongside MiniCPM5 with room for KV cache.

---

## Embedding Model

An embedding model converts text to vectors (lists of numbers) that represent
meaning. The vector DB uses these to find "the chunk most relevant to the question."

Key facts:
- Tiny: BGE-small-en-v1.5 is ~0.1 GB
- Runs on CPU, ~50ms per query
- Doesn't need a GPU
- You don't choose this based on quality — all popular ones work well for RAG

Top pick: **BGE-small-en-v1.5** — fast, good quality, runs on CPU.
Alternative: **GTE-small** — slightly better but larger.

---

## Vector DB (Undecided)

A vector database stores document chunks and finds the most relevant ones
for a given question using vector similarity search.

| Option | Type | Setup | Why |
|---|---|---|---|
| **Chroma** | Embedded (in-process) | `pip install chromadb` | Simplest — no server, file-based |
| **Qdrant** | Local server or cloud | Docker container | Scales better, has filtering |
| **LanceDB** | Embedded | `pip install lancedb` | Uses Lance columnar format, fast |
| **Pinecone** | Cloud | API key | Managed, no ops — but data leaves your machine |

For a personal project on the same machine: **Chroma** is the simplest path.
If you want to learn a production-grade tool: **Qdrant**.

---

## Document Ingestion (Firecrawl)

Firecrawl turns URLs into LLM-ready clean text (markdown). It handles
JavaScript rendering, selects only the relevant content, and returns
structured text you can feed directly into the RAG pipeline.

```
Firecrawl API → clean markdown → chunk → embed → store in vector DB
```

### What Firecrawl Does

```
Input URL: https://some-docs-page.com
     │
     ▼
Firecrawl renders the page (handles JS, popups, etc.)
     │
     ▼
Extracts the main content (strips nav, ads, footers)
     │
     ▼
Returns clean markdown
     │
     ▼
Your RAG pipeline: chunk text → embed → store
```

### Does AWS Have This?

**No.** AWS has:
- **Amazon Kendra Web Crawler** — indexes pages for Kendra search,
  not for LLM-ready extraction. It's designed for enterprise search, not RAG.
- **AWS Glue + Lambda + Playwright** — you'd build your own crawler from scratch.
  Functional but requires significant setup.

Firecrawl is purpose-built for this use case. There's no AWS equivalent.

### Alternatives to Firecrawl

| Tool | Type | Cost |
|---|---|---|
| **Firecrawl** | Cloud API | Free tier (500 pages), $19/mo |
| **Crawl4AI** | Open source, local | Free — runs on your machine |
| **Apify** | Cloud with marketplace | Free tier, pay per use |

Firecrawl and Crawl4AI are the two main options. Crawl4AI is open source and
runs entirely locally — no API key needed. Firecrawl is simpler (just call an API).

---

## Full Data Flow

```
INGESTION (one-time per document set)
═════════════════════════════════════════════

1. Gather sources:
   ├── URLs for Firecrawl
   ├── PDFs / docs (local files)
   └── Code / markdown repos

2. Convert to clean text:
   ├── Firecrawl for web pages
   └── Direct read for local files

3. Chunk into segments (~512 tokens each)
   └── Overlap chunks by ~50 tokens for context continuity

4. Embed each chunk (BGE-small, CPU)
   └── Store in vector DB with original text + metadata


QUERY (per user question)
══════════════════════════

1. Client sends question to cognitive core
2. Router (MiniCPM5-1B) decides:
   ├── Can answer directly → return
   ├── Needs tool call → dispatch to tool
   └── Knowledge question → RAG pipeline:

3. RAG pipeline:
   ├── Embed query (BGE-small, CPU, ~50ms)
   ├── Search vector DB → top 3-5 chunks
   ├── Build prompt:
   │   "Answer based on the following context:
   │    ---
   │    [chunk text]
   │    ---
   │    Question: [user question]"
   └── Send to RAG model (port 8082)
       └── Return grounded answer

4. Router returns answer to client
```

---

## VRAM Budget (7900 XTX, 24 GB)

| Component | VRAM | Notes |
|---|---|---|
| MiniCPM5-1B Q8_0 | ~1.1 GB | Always loaded (router) |
| RAG model (7-8B) Q4_K_M | ~5.5 GB | Loaded on demand |
| KV cache (32K context × 2) | ~2-3 GB | Shared between models |
| **Total running** | **~9 GB** | |
| **Free** | **~15 GB** | For other tasks |

Both models fit simultaneously on a 7900 XTX with 15 GB to spare.
