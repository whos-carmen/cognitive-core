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

## Serving Layer — Recommendation: SGLang + llama.cpp Backup

### Why SGLang for MiniCPM5

SGLang is the **recommended backend** by the OpenBMB team for MiniCPM5. The reason is
specific: MiniCPM5 emits tool calls in XML format (`<tool_call>...</tool_call>`),
and SGLang has a **native parser** (`--tool-call-parser minicpm5`) that converts
them to standard OpenAI-compatible tool_calls automatically. llama.cpp doesn't
have this parser — you'd need to handle the XML format yourself (use
[../eval/tool_parser.py](../eval/tool_parser.py) if you go that route).

Other SGLang advantages:
- **RadixAttention prefix cache** — reuses KV cache across requests with
  shared prefixes, reducing latency and memory by ~30-50% for multi-turn
- **Highest token throughput** in 2025-2026 benchmarks vs vLLM, TGI
- **OpenAI-compatible API** — works with any client

### The Catch

The `--tool-call-parser minicpm5` feature requires a **SGLang build newer than
the latest pip release** as of June 2026. Plain chat completions work fine on the
pip release — but tool parsing needs a `pip install` from the main branch or the
official Docker image.

### Alternative: llama.cpp (Simplicity Pick)

If you don't need tool-call parsing (you handle the XML yourself or use a
custom format), llama.cpp is:
- Single binary, no Python runtime
- GPU-first, handles model swapping
- Easy to run multiple servers on different ports

### Recommendation

| If you... | Pick |
|---|---|
| Want native tool parsing for MiniCPM5 | **SGLang** (build from main branch) |
| Want simplicity, don't mind XML handling | **llama.cpp** with [custom parser](../eval/tool_parser.py) |

A unified parser that handles both `<tool_call>` and `<function>` XML formats
is included at [eval/tool_parser.py](../eval/tool_parser.py). It works with any
serving layer — run it on the client side, or integrate it into your agent loop.

All four expose an OpenAI-compatible API, so the choice doesn't affect the
RAG pipeline. Switch later if needed.

### Layout

```
Port 8081 — SGLang server serving MiniCPM5-1B (router)
Port 8082 — llama.cpp / SGLang serving RAG model (knowledge)
```

VRAM fits both models simultaneously:

| Model | Quant | VRAM | Port | Role |
|---|---|---|---|---|
| MiniCPM5-1B | Q8_0 (~1.1 GB) | 8081 | Router |
| RAG model (7-8B) | Q4_K_M (~5.5 GB) | 8082 | Knowledge Q&A |
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

Recommended: **IBM Granite-embedding-english-r2**.

| Property | Value |
|---|---|
| Parameters | 149M |
| Embedding size | 768 |
| Context length | **8192 tokens** — can embed full document sections in one pass |
| Size on disk | ~0.3 GB |
| Speed (GPU) | 144 docs/sec (on H100), near-instant at 1B scale |
| Speed (CPU) | ~50-100ms per query |
| License | Apache 2.0 |
| Architecture | ModernBERT (bi-encoder) |

Benchmarks vs common alternatives:

| Model | Average | BEIR Retrieval | CoIR (Code) | MLDR (Long) | MTRAG (Conv) |
|---|---|---|---|---|---|
| BGE-base-en-v1.5 | 46.9 | 54.8 | 46.6 | 33.5 | 38.8 |
| GTE-base-en-v1.5 | 52.8 | 55.5 | 42.4 | 42.7 | 36.0 |
| **Granite-embedding-english-r2** | **59.5** | 56.4 | 54.8 | 41.6 | 57.6 |

### GPU vs CPU

At only 0.3 GB, Granite fits easily on GPU alongside both models:

| Component | VRAM |
|---|---|
| MiniCPM5-1B Q8_0 (router) | ~1.1 GB |
| Llama-3.1-8B Q4_K_M (RAG) | ~5.5 GB |
| Granite embedding 149M | ~0.3 GB |
| KV cache | ~2-3 GB |
| **Total** | **~9 GB** — **15 GB free** on 7900 XTX |

Use it via sentence-transformers:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("ibm-granite/granite-embedding-english-r2")

# Encode queries and documents
texts = ["What is OPD?", "On-Policy Distillation is..."]
embeddings = model.encode(texts, normalize_embeddings=True)
```

---

## Vector DB — Recommendation: Chroma, with Option to Switch

### Why Chroma for a Personal Project

| Factor | Chroma | Qdrant | LanceDB | Pinecone |
|---|---|---|---|---|
| Setup | `pip install chromadb` | Docker container | `pip install lancedb` | API key + cloud account |
| Server needed | No (in-process) | Yes | No (in-process) | Yes (cloud) |
| Data leaves machine | No | No | No | Yes |
| Cost | Free | Free (self-host) | Free | Pay per use |
| Scales to | Small-medium | Large | Medium | Large |
| Filters/metadata | Basic | Advanced | Basic | Advanced |

**Chroma is the right starting point because:**
1. No server to run — `import chromadb` and it works
2. Persists to disk as files — easy to back up, move, reset
3. Good enough for thousands of documents at personal scale
4. If you outgrow it, switching to Qdrant or LanceDB is straightforward
   (they all speak the same API patterns)

### Switch Path

If Chroma becomes too slow or you want metadata filtering:
```
Chroma → Qdrant (Docker, same API style, better filtering)
Chroma → LanceDB (same in-process style, columnar storage, faster at scale)
```
No need to decide today. Start with Chroma, switch later if needed.
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

4. Embed each chunk (Granite-embedding-english-r2, GPU or CPU)
   └── Store in vector DB with original text + metadata


QUERY (per user question)
══════════════════════════

1. Client sends question to cognitive core
2. Router (MiniCPM5-1B) decides:
   ├── Can answer directly → return
   ├── Needs tool call → dispatch to tool
   └── Knowledge question → RAG pipeline:

3. RAG pipeline:
   ├── Embed query (Granite-embedding-english-r2, GPU or CPU, ~50ms)
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
| Granite embedding 149M | ~0.3 GB | GPU or CPU — 17 GB free either way |
| KV cache (32K context × 2) | ~2-3 GB | Shared between models |
| **Total running** | **~9 GB** | |
| **Free** | **~15 GB** | For other tasks |

Both models fit simultaneously on a 7900 XTX with 15 GB to spare.
