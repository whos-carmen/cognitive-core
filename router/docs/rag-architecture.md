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

## Serving Layer — Reality: llama.cpp with ROCm

### The Original Plan: SGLang

SGLang was the **recommended backend** by the OpenBMB team for MiniCPM5. The reason is
specific: MiniCPM5 emits tool calls in XML format (`<tool_call>...</tool_call>`),
and SGLang has a **native parser** (`--tool-call-parser minicpm5`) that converts
them to standard OpenAI-compatible tool_calls automatically. llama.cpp doesn't
have this parser — you'd need to handle the XML format yourself (use
[../eval/tool_parser.py](../eval/tool_parser.py) if you go that route).

### Why SGLang Didn't Work

SGLang's ROCm support relies on `sgl_kernel`, which contains CUDA-like C++ kernels
compiled for specific AMD GPU architectures. As of July 2026, `sgl_kernel` only
targets **gfx942 (MI300/MI325)** and **gfx950 (MI350)** — AMD data-center GPUs.
Consumer RDNA3 GPUs like the **7900 XTX (gfx1100)** are not supported.

Attempted workarounds that failed:

1. **Force gfx1100 target in setup_rocm.py** — The kernel code uses MI300/MI350-specific
   instructions that don't exist on gfx1100. Compilation fails.
2. **Install aiter (AMD AI Edge Toolkit)** — The JIT build crashed because the
   `hipcc` compiler flags are incompatible between ROCm 7.2 and what aiter expects.
3. **SGLang pip release** — The `--tool-call-parser minicpm5` feature doesn't exist
   in the pip release (it requires building from main).

**Bottom line:** If you have a consumer AMD GPU (RX 7900 series, RX 6000/7000 series),
use llama.cpp. If you have an AMD Instinct MI300/MI350, SGLang is viable.

### What We Use Instead: llama.cpp with ROCm

llama.cpp with ROCm is the working backend for consumer AMD GPUs. It:
- Has native gfx1100 support — compiles and runs out of the box
- Achieves ~280-300 tok/s on a 7900 XTX with MiniCPM5-1B Q8_0
- Exposes an OpenAI-compatible API — works with any client
- Requires client-side tool call parsing (handled by [../eval/tool_parser.py](../eval/tool_parser.py))

```bash
# Build for consumer AMD GPU
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_HIP=ON -DGGML_HIP_GRAPH=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS="gfx1100"    # or gfx1030 (RX 6900), gfx942 (MI300), etc.
cmake --build . --config Release -j$(nproc)
```

### Recommendation (Updated)

| If you have... | Pick | Why |
|---|---|---|
| AMD consumer GPU (RX 7900 XTX, etc.) | **llama.cpp** + [custom parser](../eval/tool_parser.py) | SGLang kernels don't support gfx1100 |
| AMD data-center GPU (MI300/MI350) | **SGLang** (build from main) | Native tool parsing, higher throughput |
| NVIDIA GPU | **SGLang** or **llama.cpp** | Both work, SGLang has better tool support |

A unified parser that handles all three tool call XML formats (`<tool_call>` JSON,
`<function>` attribute JSON, and `<function><param>` native XML) is included at
[eval/tool_parser.py](../eval/tool_parser.py). It works with any serving layer —
run it on the client side, or integrate it into your agent loop.

### Layout

```
Port 8081 — SGLang server serving MiniCPM5-1B (router)
Port 8082 — llama.cpp / SGLang serving RAG model (knowledge)
```

VRAM fits both models simultaneously:

| Model | Quant | VRAM | Port | Role |
|---|---|---|---|---|
| MiniCPM5-1B | Q8_0 (~1.1 GB) | 8081 | Router |
| RAG model (7-8B) | Q4_K_M (~5.3 GB) | 8082 | Knowledge Q&A |
| **Total** | | **~6.4 GB** | out of 24 available |

---

## RAG Model

The RAG model is separate from the router. Its job is to answer questions
by reading context you provide — NOT from its own training knowledge.

Top choices for the 7900 XTX:

| Model | Quant | VRAM | Why |
|---|---|---|---|
| **Granite 4.1-8B-Instruct** | Q4_K_M | ~5.3 GB | Apache 2.0 license, strong instruction following, IBM enterprise provenance |
| Qwen2.5-7B-Instruct | Q4_K_M | ~5 GB | Excellent RAG quality, long context |
| Gemma-2-9B-it | Q4_K_M | ~6 GB | Google quality, concise |

All fit alongside MiniCPM5 with room for KV cache. **Granite 4.1-8B** is the
recommended default — the Apache 2.0 license is cleaner for redistribution and
IBM trains exclusively on permissively licensed data.

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
| Granite 4.1-8B Q4_K_M (RAG) | ~5.3 GB |
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
| Granite 4.1-8B Q4_K_M | ~5.3 GB | Loaded on demand (RAG) |
| Granite embedding 149M | ~0.3 GB | GPU or CPU — 17 GB free either way |
| KV cache (32K context × 2) | ~2-3 GB | Shared between models |
| **Total running** | **~9 GB** | |
| **Free** | **~15 GB** | For other tasks |

Both models fit simultaneously on a 7900 XTX with 15 GB to spare.

---

## Advanced RAG Techniques

### GraphRAG

[Microsoft GraphRAG](https://graphrag.com) extends vector RAG by building a
knowledge graph from your documents — extracting entities and relationships,
then generating hierarchical community summaries. It can answer multi-hop
questions that require following chains of relationships.

GraphRAG runs locally (MIT license). The **LazyGraphRAG** mode defers
summarization to query time, cutting indexing cost significantly.

For the cognitive core, GraphRAG is a potential upgrade path if vector-only
RAG isn't sufficient. Start with basic Chroma RAG, then add GraphRAG if you
need multi-hop reasoning.

### RAG Techniques Reference

The [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) repo (28K+
stars) covers 42+ techniques with runnable notebooks: query transformations,
fusion retrieval, reranking, Self-RAG, Corrective RAG, Agentic RAG, and more.
