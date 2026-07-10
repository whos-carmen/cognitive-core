#!/usr/bin/env python3
"""RAG Pipeline — ingestion and query for the Cognitive Core.

Ingestion:
    python rag_pipeline.py ingest <file_or_dir> [--source-name NAME]

Query:
    python rag_pipeline.py query "What is the cognitive core?"

Serve with RAG model on port 8082 for knowledge-backed answers.
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

# ── Globals (lazy-loaded) ──
_embed_model = None
_chroma_collection = None

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
RAG_URL = "http://localhost:8082/v1"

# System prompt injected into RAG queries
RAG_SYSTEM_PROMPT = """You are a knowledge assistant. You are given context and a question.
Answer the question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information to answer that."
Do NOT use your own knowledge — only the context below."""


# ═══════════════════════════════════════════
#  Embedding
# ═══════════════════════════════════════════

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("ibm-granite/granite-embedding-english-r2")
    return _embed_model


# ═══════════════════════════════════════════
#  Chroma
# ═══════════════════════════════════════════

def get_chroma(path=CHROMA_PATH):
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        db = chromadb.PersistentClient(path=path)
        _chroma_collection = db.get_or_create_collection("knowledge")
    return _chroma_collection


# ═══════════════════════════════════════════
#  Chunking
# ═══════════════════════════════════════════

def chunk_text(text: str, chunk_size=512, overlap=256) -> list[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []
    chunks = []
    for i in range(0, max(len(text), 1), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) >= 20:
            chunks.append(chunk)
    return chunks


# ═══════════════════════════════════════════
#  Ingestion
# ═══════════════════════════════════════════

def ingest_text(text: str, source: str, chunk_size=512, overlap=256) -> dict:
    """Chunk, embed with Granite, and store in Chroma."""
    model = get_embed_model()
    collection = get_chroma()

    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        return {"status": "error", "message": "Text too short after chunking"}

    ids = [f"{source}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": source} for _ in chunks]

    # Embed in batches of 64
    batch_size = 64
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        embeddings = model.encode(
            chunks[batch_start:batch_end],
            normalize_embeddings=True
        ).tolist()
        collection.add(
            documents=chunks[batch_start:batch_end],
            embeddings=embeddings,
            metadatas=metadatas[batch_start:batch_end],
            ids=ids[batch_start:batch_end],
        )

    return {
        "status": "ok",
        "chunks": len(chunks),
        "source": source,
        "total_chars": len(text),
    }


def ingest_file(path: str, source_name: str = None) -> dict:
    """Read a file and ingest it."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    source = source_name or path.name
    return ingest_text(text, source)


def ingest_directory(path: str) -> list[dict]:
    """Recursively ingest all text files in a directory."""
    results = []
    path = Path(path)
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.suffix in (".txt", ".md", ".py", ".js", ".ts",
                                         ".go", ".rs", ".sh", ".json", ".yaml", ".yml",
                                         ".html", ".css", ".csv"):
            result = ingest_file(str(f))
            results.append(result)
            print(f"  {result['chunks']:>3} chunks  {f.name}")
    return results


# ═══════════════════════════════════════════
#  Query
# ═══════════════════════════════════════════

def query_rag(question: str, n_results: int = 5) -> dict:
    """Query the RAG pipeline: embed → search Chroma → prompt RAG model."""
    model = get_embed_model()
    collection = get_chroma()

    total_chunks = collection.count()
    if total_chunks == 0:
        return {
            "answer": "Knowledge base is empty. Ingest documents first.",
            "chunks_used": 0,
            "total_chunks": 0,
        }

    # Embed query
    q_embedding = model.encode([question], normalize_embeddings=True).tolist()[0]

    # Search Chroma
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(n_results, total_chunks),
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return {"answer": "No relevant documents found.", "chunks_used": 0, "total_chunks": total_chunks}

    # Build context
    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        source = meta.get("source", "unknown") if meta else "unknown"
        context_parts.append(f"[Source: {source}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    # Build RAG prompt
    rag_messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    # Query RAG model on port 8082
    from openai import OpenAI
    client = OpenAI(base_url=RAG_URL, api_key="not-needed")
    response = client.chat.completions.create(
        model="granite",
        messages=rag_messages,
        max_tokens=500,
        stream=False,
    )

    answer = response.choices[0].message.content or ""

    return {
        "answer": answer,
        "chunks_used": len(documents),
        "total_chunks": total_chunks,
        "source_chunks": [
            {
                "source": (metadatas[i].get("source", "unknown") if metadatas[i] else "unknown"),
                "snippet": documents[i][:150],
            }
            for i in range(len(documents))
        ],
    }


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Cognitive Core — RAG Pipeline")
    sub = ap.add_subparsers(dest="command", required=True)

    # ingest
    ingest_p = sub.add_parser("ingest", help="Ingest documents into Chroma")
    ingest_p.add_argument("path", help="File or directory to ingest")
    ingest_p.add_argument("--source-name", help="Override source name")

    # query
    query_p = sub.add_parser("query", help="Query the RAG pipeline")
    query_p.add_argument("question", help="Your question")
    query_p.add_argument("--results", "-n", type=int, default=5, help="Number of chunks to retrieve")

    # list
    list_p = sub.add_parser("list", help="List Chroma stats")

    args = ap.parse_args()

    if args.command == "ingest":
        path = args.path
        if os.path.isfile(path):
            print(f"Ingesting file: {path}")
            result = ingest_file(path, args.source_name)
            print(f"  {result['chunks']} chunks from '{result['source']}' ({result['total_chars']} chars)")
            print(f"  Status: {result['status']}")
        elif os.path.isdir(path):
            print(f"Ingesting directory: {path}")
            results = ingest_directory(path)
            total_chunks = sum(r["chunks"] for r in results if r["status"] == "ok")
            total_files = sum(1 for r in results if r["status"] == "ok")
            print(f"\nDone: {total_files} files, {total_chunks} total chunks")
        else:
            print(f"Error: {path} not found")

    elif args.command == "query":
        result = query_rag(args.question, args.results)
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {result['answer']}")
        print(f"\n[Stats] {result['chunks_used']} chunks used, {result['total_chunks']} total in KB")

        if result.get("source_chunks"):
            print("\nSources:")
            for s in result["source_chunks"]:
                print(f"  · {s['source']}: {s['snippet']}...")

    elif args.command == "list":
        collection = get_chroma()
        count = collection.count()
        print(f"Chroma KB: {count} chunks")
        if count > 0:
            samples = collection.peek()
            print(f"\nSample chunks:")
            for doc, meta in zip(samples.get("documents", [])[:3],
                                 samples.get("metadatas", [])[:3]):
                src = meta.get("source", "?") if meta else "?"
                print(f"  [{src}]: {doc[:100]}...")


if __name__ == "__main__":
    main()
