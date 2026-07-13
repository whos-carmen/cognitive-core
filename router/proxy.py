#!/usr/bin/env python3
"""Cognitive Core — Smart Routing Proxy

A drop-in OpenAI-compatible API that routes prompts to the right
model based on complexity. Also provides coding agents.

Usage:
    PROXY_PORT=8080 python proxy.py

Then use with any OpenAI client:
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from openai import OpenAI

# ── Model endpoints ──
ROUTER_URL = "http://localhost:8081/v1"     # Qwen2.5-3B (fast)
RAG_URL = "http://localhost:8082/v1"        # Qwen2.5-7B (quality)
AGENT_URL = "http://localhost:8083/v1"      # Qwen2.5-Coder-7B (agentic)

# ── Tavily web search (via subprocess npx) ──
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via Tavily REST API."""
    import urllib.request, urllib.parse, json
    try:
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return "TAVILY_API_KEY not set."
        data = json.dumps({"api_key": api_key, "query": query, "max_results": max_results, "search_depth": "basic"}).encode()
        req = urllib.request.Request("https://api.tavily.com/search", data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        results = result.get("results", [])
        if not results:
            return "No web results found."
        return "" + "\n\n".join(f"Title: {r.get('title','')}\nURL: {r.get('url','')}\n{r.get('content','')}" for r in results[:3])
    except Exception as e:
        return f"Web search error: {e}"

# ── Chroma RAG ──
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
def rag_query(query: str) -> str:
    """Query the Chroma knowledge base."""
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
        embed = SentenceTransformer("ibm-granite/granite-embedding-english-r2")
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = db.get_or_create_collection("knowledge")
        if collection.count() == 0:
            return "Knowledge base is empty."
        q_emb = embed.encode([query], normalize_embeddings=True).tolist()[0]
        results = collection.query(query_embeddings=[q_emb], n_results=3)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if not docs:
            return "No relevant docs found."
        return "\n---\n".join(f"[{metas[i].get('source','?') if metas[i] else '?'}]\n{docs[i][:300]}" for i in range(len(docs)))
    except Exception as e:
        return f"RAG error: {e}"

# ── Complexity thresholds ──
SIMPLE_KEYWORDS = [
    "hello", "hi", "hey", "thanks", "what is 2", "what's 2", "what is the capital",
    "yes", "no", "ok", "okay", "goodbye", "bye", "who are you",
]
CODING_KEYWORDS = [
    "haskell", "python", "javascript", "typescript", "rust", "golang", "java", "cpp", "c++", "c#", "swift",
    "kotlin", "ruby", "php", "scala", "perl", "lua", "elixir", "clojure", "dart", "sql",
    "function", "class", "method", "variable", "loop", "array", "list", "dict", "string", "int",
    "code", "program", "script", "algorithm", "compile", "debug", "error", "bug", "test",
    "api", "endpoint", "route", "database", "query", "server", "client", "frontend", "backend",
    "recursion", "lambda", "monad", "functor", "applicative", "typeclass", "instance", "derive",
]

COMPLEX_PATTERNS = [
    r"\bwrite\b.*\bcode\b", r"\bimplement\b", r"\bcreate a\b.*\bfunction\b",
    r"\breview\b.*\bcode\b", r"\bsecurity\b.*\breview\b", r"\brefactor\b",
    r"\bexplain\b.*\bin detail\b", r"\bcompare\b", r"\banalyze\b",
    r"\bdebug\b", r"\btroubleshoot\b",
]

# ── Agent mode detection ──
WRITER_PATTERNS = [r"\bwrite\b", r"\bcreate\b", r"\bimplement\b", r"\bgenerate\b", r"\bproduce\b.*\bcode\b"]
REVIEWER_PATTERNS = [r"\breview\b", r"\bcheck\b.*\bcode\b", r"\bdoes this look\b", r"\bcode review\b"]
SECURITY_PATTERNS = [r"\bsecurity\b", r"\bvulnerability\b", r"\bsqli\b", r"\bxss\b", r"\binjection\b", r"\bexploit\b"]


# ═══════════════════════════════════════════
#  Complexity Classifier
# ═══════════════════════════════════════════

def classify_complexity(prompt: str) -> str:
    """Classify a prompt as 'simple', 'medium', or 'complex'."""
    p = prompt.lower().strip()

    # Simple: single facts, greetings, math
    for kw in SIMPLE_KEYWORDS:
        if p.startswith(kw) or p == kw:
            return "simple"

    # Very short questions are usually simple
    if len(p.split()) <= 5 and "?" in p:
        return "simple"

    # Complex: matches complex patterns
    for pat in COMPLEX_PATTERNS:
        if re.search(pat, p):
            return "complex"

    # Longer prompts are more likely complex
    word_count = len(p.split())
    if word_count > 50:
        return "complex"
    if word_count > 20:
        return "medium"

    return "medium"


def classify_agent_task(prompt: str) -> str | None:
    """Classify if a prompt should go to an agent. Returns agent type or None."""
    p = prompt.lower()
    is_code = any(kw in p for kw in ["code", "function", "script", "program", "app", "api", "endpoint"])
    is_writer = any(re.search(pat, p) for pat in WRITER_PATTERNS)
    is_reviewer = any(re.search(pat, p) for pat in REVIEWER_PATTERNS)
    is_security = any(re.search(pat, p) for pat in SECURITY_PATTERNS)

    if is_security and (is_code or is_reviewer):
        return "security_reviewer"
    if is_reviewer and is_code:
        return "reviewer"
    if is_writer and is_code:
        return "writer"
    return None  # Not an agent task


# ═══════════════════════════════════════════
#  Model Callers
# ═══════════════════════════════════════════

def call_router(messages: list, max_tokens: int = 500) -> str:
    """Call the fast 1B model."""
    client = OpenAI(base_url=ROUTER_URL, api_key="not-needed")
    try:
        r = client.chat.completions.create(model="qwen2.5-3b", messages=messages, max_tokens=max_tokens)
        return r.choices[0].message.content or ""
    except Exception as e:
        return f"Error: {e}"


def call_granite(messages: list, max_tokens: int = 800) -> str:
    """Call Granite 8B for quality responses."""
    client = OpenAI(base_url=RAG_URL, api_key="not-needed")
    try:
        r = client.chat.completions.create(model="qwen2.5-7b", messages=messages, max_tokens=max_tokens)
        return r.choices[0].message.content or ""
    except Exception as e:
        return f"Error: {e}"


def call_qwen(messages: list, system: str = None, max_tokens: int = 1000) -> str:
    """Call Qwen 4B for agentic tasks."""
    client = OpenAI(base_url=AGENT_URL, api_key="not-needed")
    if system:
        messages = [{"role": "system", "content": system}] + messages
    try:
        r = client.chat.completions.create(model="qwen2.5-coder-7b", messages=messages, max_tokens=max_tokens)
        content = r.choices[0].message.content or ""
        reasoning = getattr(r.choices[0].message, "reasoning_content", "") or ""
        return content or reasoning or ""
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════
#  Agent Prompts
# ═══════════════════════════════════════════

AGENT_PROMPTS = {
    "writer": """You are a code writer agent. Write clean, well-documented code.
Given a coding task, produce the code with:
- Clear comments explaining the logic
- Error handling
- Type hints where appropriate
- Follow PEP8 style
Output ONLY the code and brief usage example.""",

    "reviewer": """You are a code reviewer agent. Review code for:
- Correctness: Does it do what it's supposed to?
- Performance: Can it be optimized?
- Style: Does it follow best practices?
- Edge cases: Are there unhandled cases?
Output a structured review with findings and suggestions.""",

    "security_reviewer": """You are a security reviewer agent. Audit code for:
- SQL injection
- XSS vulnerabilities
- Command injection
- Insecure deserialization
- Hardcoded secrets
- Authentication/authorization flaws
- Rate limiting issues
Output a structured security audit with severity levels (CRITICAL, HIGH, MEDIUM, LOW).""",
}


# ═══════════════════════════════════════════
#  Request Handler
# ═══════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and len(args) > 1:
            print(f"  [{args[0]}] {args[1]} {args[2]}", flush=True)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [
                    {"id": "cognitive-core-proxy", "object": "model", "created": int(time.time()), "owned_by": "cognitive-core"},
                    {"id": "simple", "object": "model", "created": int(time.time()), "owned_by": "router"},
                    {"id": "medium", "object": "model", "created": int(time.time()), "owned_by": "granite"},
                    {"id": "complex", "object": "model", "created": int(time.time()), "owned_by": "granite"},
                    {"id": "writer", "object": "model", "created": int(time.time()), "owned_by": "agent"},
                    {"id": "reviewer", "object": "model", "created": int(time.time()), "owned_by": "agent"},
                    {"id": "security_reviewer", "object": "model", "created": int(time.time()), "owned_by": "agent"},
                ]
            })
        else:
            self.send_json({"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/v1/chat/completions":
            return self._handle_chat()
        self.send_json({"error": "not found"})

    def _handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        # Strip OpenAI-specific fields that we don't support
        body.pop("stream_options", None)
        messages = body.get("messages", [])
        model = body.get("model", "auto")
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", 1000)

        # Extract the last user message (handle both string and list content)
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                raw = m["content"]
                if isinstance(raw, str):
                    last_user = raw
                elif isinstance(raw, list):
                    parts = [p.get("text","") for p in raw if isinstance(p, dict)]
                    last_user = " ".join(parts)
                else:
                    last_user = str(raw)
                break

        if not last_user:
            self._send_error("no user message found")
            return

        t0 = time.time()
        complexity = classify_complexity(last_user)
        agent_type = classify_agent_task(last_user) if complexity == "complex" else None

        # Check if web search or RAG is needed (at any complexity level)
        needs_web = any(kw in last_user.lower() for kw in ["himeko", "hsr", "honkai", "genshin", "star rail", "zzz", "wuthering", "anime", "manga", "best teammate", "tier list", "build", "weapon", "team comp", "games", "pop culture", "current", "news", "weather", "latest", "search", "find", "who is", "what is"])
        needs_rag = any(kw in last_user.lower() for kw in ["rag", "knowledge base", "chroma", "what info", "stored in"])

        # Coding questions → coder model regardless of complexity
        if any(kw in last_user.lower() for kw in CODING_KEYWORDS):
            response = call_qwen(messages, max_tokens=max_tokens)
            route_info = "coder"
        elif needs_web:
            response = web_search(last_user)
            route_info = "web_search"
        elif needs_rag:
            response = rag_query(last_user)
            route_info = "rag"
        elif agent_type:
            prompt = AGENT_PROMPTS[agent_type]
            response = call_qwen(messages, system=prompt, max_tokens=max_tokens)
            route_info = f"agent:{agent_type}"
        elif model == "simple" or complexity == "simple":
            response = call_router(messages, max_tokens=max_tokens)
            route_info = "simple"
        elif model == "complex" or complexity == "complex":
            response = call_granite(messages, max_tokens=max_tokens)
            route_info = "complex"
        else:
            response = call_granite(messages, max_tokens=max_tokens)
            route_info = "medium"

        elapsed = round((time.time() - t0) * 1000)

        # Log
        print(f"  [{route_info}] {elapsed}ms | {last_user[:60]}")

        # Stream or return
        if stream:
            self._send_streaming(response, model)
        else:
            self._send_response(response, model)

    def _send_response(self, content: str, model: str):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }).encode())

    def _send_streaming(self, content: str, model: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        chunk_size = 20
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            data = json.dumps({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            })
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.01)

        # Final chunk with finish_reason: stop (OpenAI clients check this)
        final = json.dumps({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
        self.wfile.write(f"data: {final}\n\n".encode())
        self.wfile.write(f"data: [DONE]\n\n".encode())

    def _send_error(self, msg: str):
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": msg}).encode())

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PROXY_PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Cognitive Core Proxy → http://0.0.0.0:{port}/v1")
    print(f"  Simple queries → 1B router ({ROUTER_URL})")
    print(f"  Medium/complex → Granite 8B ({RAG_URL})")
    print(f"  Coding agents  → Qwen 4B ({AGENT_URL})")
    print(f"  Use with: OpenAI(base_url='http://localhost:{port}/v1', api_key='not-needed')")
    server.serve_forever()
