#!/usr/bin/env python3
"""Cognitive Core — Runtime Dashboard + Live Chat

Shows router log, RAG log, traces, and a live chat panel to watch
the model think, reason, and call tools in real-time.

Usage:
    python scripts/runtime_dashboard.py --port 8766
"""

import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import asyncio
import sys as _sys

# ── Paths ──
ROUTER_LOG = "/tmp/cognitive-core.log"
RAG_LOG = "/tmp/cognitive-core-rag.log"
TRACES_PATH = "/var/log/cognitive-core/traces.jsonl"
CHAT_LOG = "/var/log/cognitive-core/chat.jsonl"
TOOLS_LOG = "/var/log/cognitive-core/tools.jsonl"
RAG_LOG_STRUCTURED = "/var/log/cognitive-core/rag.jsonl"
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
ROUTER_URL = "http://localhost:8081/v1"
RAG_URL = "http://localhost:8082/v1"
MAX_TRACES = 200


def tail(path: str, n: int = 40) -> str:
    if not os.path.exists(path):
        return "[waiting for log file...]"
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except (IOError, PermissionError):
        return "[cannot read log]"


def read_traces(path: str, last_n: int = 50) -> list[dict]:
    if not os.path.exists(path):
        return []
    traces = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except (IOError, PermissionError):
        return []
    return traces[-last_n:]


def rocm_vram() -> str:
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if "VRAM Total Memory" in line and "GPU[0]" in line:
                total = int(line.split(":")[-1].strip()) // (1024**3)
            if "VRAM Total Used Memory" in line:
                used = int(line.split(":")[-1].strip()) // (1024**3)
        return f"{used}G / {total}G"
    except Exception:
        return "N/A"


def chroma_count() -> int:
    try:
        import chromadb
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        return db.get_or_create_collection("knowledge").count()
    except Exception:
        return -1


# ═══════════════════════════════════════════════════════════════
#  HTTP Handler
# ═══════════════════════════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/data":
            self.send_json(self._collect_data())
        elif path == "/health":
            self.send_json({"status": "ok"})
        elif path == "/api/vram":
            self.send_json({"vram": rocm_vram()})
        else:
            self.send_html()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/ingest":
            return self._ingest_handler()
        elif path == "/api/chat":
            return self._chat_handler()
        self.send_json({"status": "error", "message": "not found"})

    def _collect_data(self):
        return {
            "router_log": tail(ROUTER_LOG, 40),
            "rag_log": tail(RAG_LOG, 40),
            "chat_log": tail(CHAT_LOG, 20),
            "tool_log": tail(TOOLS_LOG, 20),
            "rag_structured": tail(RAG_LOG_STRUCTURED, 15),
            "traces": read_traces(TRACES_PATH)[::-1],
            "trace_count": sum(1 for _ in open(TRACES_PATH) if _.strip()) if os.path.exists(TRACES_PATH) else 0,
            "chroma_count": chroma_count(),
            "now": datetime.now().isoformat(),
        }

    # ── Chat SSE endpoint (agent loop with MCP tools) ──
    def _chat_handler(self):
        import asyncio
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        prompt = body.get("prompt", "")
        system = body.get("system", "")

        if not prompt.strip():
            self.send_json({"error": "empty prompt"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from agent_loop import Agent
            global _agent, _agent_loop
            if '_agent' not in globals() or _agent is None:
                _agent_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_agent_loop)
                _agent = Agent()
                _agent_loop.run_until_complete(_agent.start())
            result = _agent_loop.run_until_complete(
                _agent.run(prompt, system if system else None, on_token=lambda evt, txt: self._sse(evt, txt))
            )
            self._sse("done", "")
        except Exception as e:
            self._sse("error", str(e))
            self._sse("done", "")

    def _sse(self, event: str, data: str):
        try:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _write_trace(self, prompt, content, reasoning, t_start=None):
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from eval.tool_parser import ToolCallParser
            parser = ToolCallParser()
            calls = parser.parse(content + reasoning)
        except Exception:
            calls = []

        decision = "answer_directly"
        tool_info = None
        if calls:
            decision = "tool_call"
            tool_info = {"name": calls[0]["name"], "parameters": calls[0]["parameters"]}
        elif "search" in reasoning.lower() or "look up" in reasoning.lower() or "retriev" in reasoning.lower() or "not know" in reasoning.lower():
            decision = "needs_knowledge"

        latency = 0
        if t_start:
            latency = round((datetime.now() - t_start).total_seconds() * 1000)

        trace = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "user": prompt[:120],
            "latency_ms": latency,
            "tool": tool_info,
            "reasoning_snippet": reasoning[:200] if reasoning else None,
        }
        try:
            with open(TRACES_PATH, "a") as f:
                f.write(json.dumps(trace) + "\n")
        except (IOError, PermissionError):
            pass

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(HTML.encode())

    # ── Ingestion handler (same as before) ──
    def _ingest_handler(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        text = body.get("text", "")
        source = body.get("source", "paste")

        if len(text.strip()) < 10:
            self.send_json({"status": "error", "message": "Text too short"})
            return

        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            model = SentenceTransformer("ibm-granite/granite-embedding-english-r2")
            db = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = db.get_or_create_collection("knowledge")

            chunk_size, overlap = 512, 256
            chunks = []
            for i in range(0, max(len(text), 1), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk.strip()) >= 20:
                    chunks.append(chunk)

            if not chunks:
                self.send_json({"status": "error", "message": "Text too short after chunking"})
                return

            ids = [f"{source}-{uuid.uuid4().hex[:8]}-{j}" for j in range(len(chunks))]
            metadatas = [{"source": source} for _ in chunks]

            for bs in range(0, len(chunks), 64):
                be = min(bs + 64, len(chunks))
                embs = model.encode(chunks[bs:be], normalize_embeddings=True).tolist()
                collection.add(documents=chunks[bs:be], embeddings=embs, metadatas=metadatas[bs:be], ids=ids[bs:be])

            self.send_json({"status": "ok", "chunks": len(chunks), "source": source, "total_chars": len(text)})
        except Exception as e:
            self.send_json({"status": "error", "message": str(e)})


# ═══════════════════════════════════════════════════════════════
#  HTML
# ═══════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cognitive Core — Runtime</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Consolas", "Monaco", "Liberation Mono", monospace;
  background: #1a1a2e; color: #c7c7c7; font-size: 13px;
  padding: 8px; min-height: 100vh;
}
a { color: #4af; text-decoration: none; }

/* Header */
.header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 4px 8px; background: #16213e; border: 1px solid #0f3460;
  margin-bottom: 6px;
}
.header h1 { font-size: 14px; color: #e94560; font-weight: normal; }
.header .sub { color: #6a6a8a; font-size: 11px; }

/* Stats bar */
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  gap: 3px; margin-bottom: 6px;
}
.stat {
  background: #16213e; border: 1px solid #0f3460;
  padding: 3px 6px; text-align: center;
}
.stat .num { color: #4af; font-size: 14px; }
.stat .lbl { color: #6a6a8a; font-size: 10px; text-transform: uppercase; }

/* Main grid */
.main-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 4px; margin-bottom: 6px;
}
.main-grid .log-panel {
  min-height: 180px; max-height: 240px;
}

/* Log panels */
.log-panel {
  background: #0d0d1a; border: 1px solid #0f3460;
  min-height: 200px; max-height: 280px;
  display: flex; flex-direction: column;
}
.log-panel .panel-title {
  background: #16213e; color: #e94560;
  padding: 2px 6px; font-size: 11px;
  border-bottom: 1px solid #0f3460;
  flex-shrink: 0;
}
.log-panel .panel-body {
  padding: 4px 6px; font-size: 11px;
  white-space: pre-wrap; overflow-y: auto;
  flex: 1;
  color: #8f8;
  line-height: 1.35;
}

/* Live Chat */
.chat-panel {
  background: #0d0d1a; border: 1px solid #0f3460;
  margin-bottom: 6px;
}
.chat-panel .panel-title {
  background: #16213e; color: #e94560;
  padding: 2px 6px; font-size: 11px;
  border-bottom: 1px solid #0f3460;
}
.chat-input-row {
  display: flex; gap: 4px; padding: 4px;
  border-bottom: 1px solid #0f3460;
}
.chat-input-row input {
  flex: 1;
  background: #0d0d1a; border: 1px solid #0f3460;
  color: #c7c7c7; font-family: monospace; font-size: 12px;
  padding: 4px 8px;
}
.chat-input-row input:focus { outline: none; border-color: #4af; }
.chat-input-row .btn {
  background: #0f3460; border: none; color: #c7c7c7;
  padding: 4px 12px; cursor: pointer; font-family: monospace; font-size: 11px;
}
.chat-input-row .btn:hover { background: #1a5276; }

.chat-output {
  padding: 6px; max-height: 400px; overflow-y: auto;
  font-size: 12px; line-height: 1.5;
}
.chat-thinking {
  color: #887; font-style: italic;
}
.chat-content {
  color: #c7c7c7;
}
.chat-tool {
  color: #4af; background: #0f346044;
  padding: 2px 6px; margin: 2px 0; border-radius: 2px;
  display: inline-block;
  font-size: 11px;
}
.chat-error {
  color: #e94560;
}
.chat-status {
  color: #6a6a8a; font-size: 11px; padding: 4px;
}

/* Traces panel */
.traces-panel {
  background: #0d0d1a; border: 1px solid #0f3460;
  margin-bottom: 6px;
}
.traces-panel .panel-title {
  background: #16213e; color: #e94560;
  padding: 2px 6px; font-size: 11px;
  border-bottom: 1px solid #0f3460;
}
.traces-panel .panel-body { max-height: 200px; overflow-y: auto; }
.trace-row {
  display: grid;
  grid-template-columns: 140px 1fr 80px 60px;
  gap: 4px; padding: 2px 6px; font-size: 11px;
  border-bottom: 1px solid #0f346022;
}
.trace-row:hover { background: #16213e44; }
.trace-time { color: #6a6a8a; }
.trace-msg { color: #c7c7c7; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.trace-dec { font-size: 10px; padding: 0 4px; text-align: center; }
.dec-answer_directly { color: #8f8; }
.dec-tool_call { color: #4af; }
.dec-needs_knowledge, .dec-needs_rag { color: #fa0; }
.dec-memory_recall { color: #c7f; }
.dec-unknown { color: #666; }
.trace-lat { color: #6a6a8a; text-align: right; }

/* Ingestion collapsed */
.ingest-panel { background: #0d0d1a; border: 1px solid #0f3460; }
.ingest-panel .panel-title {
  background: #16213e; color: #e94560;
  padding: 2px 6px; font-size: 11px; cursor: pointer;
}
.ingest-body { display: none; padding: 6px; }
.ingest-body.open { display: block; }
.ingest-body input, .ingest-body textarea {
  background: #0d0d1a; border: 1px solid #0f3460;
  color: #c7c7c7; font-family: monospace; font-size: 11px;
  padding: 3px 6px; width: 100%;
}
.ingest-body textarea { min-height: 60px; margin-bottom: 4px; }
.ingest-body .btn {
  background: #0f3460; border: none; color: #c7c7c7;
  padding: 4px 12px; cursor: pointer; font-family: monospace; font-size: 11px;
}
.ingest-body .btn:hover { background: #1a5276; }

.footer { text-align: right; color: #484848; font-size: 10px; margin-top: 4px; }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1>⚡ Cognitive Core</h1>
  <span class="sub" id="subtitle">router:8081 · rag:8082 · kb:<span id="kbCount">?</span></span>
</div>

<!-- Stats -->
<div class="stats" id="statsRow">
  <div class="stat"><div class="num" id="sTraces">—</div><div class="lbl">Traces</div></div>
  <div class="stat"><div class="num" id="sVram">—</div><div class="lbl">VRAM</div></div>
  <div class="stat"><div class="num" id="sKb">—</div><div class="lbl">Chunks</div></div>
</div>

<!-- 4-panel grid -->
<div class="main-grid">
  <div class="log-panel">
    <div class="panel-title">💬 Router Chat — prompts &amp; responses</div>
    <div class="panel-body" id="chatLog">Loading...</div>
  </div>
  <div class="log-panel">
    <div class="panel-title">📜 RAG Server Log — Granite 4.1-8B :8082</div>
    <div class="panel-body" id="ragLog">Loading...</div>
  </div>
  <div class="log-panel">
    <div class="panel-title">🔧 Tool Calls — executed via MCP / builtin</div>
    <div class="panel-body" id="toolLog">Loading...</div>
  </div>
  <div class="log-panel">
    <div class="panel-title">📚 Chroma Recalls — RAG queries &amp; results</div>
    <div class="panel-body" id="ragStructuredLog">Loading...</div>
  </div>
</div>

<!-- Live Chat -->
<div class="chat-panel">
  <div class="panel-title">💬 Live Chat — watch the model think in real-time</div>
  <div class="chat-input-row">
    <input id="chatInput" type="text" placeholder="type a prompt here..." onkeydown="if(event.key==='Enter')sendChat()">
    <button class="btn" onclick="sendChat()">Send</button>
    <button class="btn" onclick="document.getElementById('chatOutput').innerHTML=''" style="color:#6a6a8a;">Clear</button>
  </div>
  <div class="chat-output" id="chatOutput">
    <div class="chat-status">Type a prompt above to watch the model reason and call tools live.</div>
  </div>
</div>

<!-- Traces -->
<div class="traces-panel">
  <div class="panel-title">📊 Traces · last <span id="traceCount">0</span></div>
  <div class="panel-body" id="tracesFeed"><div style="color:#484848;padding:6px;font-size:11px;">Waiting for traces...</div></div>
</div>

<!-- Ingestion -->
<div class="ingest-panel">
  <div class="panel-title" onclick="toggleIngest()">📥 Ingest &#9660;</div>
  <div class="ingest-body" id="ingestBody">
    <textarea id="pasteText" placeholder="paste text or code..."></textarea>
    <div style="display:flex;gap:4px;">
      <input id="pasteSource" placeholder="source name" style="flex:1;">
      <button class="btn" onclick="ingest()">Ingest</button>
    </div>
    <div id="ingestResult" style="margin-top:4px;font-size:11px;"></div>
  </div>
</div>

<div class="footer" id="lastUpdated"></div>

<script>
function toggleIngest() {
  document.getElementById('ingestBody').classList.toggle('open');
}

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderTraces(traces) {
  if (!traces || traces.length === 0)
    return '<div style="color:#484848;padding:6px;font-size:11px;">No traces yet.</div>';
  return traces.map(t => {
    const ts = (t.timestamp || '').replace('T',' ').slice(0,19) || '?';
    const dec = t.decision || 'unknown';
    const user = esc(t.user || t.prompt || '').slice(0, 80);
    const lat = t.latency_ms != null ? t.latency_ms + 'ms' : '';
    return `<div class="trace-row">
      <span class="trace-time">${ts}</span>
      <span class="trace-msg">${user}</span>
      <span class="trace-dec dec-${dec}">${dec}</span>
      <span class="trace-lat">${lat}</span>
    </div>`;
  }).join('');
}

function update() {
  fetch('/api/data').then(r => r.json()).then(d => {
    document.getElementById('chatLog').textContent = d.chat_log || '(empty)';
    document.getElementById('ragLog').textContent = d.rag_log || '(empty)';
    document.getElementById('toolLog').textContent = d.tool_log || '(empty)';
    document.getElementById('ragStructuredLog').textContent = d.rag_structured || '(empty)';
    document.getElementById('traceCount').textContent = d.trace_count;
    document.getElementById('tracesFeed').innerHTML = renderTraces(d.traces);
    document.getElementById('kbCount').textContent = d.chroma_count >= 0 ? d.chroma_count : '?';
    document.getElementById('sTraces').textContent = d.trace_count;
    document.getElementById('sKb').textContent = d.chroma_count >= 0 ? d.chroma_count : '-';
  });
  fetch('/api/vram').then(r => r.json()).then(d => {
    document.getElementById('sVram').textContent = d.vram || 'N/A';
  });
  document.getElementById('lastUpdated').textContent = 'updated: ' + new Date().toLocaleTimeString();
}

// ── Live Chat ──
function sendChat() {
  const input = document.getElementById('chatInput');
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = '';
  const out = document.getElementById('chatOutput');
  out.innerHTML += '<div class="chat-status" style="color:#4af;">\u25b6 <b>' + esc(prompt) + '</b></div>';

  fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt})
  }).then(async response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let eventType = '';
    let reasoningDiv = null;
    let contentDiv = null;

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});

      // SSE messages separated by blank lines
      const msgs = buf.split('\n\n');
      buf = msgs.pop() || '';

      for (const msg of msgs) {
        const lines = msg.split('\n');
        let data = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) eventType = line.slice(7).trim();
          else if (line.startsWith('data: ')) data = line.slice(6);
        }
        if (!data) continue;
        try { data = JSON.parse(data); } catch(e) {}

        if (eventType === 'reasoning') {
          if (!reasoningDiv) {
            reasoningDiv = document.createElement('div');
            reasoningDiv.className = 'chat-thinking';
            reasoningDiv.textContent = data;
            out.appendChild(reasoningDiv);
          } else {
            reasoningDiv.textContent += data;
          }
        } else if (eventType === 'content') {
          if (!contentDiv) {
            contentDiv = document.createElement('div');
            contentDiv.className = 'chat-content';
            out.appendChild(contentDiv);
          }
          contentDiv.textContent += data;
        } else if (eventType === 'error') {
          out.innerHTML += '<div class="chat-error">\u26a0 ' + esc(data) + '</div>';
        } else if (eventType === 'done') {
          out.innerHTML += '<div class="chat-status" style="color:#484848;">\u2713 done</div>';
        }
        out.scrollTop = out.scrollHeight;
      }
    }
  }).catch(err => {
    out.innerHTML += '<div class="chat-error">\u26a0 ' + esc(err.message) + '</div>';
  });
}
setInterval(update, 3000);
update();
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cognitive Core Runtime Dashboard")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    # Install requests in a thread-safe way for the chat proxy
    try:
        import requests as _req
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
        import requests as _req

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Cognitive Core Dashboard → http://{args.host}:{args.port}")
    print(f"  Live chat: type a prompt and watch the model think + call tools")
    print(f"  Router log: {ROUTER_LOG}")
    print(f"  RAG log:    {RAG_LOG}")
    print(f"  Traces:     {TRACES_PATH}")
    server.serve_forever()
