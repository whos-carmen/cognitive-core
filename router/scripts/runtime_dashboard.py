#!/usr/bin/env python3
"""Cognitive Core — Runtime Observability Dashboard + Data Ingestion

Shows the router's decision chain in real-time, and allows uploading files
to the Chroma knowledge base via the web UI.

Usage:
    python scripts/runtime_dashboard.py --port 8766
"""

import argparse
import json
import os
import re
import time
import io
import cgi
import uuid
import traceback
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

TRACES_PATH = "/var/log/cognitive-core/traces.jsonl"
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
MAX_TRACES = 500

# Lazy imports for ingestion (only loaded when a file is uploaded)
_embed_model = None
_chroma_collection = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("ibm-granite/granite-embedding-english-r2")
    return _embed_model

def get_chroma_collection(path):
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        db = chromadb.PersistentClient(path=path)
        _chroma_collection = db.get_or_create_collection("knowledge")
    return _chroma_collection


def read_traces(path: str, last_n: int = 200) -> list[dict]:
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


def tail_file(path: str, n: int = 20) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except (IOError, PermissionError):
        return ""


def ingest_text(text: str, source: str, chroma_path: str, metadata: dict = None) -> dict:
    """Chunk, embed with Granite, and store in Chroma."""
    model = get_embed_model()
    collection = get_chroma_collection(chroma_path)

    # Simple chunking: 512-char chunks with 256-char overlap
    chunk_size = 512
    overlap = 256
    chunks = []
    ids = []

    for i in range(0, max(len(text), 1), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) < 20:
            continue
        chunk_id = f"{source}-{uuid.uuid4().hex[:8]}-{i}"
        chunks.append(chunk)
        ids.append(chunk_id)

    if not chunks:
        return {"status": "error", "message": "Text too short after chunking"}

    # Embed in batches
    batch_size = 64
    total_chunks = len(chunks)
    metadatas = [{"source": source, **(metadata or {})} for _ in chunks]

    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_texts = chunks[batch_start:batch_end]
        batch_ids = ids[batch_start:batch_end]
        batch_meta = metadatas[batch_start:batch_end]

        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )

    return {
        "status": "ok",
        "chunks": total_chunks,
        "source": source,
        "total_chars": len(text),
    }


def _handler_class(traces_path: str, chroma_path: str):
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import urlparse

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/traces":
                traces = read_traces(traces_path)
                self.send_json({
                    "traces": traces,
                    "count": len(traces),
                    "now": datetime.now().isoformat(),
                })
            elif path == "/api/stats":
                traces = read_traces(traces_path)
                stats = compute_stats(traces)
                self.send_json(stats)
            elif path == "/api/ingest/stats":
                collection = get_chroma_collection(chroma_path)
                count = collection.count()
                self.send_json({"count": count})
            else:
                self.send_html()

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/ingest":
                self._handle_upload()

        def _handle_upload(self):
            content_type = self.headers.get("Content-Type", "")
            ct = content_type.split(";")[0]

            try:
                if "multipart/form-data" in content_type:
                    form = cgi.FieldStorage(
                        fp=self.rfile,
                        headers=self.headers,
                        environ={
                            "REQUEST_METHOD": "POST",
                            "CONTENT_TYPE": content_type,
                        }
                    )
                    file_item = form.getfirst("file")
                    source_name = form.getfirst("source") or file_item.filename or "upload"

                    if file_item and hasattr(file_item, "file"):
                        text = file_item.file.read().decode("utf-8", errors="replace")
                    else:
                        self.send_json({"status": "error", "message": "No file uploaded"})
                        return

                elif "application/json" in content_type:
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length).decode("utf-8")
                    data = json.loads(body)
                    text = data.get("text", "")
                    source_name = data.get("source", "paste")

                else:
                    self.send_json({"status": "error", "message": f"Unsupported content type: {ct}"})
                    return

                if len(text.strip()) < 10:
                    self.send_json({"status": "error", "message": "Text too short"})
                    return

                result = ingest_text(text, source_name, chroma_path)
                self.send_json(result)

            except Exception as e:
                self.send_json({"status": "error", "message": str(e)})

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

        def log_message(self, fmt, *args):
            pass

    return Handler


def compute_stats(traces: list[dict]) -> dict:
    total = len(traces)
    if total == 0:
        return {"total": 0, "by_decision": {}, "avg_latency_ms": 0, "rag_count": 0, "tool_count": 0, "memory_count": 0}

    decisions = {}
    rag_count = 0
    tool_count = 0
    memory_count = 0
    latencies = []

    for t in traces:
        dec = t.get("decision", "unknown")
        decisions[dec] = decisions.get(dec, 0) + 1
        if t.get("rag"): rag_count += 1
        if t.get("tool"): tool_count += 1
        if t.get("memory"): memory_count += 1
        lat = t.get("latency_ms")
        if lat is not None:
            latencies.append(lat)

    return {
        "total": total,
        "by_decision": decisions,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "rag_count": rag_count,
        "tool_count": tool_count,
        "memory_count": memory_count,
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cognitive Core — Runtime Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background: #0d1117; color: #e1e4eb; line-height: 1.5;
    padding: 1rem; max-width: 1200px; margin: 0 auto;
  }
  h1 { font-size: 1.2rem; margin-bottom: .25rem; display: flex; align-items: center; gap: .5rem; }
  .subtitle { color: #8b949e; font-size: .8rem; margin-bottom: 1rem; }

  /* Tabs */
  .tabs { display: flex; gap: .25rem; margin-bottom: 1rem; }
  .tab {
    padding: .4rem 1rem; border-radius: 6px 6px 0 0;
    cursor: pointer; font-size: .85rem; border: 1px solid #21262d; border-bottom: none;
    background: #161b22; color: #8b949e;
  }
  .tab.active { background: #0d1117; color: #c9d1d9; border-bottom: 1px solid #0d1117; margin-bottom: -1px; }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }

  /* Stats bar */
  .stats-bar {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: .5rem; margin-bottom: 1rem;
  }
  .stat-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: .6rem 1rem; text-align: center;
  }
  .stat-card .number { font-size: 1.3rem; font-weight: 600; color: #58a6ff; font-family: monospace; }
  .stat-card .label { font-size: .7rem; color: #8b949e; text-transform: uppercase; letter-spacing: .3px; }

  /* Trace feed */
  .trace-feed { display: flex; flex-direction: column; gap: .5rem; }
  .trace {
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: .75rem 1rem; font-size: .85rem;
  }
  .trace-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: .4rem; gap: .5rem;
  }
  .trace-time { color: #484f58; font-family: monospace; font-size: .75rem; white-space: nowrap; }
  .trace-user { color: #c9d1d9; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .trace-decision {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: .7rem; font-weight: 600; white-space: nowrap;
  }
  .decision-answer_directly { background: #1a4731; color: #7ee787; }
  .decision-tool_call { background: #1a3a5c; color: #79c0ff; }
  .decision-needs_knowledge { background: #3b2e00; color: #d29922; }
  .decision-needs_rag { background: #3b2e00; color: #d29922; }
  .decision-memory_recall { background: #2a1a5c; color: #a379ff; }
  .decision-unknown { background: #21262d; color: #8b949e; }

  .trace-body { color: #8b949e; font-size: .8rem; line-height: 1.4; }
  .trace-body .rag-block, .trace-body .tool-block, .trace-body .memory-block {
    margin-top: .3rem; padding: .4rem .6rem; background: #0d1117;
    border-radius: 4px; font-family: monospace; font-size: .75rem;
    overflow-x: auto; white-space: pre-wrap; word-break: break-word;
  }
  .trace-body .rag-block { border-left: 2px solid #d29922; }
  .trace-body .tool-block { border-left: 2px solid #79c0ff; }
  .trace-body .memory-block { border-left: 2px solid #a379ff; }

  .latency { font-size: .75rem; color: #484f58; font-family: monospace; }

  /* Ingestion */
  .ingest-area {
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 1.5rem; margin-bottom: 1rem;
  }
  .ingest-area h2 { font-size: 1rem; margin-bottom: .5rem; }
  .ingest-area p { color: #8b949e; font-size: .85rem; margin-bottom: 1rem; }
  .upload-zone {
    border: 2px dashed #21262d; border-radius: 8px;
    padding: 2rem; text-align: center; cursor: pointer;
    margin-bottom: 1rem;
  }
  .upload-zone:hover { border-color: #58a6ff; background: #1a1d27; }
  .upload-zone.dragover { border-color: #3fb950; background: #1a4731; }
  .upload-zone .icon { font-size: 2rem; margin-bottom: .5rem; }
  .upload-zone .hint { color: #484f58; font-size: .8rem; margin-top: .3rem; }

  .paste-area { margin-bottom: 1rem; }
  .paste-area textarea {
    width: 100%; min-height: 120px;
    background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
    color: #c9d1d9; padding: .75rem; font-family: monospace; font-size: .8rem;
    resize: vertical;
  }
  .paste-area textarea:focus { outline: none; border-color: #58a6ff; }

  .ingest-actions { display: flex; gap: .5rem; align-items: center; }
  .btn {
    padding: .4rem 1rem; border-radius: 6px; border: none;
    font-size: .85rem; cursor: pointer; font-weight: 500;
  }
  .btn-primary { background: #238636; color: #fff; }
  .btn-primary:hover { background: #2ea043; }
  .btn-primary:disabled { opacity: .5; cursor: not-allowed; }

  .ingest-result {
    margin-top: .75rem; padding: .5rem .75rem; border-radius: 6px;
    font-size: .85rem; display: none;
  }
  .ingest-result.ok { display: block; background: #1a4731; color: #7ee787; }
  .ingest-result.err { display: block; background: #3b1a1a; color: #f85149; }

  .last-updated { font-size: .7rem; color: #484f58; text-align: right; margin-top: .5rem; }

  .empty-state { text-align: center; padding: 3rem 1rem; color: #484f58; }
  .empty-state code { background: #161b22; padding: 2px 6px; border-radius: 4px; }

  @media (max-width: 640px) {
    .trace-header { flex-wrap: wrap; }
    .trace-user { white-space: normal; }
  }
</style>
</head>
<body>

<h1>
  <span>🔍 Cognitive Core</span>
  <span id="traceCount" style="font-size:.8rem;color:#8b949e;font-family:monospace;">—</span>
</h1>
<p class="subtitle">Runtime observability + RAG data ingestion</p>

<!-- Tab bar -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('traces')">📊 Traces</div>
  <div class="tab" onclick="switchTab('ingest')">📥 Ingest</div>
</div>

<!-- Traces tab -->
<div id="tab-traces" class="tab-pane active">
  <!-- Stats -->
  <div class="stats-bar">
    <div class="stat-card"><div class="number" id="statTotal">—</div><div class="label">Total Requests</div></div>
    <div class="stat-card"><div class="number" id="statAnswer">—</div><div class="label">Answered Directly</div></div>
    <div class="stat-card"><div class="number" id="statTool">—</div><div class="label">Tool Calls</div></div>
    <div class="stat-card"><div class="number" id="statRag">—</div><div class="label">RAG Queries</div></div>
    <div class="stat-card"><div class="number" id="statMemory">—</div><div class="label">Memory Accesses</div></div>
    <div class="stat-card"><div class="number" id="statLatency">—</div><div class="label">Avg Latency (ms)</div></div>
  </div>

  <div id="traceFeed" class="trace-feed">
    <div class="empty-state" id="emptyState">
      <p style="margin-bottom:.5rem;">Waiting for traces...</p>
      <p style="font-size:.75rem;">The cognitive core logs decisions to <code>/var/log/cognitive-core/traces.jsonl</code></p>
    </div>
  </div>
</div>

<!-- Ingest tab -->
<div id="tab-ingest" class="tab-pane">
  <div class="ingest-area">
    <h2>📥 Upload File</h2>
    <p>Upload code samples, documentation, man pages, past projects — anything you want the router to know about.</p>

    <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
      <div class="icon">📄</div>
      <div>Click or drop a file here</div>
      <div class="hint">.txt, .md, .py, .js, .sh, .jsonl, .pdf</div>
    </div>
    <input type="file" id="fileInput" style="display:none" accept=".txt,.md,.py,.js,.ts,.go,.rs,.sh,.jsonl,.csv,.json,.yaml" onchange="uploadFile(this.files[0])">

    <details style="margin-bottom:.75rem;">
      <summary style="cursor:pointer;color:#8b949e;font-size:.85rem;">Or paste text directly</summary>
      <div class="paste-area">
        <textarea id="pasteText" placeholder="Paste code, documentation, or notes here..."></textarea>
        <div class="ingest-actions">
          <input id="pasteSource" type="text" placeholder="Source name (e.g. ffmpeg-man, project-alpha)" style="flex:1;background:#0d1117;border:1px solid #21262d;border-radius:6px;color:#c9d1d9;padding:.4rem .6rem;font-size:.85rem;">
          <button class="btn btn-primary" onclick="pasteIngest()">Ingest</button>
        </div>
      </div>
    </details>

    <div id="ingestResult" class="ingest-result"></div>

    <div style="margin-top:.5rem;font-size:.8rem;color:#484f58;">
      <span id="kbCount">Chroma KB: —</span>
    </div>
  </div>
</div>

<div class="last-updated" id="lastUpdated">Last updated: —</div>

<script>
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[onclick*="'${name}'"]`).classList.add('active');
  document.getElementById(`tab-${name}`).classList.add('active');
  if (name === 'ingest') updateKbCount();
}

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderTrace(t) {
  const ts = t.timestamp ? t.timestamp.replace('T', ' ').slice(0,19) : '?';
  const decision = t.decision || 'unknown';
  const user = esc(t.user || '(no prompt)');
  const lat = t.latency_ms != null ? t.latency_ms + 'ms' : '';
  let bodyHtml = '';

  if (t.router_response) {
    const snippet = t.router_response.length > 200
      ? esc(t.router_response.slice(0,200)) + '...'
      : esc(t.router_response);
    bodyHtml += `<div style="margin-top:.3rem;color:#c9d1d9;">${snippet}</div>`;
  }

  if (t.rag) {
    const rag = t.rag;
    let ragHtml = `<span class="label">RAG</span>`;
    if (rag.collection) ragHtml += `\n  Collection: ${esc(rag.collection)}`;
    if (rag.query) ragHtml += `\n  Query: ${esc(rag.query)}`;
    if (rag.retrieved_chunks) {
      ragHtml += `\n  Chunks: ${rag.retrieved_chunks.length} retrieved`;
      rag.retrieved_chunks.slice(0, 3).forEach(c =>
        ragHtml += `\n    · ${esc(String(c).slice(0, 100))}`
      );
    }
    if (rag.model) ragHtml += `\n  Model: ${esc(rag.model)}`;
    bodyHtml += `<div class="rag-block">${ragHtml}</div>`;
  }

  if (t.tool) {
    const tool = t.tool;
    let toolHtml = `<span class="label">Tool</span>`;
    toolHtml += `\n  Call: ${esc(tool.name)}(${JSON.stringify(tool.parameters || {})})`;
    if (tool.result) {
      const r = String(tool.result);
      toolHtml += `\n  Result: ${esc(r.length > 150 ? r.slice(0,150) + '...' : r)}`;
    }
    bodyHtml += `<div class="tool-block">${toolHtml}</div>`;
  }

  if (t.memory) {
    const mem = t.memory;
    let memHtml = `<span class="label">Memory</span>`;
    if (mem.recalled && mem.recalled.length > 0) memHtml += `\n  Recalled: ${mem.recalled.length} items`;
    if (mem.stored) memHtml += `\n  Stored: ✓`;
    bodyHtml += `<div class="memory-block">${memHtml}</div>`;
  }

  return `
    <div class="trace">
      <div class="trace-header">
        <span class="trace-time">${ts}</span>
        <span class="trace-user">${user}</span>
        <span class="trace-decision decision-${decision}">${decision}</span>
        ${lat ? `<span class="latency">${lat}</span>` : ''}
      </div>
      ${bodyHtml ? `<div class="trace-body">${bodyHtml}</div>` : ''}
    </div>
  `;
}

function update() {
  fetch('/api/traces').then(r => r.json()).then(d => {
    const feed = document.getElementById('traceFeed');
    const empty = document.getElementById('emptyState');
    if (d.traces.length === 0) {
      if (empty) empty.style.display = 'block';
      return;
    }
    if (empty) empty.style.display = 'none';
    document.getElementById('traceCount').textContent = d.count + ' traces';
    feed.innerHTML = d.traces.slice().reverse().map(renderTrace).join('');
  });

  fetch('/api/stats').then(r => r.json()).then(d => {
    document.getElementById('statTotal').textContent = d.total || 0;
    document.getElementById('statAnswer').textContent = (d.by_decision && d.by_decision.answer_directly) || 0;
    document.getElementById('statTool').textContent = d.tool_count || 0;
    document.getElementById('statRag').textContent = d.rag_count || 0;
    document.getElementById('statMemory').textContent = d.memory_count || 0;
    document.getElementById('statLatency').textContent = d.avg_latency_ms || 0;
  });

  document.getElementById('lastUpdated').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
}

function updateKbCount() {
  fetch('/api/ingest/stats').then(r => r.json()).then(d => {
    document.getElementById('kbCount').textContent = 'Chroma KB: ' + (d.count || 0) + ' chunks';
  }).catch(() => {});
}

function uploadFile(file) {
  if (!file) return;
  const formData = new FormData();
  formData.append('file', file);
  formData.append('source', file.name);
  doIngest(formData);
}

function pasteIngest() {
  const text = document.getElementById('pasteText').value;
  const source = document.getElementById('pasteSource').value || 'paste-' + Date.now();
  if (!text.trim()) return;
  const formData = new FormData();
  formData.append('file', new Blob([text], {type: 'text/plain'}), source);
  formData.append('source', source);
  doIngest(formData);
}

function doIngest(formData) {
  const btn = document.querySelector('.btn-primary');
  const result = document.getElementById('ingestResult');
  btn.disabled = true;
  result.className = 'ingest-result';
  result.textContent = 'Ingesting...';

  fetch('/api/ingest', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(d => {
      if (d.status === 'ok') {
        result.className = 'ingest-result ok';
        result.textContent = `✓ Ingested ${d.chunks} chunks from "${d.source}" (${d.total_chars} chars)`;
        updateKbCount();
      } else {
        result.className = 'ingest-result err';
        result.textContent = '✗ ' + (d.message || 'Unknown error');
      }
    })
    .catch(e => {
      result.className = 'ingest-result err';
      result.textContent = '✗ ' + e.message;
    })
    .finally(() => { btn.disabled = false; });
}

// Drag & drop
const zone = document.getElementById('uploadZone');
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
zone.addEventListener('drop', e => {
  e.preventDefault();
  zone.classList.remove('dragover');
  uploadFile(e.dataTransfer.files[0]);
});

setInterval(update, 2000);
update();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cognitive Core Runtime Dashboard")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--traces", default=TRACES_PATH, help="Path to traces JSONL file")
    ap.add_argument("--chroma-path", default=CHROMA_PATH, help="Path to Chroma DB directory")
    args = ap.parse_args()

    Handler = _handler_class(args.traces, args.chroma_path)
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Cognitive Core Dashboard → http://{args.host}:{args.port}")
    print(f"  Traces: {args.traces}")
    print(f"  Chroma: {args.chroma_path}")
    server.serve_forever()
