#!/usr/bin/env python3
"""Cognitive Core — Runtime Observability Dashboard

Shows the router's decision chain in real-time: what the router decided,
whether it delegated to RAG, what was retrieved, what tools were called,
and memory activity.

Usage:
    python scripts/runtime_dashboard.py [--port 8766]

The cognitive core should log decisions as JSONL to:
    /var/log/cognitive-core/traces.jsonl

Each trace line:
{
    "timestamp": "2026-07-09T14:32:01Z",
    "user": "What does OPD mean?",
    "decision": "needs_knowledge",       // answer_directly | tool_call | needs_knowledge | needs_rag | memory_recall
    "router_model": "MiniCPM5-1B",
    "router_response": "...",
    "rag": {
        "collection": "papers",
        "query": "OPD on-policy distillation",
        "retrieved_chunks": ["chunk_1", "chunk_2"],
        "model": "Llama-3.1-8B"
    },
    "tool": {
        "name": "web_search",
        "parameters": {"query": "..."},
        "result": "..."
    },
    "memory": {
        "recalled": ["previous context"],
        "stored": true
    },
    "latency_ms": 2430
}
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

TRACES_PATH = "/var/log/cognitive-core/traces.jsonl"
MAX_TRACES = 500


def read_traces(path: str, last_n: int = 200) -> list[dict]:
    """Read the last N trace entries from the JSONL log file."""
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
    """Tail the last N lines of a file."""
    if not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except (IOError, PermissionError):
        return ""


def _handler_class(traces_path: str):
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
            else:
                self.send_html()

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
    """Compute aggregate stats from traces."""
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

        if t.get("rag"):
            rag_count += 1
        if t.get("tool"):
            tool_count += 1
        if t.get("memory"):
            memory_count += 1

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
  .stat-card .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }

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
  .trace-body .label { color: #484f58; }
  .trace-body .rag-block, .trace-body .tool-block, .trace-body .memory-block {
    margin-top: .3rem; padding: .4rem .6rem; background: #0d1117;
    border-radius: 4px; font-family: monospace; font-size: .75rem;
    overflow-x: auto; white-space: pre-wrap; word-break: break-word;
  }
  .trace-body .rag-block { border-left: 2px solid #d29922; }
  .trace-body .tool-block { border-left: 2px solid #79c0ff; }
  .trace-body .memory-block { border-left: 2px solid #a379ff; }

  .latency { font-size: .75rem; color: #484f58; font-family: monospace; }

  .empty-state {
    text-align: center; padding: 3rem 1rem; color: #484f58;
  }
  .empty-state code { background: #161b22; padding: 2px 6px; border-radius: 4px; }

  .last-updated { font-size: .7rem; color: #484f58; text-align: right; margin-top: .5rem; }

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
<p class="subtitle">Runtime observability — every routing decision, RAG query, tool call, and memory access</p>

<!-- Stats -->
<div class="stats-bar" id="statsBar">
  <div class="stat-card"><div class="number" id="statTotal">—</div><div class="label">Total Requests</div></div>
  <div class="stat-card"><div class="number" id="statAnswer">—</div><div class="label">Answered Directly</div></div>
  <div class="stat-card"><div class="number" id="statTool">—</div><div class="label">Tool Calls</div></div>
  <div class="stat-card"><div class="number" id="statRag">—</div><div class="label">RAG Queries</div></div>
  <div class="stat-card"><div class="number" id="statMemory">—</div><div class="label">Memory Accesses</div></div>
  <div class="stat-card"><div class="number" id="statLatency">—</div><div class="label">Avg Latency (ms)</div></div>
</div>

<!-- Trace feed -->
<div id="traceFeed" class="trace-feed">
  <div class="empty-state" id="emptyState">
    <p style="margin-bottom:.5rem;">Waiting for traces...</p>
    <p style="font-size:.75rem;">The cognitive core logs decisions to <code>/var/log/cognitive-core/traces.jsonl</code></p>
    <p style="font-size:.75rem;">Once traces appear, this dashboard updates every 2 seconds.</p>
  </div>
</div>

<div class="last-updated" id="lastUpdated">Last updated: —</div>

<script>
function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderTrace(t) {
  const ts = t.timestamp ? t.timestamp.replace('T', ' ').slice(0,19) : '?';
  const decision = t.decision || 'unknown';
  const user = esc(t.user || '(no prompt)');
  const lat = t.latency_ms != null ? t.latency_ms + 'ms' : '';

  // Build body sections
  let bodyHtml = '';

  // Router response snippet
  if (t.router_response) {
    const snippet = t.router_response.length > 200
      ? esc(t.router_response.slice(0,200)) + '...'
      : esc(t.router_response);
    bodyHtml += `<div style="margin-top:.3rem;color:#c9d1d9;">${snippet}</div>`;
  }

  // RAG block
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

  // Tool block
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

  // Memory block
  if (t.memory) {
    const mem = t.memory;
    let memHtml = `<span class="label">Memory</span>`;
    if (mem.recalled && mem.recalled.length > 0)
      memHtml += `\n  Recalled: ${mem.recalled.length} items`;
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
  // Traces
  fetch('/api/traces')
    .then(r => r.json())
    .then(d => {
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

  // Stats
  fetch('/api/stats')
    .then(r => r.json())
    .then(d => {
      document.getElementById('statTotal').textContent = d.total || 0;
      document.getElementById('statAnswer').textContent = (d.by_decision && d.by_decision.answer_directly) || 0;
      document.getElementById('statTool').textContent = d.tool_count || 0;
      document.getElementById('statRag').textContent = d.rag_count || 0;
      document.getElementById('statMemory').textContent = d.memory_count || 0;
      document.getElementById('statLatency').textContent = d.avg_latency_ms || 0;
    });

  document.getElementById('lastUpdated').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
}

// Poll every 2 seconds
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
    ap.add_argument("--traces", default=TRACES_PATH,
                    help="Path to traces JSONL file")
    args = ap.parse_args()

    Handler = _handler_class(args.traces)
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Cognitive Core Runtime Dashboard → http://{args.host}:{args.port}")
    print(f"Watching: {args.traces}")
    server.serve_forever()
