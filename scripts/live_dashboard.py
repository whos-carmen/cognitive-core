#!/usr/bin/env python3
"""Cognitive Core — Live Training Dashboard Server
Serves a real-time-updating dashboard for SFT/DPO training runs.

Usage:
    python scripts/dashboard.py [--port 8765]
"""
import argparse, json, os, datetime, subprocess, glob
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJ, "models", "logs")
SFT_LOG = os.path.join(LOG_DIR, "sft.log")
SFT_METRICS = os.path.join(LOG_DIR, "sft_metrics.jsonl")
DPO_METRICS = os.path.join(LOG_DIR, "dpo_metrics.jsonl")
TRAIN_DIR = os.path.join(PROJ, "train")

def parse_metrics(path, last_n=50):
    """Parse a metrics.jsonl file, return last N entries."""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except:
                    pass
    return entries[-last_n:]

def get_gpu_info():
    """Get GPU utilization from rocm-smi (concise table format)."""
    try:
        out = subprocess.run(
            ["rocm-smi"],
            capture_output=True, text=True, timeout=5
        ).stdout
        lines = out.strip().split("\n")
        devices = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 9 and parts[0].isdigit():
                devices.append({
                    "id": parts[0],
                    "name": f"Device {parts[0]}",
                    "temp": parts[4],
                    "power": parts[5],
                    "vram_used": parts[14].replace("%",""),
                    "gpu_usage": parts[15].replace("%",""),
                })
        if not devices:
            # fallback: try rocminfo
            return [{"name": "RX 7900 XTX", "temp": "?", "vram_used": "?", "vram_total": "?", "gpu_usage": "?", "power": "?"}]
        return devices
    except:
        return [{"name": "RX 7900 XTX", "temp": "?", "vram_used": "?", "vram_total": "?", "gpu_usage": "?", "power": "?"}]

def find_checkpoints():
    """Find training checkpoints."""
    outputs_dir = os.path.join(TRAIN_DIR, "outputs")
    if not os.path.isdir(outputs_dir):
        return []
    checkpoints = []
    for d in os.listdir(outputs_dir):
        root = os.path.join(outputs_dir, d)
        if not os.path.isdir(root):
            continue
        ckpts = glob.glob(os.path.join(root, "checkpoint-*"))
        for ckpt in ckpts:
            step = os.path.basename(ckpt).replace("checkpoint-", "")
            checkpoints.append({"dir": d, "step": step, "path": ckpt})
    return checkpoints

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.send_json(self._get_status())
        elif path == "/api/metrics":
            self.send_json({
                "sft": parse_metrics(SFT_METRICS),
                "dpo": parse_metrics(DPO_METRICS),
            })
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

    def _get_status(self):
        gpu = get_gpu_info()
        ckpts = find_checkpoints()

        # SFT log tail
        sft_tail = ""
        if os.path.exists(SFT_LOG):
            with open(SFT_LOG) as f:
                lines = f.readlines()
                sft_tail = "".join(lines[-20:])

        # Check if training is running
        running = False
        try:
            out = subprocess.run(["pgrep", "-f", "sft.py|dpo.py"], capture_output=True, text=True, timeout=3)
            running = bool(out.stdout.strip())
        except:
            pass

        return {
            "gpu": gpu,
            "running": running,
            "checkpoints": ckpts,
            "sft_tail": sft_tail,
            "now": datetime.datetime.now().isoformat(),
        }

    def log_message(self, format, *args):
        pass  # quiet


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cognitive Core — Live Training Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background: #0f1117; color: #e1e4eb; line-height: 1.5;
    padding: 1.5rem; max-width: 960px; margin: 0 auto;
  }
  h1 { font-size: 1.3rem; margin-bottom: .25rem; display: flex; align-items: center; gap: .75rem; }
  .subtitle { color: #8b929a; font-size: .85rem; margin-bottom: 1rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; margin-bottom: .75rem; }
  .card {
    background: #1a1d27; border: 1px solid #2a2e3a; border-radius: 8px;
    padding: 1rem;
  }
  .card.full { grid-column: 1 / -1; }
  .card h2 { font-size: .9rem; margin-bottom: .5rem; color: #c9d1d9; }
  .stat-row { display: flex; justify-content: space-between; padding: .2rem 0; font-size: .85rem; }
  .stat-row .val { font-family: monospace; color: #79c0ff; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: .75rem; font-weight: 600;
  }
  .badge.running { background: #1a4731; color: #7ee787; }
  .badge.stopped { background: #3b2e00; color: #d29922; }
  .badge.none { background: #2a2e3a; color: #8b929a; }
  pre.log {
    background: #0d1117; border: 1px solid #2a2e3a; border-radius: 6px;
    padding: .75rem; font-size: .75rem; max-height: 300px; overflow-y: auto;
    color: #c9d1d9; line-height: 1.4;
  }
  .metric-list { max-height: 250px; overflow-y: auto; font-size: .8rem; }
  .metric-row { display: flex; justify-content: space-between; padding: .15rem 0; border-bottom: 1px solid #1a1d27; }
  .metric-row .step { color: #8b929a; }
  .metric-row .loss { color: #ffa657; font-family: monospace; }
  .status-indicator { display: flex; align-items: center; gap: .5rem; }
  .dot { width: 10px; height: 10px; border-radius: 50%; }
  .dot.green { background: #3fb950; animation: pulse 2s infinite; }
  .dot.yellow { background: #d29922; }
  .dot.red { background: #f85149; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .4; } }
  .query-btn {
    background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
    padding: .25rem .75rem; border-radius: 6px; cursor: pointer; font-size: .8rem;
  }
  .query-btn:hover { background: #30363d; }
  .last-updated { font-size: .75rem; color: #484f58; text-align: right; margin-top: .5rem; }
  .gpu-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .25rem; font-size: .8rem; }
  @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<h1>
  <span>🧠 Cognitive Core</span>
  <span id="statusBadge" class="badge none">Loading...</span>
</h1>
<p class="subtitle">Live Training Dashboard — auto-refreshes every 5s</p>

<div class="grid">
  <div class="card">
    <h2>🎮 GPU</h2>
    <div id="gpuInfo"><div class="stat-row">Loading...</div></div>
  </div>
  <div class="card">
    <h2>📊 Training</h2>
    <div id="trainInfo">
      <div class="stat-row"><span>Status</span><span id="trainStatus" class="badge none">—</span></div>
      <div class="stat-row"><span>Checkpoints</span><span id="ckptCount">0</span></div>
    </div>
  </div>
</div>

<div class="card full">
  <h2>📈 Loss</h2>
  <div id="lossContainer" style="height: 200px; display: flex; align-items: center; justify-content: center; color: #484f58;">
    Waiting for first metrics...
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>📋 Recent Metrics</h2>
    <div id="metricsList" class="metric-list">No data yet</div>
  </div>
  <div class="card">
    <h2>📝 Log (tail)</h2>
    <pre id="logTail" class="log">Waiting for training to start...</pre>
  </div>
</div>

<div class="last-updated" id="lastUpdated">Last updated: —</div>

<script>
function fmt(v) { return v == null || v === '?' ? '—' : v; }

function drawChart(canvas, metrics) {
  if (metrics.length < 2) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * (devicePixelRatio || 1);
  const H = canvas.height = canvas.clientHeight * (devicePixelRatio || 1);
  ctx.scale(devicePixelRatio || 1, devicePixelRatio || 1);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const pad = { top: 20, right: 20, bottom: 25, left: 45 };

  const steps = metrics.map(m => m.step);
  const losses = metrics.map(m => m.loss);
  const maxStep = Math.max(...steps);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const range = maxLoss - minLoss || 1;
  const margin = range * 0.15;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#e1e4eb';
  ctx.font = `${11}px monospace`;
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (h - pad.top - pad.bottom) * (1 - i/4);
    const val = (maxLoss + margin) - (range + 2*margin) * i/4;
    ctx.fillText(val.toFixed(2), pad.left - 5, y + 4);
    ctx.strokeStyle = '#2a2e3a';
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
  }

  ctx.strokeStyle = '#58a6ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < steps.length; i++) {
    const x = pad.left + (w - pad.left - pad.right) * (steps[i] / maxStep);
    const y = pad.top + (h - pad.top - pad.bottom) * (1 - (losses[i] - (minLoss - margin)) / (range + 2*margin));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // dots
  ctx.fillStyle = '#58a6ff';
  for (let i = 0; i < steps.length; i++) {
    const x = pad.left + (w - pad.left - pad.right) * (steps[i] / maxStep);
    const y = pad.top + (h - pad.top - pad.bottom) * (1 - (losses[i] - (minLoss - margin)) / (range + 2*margin));
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI*2); ctx.fill();
  }
}

function update() {
  fetch('/api/status')
    .then(r => r.json())
    .then(d => {
      const gpu = d.gpu[0] || {};
      document.getElementById('gpuInfo').innerHTML = `
        <div class="gpu-grid">
          <span>GPU</span><span class="val">${fmt(gpu.name)}</span>
          <span>Temp</span><span class="val">${fmt(gpu.temp)}</span>
          <span>VRAM%</span><span class="val">${fmt(gpu.vram_used)}%</span>
          <span>Usage</span><span class="val">${fmt(gpu.gpu_usage)}%</span>
          <span>Power</span><span class="val">${fmt(gpu.power)}</span>
        </div>`;

      const badge = document.getElementById('statusBadge');
      const ts = document.getElementById('trainStatus');
      if (d.running) {
        badge.className = 'badge running'; badge.textContent = '● Running';
        ts.className = 'badge running'; ts.textContent = '● Running';
      } else if (d.checkpoints.length > 0) {
        badge.className = 'badge stopped'; badge.textContent = '● Stopped';
        ts.className = 'badge stopped'; ts.textContent = '● Stopped (checkpoints exist)';
      } else {
        badge.className = 'badge none'; badge.textContent = '● Idle';
        ts.className = 'badge none'; ts.textContent = '● Idle';
      }
      document.getElementById('ckptCount').textContent = d.checkpoints.length;

      if (d.sft_tail) {
        document.getElementById('logTail').textContent = d.sft_tail;
      }
    });

  fetch('/api/metrics')
    .then(r => r.json())
    .then(d => {
      const sft = d.sft || [];
      const list = document.getElementById('metricsList');
      if (sft.length > 0) {
        list.innerHTML = sft.slice().reverse().slice(0, 30).map(m => {
          const loss = m.loss != null ? `<span class="loss">${Number(m.loss).toFixed(4)}</span>` : '';
          const acc = m.acc != null ? ` acc=${m.acc}` : '';
          return `<div class="metric-row"><span class="step">step ${m.step}</span>${loss}${acc ? `<span>${acc}</span>` : ''}</div>`;
        }).join('');

        // Draw chart
        const container = document.getElementById('lossContainer');
        container.innerHTML = '';
        const canvas = document.createElement('canvas');
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        container.appendChild(canvas);
        drawChart(canvas, sft);
      }

      /* DPO metrics if present */
      const dpo = d.dpo || [];
      if (dpo.length > 0) {
        // Append DPO info
      }
    });

  document.getElementById('lastUpdated').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
}

// Poll every 5 seconds
setInterval(update, 5000);
update();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    server = HTTPServer((args.host, args.port), DashboardHandler)
    print(f"Cognitive Core Dashboard → http://{args.host}:{args.port}")
    print(f"Open in browser to monitor SFT/DPO training in real-time.")
    server.serve_forever()
