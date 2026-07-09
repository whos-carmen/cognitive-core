#!/usr/bin/env python3
"""Cognitive Core — Live Training Dashboard Server
Serves a real-time-updating dashboard for SFT/DPO training runs.

Usage:
    python scripts/dashboard.py [--port 8765]
"""
import argparse, json, os, datetime, subprocess, glob, time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJ, "models", "logs")
SFT_LOG = os.path.join(LOG_DIR, "sft.log")
SFT_METRICS = os.path.join(LOG_DIR, "sft_metrics.jsonl")
DPO_METRICS = os.path.join(LOG_DIR, "dpo_metrics.jsonl")
TRAIN_DIR = os.path.join(PROJ, "train")

# AWS g7e.2xlarge pricing (on-demand, us-east-1 — update for your region)
HOURLY_RATE = 2.81  # USD/hour on-demand

def parse_metrics(path, last_n=200):
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
    """Get GPU utilization from nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        ).stdout
        devices = []
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                devices.append({
                    "name": parts[0],
                    "temp": parts[1],
                    "gpu_usage": parts[2],
                    "vram_used_mb": parts[3],
                    "vram_total_mb": parts[4],
                    "vram_pct": round(float(parts[3]) / float(parts[4]) * 100, 1) if parts[4] != "0" else 0,
                    "power_w": parts[5],
                })
        if not devices:
            return [{"name": "GPU", "temp": "?", "gpu_usage": "?", "vram_used_mb": "?", "vram_total_mb": "?", "vram_pct": 0, "power_w": "?"}]
        return devices
    except:
        return [{"name": "GPU", "temp": "?", "gpu_usage": "?", "vram_used_mb": "?", "vram_total_mb": "?", "vram_pct": 0, "power_w": "?"}]

def get_system_info():
    """Get CPU/memory/disk usage."""
    info = {}
    try:
        # CPU load
        load = os.getloadavg()
        info["cpu_load"] = f"{load[0]:.1f} / {load[1]:.1f} / {load[2]:.1f}"
        info["cpu_pct"] = round(load[0] / os.cpu_count() * 100, 1)
    except:
        info["cpu_load"] = "?"
        info["cpu_pct"] = 0
    try:
        # Memory
        out = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=3).stdout
        lines = out.strip().split("\n")
        parts = lines[1].split()
        info["ram_used"] = int(parts[2])
        info["ram_total"] = int(parts[1])
        info["ram_pct"] = round(int(parts[2]) / int(parts[1]) * 100, 1)
    except:
        info["ram_used"] = 0
        info["ram_total"] = 0
        info["ram_pct"] = 0
    try:
        # Disk
        st = os.statvfs("/")
        info["disk_used_gb"] = round((st.f_blocks - st.f_bavail) * st.f_frsize / (1024**3), 1)
        info["disk_total_gb"] = round(st.f_blocks * st.f_frsize / (1024**3), 1)
        info["disk_pct"] = round(info["disk_used_gb"] / info["disk_total_gb"] * 100, 1) if info["disk_total_gb"] > 0 else 0
    except:
        info["disk_used_gb"] = 0
        info["disk_total_gb"] = 0
        info["disk_pct"] = 0
    return info

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

def get_training_info(metrics):
    """Compute training stats from metrics."""
    if not metrics:
        return {"steps": 0, "elapsed": "?", "elapsed_seconds": 0, "cost": 0}
    first_ts = metrics[0].get("timestamp") or metrics[0].get("time")
    last_ts = metrics[-1].get("timestamp") or metrics[-1].get("time")
    steps = metrics[-1].get("step", 0)
    elapsed_s = 0
    if first_ts and last_ts:
        try:
            if isinstance(first_ts, str):
                t1 = datetime.datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                t2 = datetime.datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                elapsed_s = (t2 - t1).total_seconds()
        except:
            pass
    hours = elapsed_s / 3600 if elapsed_s > 0 else 0
    cost = hours * HOURLY_RATE
    return {
        "steps": steps,
        "elapsed_seconds": int(elapsed_s),
        "elapsed": f"{int(elapsed_s//3600)}h {int((elapsed_s%3600)//60)}m" if elapsed_s > 0 else "?",
        "cost": round(cost, 2),
    }

class DashboardHandler:
    """Simple HTTP handler using http.server."""
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import urlparse

    def __init__(self, *args, **kwargs):
        from http.server import BaseHTTPRequestHandler
        super().__init__(*args, **kwargs)

def _handler_class():
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import urlparse

    class Handler(BaseHTTPRequestHandler):
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
            sys = get_system_info()
            ckpts = find_checkpoints()

            sft_metrics = parse_metrics(SFT_METRICS, last_n=200)
            dpo_metrics = parse_metrics(DPO_METRICS, last_n=200)

            sft_info = get_training_info(sft_metrics)
            dpo_info = get_training_info(dpo_metrics)

            sft_tail = ""
            if os.path.exists(SFT_LOG):
                with open(SFT_LOG) as f:
                    lines = f.readlines()
                    sft_tail = "".join(lines[-30:])

            running = False
            try:
                out = subprocess.run(["pgrep", "-f", "sft.py|dpo.py"], capture_output=True, text=True, timeout=3)
                running = bool(out.stdout.strip())
            except:
                pass

            return {
                "gpu": gpu,
                "system": sys,
                "running": running,
                "checkpoints": ckpts,
                "sft_tail": sft_tail,
                "sft_info": sft_info,
                "dpo_info": dpo_info,
                "hourly_rate": HOURLY_RATE,
                "now": datetime.datetime.now().isoformat(),
            }

        def log_message(self, format, *args):
            pass

    return Handler

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cognitive Core — Training Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background: #0d1117; color: #e1e4eb; line-height: 1.5;
    padding: 1.5rem; max-width: 1100px; margin: 0 auto;
  }
  h1 { font-size: 1.3rem; margin-bottom: .25rem; display: flex; align-items: center; gap: .75rem; }
  .subtitle { color: #8b929a; font-size: .85rem; margin-bottom: 1rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: .75rem; margin-bottom: .75rem; }
  .card {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 1rem;
  }
  .card.full { grid-column: 1 / -1; }
  .card h2 { font-size: .85rem; margin-bottom: .5rem; color: #8b949e; text-transform: uppercase; letter-spacing: .5px; }
  .stat-row { display: flex; justify-content: space-between; padding: .2rem 0; font-size: .85rem; }
  .stat-row .val { font-family: monospace; color: #58a6ff; }
  .stat-row .label { color: #8b949e; }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: .75rem; font-weight: 600;
  }
  .badge.running { background: #1a4731; color: #7ee787; }
  .badge.stopped { background: #3b2e00; color: #d29922; }
  .badge.none { background: #21262d; color: #8b949e; }
  .badge.done { background: #1a3a5c; color: #79c0ff; }
  pre.log {
    background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
    padding: .75rem; font-size: .7rem; max-height: 300px; overflow-y: auto;
    color: #8b949e; line-height: 1.4; font-family: 'SF Mono',Consolas,monospace;
  }
  .metric-list { max-height: 300px; overflow-y: auto; font-size: .8rem; }
  .metric-row { display: flex; justify-content: space-between; padding: .15rem 0; border-bottom: 1px solid #0d1117; }
  .metric-row .step { color: #8b949e; font-family: monospace; font-size: .75rem; }
  .metric-row .loss { color: #ffa657; font-family: monospace; }
  .status-indicator { display: flex; align-items: center; gap: .5rem; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .dot.green { background: #3fb950; animation: pulse 2s infinite; }
  .dot.yellow { background: #d29922; }
  .dot.red { background: #f85149; animation: pulse 2s infinite; }
  .dot.blue { background: #58a6ff; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
  .progress-bar {
    width: 100%; height: 6px; background: #21262d; border-radius: 3px;
    margin: .5rem 0; overflow: hidden;
  }
  .progress-bar .fill {
    height: 100%; background: linear-gradient(90deg, #238636, #3fb950);
    border-radius: 3px; transition: width 0.5s ease;
  }
  .last-updated { font-size: .75rem; color: #484f58; text-align: right; margin-top: .5rem; }
  .gpu-grid { display: grid; grid-template-columns: auto 1fr; gap: .15rem .75rem; font-size: .82rem; }
  .gpu-bar { height: 4px; background: #21262d; border-radius: 2px; margin-top: 2px; }
  .gpu-bar .fill { height: 100%; border-radius: 2px; background: #58a6ff; }
  .sys-grid { display: grid; grid-template-columns: auto 1fr; gap: .15rem .75rem; font-size: .82rem; }
  .cost-display { font-size: 1.5rem; font-family: monospace; color: #58a6ff; text-align: center; margin: .5rem 0; }
  .cost-label { font-size: .75rem; color: #8b949e; text-align: center; }
  @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<h1>
  <span>🧠 Cognitive Core</span>
  <span id="statusBadge" class="badge none">Loading...</span>
</h1>
<p class="subtitle">Training Dashboard — auto-refreshes every 3s</p>

<div class="grid">
  <!-- GPU -->
  <div class="card">
    <h2>🎮 GPU</h2>
    <div id="gpuInfo">Loading...</div>
  </div>

  <!-- System -->
  <div class="card">
    <h2>💻 System</h2>
    <div id="sysInfo">Loading...</div>
  </div>

  <!-- Cost -->
  <div class="card">
    <h2>💰 Training Cost</h2>
    <div id="costInfo">
      <div class="cost-display">$0.00</div>
      <div class="cost-label">estimated spend</div>
    </div>
  </div>
</div>

<!-- Loss chart -->
<div class="card full">
  <h2>📈 Loss</h2>
  <div id="lossContainer" style="height: 220px; display: flex; align-items: center; justify-content: center; color: #484f58;">
    Waiting for training to start...
  </div>
</div>

<div class="grid">
  <!-- Metrics -->
  <div class="card">
    <h2>📋 Recent Metrics</h2>
    <div id="metricsList" class="metric-list">No data yet</div>
  </div>

  <!-- Checkpoints -->
  <div class="card">
    <h2>💾 Checkpoints</h2>
    <div id="ckptList" class="metric-list">No checkpoints yet</div>
  </div>

  <!-- Log tail -->
  <div class="card">
    <h2>📝 Training Log</h2>
    <pre id="logTail" class="log">Waiting for training...</pre>
  </div>
</div>

<div class="last-updated" id="lastUpdated">Last updated: —</div>

<script>
function fmt(v) { return v == null || v === '?' || v === '' ? '—' : v; }

function makeBar(pct, color) {
  const c = color || '#58a6ff';
  return `<div class="gpu-bar"><div class="fill" style="width:${Math.min(pct,100)}%;background:${c}"></div></div>`;
}

function drawChart(canvas, metrics) {
  if (metrics.length < 2) return;
  const ctx = canvas.getContext('2d');
  const dpr = devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = canvas.clientHeight * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const pad = { top: 25, right: 20, bottom: 30, left: 55 };

  const steps = metrics.map(m => m.step || 0);
  const losses = metrics.map(m => m.loss || 0);
  const maxStep = Math.max(...steps) || 1;
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const range = maxLoss - minLoss || 1;
  const margin = range * 0.15;

  ctx.clearRect(0, 0, w, h);

  // Grid
  ctx.fillStyle = '#8b949e';
  ctx.font = `${11}px monospace`;
  ctx.textAlign = 'right';
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (h - pad.top - pad.bottom) * (1 - i/5);
    const val = (maxLoss + margin) - (range + 2*margin) * i/5;
    ctx.fillText(val.toFixed(3), pad.left - 8, y + 4);
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
  }

  // X axis labels
  ctx.textAlign = 'center';
  ctx.fillStyle = '#484f58';
  for (let i = 0; i <= 4; i++) {
    const x = pad.left + (w - pad.left - pad.right) * i/4;
    const val = Math.round(maxStep * i/4);
    ctx.fillText(`step ${val}`, x, h - 8);
  }

  // Line
  ctx.strokeStyle = '#58a6ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < steps.length; i++) {
    const x = pad.left + (w - pad.left - pad.right) * (steps[i] / maxStep);
    const y = pad.top + (h - pad.top - pad.bottom) * (1 - (losses[i] - (minLoss - margin)) / (range + 2*margin));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Dots
  ctx.fillStyle = '#58a6ff';
  const maxDots = 60;
  const step = Math.max(1, Math.floor(steps.length / maxDots));
  for (let i = 0; i < steps.length; i += step) {
    const x = pad.left + (w - pad.left - pad.right) * (steps[i] / maxStep);
    const y = pad.top + (h - pad.top - pad.bottom) * (1 - (losses[i] - (minLoss - margin)) / (range + 2*margin));
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI*2); ctx.fill();
  }

  // Latest point highlighted
  if (steps.length > 0) {
    const li = steps.length - 1;
    const x = pad.left + (w - pad.left - pad.right) * (steps[li] / maxStep);
    const y = pad.top + (h - pad.top - pad.bottom) * (1 - (losses[li] - (minLoss - margin)) / (range + 2*margin));
    ctx.fillStyle = '#ffa657';
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI*2); ctx.fill();
  }
}

function update() {
  fetch('/api/status').then(r => r.json()).then(d => {
    const gpu = d.gpu[0] || {};
    const sys = d.system || {};

    // GPU info
    document.getElementById('gpuInfo').innerHTML = `
      <div class="gpu-grid">
        <span class="label">Name</span><span class="val">${fmt(gpu.name)}</span>
        <span class="label">Temp</span><span class="val">${fmt(gpu.temp)}°C</span>
        <span class="label">VRAM</span><span class="val">${fmt(gpu.vram_used_mb)} / ${fmt(gpu.vram_total_mb)} MB (${fmt(gpu.vram_pct)}%)</span>
        <span class="label">Usage</span><span class="val">${fmt(gpu.gpu_usage)}%</span>
        <span class="label">Power</span><span class="val">${fmt(gpu.power_w)} W</span>
      </div>`;

    // System info
    document.getElementById('sysInfo').innerHTML = `
      <div class="sys-grid">
        <span class="label">CPU</span><span class="val">${fmt(sys.cpu_pct)}% (load: ${fmt(sys.cpu_load)})</span>
        <span class="label">RAM</span><span class="val">${fmt(sys.ram_used)} / ${fmt(sys.ram_total)} MB (${fmt(sys.ram_pct)}%)</span>
        <span class="label">Disk</span><span class="val">${fmt(sys.disk_used_gb)} / ${fmt(sys.disk_total_gb)} GB (${fmt(sys.disk_pct)}%)</span>
      </div>`;

    // Status badge
    const badge = document.getElementById('statusBadge');
    if (d.running) {
      badge.className = 'badge running'; badge.textContent = '● TRAINING';
    } else if (d.checkpoints.length > 0) {
      badge.className = 'badge stopped'; badge.textContent = '● STOPPED';
    } else {
      badge.className = 'badge none'; badge.textContent = '● IDLE';
    }

    // Cost
    const sftCost = d.sft_info ? d.sft_info.cost : 0;
    const dpoCost = d.dpo_info ? d.dpo_info.cost : 0;
    const totalCost = sftCost + dpoCost;
    const sftTime = d.sft_info ? d.sft_info.elapsed : '?';
    const dpoTime = d.dpo_info ? d.dpo_info.elapsed : '?';
    document.getElementById('costInfo').innerHTML = `
      <div class="cost-display">$${totalCost.toFixed(2)}</div>
      <div class="cost-label">@ $${d.hourly_rate}/hr on-demand</div>
      <div style="margin-top:.5rem;font-size:.8rem;color:#8b949e;">
        SFT: ${sftTime} ($${sftCost.toFixed(2)})<br>
        DPO: ${dpoTime} ($${dpoCost.toFixed(2)})
      </div>`;

    // Checkpoints
    const ckptList = document.getElementById('ckptList');
    if (d.checkpoints.length > 0) {
      ckptList.innerHTML = d.checkpoints.slice(-10).reverse().map(c =>
        `<div class="metric-row"><span class="step">${c.dir}</span><span class="loss">step ${c.step}</span></div>`
      ).join('');
    }

    // Log tail
    if (d.sft_tail) {
      document.getElementById('logTail').textContent = d.sft_tail;
    }
  });

  fetch('/api/metrics').then(r => r.json()).then(d => {
    const all = [...(d.sft || []), ...(d.dpo || [])];
    all.sort((a, b) => (a.step || 0) - (b.step || 0));

    if (all.length > 0) {
      // Metrics list
      const list = document.getElementById('metricsList');
      list.innerHTML = all.slice().reverse().slice(0, 40).map(m => {
        const loss = m.loss != null ? `<span class="loss">${Number(m.loss).toFixed(4)}</span>` : '';
        const phase = m.phase || '';
        return `<div class="metric-row"><span class="step">${phase} step ${m.step}</span>${loss}</div>`;
      }).join('');

      // Loss chart
      const container = document.getElementById('lossContainer');
      container.innerHTML = '';
      const canvas = document.createElement('canvas');
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      container.appendChild(canvas);
      drawChart(canvas, all);
    }
  });

  document.getElementById('lastUpdated').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
}

setInterval(update, 3000);
update();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    from http.server import HTTPServer

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    Handler = _handler_class()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Cognitive Core Dashboard → http://{args.host}:{args.port}")
    print(f"Open in browser to monitor training in real-time.")
    server.serve_forever()
