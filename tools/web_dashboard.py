#!/usr/bin/env python3
"""Flask web dashboard for monitoring NeuralSpell training.

Auto-refreshing charts for loss, accuracy, throughput, GPU memory,
gradient norms, and sample predictions. Reads from CSV logs and
samples.log produced by the training scripts.

Usage:
    PYTHONPATH=. python tools/web_dashboard.py
    PYTHONPATH=. python tools/web_dashboard.py --port 8080
    PYTHONPATH=. python tools/web_dashboard.py --host 0.0.0.0  # network access
"""

import argparse
import csv
import re
import time
from pathlib import Path

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

# Base directory for checkpoints
CHECKPOINT_BASE = Path("checkpoints")
LOG_DIRS = {
    "pretrain": CHECKPOINT_BASE / "pretrain" / "logs",
    "finetune": CHECKPOINT_BASE / "finetune" / "logs",
}

# Also check old log locations
OLD_LOG_DIRS = {
    "pretrain_old": Path("logs/pretrain"),
    "finetune_old": Path("logs/finetune"),
}


def read_metrics(csv_path: Path) -> list[dict]:
    """Read metrics CSV into list of row dicts."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    if "." in v or "e" in v.lower():
                        parsed[k] = float(v)
                    else:
                        parsed[k] = int(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def read_samples(samples_path: Path, last_n: int = 5) -> list[dict]:
    """Parse the last N sample blocks from samples.log."""
    if not samples_path.exists():
        return []
    text = samples_path.read_text()
    blocks = text.split("=" * 60)

    samples = []
    current_step = None
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        step_match = re.search(r"Step (\d+)", block)
        if step_match:
            current_step = int(step_match.group(1))
            continue
        # Parse sample entries
        corrupted = re.findall(r"Corrupted:\s*(.+)", block)
        generated = re.findall(r"(?:Generated|Predicted):\s*(.+)", block)
        original = re.findall(r"Original:\s*(.+)", block)
        for c, g, o in zip(corrupted, generated, original):
            samples.append({
                "step": current_step,
                "corrupted": c.strip(),
                "generated": g.strip(),
                "original": o.strip(),
            })

    return samples[-last_n:]


def find_active_phase() -> str:
    """Detect which phase is currently training based on file modification times."""
    latest_phase = "pretrain"
    latest_mtime = 0
    for phase, log_dir in {**LOG_DIRS, **OLD_LOG_DIRS}.items():
        csv_path = log_dir / "metrics.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            mtime = csv_path.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_phase = phase.replace("_old", "")
    return latest_phase


def get_all_metrics() -> dict:
    """Get metrics for all phases."""
    result = {}
    for phase, log_dir in {**LOG_DIRS, **OLD_LOG_DIRS}.items():
        csv_path = log_dir / "metrics.csv"
        samples_path = log_dir / "samples.log"
        phase_name = phase.replace("_old", "")
        if phase_name not in result:
            rows = read_metrics(csv_path)
            if rows:
                result[phase_name] = {
                    "metrics": rows,
                    "samples": read_samples(samples_path),
                }
        elif not result[phase_name]["metrics"]:
            rows = read_metrics(csv_path)
            if rows:
                result[phase_name] = {
                    "metrics": rows,
                    "samples": read_samples(samples_path),
                }
    return result


# ─── HTML Template ───────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuralSpell Training Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9; line-height: 1.5;
  }
  header {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 16px 24px; display: flex; align-items: center; justify-content: space-between;
  }
  header h1 { font-size: 20px; font-weight: 600; }
  header h1 span { color: #58a6ff; }
  .status { font-size: 13px; color: #8b949e; }
  .status .live { color: #3fb950; font-weight: 600; }
  .tabs {
    display: flex; gap: 4px; padding: 8px 24px; background: #161b22;
    border-bottom: 1px solid #30363d;
  }
  .tab {
    padding: 8px 16px; border-radius: 6px 6px 0 0; cursor: pointer;
    font-size: 14px; color: #8b949e; border: 1px solid transparent;
    background: transparent; transition: all 0.2s;
  }
  .tab:hover { color: #c9d1d9; }
  .tab.active { color: #c9d1d9; background: #0d1117; border-color: #30363d; border-bottom-color: #0d1117; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px 24px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px; position: relative;
  }
  .card h3 { font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
  .card .value { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
  .card .subtext { font-size: 12px; color: #8b949e; }
  .chart-card { min-height: 280px; }
  canvas { width: 100% !important; height: 240px !important; }
  .samples-card { grid-column: 1 / -1; }
  .sample {
    border: 1px solid #30363d; border-radius: 6px; padding: 12px;
    margin-bottom: 8px; font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 13px;
  }
  .sample .label { font-size: 11px; color: #8b949e; text-transform: uppercase; margin-bottom: 2px; }
  .sample .corrupted { color: #f85149; }
  .sample .generated { color: #3fb950; }
  .sample .original { color: #58a6ff; }
  .sample .step-badge {
    float: right; background: #30363d; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; color: #8b949e;
  }
  .phase-hidden { display: none; }
  @media (max-width: 768px) { .grid, .grid-3 { grid-template-columns: 1fr; } }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>

<header>
  <h1><span>NeuralSpell</span> Training Dashboard</h1>
  <div class="status">
    <span class="live" id="status-dot">●</span>
    <span id="status-text">Loading...</span>
  </div>
</header>

<div class="tabs">
  <div class="tab active" data-phase="pretrain" onclick="switchPhase('pretrain')">Phase 1: Pretrain</div>
  <div class="tab" data-phase="finetune" onclick="switchPhase('finetune')">Phase 2: Finetune</div>
</div>

<div class="container">
  <!-- Summary stats -->
  <div class="grid-3" id="stats"></div>

  <!-- Charts -->
  <div class="grid">
    <div class="card chart-card"><h3>Loss</h3><canvas id="chart-loss"></canvas></div>
    <div class="card chart-card"><h3>Token Accuracy</h3><canvas id="chart-accuracy"></canvas></div>
    <div class="card chart-card"><h3>Throughput (tokens/sec)</h3><canvas id="chart-tps"></canvas></div>
    <div class="card chart-card"><h3>Learning Rate</h3><canvas id="chart-lr"></canvas></div>
    <div class="card chart-card"><h3>GPU Memory (GB)</h3><canvas id="chart-gpu"></canvas></div>
    <div class="card chart-card"><h3>Gradient Norm</h3><canvas id="chart-grad"></canvas></div>
  </div>

  <!-- Sample predictions -->
  <div class="card samples-card">
    <h3>Recent Correction Samples</h3>
    <div id="samples"></div>
  </div>
</div>

<script>
const CHART_COLORS = {
  loss: '#f85149', accuracy: '#3fb950', tps: '#58a6ff',
  lr: '#d29922', gpu: '#bc8cff', grad: '#f0883e'
};

const chartConfig = (label, color) => ({
  type: 'line',
  data: { labels: [], datasets: [{ label, data: [], borderColor: color, backgroundColor: color + '15',
    pointRadius: 0, borderWidth: 1.5, fill: true, tension: 0.3 }] },
  options: {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 0 },
    scales: {
      x: { display: true, grid: { color: '#21262d' }, ticks: { color: '#8b949e', maxTicksLimit: 8 } },
      y: { display: true, grid: { color: '#21262d' }, ticks: { color: '#8b949e' } }
    },
    plugins: { legend: { display: false } }
  }
});

const charts = {};
let currentPhase = 'pretrain';
let allData = {};

function initCharts() {
  charts.loss = new Chart(document.getElementById('chart-loss'), chartConfig('Loss', CHART_COLORS.loss));
  charts.accuracy = new Chart(document.getElementById('chart-accuracy'), chartConfig('Token Accuracy', CHART_COLORS.accuracy));
  charts.tps = new Chart(document.getElementById('chart-tps'), chartConfig('Tokens/sec', CHART_COLORS.tps));
  charts.lr = new Chart(document.getElementById('chart-lr'), chartConfig('Learning Rate', CHART_COLORS.lr));
  charts.gpu = new Chart(document.getElementById('chart-gpu'), chartConfig('GPU Memory', CHART_COLORS.gpu));
  charts.grad = new Chart(document.getElementById('chart-grad'), chartConfig('Gradient Norm', CHART_COLORS.grad));
}

function updateCharts(metrics) {
  if (!metrics || !metrics.length) return;

  // Downsample for performance if needed
  let data = metrics;
  if (data.length > 2000) {
    const step = Math.ceil(data.length / 2000);
    data = data.filter((_, i) => i % step === 0);
  }

  const steps = data.map(r => r.step);
  const update = (chart, key) => {
    const vals = data.map(r => r[key]).filter(v => v !== undefined && v !== null);
    if (vals.length === 0) return;
    chart.data.labels = steps.slice(0, vals.length);
    chart.data.datasets[0].data = vals;
    chart.update('none');
  };

  update(charts.loss, 'loss');
  update(charts.accuracy, 'token_accuracy');
  update(charts.tps, 'tokens_per_sec');
  update(charts.lr, 'lr');
  update(charts.gpu, 'gpu_mem_gb');
  update(charts.grad, 'grad_norm');
}

function updateStats(metrics) {
  const el = document.getElementById('stats');
  if (!metrics || !metrics.length) {
    el.innerHTML = '<div class="card"><div class="value">—</div><div class="subtext">No data yet</div></div>';
    return;
  }
  const latest = metrics[metrics.length - 1];
  const step = latest.step || 0;
  const totalSteps = currentPhase === 'pretrain' ? 300000 : 100000;
  const pct = ((step / totalSteps) * 100).toFixed(1);
  const loss = (latest.loss || 0).toFixed(4);
  const acc = latest.token_accuracy ? (latest.token_accuracy * 100).toFixed(1) + '%' : '—';
  const tps = latest.tokens_per_sec ? Math.round(latest.tokens_per_sec).toLocaleString() : '—';
  const elapsed = latest.elapsed_sec ? (latest.elapsed_sec / 3600).toFixed(1) + 'h' : '—';
  const eta = (step > 0 && latest.elapsed_sec) ? (((totalSteps - step) * latest.elapsed_sec / step) / 3600).toFixed(1) + 'h' : '—';
  const gpu = latest.gpu_mem_gb ? latest.gpu_mem_gb.toFixed(1) + ' GB' : '—';

  el.innerHTML = `
    <div class="card"><div class="value">${step.toLocaleString()} <span style="font-size:14px;color:#8b949e">/ ${totalSteps.toLocaleString()}</span></div><div class="subtext">Step (${pct}%)</div></div>
    <div class="card"><div class="value" style="color:#f85149">${loss}</div><div class="subtext">Loss</div></div>
    <div class="card"><div class="value" style="color:#3fb950">${acc}</div><div class="subtext">Token Accuracy</div></div>
    <div class="card"><div class="value" style="color:#58a6ff">${tps}</div><div class="subtext">Tokens/sec</div></div>
    <div class="card"><div class="value">${elapsed}</div><div class="subtext">Elapsed</div></div>
    <div class="card"><div class="value">${eta}</div><div class="subtext">ETA</div></div>
  `;
}

function updateSamples(samples) {
  const el = document.getElementById('samples');
  if (!samples || !samples.length) {
    el.innerHTML = '<div style="color:#8b949e;padding:12px;">No samples yet. Samples appear every few thousand steps.</div>';
    return;
  }
  el.innerHTML = samples.map(s => `
    <div class="sample">
      ${s.step ? '<span class="step-badge">Step ' + s.step.toLocaleString() + '</span>' : ''}
      <div class="label">Corrupted</div>
      <div class="corrupted">${escapeHtml(s.corrupted)}</div>
      <div class="label" style="margin-top:6px">Generated</div>
      <div class="generated">${escapeHtml(s.generated)}</div>
      <div class="label" style="margin-top:6px">Original</div>
      <div class="original">${escapeHtml(s.original)}</div>
    </div>
  `).join('');
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function switchPhase(phase) {
  currentPhase = phase;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.phase === phase));
  renderPhase();
}

function renderPhase() {
  const data = allData[currentPhase];
  if (data) {
    updateStats(data.metrics);
    updateCharts(data.metrics);
    updateSamples(data.samples);
  } else {
    updateStats([]);
    updateCharts([]);
    updateSamples([]);
  }
}

async function fetchData() {
  try {
    const resp = await fetch('/api/metrics');
    allData = await resp.json();

    // Auto-detect active phase
    const activePhase = allData._active_phase || 'pretrain';
    document.getElementById('status-text').textContent =
      `Phase: ${activePhase} | Last update: ${new Date().toLocaleTimeString()}`;

    renderPhase();
  } catch (e) {
    document.getElementById('status-text').textContent = 'Error: ' + e.message;
    document.getElementById('status-dot').style.color = '#f85149';
  }
}

initCharts();
fetchData();
setInterval(fetchData, 15000);  // refresh every 15s
</script>
</body>
</html>
"""


# ─── Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/metrics")
def api_metrics():
    data = get_all_metrics()
    data["_active_phase"] = find_active_phase()
    return jsonify(data)


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NeuralSpell Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = parser.parse_args()

    print(f"\n  NeuralSpell Training Dashboard")
    print(f"  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
