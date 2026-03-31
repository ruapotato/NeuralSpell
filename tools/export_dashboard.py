#!/usr/bin/env python3
"""Export a static HTML snapshot of the training dashboard.

Embeds all metrics and samples directly in the HTML — no server needed.
Open the output file in any browser.

Usage:
    PYTHONPATH=. python tools/export_dashboard.py
    PYTHONPATH=. python tools/export_dashboard.py --output docs/dashboard.html
"""

import argparse
import csv
import json
import re
from pathlib import Path

CHECKPOINT_BASE = Path("checkpoints")
LOG_DIRS = {
    "pretrain": CHECKPOINT_BASE / "pretrain" / "logs",
    "finetune": CHECKPOINT_BASE / "finetune" / "logs",
}


def read_metrics(csv_path: Path) -> list[dict]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
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


def read_samples(samples_path: Path, last_n: int = 15) -> list[dict]:
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
        m = re.search(r"Step (\d+)", block)
        if m:
            current_step = int(m.group(1))
            continue
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


def gather_data() -> dict:
    data = {}
    for phase, log_dir in LOG_DIRS.items():
        metrics = read_metrics(log_dir / "metrics.csv")
        samples = read_samples(log_dir / "samples.log")
        data[phase] = {"metrics": metrics, "samples": samples}
    return data


def generate_html(data: dict) -> str:
    data_json = json.dumps(data)
    # Summary for header
    pretrain = data.get("pretrain", {}).get("metrics", [])
    latest = pretrain[-1] if pretrain else {}
    step = latest.get("step", 0)
    total = 150000
    loss = latest.get("loss", 0)
    acc = latest.get("token_accuracy", 0)
    elapsed_h = latest.get("elapsed_sec", 0) / 3600

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuralSpell Training Progress</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9; line-height: 1.5;
  }}
  header {{
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 16px 24px; display: flex; align-items: center; justify-content: space-between;
  }}
  header h1 {{ font-size: 20px; font-weight: 600; }}
  header h1 span {{ color: #58a6ff; }}
  .snapshot {{ font-size: 13px; color: #8b949e; }}
  .tabs {{
    display: flex; gap: 4px; padding: 8px 24px; background: #161b22;
    border-bottom: 1px solid #30363d;
  }}
  .tab {{
    padding: 8px 16px; border-radius: 6px 6px 0 0; cursor: pointer;
    font-size: 14px; color: #8b949e; border: 1px solid transparent;
    background: transparent; transition: all 0.2s;
  }}
  .tab:hover {{ color: #c9d1d9; }}
  .tab.active {{ color: #c9d1d9; background: #0d1117; border-color: #30363d; border-bottom-color: #0d1117; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px 24px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 16px; }}
  .card {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px;
  }}
  .card h3 {{ font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}
  .card .value {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
  .card .subtext {{ font-size: 12px; color: #8b949e; }}
  .chart-card {{ min-height: 280px; }}
  canvas {{ width: 100% !important; height: 240px !important; }}
  .samples-card {{ grid-column: 1 / -1; }}
  .sample {{
    border: 1px solid #30363d; border-radius: 6px; padding: 12px;
    margin-bottom: 8px; font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 13px;
  }}
  .sample .label {{ font-size: 11px; color: #8b949e; text-transform: uppercase; margin-bottom: 2px; }}
  .sample .step-badge {{
    float: right; background: #30363d; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; color: #8b949e;
  }}
  .sample .score-badge {{
    display: inline-block; background: #30363d; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; margin-left: 8px;
  }}
  .sample .score-badge.good {{ color: #3fb950; }}
  .sample .score-badge.partial {{ color: #d29922; }}
  .sample .score-badge.bad {{ color: #f85149; }}
  .w-ok {{ color: #c9d1d9; }}
  .w-err {{ color: #f85149; text-decoration: underline; text-decoration-style: wavy; text-underline-offset: 3px; }}
  .w-fixed {{ color: #3fb950; font-weight: 600; }}
  .w-missed {{ color: #f85149; font-weight: 600; }}
  .w-broken {{ color: #d29922; font-weight: 600; text-decoration: underline; text-decoration-style: wavy; text-underline-offset: 3px; }}
  @media (max-width: 768px) {{ .grid, .grid-3 {{ grid-template-columns: 1fr; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>

<header>
  <h1><span>NeuralSpell</span> Training Progress</h1>
  <div class="snapshot">Static snapshot &mdash; {step:,} / {total:,} steps ({step/total*100:.1f}%) &mdash; {elapsed_h:.1f}h elapsed</div>
</header>

<div class="tabs">
  <div class="tab active" data-phase="pretrain" onclick="switchPhase('pretrain')">Phase 1: Pretrain</div>
  <div class="tab" data-phase="finetune" onclick="switchPhase('finetune')">Phase 2: Finetune</div>
</div>

<div class="container">
  <div class="grid-3" id="stats"></div>
  <div class="grid">
    <div class="card chart-card"><h3>Loss</h3><canvas id="chart-loss"></canvas></div>
    <div class="card chart-card"><h3>Token Accuracy</h3><canvas id="chart-accuracy"></canvas></div>
    <div class="card chart-card"><h3>Throughput (tokens/sec)</h3><canvas id="chart-tps"></canvas></div>
    <div class="card chart-card"><h3>Learning Rate</h3><canvas id="chart-lr"></canvas></div>
    <div class="card chart-card"><h3>GPU Memory (GB)</h3><canvas id="chart-gpu"></canvas></div>
    <div class="card chart-card"><h3>Gradient Norm</h3><canvas id="chart-grad"></canvas></div>
  </div>
  <div class="card samples-card">
    <h3>Correction Samples</h3>
    <div id="samples"></div>
  </div>
</div>

<script>
const ALL_DATA = {data_json};

const CHART_COLORS = {{
  loss: '#f85149', accuracy: '#3fb950', tps: '#58a6ff',
  lr: '#d29922', gpu: '#bc8cff', grad: '#f0883e'
}};

const chartConfig = (label, color) => ({{
  type: 'line',
  data: {{ labels: [], datasets: [{{ label, data: [], borderColor: color, backgroundColor: color + '15',
    pointRadius: 0, borderWidth: 1.5, fill: true, tension: 0.3 }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: {{ duration: 0 }},
    scales: {{
      x: {{ display: true, grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e', maxTicksLimit: 8 }} }},
      y: {{ display: true, grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }}
    }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

const charts = {{}};
let currentPhase = 'pretrain';

function initCharts() {{
  charts.loss = new Chart(document.getElementById('chart-loss'), chartConfig('Loss', CHART_COLORS.loss));
  charts.accuracy = new Chart(document.getElementById('chart-accuracy'), chartConfig('Token Accuracy', CHART_COLORS.accuracy));
  charts.tps = new Chart(document.getElementById('chart-tps'), chartConfig('Tokens/sec', CHART_COLORS.tps));
  charts.lr = new Chart(document.getElementById('chart-lr'), chartConfig('Learning Rate', CHART_COLORS.lr));
  charts.gpu = new Chart(document.getElementById('chart-gpu'), chartConfig('GPU Memory', CHART_COLORS.gpu));
  charts.grad = new Chart(document.getElementById('chart-grad'), chartConfig('Gradient Norm', CHART_COLORS.grad));
}}

function updateCharts(metrics) {{
  if (!metrics || !metrics.length) return;
  let data = metrics;
  if (data.length > 2000) {{
    const step = Math.ceil(data.length / 2000);
    data = data.filter((_, i) => i % step === 0);
  }}
  const steps = data.map(r => r.step);
  const update = (chart, key) => {{
    const vals = data.map(r => r[key]).filter(v => v !== undefined && v !== null);
    if (vals.length === 0) return;
    chart.data.labels = steps.slice(0, vals.length);
    chart.data.datasets[0].data = vals;
    chart.update('none');
  }};
  update(charts.loss, 'loss');
  update(charts.accuracy, 'token_accuracy');
  update(charts.tps, 'tokens_per_sec');
  update(charts.lr, 'lr');
  update(charts.gpu, 'gpu_mem_gb');
  update(charts.grad, 'grad_norm');
}}

function updateStats(metrics) {{
  const el = document.getElementById('stats');
  if (!metrics || !metrics.length) {{
    el.innerHTML = '<div class="card"><div class="value">&mdash;</div><div class="subtext">No data yet</div></div>';
    return;
  }}
  const latest = metrics[metrics.length - 1];
  const step = latest.step || 0;
  const totalSteps = currentPhase === 'pretrain' ? 150000 : 50000;
  const pct = ((step / totalSteps) * 100).toFixed(1);
  const loss = (latest.loss || 0).toFixed(4);
  const acc = latest.token_accuracy ? (latest.token_accuracy * 100).toFixed(1) + '%' : '&mdash;';
  const tps = latest.tokens_per_sec ? Math.round(latest.tokens_per_sec).toLocaleString() : '&mdash;';
  const totalElapsed = metrics.reduce((sum, r, i) => {{
    if (i === 0) return r.elapsed_sec || 0;
    const prev = metrics[i-1].elapsed_sec || 0;
    const curr = r.elapsed_sec || 0;
    return curr < prev ? sum + curr : sum + (curr - prev);
  }}, 0);
  const elapsed = totalElapsed > 0 ? (totalElapsed / 3600).toFixed(1) + 'h' : '&mdash;';
  let eta = '&mdash;';
  if (metrics.length >= 2) {{
    const prev = metrics[metrics.length - 2];
    const ds = latest.step - prev.step;
    const dt = (latest.elapsed_sec || 0) - (prev.elapsed_sec || 0);
    if (ds > 0 && dt > 0) {{
      const secPerStep = dt / ds;
      eta = ((totalSteps - step) * secPerStep / 3600).toFixed(1) + 'h';
    }}
  }}
  el.innerHTML = `
    <div class="card"><div class="value">${{step.toLocaleString()}} <span style="font-size:14px;color:#8b949e">/ ${{totalSteps.toLocaleString()}}</span></div><div class="subtext">Step (${{pct}}%)</div></div>
    <div class="card"><div class="value" style="color:#f85149">${{loss}}</div><div class="subtext">Loss</div></div>
    <div class="card"><div class="value" style="color:#3fb950">${{acc}}</div><div class="subtext">Token Accuracy</div></div>
    <div class="card"><div class="value" style="color:#58a6ff">${{tps}}</div><div class="subtext">Tokens/sec</div></div>
    <div class="card"><div class="value">${{elapsed}}</div><div class="subtext">Elapsed</div></div>
    <div class="card"><div class="value">${{eta}}</div><div class="subtext">ETA</div></div>
  `;
}}

function escapeHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

function diffWords(corrupted, original, generated) {{
  const cw = corrupted.split(/\s+/).filter(Boolean);
  const ow = original.split(/\s+/).filter(Boolean);
  const gw = generated.split(/\s+/).filter(Boolean);
  function align(a, b) {{
    const m = a.length, n = b.length;
    const dp = Array.from({{length: m+1}}, () => new Uint16Array(n+1));
    for (let i=1; i<=m; i++)
      for (let j=1; j<=n; j++)
        dp[i][j] = a[i-1].toLowerCase() === b[j-1].toLowerCase()
          ? dp[i-1][j-1]+1 : Math.max(dp[i-1][j], dp[i][j-1]);
    const pairs = [];
    let i=m, j=n;
    while (i>0 || j>0) {{
      if (i>0 && j>0 && a[i-1].toLowerCase() === b[j-1].toLowerCase()) {{ pairs.push([i-1, j-1]); i--; j--; }}
      else if (j>0 && (i===0 || dp[i][j-1] >= dp[i-1][j])) {{ pairs.push([-1, j-1]); j--; }}
      else {{ pairs.push([i-1, -1]); i--; }}
    }}
    return pairs.reverse();
  }}
  const co = align(cw, ow);
  const errIndicesOrig = new Set();
  co.forEach(([ci, oi]) => {{
    if (ci === -1 || oi === -1) {{ if (oi >= 0) errIndicesOrig.add(oi); }}
    else if (cw[ci] !== ow[oi]) errIndicesOrig.add(oi);
  }});
  const go = align(gw, ow);
  let fixed = 0, missed = 0, broken = 0, totalErrs = errIndicesOrig.size;
  const corrHtml = co.map(([ci, oi]) => {{
    if (ci === -1) return '';
    const w = escapeHtml(cw[ci]);
    if (oi === -1) return `<span class="w-err">${{w}}</span>`;
    return cw[ci] !== ow[oi] ? `<span class="w-err">${{w}}</span>` : `<span class="w-ok">${{w}}</span>`;
  }}).filter(Boolean).join(' ');
  const genHtml = go.map(([gi, oi]) => {{
    if (gi === -1) return '';
    const w = escapeHtml(gw[gi]);
    if (oi === -1) {{ broken++; return `<span class="w-broken">${{w}}</span>`; }}
    const wasErr = errIndicesOrig.has(oi);
    const isCorrect = gw[gi].toLowerCase() === ow[oi].toLowerCase();
    if (wasErr && isCorrect) {{ fixed++; return `<span class="w-fixed">${{w}}</span>`; }}
    if (wasErr && !isCorrect) {{ missed++; return `<span class="w-missed">${{w}}</span>`; }}
    if (!wasErr && !isCorrect) {{ broken++; return `<span class="w-broken">${{w}}</span>`; }}
    return `<span class="w-ok">${{w}}</span>`;
  }}).filter(Boolean).join(' ');
  go.forEach(([gi, oi]) => {{
    if (gi === -1 && oi >= 0 && errIndicesOrig.has(oi)) missed++;
  }});
  const scorePct = totalErrs > 0 ? Math.round(fixed / totalErrs * 100) : 100;
  const scoreClass = scorePct >= 80 ? 'good' : scorePct >= 40 ? 'partial' : 'bad';
  const scoreHtml = totalErrs > 0
    ? `<span class="score-badge ${{scoreClass}}">${{fixed}}/${{totalErrs}} fixed (${{scorePct}}%)</span>`
    : `<span class="score-badge good">no errors</span>`;
  const origHtml = ow.map(w => escapeHtml(w)).join(' ');
  return {{ corrHtml, genHtml, origHtml, scoreHtml }};
}}

function updateSamples(samples) {{
  const el = document.getElementById('samples');
  if (!samples || !samples.length) {{
    el.innerHTML = '<div style="color:#8b949e;padding:12px;">No samples yet.</div>';
    return;
  }}
  el.innerHTML = samples.map(s => {{
    const d = diffWords(s.corrupted, s.original, s.generated);
    return `
    <div class="sample">
      ${{s.step ? '<span class="step-badge">Step ' + s.step.toLocaleString() + '</span>' : ''}}
      <div class="label">Corrupted</div>
      <div>${{d.corrHtml}}</div>
      <div class="label" style="margin-top:6px">Generated ${{d.scoreHtml}}</div>
      <div>${{d.genHtml}}</div>
      <div class="label" style="margin-top:6px">Original</div>
      <div style="color:#8b949e">${{d.origHtml}}</div>
    </div>`;
  }}).join('');
}}

function switchPhase(phase) {{
  currentPhase = phase;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.phase === phase));
  const data = ALL_DATA[phase] || {{}};
  updateStats(data.metrics || []);
  updateCharts(data.metrics || []);
  updateSamples(data.samples || []);
}}

initCharts();
switchPhase('pretrain');
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Export static training dashboard")
    parser.add_argument("--output", type=Path, default=Path("docs/dashboard.html"))
    args = parser.parse_args()

    data = gather_data()
    html = generate_html(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)

    pretrain = data.get("pretrain", {}).get("metrics", [])
    step = pretrain[-1]["step"] if pretrain else 0
    print(f"Exported dashboard to {args.output} ({len(pretrain)} data points, step {step:,})")


if __name__ == "__main__":
    main()
