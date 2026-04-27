#!/usr/bin/env python3
"""
Plot training loss from distillation logs.

Usage:
    python plot_loss.py                          # auto-find latest checkpoint log
    python plot_loss.py training.log               # specific log file
    python plot_loss.py /path/to/training.log      # full path
    python plot_loss.py --output loss.html         # custom output name
"""

import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


def parse_log(log_path: str) -> list[dict]:
    """Parse training log and extract metrics per step."""
    pattern = re.compile(
        r'Epoch\s+(\d+)\s+\|\s+Step\s+(\d+)\s+\|\s+'
        r'Loss:\s+([\d.eE+-]+)\s+\|\s+'
        r'Final\s+MSE:\s+([\d.eE+-]+)\s+\|\s+'
        r'Deepstack\s+MSE:\s+([\d.eE+-]+)\s+\|\s+'
        r'LR:\s+([\d.eE+-]+)'
    )

    records = []
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch, step, loss, final_mse, deepstack_mse, lr = m.groups()
                records.append({
                    'epoch': int(epoch),
                    'step': int(step),
                    'loss': float(loss),
                    'final_mse': float(final_mse),
                    'deepstack_mse': float(deepstack_mse),
                    'lr': float(lr),
                })
    return records


def generate_html(records: list[dict], output_path: str, title: str = "ViT Distillation Loss"):
    """Generate interactive HTML with Plotly.js."""

    steps = [r['step'] for r in records]
    loss = [r['loss'] for r in records]
    final_mse = [r['final_mse'] for r in records]
    deepstack_mse = [r['deepstack_mse'] for r in records]
    lr = [r['lr'] for r in records]

    data_json = json.dumps({
        'steps': steps,
        'loss': loss,
        'final_mse': final_mse,
        'deepstack_mse': deepstack_mse,
        'lr': lr,
    })

    # Build HTML using string formatting instead of f-strings to avoid escaping issues
    html = (
        'eJy9VG1v2jAQ/isnvyA0ErQKoZq2adImTdo+9LFqH1A1ycWxF8eO7AtQIf77znZIS9tK'
        '6Id9iO+5+57nzklyfX19fXt7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7'
        'e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7'
        'e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7e3t7'
        # ... (truncated for brevity, will use proper template below)
    )

    # Actually, let's use proper string formatting with .format() to avoid f-string issues
    template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
  .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  h1 {{ margin-top: 0; color: #333; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }}
  .card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
  .card .label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card .value {{ font-size: 22px; font-weight: 600; color: #1a1a2e; margin-top: 5px; }}
  .card .delta {{ font-size: 13px; margin-top: 3px; }}
  .delta.pos {{ color: #e74c3c; }}
  .delta.neg {{ color: #27ae60; }}
  .chart {{ margin-bottom: 30px; }}
  .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 20px; }}
</style>
</head>
<body>
<div class="container">
  <h1>🐢 ViT Distillation Training Loss</h1>
  <div class="summary" id="summary"></div>
  <div id="loss-chart" class="chart"></div>
  <div id="lr-chart" class="chart"></div>
  <div class="footer">Generated at {timestamp}</div>
</div>

<script>
const data = {data_json};

function fmt(v) {{
  if (v < 0.01) return v.toExponential(2);
  return v.toFixed(4);
}}

const first = {{
  loss: data.loss[0],
  final_mse: data.final_mse[0],
  deepstack_mse: data.deepstack_mse[0]
}};

const last = {{
  loss: data.loss[data.loss.length - 1],
  final_mse: data.final_mse[data.final_mse.length - 1],
  deepstack_mse: data.deepstack_mse[data.deepstack_mse.length - 1]
}};

function delta(oldv, newv) {{
  const d = newv - oldv;
  const pct = oldv !== 0 ? (d / oldv * 100).toFixed(1) : '0.0';
  const cls = d > 0 ? 'pos' : 'neg';
  const arrow = d > 0 ? '▲' : '▼';
  return `<span class="delta ${{cls}}">${{arrow}} ${{Math.abs(d).toExponential(2)}} (${{Math.abs(pct)}}%)</span>`;
}}

document.getElementById('summary').innerHTML = `
  <div class="card"><div class="label">Total Steps</div><div class="value">${{data.steps[data.steps.length - 1]}}</div></div>
  <div class="card"><div class="label">Total Loss</div><div class="value">${{fmt(last.loss)}}</div>${{delta(first.loss, last.loss)}}</div>
  <div class="card"><div class="label">Final MSE</div><div class="value">${{fmt(last.final_mse)}}</div>${{delta(first.final_mse, last.final_mse)}}</div>
  <div class="card"><div class="label">Deepstack MSE</div><div class="value">${{fmt(last.deepstack_mse)}}</div>${{delta(first.deepstack_mse, last.deepstack_mse)}}</div>
  <div class="card"><div class="label">Learning Rate</div><div class="value">${{last.lr.toExponential(2)}}</div></div>
`;

Plotly.newPlot('loss-chart', [
  {{
    x: data.steps,
    y: data.loss,
    mode: 'lines',
    name: 'Total Loss',
    line: {{ color: '#e74c3c', width: 2 }},
    hovertemplate: 'Step %{{x}}<br>Loss: %{{y:.6f}}<extra></extra>'
  }},
  {{
    x: data.steps,
    y: data.final_mse,
    mode: 'lines',
    name: 'Final MSE',
    line: {{ color: '#3498db', width: 2 }},
    hovertemplate: 'Step %{{x}}<br>Final MSE: %{{y:.6f}}<extra></extra>'
  }},
  {{
    x: data.steps,
    y: data.deepstack_mse,
    mode: 'lines',
    name: 'Deepstack MSE',
    line: {{ color: '#2ecc71', width: 2 }},
    hovertemplate: 'Step %{{x}}<br>Deepstack MSE: %{{y:.6f}}<extra></extra>'
  }}
], {{
  title: {{ text: 'Loss Curve', font: {{ size: 18 }} }},
  xaxis: {{ title: 'Training Step', gridcolor: '#eee' }},
  yaxis: {{ title: 'Loss / MSE', type: 'log', gridcolor: '#eee' }},
  legend: {{ x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)', bordercolor: '#ddd', borderwidth: 1 }},
  plot_bgcolor: '#fafafa',
  paper_bgcolor: '#fafafa',
  margin: {{ t: 50, b: 50, l: 60, r: 20 }},
  hovermode: 'x unified'
}}, {{ responsive: true }});

Plotly.newPlot('lr-chart', [
  {{
    x: data.steps,
    y: data.lr,
    mode: 'lines',
    name: 'Learning Rate',
    line: {{ color: '#9b59b6', width: 2 }},
    fill: 'tozeroy',
    fillcolor: 'rgba(155, 89, 182, 0.1)',
    hovertemplate: 'Step %{{x}}<br>LR: %{{y:.2e}}<extra></extra>'
  }}
], {{
  title: {{ text: 'Learning Rate Schedule', font: {{ size: 18 }} }},
  xaxis: {{ title: 'Training Step', gridcolor: '#eee' }},
  yaxis: {{ title: 'Learning Rate', type: 'log', gridcolor: '#eee' }},
  plot_bgcolor: '#fafafa',
  paper_bgcolor: '#fafafa',
  margin: {{ t: 50, b: 50, l: 60, r: 20 }},
  showlegend: false
}}, {{ responsive: true }});
</script>
</body>
</html>
""".format(
        title=title,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        data_json=data_json,
    )

    with open(output_path, 'w') as f:
        f.write(template)

    print(f"✅ Plot saved to: {output_path}")
    print(f"   Records: {len(records)}")
    print(f"   Steps: {records[0]['step']} → {records[-1]['step']}")
    print(f"   Loss: {records[0]['loss']:.4f} → {records[-1]['loss']:.4f}")
    print(f"\nOpen in browser:")
    print(f"   file://{Path(output_path).resolve()}")


def find_latest_checkpoint_log() -> str | None:
    """Find the most recent training.log in checkpoint directories."""
    output_dir = Path("/gpfs-data/mikelee/distillation_output")
    if not output_dir.exists():
        return None

    checkpoints = sorted(output_dir.glob("checkpoint_*/training.log"))
    if checkpoints:
        return str(checkpoints[-1])

    # Fallback: current directory
    local_log = Path("training.log")
    if local_log.exists():
        return str(local_log)

    return None


def main():
    parser = argparse.ArgumentParser(description="Plot ViT distillation training loss")
    parser.add_argument("log", nargs="?", default=None, help="Path to training.log")
    parser.add_argument("-o", "--output", default="loss_plot.html", help="Output HTML file")
    parser.add_argument("--title", default="ViT Distillation Loss", help="Chart title")
    args = parser.parse_args()

    log_path = args.log
    if log_path is None:
        log_path = find_latest_checkpoint_log()
        if log_path is None:
            print("❌ No training.log found. Please specify the path:")
            print("   python plot_loss.py /path/to/training.log")
            sys.exit(1)
        print(f"📁 Auto-found log: {log_path}")

    if not Path(log_path).exists():
        print(f"❌ File not found: {log_path}")
        sys.exit(1)

    records = parse_log(log_path)
    if not records:
        print("❌ No valid training records found in the log file.")
        sys.exit(1)

    generate_html(records, args.output, args.title)


if __name__ == "__main__":
    main()
