"""
Plot energy vs pipeline width using gem5 outputs under `results/out`.

Energy formula used:
  energy = (Runtime Dynamic + Subthreshold Leakage + Gate Leakage) * Simulated seconds

Outputs:
  - results/plots/pipeline_width_energy.csv
  - results/plots/pipeline_width_energy.png / .pdf
"""

from __future__ import annotations

import os
import re
import csv
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt


ROOT = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, "results", "out")
PLOTS_DIR = os.path.join(ROOT, "results", "plots")


def parse_metrics(text: str) -> Optional[Tuple[float, float, float, float]]:
    def g(pat: str):
        m = re.search(pat, text)
        return float(m.group(1)) if m else None

    sim = g(r"Simulated seconds\s*=\s*([0-9.eE+\-]+)")
    runtime = g(r"Runtime Dynamic\s*=\s*([0-9.eE+\-]+)\s*W")
    sub = g(r"Subthreshold Leakage\s*=\s*([0-9.eE+\-]+)\s*W")
    gate = g(r"Gate Leakage\s*=\s*([0-9.eE+\-]+)\s*W")

    if None in (sim, runtime, sub, gate):
        return None
    return sim, runtime, sub, gate


def extract_width(name: str) -> Optional[int]:
    m = re.search(r"PIPELINE_WIDTH_([0-9]+)", name)
    return int(m.group(1)) if m else None


def collect(out_dir: str) -> Dict[int, float]:
    vals: Dict[int, float] = {}
    if not os.path.isdir(out_dir):
        return vals

    for entry in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, entry)
        # log files
        if os.path.isfile(path) and entry.endswith('.log'):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
            w = extract_width(entry)
            parsed = parse_metrics(txt)
            if w is not None and parsed is not None:
                sim, runtime, sub, gate = parsed
                energy = (runtime + sub + gate) * sim
                vals[w] = energy
            continue

        if os.path.isdir(path):
            candidates = [
                os.path.join(path, 'results'),
                os.path.join(path, 'gem5.out'),
                os.path.join(path, 'gem5.out', 'results'),
                os.path.join(path, 'gem5.out', 'stats.txt'),
            ]
            txt = None
            for c in candidates:
                if os.path.isfile(c):
                    with open(c, 'r', encoding='utf-8', errors='ignore') as f:
                        txt = f.read()
                    break

            if txt is None:
                for sf in os.listdir(path):
                    sp = os.path.join(path, sf)
                    if os.path.isfile(sp):
                        with open(sp, 'r', encoding='utf-8', errors='ignore') as f:
                            data = f.read()
                        if 'Simulated seconds' in data or 'Runtime Dynamic' in data:
                            txt = data
                            break

            if txt is None:
                continue

            w = extract_width(entry)
            parsed = parse_metrics(txt)
            if w is not None and parsed is not None:
                sim, runtime, sub, gate = parsed
                energy = (runtime + sub + gate) * sim
                vals[w] = energy

    return vals


def plot(vals: Dict[int, float], out_dir: str) -> None:
    if not vals:
        print('No pipeline width data found in', OUT_DIR)
        return

    os.makedirs(out_dir, exist_ok=True)
    items = sorted(vals.items())
    widths = [k for k, _ in items]
    energy = [v for _, v in items]

    # CSV
    csv_path = os.path.join(out_dir, 'pipeline_width_energy.csv')
    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['pipeline_width', 'energy_joules'])
        for a, b in zip(widths, energy):
            w.writerow([a, b])

    plt.figure(figsize=(7, 4.5))
    plt.plot(widths, energy, marker='o', linestyle='-')
    plt.xlabel('Pipeline width')
    plt.ylabel('Energy (J)')
    plt.title('Energy vs Pipeline Width')
    plt.grid(alpha=0.4)

    # mark minimum
    try:
        mv = min(energy)
        midx = energy.index(mv)
        mw = widths[midx]
        plt.scatter([mw], [mv], color='red', s=80, zorder=5, label='min energy')
        plt.annotate(f"min={mv:.6g} J\nwidth={mw}", xy=(mw, mv), xytext=(10, -30), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'))
        plt.legend(fontsize='small')
    except Exception:
        pass

    png = os.path.join(out_dir, 'pipeline_width_energy.png')
    pdf = os.path.join(out_dir, 'pipeline_width_energy.pdf')
    plt.tight_layout()
    plt.savefig(png, dpi=200)
    plt.savefig(pdf)
    print('Wrote', png, pdf, csv_path)


def main() -> None:
    vals = collect(OUT_DIR)
    if not vals:
        print('No data parsed. Checked', OUT_DIR)
        return
    plot(vals, PLOTS_DIR)


if __name__ == '__main__':
    main()
