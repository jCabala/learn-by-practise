"""
Plot energy vs IQ size using gem5 outputs under `results/out`.

This script finds runs named like `IQSIZE_<n>` (either top-level .log files or
directories under `results/out`), parses the following fields from the
gem5 output: `Simulated seconds`, `Runtime Dynamic`, `Subthreshold Leakage`,
and `Gate Leakage` (all in watts except seconds). It computes total energy as:

    energy_joules = (Runtime Dynamic + Subthreshold Leakage + Gate Leakage) * Simulated seconds

The script writes `results/plots/iqsize_energy.csv` and `iqsize_energy.png`/`.pdf`.
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


def parse_metrics_from_text(text: str) -> Optional[Tuple[float, float, float, float]]:
    """Return (sim_seconds, runtime_W, sub_W, gate_W) or None if not all present."""
    def grab(pattern: str):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    sim = grab(r"Simulated seconds\s*=\s*([0-9.eE+\-]+)")
    runtime = grab(r"Runtime Dynamic\s*=\s*([0-9.eE+\-]+)\s*W")
    sub = grab(r"Subthreshold Leakage\s*=\s*([0-9.eE+\-]+)\s*W")
    gate = grab(r"Gate Leakage\s*=\s*([0-9.eE+\-]+)\s*W")

    if None in (sim, runtime, sub, gate):
        return None
    return sim, runtime, sub, gate


def extract_iq_size(name: str) -> Optional[int]:
    m = re.search(r"IQSIZE_([0-9]+)", name)
    return int(m.group(1)) if m else None


def collect_values(out_dir: str) -> Dict[int, float]:
    values: Dict[int, float] = {}
    if not os.path.isdir(out_dir):
        return values

    for entry in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, entry)

        # top-level .log files
        if os.path.isfile(path) and entry.endswith('.log'):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
            iq = extract_iq_size(entry)
            parsed = parse_metrics_from_text(txt)
            if iq is not None and parsed is not None:
                sim, runtime, sub, gate = parsed
                energy = (runtime + sub + gate) * sim
                values[iq] = energy
            continue

        # directories
        if os.path.isdir(path):
            # try common candidates
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
                for subf in os.listdir(path):
                    subp = os.path.join(path, subf)
                    if os.path.isfile(subp):
                        with open(subp, 'r', encoding='utf-8', errors='ignore') as f:
                            data = f.read()
                        if 'Simulated seconds' in data or 'Runtime Dynamic' in data:
                            txt = data
                            break

            if txt is None:
                continue

            iq = extract_iq_size(entry)
            parsed = parse_metrics_from_text(txt)
            if iq is not None and parsed is not None:
                sim, runtime, sub, gate = parsed
                energy = (runtime + sub + gate) * sim
                values[iq] = energy

    return values


def plot(values: Dict[int, float], out_dir: str) -> None:
    if not values:
        print('No data found under', OUT_DIR)
        return

    os.makedirs(out_dir, exist_ok=True)

    items = sorted(values.items())
    iq = [k for k, _ in items]
    energy = [v for _, v in items]

    # save CSV
    csv_path = os.path.join(out_dir, 'iqsize_energy.csv')
    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['iq_size', 'energy_joules'])
        for a, b in zip(iq, energy):
            w.writerow([a, b])

    plt.figure(figsize=(7, 4.5))
    plt.plot(iq, energy, marker='o', linestyle='-')
    plt.xlabel('IQ size')
    plt.ylabel('Energy (J)')
    plt.title('Energy vs IQ size')
    plt.grid(alpha=0.4)

    # mark minimum
    try:
        mval = min(energy)
        midx = energy.index(mval)
        miq = iq[midx]
        plt.scatter([miq], [mval], color='red', s=80, zorder=5, label='min energy')
        plt.annotate(f"min={mval:.6g} J\niq={miq}", xy=(miq, mval), xytext=(10, -30), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'))
        plt.legend(fontsize='small')
    except Exception:
        pass

    png = os.path.join(out_dir, 'iqsize_energy.png')
    pdf = os.path.join(out_dir, 'iqsize_energy.pdf')
    plt.tight_layout()
    plt.savefig(png, dpi=200)
    plt.savefig(pdf)
    print('Wrote', png, pdf, csv_path)


def main() -> None:
    vals = collect_values(OUT_DIR)
    if not vals:
        print('No IQ size data parsed. Checked', OUT_DIR)
        return
    plot(vals, PLOTS_DIR)


if __name__ == '__main__':
    main()
