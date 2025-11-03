"""
Parse gem5 output under `results/out` and plot energy (J) vs cache size for three families:
 - L1 data cache (out/c1)
 - L2 cache (out/c2)
 - L1 instruction cache (out/instr)

Energy calculation matches other scripts in this project:
 energy_joules = (Runtime Dynamic + Subthreshold Leakage + Gate Leakage) * Simulated seconds

Usage: run this script from the `part_3/cache_sizes` directory. It will write CSV and
PNG/PDF files into `results/plots`.
"""

from __future__ import annotations

import os
import re
import csv
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt


ROOT = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, "results", "out")
PLOTS_DIR = os.path.join(ROOT, "results", "plots")


def parse_metrics_from_text(text: str) -> Optional[Tuple[float, float, float, float]]:
    """Return (sim_seconds, runtime_dynamic_W, subthreshold_W, gate_W) or None if missing."""
    def find_float(pattern: str):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    sim = find_float(r"Simulated seconds\s*=\s*([0-9.eE+\-]+)")
    runtime = find_float(r"Runtime Dynamic\s*=\s*([0-9.eE+\-]+)\s*W")
    sub = find_float(r"Subthreshold Leakage\s*=\s*([0-9.eE+\-]+)\s*W")
    gate = find_float(r"Gate Leakage\s*=\s*([0-9.eE+\-]+)\s*W")

    if None in (sim, runtime, sub, gate):
        return None
    return sim, runtime, sub, gate


def extract_cache_size(name: str, prefix: str) -> Optional[int]:
    """Extract numeric cache size from names like C1_64 or 0_C1_64.log or INSTR_128."""
    # look for e.g. C1_64 or INSTR_64
    m = re.search(rf"{re.escape(prefix)}_([0-9]+)", name)
    if m:
        return int(m.group(1))
    return None


def collect_energy_values(out_dir: str, family: str, prefix: str) -> Dict[int, float]:
    """Scan files and directories under out_dir/<family> and return mapping size -> energy_joules."""
    values: Dict[int, float] = {}

    fam_dir = os.path.join(out_dir, family)
    if not os.path.isdir(fam_dir):
        print('Skipping missing directory', fam_dir)
        return values

    for entry in sorted(os.listdir(fam_dir)):
        path = os.path.join(fam_dir, entry)

        # top-level .log files
        if os.path.isfile(path) and entry.endswith('.log'):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
            size = extract_cache_size(entry, prefix)
            parsed = parse_metrics_from_text(txt)
            if size is not None and parsed is not None:
                sim, runtime, sub, gate = parsed
                energy = (runtime + sub + gate) * sim
                values[size] = energy
            continue

        # directory case - look for results or gem5.out/stats.txt
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
                # try files in the dir for anything that looks like gem5 output
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

            size = extract_cache_size(entry, prefix)
            parsed = parse_metrics_from_text(txt)
            if size is not None and parsed is not None:
                sim, runtime, sub, gate = parsed
                energy = (runtime + sub + gate) * sim
                values[size] = energy

    return values


def plot_values(values: Dict[int, float], out_dir: str, basename: str, title: str, show_min: bool = True) -> None:
    if not values:
        print('No valid data for', basename)
        return

    os.makedirs(out_dir, exist_ok=True)

    items = sorted(values.items())
    sizes = [k for k, _ in items]
    energy = [v for _, v in items]

    # save CSV
    csv_path = os.path.join(out_dir, f'{basename}_energy.csv')
    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['cache_size_kb', 'energy_joules'])
        for s, e in zip(sizes, energy):
            w.writerow([s, e])

    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, energy, marker='o', linestyle='-')
    plt.xlabel('Cache size (KB)')
    plt.ylabel('Energy (Joules)')
    plt.title(title)
    plt.grid(alpha=0.4)

    # mark minimum (optional)
    if show_min:
        try:
            min_val = min(energy)
            min_idx = energy.index(min_val)
            min_size = sizes[min_idx]
            plt.scatter([min_size], [min_val], color='red', s=90, zorder=5, label='min energy')
            plt.annotate(
                f"min={min_val:.6g} J\nsize={min_size}",
                xy=(min_size, min_val),
                xytext=(10, -30),
                textcoords='offset points',
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red'),
            )
            plt.legend(fontsize='small')
        except Exception:
            pass

    png = os.path.join(out_dir, f'{basename}_energy.png')
    pdf = os.path.join(out_dir, f'{basename}_energy.pdf')
    plt.tight_layout()
    plt.savefig(png, dpi=200)
    plt.savefig(pdf)
    print('Wrote', png, 'and', pdf, 'and', csv_path)


def main() -> None:
    datasets = [
        ('c1', 'C1', 'L1 data cache (C1)'),
        ('c2', 'C2', 'L2 cache (C2)'),
        ('instr', 'INSTR', 'L1 instruction cache (INSTR)'),
    ]

    for family_dir, prefix, title in datasets:
        values = collect_energy_values(OUT_DIR, family_dir, prefix)
        # For L2 (c2) we do not show the minimum marker per request
        show_min = False if family_dir == 'c2' else True
        plot_values(values, PLOTS_DIR, family_dir, title, show_min=show_min)


if __name__ == '__main__':
    main()
