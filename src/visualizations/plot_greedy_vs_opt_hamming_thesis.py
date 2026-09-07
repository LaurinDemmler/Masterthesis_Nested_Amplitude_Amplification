"""Thesis plot: average Hamming distance between greedy and optimal solutions.

Compares two datasets:
  - Paper_unc_10_35   -> labelled "Uncorrelated"
  - Paper_weak_10_35  -> labelled "Weakly Correlated"

For every instance the greedy solution (density-sorted heuristic) and the exact
optimum (Gurobi) are computed, and the Hamming distance between their bitstrings
is recorded. The script plots the mean Hamming distance versus knapsack size n
for both datasets, using the corporate colours of the Technical University of
Munich (TUM).

Run from the ``src`` directory:
    python visualizations/plot_greedy_vs_opt_hamming_thesis.py
"""
import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Ensure the src/ directory is importable when run from anywhere.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datastructures.knapsack import KnapsackInstance
from generator.greedy import greedy_solver
from generator.GurobiSolver import GurobiSolver


# --- Paths -----------------------------------------------------------------
CODE_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_ROOT = CODE_ROOT / "config"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "Thesis_Results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset dir -> display label
DATASETS = {
    "Uncorrelated": CONFIG_ROOT / "Paper_unc_10_35",
    "Weakly Correlated": CONFIG_ROOT / "Paper_weak_10_35",
}

# --- TUM corporate colours -------------------------------------------------
TUM_BLUE = "#0065BD"        # primary
TUM_ORANGE = "#E37222"      # accent orange
TUM_DARK_BLUE = "#005293"
TUM_GRAY = "#999999"

DATASET_COLORS = {
    "Uncorrelated": TUM_BLUE,
    "Weakly Correlated": TUM_ORANGE,
}


def hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def _size_from_name(name: str) -> int | None:
    """Extract the knapsack size n from a filename like 'n10_r400_...'."""
    try:
        return int(name.split("_", 1)[0].lstrip("n"))
    except (ValueError, IndexError):
        return None


def analyze_dataset(config_dir: pathlib.Path, gurobi: GurobiSolver):
    """Compute Hamming distances for every instance in ``config_dir``.

    Returns a dict n -> list of Hamming distances.
    """
    yamls = sorted(config_dir.rglob("*.yaml"))
    print(f"  {config_dir.name}: {len(yamls)} instances", flush=True)

    hd_by_n: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(yamls):
        n = _size_from_name(p.name)
        if n is None:
            continue
        k = KnapsackInstance(p)
        greedy_sol = greedy_solver(k)
        try:
            opt_sol = gurobi.gurobi_optimal_solution(k)
        except Exception as e:  # noqa: BLE001 - keep going on solver failure
            print(f"    skip {p.name}: {e}", flush=True)
            continue
        hd_by_n[k.num_items].append(hamming(greedy_sol.bitstring, opt_sol.bitstring))
        if (i + 1) % 50 == 0 or i == len(yamls) - 1:
            print(f"    [{i + 1}/{len(yamls)}]", flush=True)

    return hd_by_n


def main():
    gurobi = GurobiSolver()

    results = {}
    for label, config_dir in DATASETS.items():
        if not config_dir.is_dir():
            print(f"WARNING: missing dataset dir {config_dir}", flush=True)
            continue
        print(f"Analyzing '{label}' ...", flush=True)
        results[label] = analyze_dataset(config_dir, gurobi)

    if not results:
        print("No data collected - aborting.")
        return

    # --- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, hd_by_n in results.items():
        ns = sorted(hd_by_n)
        means = np.array([np.mean(hd_by_n[n]) for n in ns])
        sems = np.array([
            np.std(hd_by_n[n], ddof=1) / np.sqrt(len(hd_by_n[n]))
            if len(hd_by_n[n]) > 1 else 0.0
            for n in ns
        ])
        color = DATASET_COLORS.get(label, TUM_DARK_BLUE)
        ax.plot(ns, means, marker="o", color=color, label=label, linewidth=2)
        ax.fill_between(ns, means - sems, means + sems, color=color, alpha=0.2)

    ax.set_xlabel("Knapsack size $n$")
    ax.set_ylabel("Mean Hamming distance (greedy vs. optimum)")
    ax.set_title("Average Hamming distance between greedy and optimal solution")
    ax.grid(alpha=0.3, color=TUM_GRAY)
    ax.legend(frameon=False)
    fig.tight_layout()

    out_png = OUT_DIR / "greedy_vs_optimum_hamming_thesis.png"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"\nSaved plot to: {out_png}", flush=True)

    # --- Dump raw means to CSV --------------------------------------------
    out_csv = OUT_DIR / "greedy_vs_optimum_hamming_thesis.csv"
    with open(out_csv, "w") as f:
        f.write("dataset,n,count,mean_hamming,std_hamming\n")
        for label, hd_by_n in results.items():
            for n in sorted(hd_by_n):
                vals = np.array(hd_by_n[n], dtype=float)
                std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                f.write(f"{label},{n},{len(vals)},{vals.mean():.4f},{std:.4f}\n")
    print(f"Saved data to: {out_csv}", flush=True)

    # --- Dump per-instance Hamming distances (for downstream filtering) ---
    per_inst_csv = OUT_DIR / "greedy_vs_optimum_hamming_per_instance.csv"
    with open(per_inst_csv, "w") as f:
        f.write("dataset,n,hamming\n")
        for label, hd_by_n in results.items():
            for n in sorted(hd_by_n):
                for hd in hd_by_n[n]:
                    f.write(f"{label},{n},{hd}\n")
    print(f"Saved per-instance data to: {per_inst_csv}", flush=True)

    # --- Console summary ---------------------------------------------------
    for label, hd_by_n in results.items():
        allhd = np.concatenate([np.array(v) for v in hd_by_n.values()])
        pct_zero = float((allhd == 0).mean()) * 100
        print(f"\n{label}: N={len(allhd)}  mean HD={allhd.mean():.3f}  "
              f"median={np.median(allhd):.1f}  greedy==opt: {pct_zero:.1f}%")


if __name__ == "__main__":
    main()
