"""
Run from the repository root:
    python src/visualizations/plot_bias_hamming_support.py
"""
import csv
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# --- Paths -----------------------------------------------------------------
OUT_DIR = pathlib.Path(__file__).resolve().parent / "Thesis_Results"
IN_CSV = OUT_DIR / "greedy_vs_optimum_hamming_per_instance.csv"

# --- TUM corporate colours -------------------------------------------------
TUM_BLUE = "#0065BD"        # primary
TUM_ORANGE = "#E37222"      # accent orange
TUM_DARK_BLUE = "#005293"
TUM_GRAY = "#999999"

DATASET_COLORS = {
    "Uncorrelated": TUM_BLUE,
    "Weakly Correlated": TUM_ORANGE,
}

# Filters to suppress noisy points. Small-n bins and bins with only a handful
# of (greedy != optimum) instances have very unstable means; both are dropped.
MIN_N = 10        # ignore knapsack sizes below this value
MIN_COUNT = 10    # ignore n-bins with fewer than this many instances


def load_per_instance(csv_path: pathlib.Path):
    """Return {dataset: {n: [hamming, ...]}}, keeping only Delta > 0."""
    data: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            hd = int(row["hamming"])
            if hd == 0:
                continue  # greedy already optimal -> bias irrelevant
            n = int(row["n"])
            if n < MIN_N:
                continue  # drop small-n outliers
            data[row["dataset"]][n].append(hd)
    # drop n-bins with too few instances to give a stable mean
    for per_n in data.values():
        for n in [n for n, v in per_n.items() if len(v) < MIN_COUNT]:
            del per_n[n]
    return data


def main():
    if not IN_CSV.is_file():
        raise SystemExit(
            f"Missing {IN_CSV}. Run plot_greedy_vs_opt_hamming_thesis.py first."
        )
    data = load_per_instance(IN_CSV)

    fig, ax = plt.subplots(figsize=(9, 6))

    for label, per_n in data.items():
        ns = np.array(sorted(per_n))
        counts = np.array([len(per_n[n]) for n in ns])
        means = np.array([np.mean(per_n[n]) for n in ns])
        sems = np.array([
            np.std(per_n[n], ddof=1) / np.sqrt(len(per_n[n]))
            if len(per_n[n]) > 1 else 0.0
            for n in ns
        ])

        color = DATASET_COLORS.get(label, TUM_DARK_BLUE)
        ax.plot(ns, means, marker="o", markersize=6, color=color,
                linewidth=2, label=label)
        ax.fill_between(ns, means - sems, means + sems, color=color, alpha=0.2)

    ax.set_xlabel("Knapsack size $n$", fontsize=14)
    ax.set_ylabel(r"Hamming distance $\Delta(x^{\mathrm{greedy}}, x^{*})$",
                  fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.margins(x=0.08)
    ax.grid(alpha=0.3, color=TUM_GRAY)
    ax.legend(frameon=False, loc="upper left", fontsize=13)
    fig.tight_layout()

    out_png = OUT_DIR / "bias_hamming_support_thesis.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved plot to: {out_png}")

    # console summary
    for label, per_n in data.items():
        allhd = np.concatenate([np.array(v) for v in per_n.values()])
        ns = sorted(per_n)
        counts = np.array([len(per_n[n]) for n in ns])
        means = np.array([np.mean(per_n[n]) for n in ns])
        pooled = float(np.sum(counts * means) / np.sum(counts))
        print(f"{label:>18}: N={len(allhd)} (greedy != opt)  "
              f"pooled mean Delta = {pooled:.2f}  "
              f"(range {means.min():.2f}-{means.max():.2f} over n={ns[0]}-{ns[-1]})")


if __name__ == "__main__":
    main()
