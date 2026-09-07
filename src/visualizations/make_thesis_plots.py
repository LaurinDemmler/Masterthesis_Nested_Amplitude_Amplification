"""Generate the thesis figures without running the analysis notebooks.

Reproduces the figures currently used in the thesis:

* bias sweep  -> ``bias_sweep_<METHOD>_<CORR>.pdf/.png``
* conditional best depth (inner iterations)
  -> ``inner_iterations_conditional_best_depth_<METHOD>_<CORR>.pdf/.png``
* depth candidate coverage
  -> ``best_depth_fraction_distribution_<METHOD>_<CORR>.pdf/.png`` and
     ``depth_candidate_coverage_<METHOD>_<CORR>.pdf/.png``
* gatecount ratio vs. depth fraction
  -> ``depth_fraction_advantage_<METHOD>_<CORR>.pdf/.png``

All figure families are written by ``thesis_nested_plots`` into
``src/visualizations/Thesis_Results`` - the same location the notebooks use.

Usage (from anywhere):

    python src/visualizations/make_thesis_plots.py
    python src/visualizations/make_thesis_plots.py --dataset uncorrelated
    python src/visualizations/make_thesis_plots.py --figures bias
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

# Non-interactive backend: plt.show() inside the plotting helpers becomes a no-op
# so the script never blocks on a GUI window.
matplotlib.use("Agg")

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS = REPO_ROOT / "log" / "results"

# thesis_nested_plots imports its siblings (plot_bias_sweep, ...) by bare name.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import thesis_nested_plots as tnp  # noqa: E402


# Dataset -> input CSVs, mirroring the two narrative notebooks.
DATASETS = {
    "uncorrelated": {
        "bias_sweep": [
            RESULTS / "capweight_bias_sweep" / "bias_sweep_capweight_results_id1.csv",
        ],
        "capweight_multi": (
            RESULTS
            / "capweight_multi_method_gate_based"
            / "capweight_multi_method_results_id7.csv"
        ),
        "depth_fraction": (
            RESULTS
            / "depth_fraction_multi_method_gate_based"
            / "depth_fraction_multi_method_results_id1.csv"
        ),
        "fixed_cost": (
            RESULTS
            / "fixed_cost_multi_method_gate_based"
            / "fixed_cost_multi_method_results_id1.csv"
        ),
        "unbiased_fixed_cost": (
            RESULTS
            / "fixed_cost_multi_method_gate_based"
            / "fixed_cost_multi_method_results_id3.csv"
        ),
        "unbiased_capweight": (
            RESULTS
            / "capweight_multi_method_gate_based"
            / "capweight_multi_method_results_id9.csv"
        ),
    },
    "weakly_correlated": {
        "bias_sweep": [
            RESULTS / "capweight_bias_sweep" / "bias_sweep_capweight_results_id4.csv",
        ],
        "capweight_multi": (
            RESULTS
            / "capweight_multi_method_gate_based"
            / "capweight_multi_method_results_id8.csv"
        ),
        "depth_fraction": (
            RESULTS
            / "depth_fraction_multi_method_gate_based"
            / "depth_fraction_multi_method_results_id2.csv"
        ),
        "fixed_cost": (
            RESULTS
            / "fixed_cost_multi_method_gate_based"
            / "fixed_cost_multi_method_results_id2.csv"
        ),
        "unbiased_fixed_cost": (
            RESULTS
            / "fixed_cost_multi_method_gate_based"
            / "fixed_cost_multi_method_results_id4.csv"
        ),
        "unbiased_capweight": (
            RESULTS
            / "capweight_multi_method_gate_based"
            / "capweight_multi_method_results_id10.csv"
        ),
    },
}


def _check(path: pathlib.Path) -> bool:
    if path.exists():
        return True
    print(f"  [skip] missing input file: {path}")
    return False


def run_bias_sweep(dataset: str, save_dir: pathlib.Path | None) -> None:
    print(f"\n### Bias sweep - {dataset}")
    for csv_path in DATASETS[dataset]["bias_sweep"]:
        if not _check(csv_path):
            continue
        print(f"  source: {csv_path.name}")
        tnp.plot_figure1_bias_sweep(str(csv_path), save_dir=save_dir)


def run_conditional_best_depth(dataset: str, save_dir: pathlib.Path | None) -> None:
    print(f"\n### Conditional best depth (inner iterations) - {dataset}")
    csv_path = DATASETS[dataset]["capweight_multi"]
    if not _check(csv_path):
        return
    print(f"  source: {csv_path.name}")
    _, df_valid = tnp.load_and_prepare_capweight_data(str(csv_path))
    tnp.plot_figure3c_conditional_best_depth(df_valid, save_dir=save_dir)


def run_candidate_coverage(dataset: str, save_dir: pathlib.Path | None) -> None:
    print(f"\n### Depth candidate coverage - {dataset}")
    csv_path = DATASETS[dataset]["depth_fraction"]
    if not _check(csv_path):
        return
    print(f"  source: {csv_path.name}")
    df_depth = tnp.analyze_depth_correlations(str(csv_path))
    tnp.plot_figure5_candidate_coverage(df_depth, save_dir=save_dir)


def run_depth_fraction(dataset: str, save_dir: pathlib.Path | None) -> None:
    print(f"\n### Gatecount ratio vs. depth fraction - {dataset}")
    csv_path = DATASETS[dataset]["depth_fraction"]
    if not _check(csv_path):
        return
    print(f"  source: {csv_path.name}")
    tnp.plot_figure4_depth_fraction(str(csv_path), save_dir=save_dir)


def run_fixed_cost(dataset: str, save_dir: pathlib.Path | None) -> None:
    print(f"\n### Fixed cost performance - {dataset}")
    csv_path = DATASETS[dataset]["fixed_cost"]
    if csv_path is None:
        print(f"  [skip] no fixed cost data for {dataset}")
        return
    if not _check(csv_path):
        return
    print(f"  source: {csv_path.name}")
    tnp.plot_figure6_fixed_cost(str(csv_path), save_dir=save_dir)


def run_fixed_budget_pbc_only(save_dir: pathlib.Path | None) -> None:
    """Fixed budget plot for the unbiased uncorrelated run (ID3)."""
    print("\n### Fixed budget unbiased (uncorrelated)")
    csv_path = DATASETS["uncorrelated"]["unbiased_fixed_cost"]
    if not _check(csv_path):
        return
    print(f"  source: {csv_path.name}")
    tnp.plot_figure6_fixed_cost(
        str(csv_path),
        save_dir=save_dir,
        filename_suffix='unbiased_uncorrelated_PBC_only',
    )


def run_fixed_budget_pcbc_only(save_dir: pathlib.Path | None) -> None:
    """Fixed budget plot for the unbiased weakly correlated run (ID4), capweight <= 0.5."""
    print("\n### Fixed budget unbiased (weakly correlated, capweight <= 0.5)")
    csv_path = DATASETS["weakly_correlated"]["unbiased_fixed_cost"]
    if not _check(csv_path):
        return
    print(f"  source: {csv_path.name}")
    tnp.plot_figure6_fixed_cost(
        str(csv_path),
        save_dir=save_dir,
        max_capweight=0.5,
        filename_suffix='unbiased_weakly_correlated_PCBC_only_cw05',
    )


def run_size_scaling(save_dir: pathlib.Path | None) -> None:
    """Budget-exponent saving over the knapsack size, for both subset settings."""
    print("\n### Size scaling of the budget-exponent saving")
    unc = DATASETS["uncorrelated"]["unbiased_fixed_cost"]
    if _check(unc):
        print(f"  source: {unc.name}")
        tnp.plot_unbiased_size_scaling(
            str(unc),
            method='nested',
            save_dir=save_dir,
            auto_scope='global',
            filename_suffix='PBC_uncorrelated',
        )
    weak = DATASETS["weakly_correlated"]["unbiased_fixed_cost"]
    if _check(weak):
        print(f"  source: {weak.name}")
        tnp.plot_unbiased_size_scaling(
            str(weak),
            method='cut',
            save_dir=save_dir,
            max_capweight=0.5,
            auto_scope='global',
            filename_suffix='PCBC_weakly_correlated_cw05',
        )


def run_unbiased_capweight(dataset: str, save_dir: pathlib.Path | None) -> None:
    print(f"\n### Unbiased capweight performance - {dataset}")
    csv_path = DATASETS[dataset]["unbiased_capweight"]
    if csv_path is None:
        print(f"  [skip] no unbiased capweight data for {dataset}")
        return
    if not _check(csv_path):
        return
    print(f"  source: {csv_path.name}")
    tnp.plot_unbiased_capweight(str(csv_path), save_dir=save_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        choices=["uncorrelated", "weakly_correlated", "all"],
        default="all",
        help="Which instance family to plot (default: all).",
    )
    parser.add_argument(
        "--figures",
        choices=["bias", "inner", "coverage", "depth", "fixed", "fixed_subset",
                 "unbiased", "size_scaling", "all"],
        default="all",
        help="Which figure family to produce (default: all).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help=f"Output directory (default: {tnp.THESIS_RESULTS_DIR}).",
    )
    args = parser.parse_args()

    save_dir = pathlib.Path(args.save_dir) if args.save_dir else None
    out_dir = save_dir if save_dir is not None else tnp.THESIS_RESULTS_DIR
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    tnp.setup_latex_style()

    for dataset in datasets:
        if args.figures in ("bias", "all"):
            run_bias_sweep(dataset, save_dir)
        if args.figures in ("inner", "all"):
            run_conditional_best_depth(dataset, save_dir)
        if args.figures in ("coverage", "all"):
            run_candidate_coverage(dataset, save_dir)
        if args.figures in ("depth", "all"):
            run_depth_fraction(dataset, save_dir)
        if args.figures in ("fixed", "all"):
            run_fixed_cost(dataset, save_dir)
        if args.figures in ("unbiased", "all"):
            run_unbiased_capweight(dataset, save_dir)

    # Fixed budget subset plots (not dataset-specific)
    if args.figures in ("fixed_subset", "all"):
        run_fixed_budget_pbc_only(save_dir)
        run_fixed_budget_pcbc_only(save_dir)
    if args.figures in ("size_scaling", "all"):
        run_size_scaling(save_dir)

    print(f"\nDone. Figures written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
