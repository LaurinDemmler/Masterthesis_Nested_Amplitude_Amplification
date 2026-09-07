import argparse
import pathlib
import sys

# Allow `python src/main.py` from any working directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pipelines.runner import run

# Every run behind the thesis figures.
THESIS_RUNS = [
    # Bias sweep -> bias_sweep_{PBC,PCBC}_{uncorrelated,weakly_correlated}
    ("simulate_capweight_bias_sweep", 1),                    # uncorrelated, nested (PBC)
    ("simulate_capweight_bias_sweep", 3),                    # uncorrelated, cut (PCBC)
    ("simulate_capweight_bias_sweep", 4),                    # weakly correlated, nested (PBC)
    ("simulate_capweight_bias_sweep", 5),                    # weakly correlated, cut (PCBC)

    # Capweight, biased -> inner_iterations_conditional_best_depth_*
    ("simulate_capweight_multi_method_gate_based", 7),       # uncorrelated
    ("simulate_capweight_multi_method_gate_based", 8),       # weakly correlated

    # Capweight, unbiased -> unbiased_capweight_performance_*
    ("simulate_capweight_multi_method_gate_based", 9),       # uncorrelated
    ("simulate_capweight_multi_method_gate_based", 10),      # weakly correlated

    # Depth fraction -> depth_fraction_advantage_*,
    # best_depth_fraction_distribution_*, depth_candidate_coverage_*
    ("simulate_depth_fraction_multi_method_gate_based", 1),  # uncorrelated
    ("simulate_depth_fraction_multi_method_gate_based", 2),  # weakly correlated

    # Fixed cost, biased -> fixed_cost_performance_*
    ("simulate_fixed_cost_multi_method_gate_based", 1),      # uncorrelated
    ("simulate_fixed_cost_multi_method_gate_based", 2),      # weakly correlated

    # Fixed cost, unbiased -> fixed_cost_performance_unbiased_*,
    # unbiased_size_scaling_*
    ("simulate_fixed_cost_multi_method_gate_based", 3),      # uncorrelated
    ("simulate_fixed_cost_multi_method_gate_based", 4),      # weakly correlated
]


def main():
    parser = argparse.ArgumentParser(description="Run knapsack simulation experiments.")
    parser.add_argument("-e", "--experiment", help="Experiment name, e.g. simulate_fixed_cost_multi_method_gate_based.")
    parser.add_argument("-i", "--id", help="Experiment id within that experiment's JSON config.")
    parser.add_argument("--all", action="store_true", help="Run every experiment behind the thesis figures (expensive).")
    args = parser.parse_args()

    if args.all:
        for experiment, id in THESIS_RUNS:
            run(experiment, id)
    elif args.experiment and args.id:
        run(args.experiment, args.id)
    else:
        parser.error("give both --experiment and --id, or --all")


if __name__ == "__main__":
    main()
