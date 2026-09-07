import pathlib
import sys

# Allow `python src/main.py` from any working directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pipelines.runner import run


if __name__ == "__main__":
    """# Bias sweep -> bias_sweep_{PBC,PCBC}_{uncorrelated,weakly_correlated}
    run("simulate_capweight_bias_sweep", 1)   # uncorrelated, nested (PBC)
    run("simulate_capweight_bias_sweep", 3)   # uncorrelated, cut (PCBC)
    run("simulate_capweight_bias_sweep", 4)   # weakly correlated, nested (PBC)
    run("simulate_capweight_bias_sweep", 5)   # weakly correlated, cut (PCBC)"""

    # Capweight, biased -> inner_iterations_conditional_best_depth_*
    run("simulate_capweight_multi_method_gate_based", 7)   # uncorrelated
    run("simulate_capweight_multi_method_gate_based", 8)   # weakly correlated

    # Capweight, unbiased -> unbiased_capweight_performance_*
    run("simulate_capweight_multi_method_gate_based", 9)   # uncorrelated
    run("simulate_capweight_multi_method_gate_based", 10)  # weakly correlated

    # Depth fraction -> depth_fraction_advantage_*,
    # best_depth_fraction_distribution_*, depth_candidate_coverage_*
    run("simulate_depth_fraction_multi_method_gate_based", 1)  # uncorrelated
    run("simulate_depth_fraction_multi_method_gate_based", 2)  # weakly correlated

    # Fixed cost, biased -> fixed_cost_performance_*
    run("simulate_fixed_cost_multi_method_gate_based", 1)  # uncorrelated
    run("simulate_fixed_cost_multi_method_gate_based", 2)  # weakly correlated

    # Fixed cost, unbiased -> fixed_cost_performance_unbiased_*,
    # unbiased_size_scaling_*
    run("simulate_fixed_cost_multi_method_gate_based", 3)  # uncorrelated
    run("simulate_fixed_cost_multi_method_gate_based", 4)  # weakly correlated
