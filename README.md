# Masterthesis Laurin Demmler - A Nested Amplitude Amplification Protocol for the binary Knapsack Problem

This is the codebase for my thesis. It includes the minimal code examples for all generated results. It does not include all the code I produced during my research to get to where it is now. That code is available upon request.

## Setup

Python 3.11. Gurobi (`gurobipy`, size-limited license suffices for n <= 35) is required to run
experiments. A working LaTeX installation (`latex` + `dvipng` on PATH) is required for plotting.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Plotting

All result CSVs from the thesis are available here, so figures regenerate without re-running any simulation:

```powershell
python src/visualizations/make_thesis_plots.py
```

Options: `--dataset {uncorrelated,weakly_correlated,all}`, `--figures {bias,inner,coverage,depth,fixed,fixed_subset,unbiased,size_scaling,all}`, `--save-dir PATH`.

Output (`.pdf` + `.png`): [src/visualizations/Thesis_Results](src/visualizations/Thesis_Results).

The figures regarding the Hamming distance evaluation are standalone (they recompute from instance YAMLs via Gurobi) and must be run from `src/`, in this order:

```powershell
cd src
python visualizations/plot_greedy_vs_opt_hamming_thesis.py
python visualizations/plot_bias_hamming_support.py
```

## Running experiments

Entry point: [src/main.py](src/main.py). Experiment names and IDs are argparsed

```powershell
python src/main.py --experiment simulate_fixed_cost_multi_method_gate_based --id 1
```
or for all experiments
```powershell
python src/main.py --all
```

Depending on the experiment and the number of instances simulated this can take hours.

### Figure -> experiment ID mapping

`<METHOD>` = `PBC` (nested) / `PCBC` (cut), `<CORR>` = `uncorrelated` / `weakly_correlated`.

| Figure family | `--experiment` | `--id` |
|---|---|---|
| `bias_sweep_PBC_<CORR>` | `simulate_capweight_bias_sweep` | 1 / 4 |
| `bias_sweep_PCBC_<CORR>` | `simulate_capweight_bias_sweep` | 3 / 5 |
| `inner_iterations_conditional_best_depth_<METHOD>_<CORR>` | `simulate_capweight_multi_method_gate_based` | 7 / 8 |
| `unbiased_capweight_performance_<CORR>` | `simulate_capweight_multi_method_gate_based` | 9 / 10 |
| `depth_fraction_advantage_*`, `best_depth_fraction_distribution_*`, `depth_candidate_coverage_*` | `simulate_depth_fraction_multi_method_gate_based` | 1 / 2 |
| `fixed_cost_performance_<CORR>` | `simulate_fixed_cost_multi_method_gate_based` | 1 / 2 |
| `fixed_cost_performance_unbiased_*`, `unbiased_size_scaling_*` | `simulate_fixed_cost_multi_method_gate_based` | 3 / 4 |

In each row the first id is uncorrelated, the second weakly correlated. Experiment definitions
(instance folder, sample count, bias factors) live in [config/experimentalConfigs](config/experimentalConfigs). Biases and the number of instances to be simulated should be changed here.


## Instance generation

[src/generator/InstanceGenerator](src/generator/InstanceGenerator). Pisinger's generator
(`InstanceGenerator.c`) must be compiled first - the binary is git-ignored.
`InstanceBatchGenerator.py` shells out to it and writes YAML.

Its `__main__` block runs a hardcoded batch; adapt the parameters (type, n range, r, series,
count, output folder) to your needs. 

## Datasets

YAML instances:

| Folder | Count | Contents |
|---|---|---|
| [config/Paper_unc_10_35](config/Paper_unc_10_35) | 1082 | uncorrelated, n = 10..35, r = 400 |
| [config/Paper_weak_10_35](config/Paper_weak_10_35) | 1082 | weakly correlated, n = 10..35, r = 400 |
| [config/Debug_bias_weak_10_20](config/Debug_bias_weak_10_20) | 464 | weakly correlated, n = 10..20; bias sweep only for runtime reasons |


## Results

CSVs in `log/results/<experiment_family>/<family>_results_id<ID>.csv`. The ones used by the
thesis figures:

| CSV | |
|---|---|
| `capweight_bias_sweep/bias_sweep_capweight_results_id{1,4}.csv` | bias sweep |
| `capweight_multi_method_gate_based/capweight_multi_method_results_id{7,8,9,10}.csv` | capacity/weight ratio |
| `depth_fraction_multi_method_gate_based/depth_fraction_multi_method_results_id{1,2}.csv` | depth fraction |
| `fixed_cost_multi_method_gate_based/fixed_cost_multi_method_results_id{1,2,3,4}.csv` | fixed gate budget |


