import json
import pathlib
from datastructures.knapsack import KnapsackInstance
from generator.GurobiSolver import GurobiSolver
from generator.QTGHotStarter import QTGHotStarter
from grover.grover import Grover
from Simulator.MultiMethodSimulator import MultiMethodSimulator

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _resolve(path_from_config: str) -> pathlib.Path:
    """Config paths are repo-root relative; absolute paths are still honoured."""
    path = pathlib.Path(path_from_config)
    return path if path.is_absolute() else REPO_ROOT / path


def run(experiment_name, id):
    id = str(id)
    experiments_json = CONFIG_DIR / "experimentalConfigs" / f"{experiment_name}.json"
    instancepath, resultpath, settings = _get_experiment_config(experiments_json, id)
    instancepath = _resolve(instancepath)
    resultpath = _resolve(resultpath)

    print("=" * 70)
    print(f"Experiment : {experiment_name}")
    print(f"Id         : {id}")
    print(f"Config     : {experiments_json}")
    print(f"Instances  : {instancepath}")
    print(f"Results    : {resultpath}")
    print("Settings   :")
    for key, value in settings.items():
        print(f"    {key:<28}= {value}")
    print("=" * 70)

    msim = MultiMethodSimulator(path_to_instances=pathlib.Path(instancepath), path_to_results=pathlib.Path(resultpath))

    if experiment_name in ("simulate_capweight_multi_method", "simulate_capweight_multi_method_gate_based"):
        msim.simulate_capweight_multi_method(**settings, id=id)
    elif experiment_name in ("simulate_remainingvalue_multi_method", "simulate_remainingvalue_multi_method_gate_based"):
        msim.simulate_remainingvalue_multi_method(**settings, id=id)
    elif experiment_name in ("simulate_depth_fraction_multi_method", "simulate_depth_fraction_multi_method_gate_based"):
        msim.simulate_depth_fraction_multi_method(**settings, id=id)
    elif experiment_name == "simulate_capweight_bias_sweep":
        msim.simulate_bias_sweep_capweight(**settings, id=id)
    elif experiment_name in ("simulate_fixed_cost_multi_method", "simulate_fixed_cost_multi_method_gate_based"):
        msim.simulate_fixed_cost_multi_method(**settings, id=id)
    else:
        raise ValueError(f"Experiment name '{experiment_name}' is not recognized.")

    print(f"Completed experiment '{experiment_name}' with id '{id}'. Results saved to {resultpath}.")


def _get_experiment_config(experiments_json_path: pathlib.Path, experiment_id: str):
    with experiments_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data.get("experiments", [])
    exp = next((e for e in experiments if str(e.get("id")) == str(experiment_id)), None)
    if exp is None:
        raise ValueError(f"Experiment id '{experiment_id}' not found in {experiments_json_path}")

    settings = {k: v for k, v in exp.items() if k != "id"}
    instance = settings.pop("path_to_instances")
    resultsfile = settings.pop("path_to_results")
    return (instance, resultsfile, settings)