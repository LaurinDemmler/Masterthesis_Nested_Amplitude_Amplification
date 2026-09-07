from dataclasses import asdict, dataclass, field
import random
import pathlib
import csv

import tqdm
from generator.QTGHotStarter import QTGHotStarter
from generator.InstanceGenerator.InstanceFilter import InstanceFilter
from datastructures.OptimizerSolution import OptimizerSolution
from datastructures.knapsack import KnapsackInstance
from grover.grover import Grover
from Simulator.Solver import (_prepare_knapsack,
                              cost_factor_nested, cost_factor_cut,
                              compute_inner_cost, compute_outer_cost, compute_global_cost,
                              compute_inner_gatecost, compute_outer_gatecost, compute_global_gatecost,
                              compute_inner_gatecost_cut, compute_outer_gatecost_cut,
                              build_resource_estimator,
                              find_depth_for_remaining_value,
                              StatisticalSmartSolver)
from generator.GurobiSolver import _solution_from_bitstring
from typing import List, Union


@dataclass
class MultiMethodResult:
    instance_name: str
    knapsack_size: int
    bias: float
    depth: int
    capweight: float
    T: int = -1
    remaining_value_ratio: float = -1.0

    # --- Global ---
    global_cost: int = -1
    optimal_global_iterations_classically: List[int] = field(default_factory=list)
    global_statistics: List[int] = field(default_factory=list)
    to_optimality_global: bool = False

    # --- Nested ---
    nested_cost: int = -1
    optimal_inner_iterations_classically: List[int] = field(default_factory=list)
    optimal_outer_iterations_classically: List[int] = field(default_factory=list)
    optimal_inner_iterations: List[int] = field(default_factory=list)
    inner_statistics: List[int] = field(default_factory=list)
    outer_statistics: List[int] = field(default_factory=list)
    optimal_nested_cost: int = -1
    optimal_global_cost: int = -1
    to_optimality_nested: bool = False

    # --- Cut ---
    cut_cost: int = -1
    cut_inner_iterations: List[int] = field(default_factory=list)
    cut_inner_statistics: List[int] = field(default_factory=list)
    cut_outer_statistics: List[int] = field(default_factory=list)
    cut_optimal_inner_classically: List[int] = field(default_factory=list)
    cut_optimal_outer_classically: List[int] = field(default_factory=list)
    to_optimality_cut: bool = False

    # --- Gate-count costs (new) ---
    global_gatecost: int = -1
    nested_gatecost: int = -1
    cut_gatecost: int = -1

    greedy_is_optimal: bool = False


@dataclass
class MultiMethodApproximationResult:
    instance_name: str
    knapsack_size: int
    bias: float
    capweight: float
    threshold_cost: int
    termination_cost_constant: int
    termination_cost_exponent: float
    approximation_ratio_greedy: float = -1.0
    greedy_is_optimal: bool = False

    # --- Global ---
    global_cost: int = -1
    optimal_global_iterations_classically: List[int] = field(default_factory=list)
    global_statistics: List[int] = field(default_factory=list)
    approximation_ratio_global: float = -1.0
    better_than_greedy_global: bool = False
    to_optimality_global: bool = False
    global_stepwise_cost: List[int] = field(default_factory=list)

    # --- Nested ---
    nested_cost: int = -1
    optimal_inner_iterations_classically: List[int] = field(default_factory=list)
    optimal_outer_iterations_classically: List[int] = field(default_factory=list)
    optimal_inner_iterations: List[int] = field(default_factory=list)
    inner_statistics: List[int] = field(default_factory=list)
    outer_statistics: List[int] = field(default_factory=list)
    smart_global_statistics: List[int] = field(default_factory=list)
    approximation_ratio_nested: float = -1.0
    better_than_greedy_nested: bool = False
    to_optimality_nested: bool = False
    nested_stepwise_cost: List[int] = field(default_factory=list)


    # --- Cut ---
    cut_cost: int = -1
    cut_inner_iterations: List[int] = field(default_factory=list)
    cut_inner_statistics: List[int] = field(default_factory=list)
    cut_outer_statistics: List[int] = field(default_factory=list)
    cut_optimal_inner_classically: List[int] = field(default_factory=list)
    cut_optimal_outer_classically: List[int] = field(default_factory=list)
    approximation_ratio_cut: float = -1.0
    better_than_greedy_cut: bool = False
    to_optimality_cut: bool = False
    cut_stepwise_cost: List[int] = field(default_factory=list)


@dataclass
class MultiMethodBiasSweepResult:
    instance_name: str
    knapsack_size: int
    global_bias: float
    nested_bias_inner: float
    nested_bias_outer: float
    cut_bias_inner: float
    cut_bias_outer: float
    bias_label: str = ""
    depth: int = -1
    capweight: float = -1.0
    T: int = -1
    remaining_value_ratio: float = -1.0

    # --- Global ---
    global_cost: int = -1
    optimal_global_iterations_classically: List[int] = field(default_factory=list)
    global_statistics: List[int] = field(default_factory=list)
    to_optimality_global: bool = False

    # --- Nested ---
    nested_cost: int = -1
    optimal_inner_iterations_classically: List[int] = field(default_factory=list)
    optimal_outer_iterations_classically: List[int] = field(default_factory=list)
    optimal_inner_iterations: List[int] = field(default_factory=list)
    inner_statistics: List[int] = field(default_factory=list)
    outer_statistics: List[int] = field(default_factory=list)
    optimal_nested_cost: int = -1
    optimal_global_cost: int = -1
    to_optimality_nested: bool = False

    # --- Cut ---
    cut_cost: int = -1
    cut_inner_iterations: List[int] = field(default_factory=list)
    cut_inner_statistics: List[int] = field(default_factory=list)
    cut_outer_statistics: List[int] = field(default_factory=list)
    cut_optimal_inner_classically: List[int] = field(default_factory=list)
    cut_optimal_outer_classically: List[int] = field(default_factory=list)
    to_optimality_cut: bool = False

    # --- Gate-count costs ---
    global_gatecost: int = -1
    nested_gatecost: int = -1
    cut_gatecost: int = -1

    greedy_is_optimal: bool = False


class MultiMethodSimulator:
    """Runs multiple solving methods on the same knapsack instances and records
    comparable results."""

    def __init__(self, path_to_instances, path_to_results):
        self.path_to_instances = pathlib.Path(path_to_instances)
        self.path_to_results = pathlib.Path(path_to_results)

    def simulate_capweight_multi_method(self, number_of_samples: int,
                                        global_bias_factor: float,
                                        nested_inner_bias_factor: float,
                                        nested_outer_bias_factor: float,
                                        cut_inner_bias_factor: float,
                                        cut_outer_bias_factor: float,
                                        order: str = "value",
                                        id=None):
        """Compare nested, global, and BnB across depth fractions, keyed by capweight ratio."""
        yaml_paths = list(self.path_to_instances.rglob("*.yml")) + list(self.path_to_instances.rglob("*.yaml"))
        if not yaml_paths:
            raise FileNotFoundError(f"No YAML files found under {self.path_to_instances}")
        if number_of_samples > len(yaml_paths) or number_of_samples <= 0:
            raise ValueError(f"Requested {number_of_samples} samples but only found {len(yaml_paths)} YAML files.")
        sampled_paths = random.sample(yaml_paths, number_of_samples)

        results: list[MultiMethodResult] = []
        print("------------------------------")
        print(f"Multi-method capweight simulation on {number_of_samples} instances...")

        for yaml_path in tqdm.tqdm(sampled_paths):
            knapsack = KnapsackInstance(yaml_path)
            bias = global_bias_factor * knapsack.num_items
            bias_in_nested = nested_inner_bias_factor * knapsack.num_items
            bias_out_nested = nested_outer_bias_factor * knapsack.num_items
            bias_in_cut = cut_inner_bias_factor * knapsack.num_items
            bias_out_cut = cut_outer_bias_factor * knapsack.num_items
            hot_starter = QTGHotStarter(knapsack, depth=knapsack.num_items)
            optimal_solution = hot_starter.get_optimal_solution()
            greedy_solution = hot_starter.greedy_solution
            greedy_is_optimal = bool(greedy_solution.total_value == optimal_solution.total_value)

            if greedy_is_optimal:
                continue

            capweight_ratio = knapsack.capacity / sum(item.weight for item in knapsack.items)

            for depth_fraction in [x * 0.1 for x in range(1, 9)]:
                depth = int(depth_fraction * knapsack.num_items)
                result = self._unitSim_statistics_multi(
                    knapsack, bias, bias_in_nested, bias_out_nested, bias_in_cut, bias_out_cut, depth, order=order,
                    current_best_solution=greedy_solution,
                    instance_name=yaml_path.stem,
                    capweight=capweight_ratio,
                    greedy_is_optimal=greedy_is_optimal,
                )
                results.append(result)

        if id is None:
            raise ValueError("An 'id' must be provided to save the results.")
        result_file = self.path_to_results / f"capweight_multi_method_results_id{id}.csv"
        _write_results_csv(results, result_file)

    def simulate_remainingvalue_multi_method(self, number_of_samples: int,
                                             global_bias_factor: float,
                                             nested_inner_bias_factor: float,
                                             nested_outer_bias_factor: float,
                                             cut_inner_bias_factor: float,
                                             cut_outer_bias_factor: float,
                                             order: str = "value",
                                             id=None):
        """Compare nested, global, and BnB across remaining-value ratios and T values."""
        yaml_paths = list(self.path_to_instances.rglob("*.yml")) + list(self.path_to_instances.rglob("*.yaml"))
        if not yaml_paths:
            raise FileNotFoundError(f"No YAML files found under {self.path_to_instances}")
        if number_of_samples > len(yaml_paths) or number_of_samples <= 0:
            raise ValueError(f"Requested {number_of_samples} samples but only found {len(yaml_paths)} YAML files.")
        sampled_paths = random.sample(yaml_paths, number_of_samples)

        results: list[MultiMethodResult] = []
        print("------------------------------")
        print(f"Multi-method remaining-value simulation on {number_of_samples} instances...")

        for yaml_path in tqdm.tqdm(sampled_paths):
            knapsack = KnapsackInstance(yaml_path)
            bias = global_bias_factor * knapsack.num_items
            bias_in_nested = nested_inner_bias_factor * knapsack.num_items
            bias_out_nested = nested_outer_bias_factor * knapsack.num_items
            bias_in_cut = cut_inner_bias_factor * knapsack.num_items
            bias_out_cut = cut_outer_bias_factor * knapsack.num_items
            hot_starter = QTGHotStarter(knapsack, depth=knapsack.num_items)
            optimal_solution = hot_starter.get_optimal_solution()
            greedy_solution = hot_starter.greedy_solution
            greedy_is_optimal = bool(greedy_solution.total_value == optimal_solution.total_value)
            seen_T_values = set()
            for T in range(int(1 * greedy_solution.total_value), int(optimal_solution.total_value),
                           int((optimal_solution.total_value - greedy_solution.total_value) / 5) + 1):
                feasible_solution_T = hot_starter.find_solution_with_value_T(T)
                if feasible_solution_T is None:
                    raise ValueError(f"No feasible solution found with value at least T={T} for instance {yaml_path.stem}.")
                actual_T = feasible_solution_T.total_value
                if actual_T in seen_T_values:
                    continue
                seen_T_values.add(actual_T)
                for remaining_value_ratio in [x * 0.1 for x in range(0, 11)]:
                    target_remaining_value = remaining_value_ratio * actual_T
                    depth = find_depth_for_remaining_value(knapsack, target_remaining_value)
                    if depth < 1 or depth >= knapsack.num_items:
                        continue
                    instance_sort = InstanceFilter(knapsack) 
                    #CURRENTLY ALL CAPWEIGHTS
                    if greedy_is_optimal and not instance_sort.filter_capweight_ratio_range(0.6, 1):
                        continue
                    actual_remaining_value_ratio = knapsack.get_remaining_value(depth) / actual_T
                    capweight = knapsack.capacity / sum(item.weight for item in knapsack.items)

                    result = self._unitSim_statistics_multi(
                        knapsack, bias, bias_in_nested, bias_out_nested, bias_in_cut, bias_out_cut, depth, order=order,
                        current_best_solution=feasible_solution_T,
                        instance_name=yaml_path.stem,
                        capweight=capweight,
                        greedy_is_optimal=greedy_is_optimal,
                        optimality_threshold=T,
                        T=actual_T,
                        remaining_value_ratio=actual_remaining_value_ratio,
                    )
                    results.append(result)

        if id is None:
            raise ValueError("An 'id' must be provided to save the results.")
        result_file = self.path_to_results / f"remainingvalue_multi_method_results_id{id}.csv"
        _write_results_csv(results, result_file)

    def simulate_depth_fraction_multi_method(self, number_of_samples: int,
                                             global_bias_factor: float,
                                             nested_inner_bias_factor: float,
                                             nested_outer_bias_factor: float,
                                             cut_inner_bias_factor: float,
                                             cut_outer_bias_factor: float,
                                             order: str = "value",
                                             id=None):
        """Compare nested, global, and BnB across depth fractions k/n."""
        yaml_paths = list(self.path_to_instances.rglob("*.yml")) + list(self.path_to_instances.rglob("*.yaml"))
        if not yaml_paths:
            raise FileNotFoundError(f"No YAML files found under {self.path_to_instances}")
        if number_of_samples > len(yaml_paths) or number_of_samples <= 0:
            raise ValueError(f"Requested {number_of_samples} samples but only found {len(yaml_paths)} YAML files.")
        sampled_paths = random.sample(yaml_paths, number_of_samples)

        results: list[MultiMethodResult] = []
        print("------------------------------")
        print(f"Multi-method depth-fraction simulation on {number_of_samples} instances...")

        for yaml_path in tqdm.tqdm(sampled_paths):
            knapsack = KnapsackInstance(yaml_path)
            bias = global_bias_factor * knapsack.num_items
            bias_in_nested = nested_inner_bias_factor * knapsack.num_items
            bias_out_nested = nested_outer_bias_factor * knapsack.num_items
            bias_in_cut = cut_inner_bias_factor * knapsack.num_items
            bias_out_cut = cut_outer_bias_factor * knapsack.num_items
            hot_starter = QTGHotStarter(knapsack, depth=knapsack.num_items)
            optimal_solution = hot_starter.get_optimal_solution()
            greedy_solution = hot_starter.greedy_solution
            greedy_is_optimal = bool(greedy_solution.total_value == optimal_solution.total_value)
            capweight = knapsack.capacity / sum(item.weight for item in knapsack.items)
            instance_sort = InstanceFilter(knapsack)
            
            if greedy_is_optimal or not instance_sort.filter_capweight_ratio_range(0, 1.0):
                continue

            

            seen_depths = set()
            for depth_fraction in [x * 0.1 for x in range(1, 9)]:
                depth = int(depth_fraction * knapsack.num_items)
                if depth < 1 or depth >= knapsack.num_items:
                    continue
                if depth in seen_depths:
                    continue
                seen_depths.add(depth)
                result = self._unitSim_statistics_multi(
                    knapsack, bias, bias_in_nested, bias_out_nested, bias_in_cut, bias_out_cut, depth, order=order,
                    current_best_solution=greedy_solution,
                    instance_name=yaml_path.stem,
                    capweight=capweight,
                    greedy_is_optimal=greedy_is_optimal,
                )
                results.append(result)

        if id is None:
            raise ValueError("An 'id' must be provided to save the results.")
        result_file = self.path_to_results / f"depth_fraction_multi_method_results_id{id}.csv"
        _write_results_csv(results, result_file)

    def _unitSim_statistics_multi(
        self,
        knapsack: KnapsackInstance,
        bias: float,
        bias_in_nested: float,
        bias_out_nested: float,
        bias_in_cut: float,
        bias_out_cut: float,
        depth: int,
        order: str = "default",
        current_best_solution: Union[None, OptimizerSolution] = None,
        instance_name: str = "",
        capweight: float = -1.0,
        greedy_is_optimal: bool = False,
        optimality_threshold: Union[None, float] = None,
        T: int = -1,
        remaining_value_ratio: float = -1.0,
    ) -> MultiMethodResult:
        """Single-shot simulation of global, nested, and BnB on one (instance, depth) pair.
        Mirrors _unitSim_statistics but runs all three methods.
        If optimality_threshold is given, use it for the to_optimality checks
        instead of current_best_solution.total_value."""

        threshold = optimality_threshold if optimality_threshold is not None else (
            current_best_solution.total_value if current_best_solution is not None else 0
        )


        # ── 1. Global ────────────────────────────────────────────────────
        ks = _prepare_knapsack(knapsack, "value")
        n = len(ks.items)
        HotStart_global = QTGHotStarter(ks, depth=n, verbose=False, current_best_solution=current_best_solution)
        post_QTG_state_good_global = HotStart_global.hot_start(bias=bias)
        grover_global = Grover(post_QTG_state_good_global)
        _, measurement_global, _ = grover_global.get_result_outer_statistics(current_best_solution, ks)

        global_iterations_statistics = [grover_global.outer_iterations_confidence]
        optimal_global_iterations_classically = [grover_global._iterations]
        global_cost = sum(compute_global_cost(global_iterations_statistics, n))

        if measurement_global is not None:
            sol_global = _solution_from_bitstring(knapsack, measurement_global)
            to_optimality_global = bool(sol_global.total_value > threshold)
        else:
            to_optimality_global = False

        # ── 2. Nested ────────────────────────────────────────────────────
        ks = _prepare_knapsack(knapsack, "value")
        n = len(ks.items)
        HotStart = QTGHotStarter(ks, depth=depth, verbose=False, current_best_solution=current_best_solution)
        post_QTGk_states_good = HotStart.partial_hot_start_QTG_k(bias=bias_in_nested)

        globally_good_states = HotStart.get_globally_marked_states()


        grover1 = Grover(post_QTGk_states_good)
        post_inner_grover_state = grover1.get_result_inner_statistics()
        inner_iterations_statistics = [grover1.inner_iterations_confidence]
        inner_iterations = [grover1.inner_iterations]
        optimal_inner_iterations_classically = [grover1._iterations]

        postQTGnk_states = HotStart.partial_hot_start_QTG_nk(post_inner_grover_state, globally_good_states, bias=bias_out_nested)
        grover2 = Grover(postQTGnk_states)
        _, measurement_nested, _ = grover2.get_result_outer_statistics(
            current_best_solution, ks, optimal_inner_iterations=grover1.inner_iterations, depth=depth)
        outer_iterations_statistics = [grover2.outer_iterations_confidence]
        optimal_outer_iterations_classically = [grover2._iterations]

        depths = [depth]
        oc = [cost_factor_nested(d) for d in depths]
        nested_cost = sum(compute_inner_cost(oc, inner_iterations_statistics)) + \
                      sum(compute_outer_cost(oc, outer_iterations_statistics, inner_iterations, n))
        optimal_global_cost = n * sum(2 * it + 1 for it in optimal_global_iterations_classically)
        optimal_nested_cost = sum(
            (2 * oi + 1) * (2 * ii * depth + n)
            for oi, ii in zip(optimal_outer_iterations_classically, optimal_inner_iterations_classically))

        if measurement_nested is not None:
            sol_nested = _solution_from_bitstring(knapsack, measurement_nested)
            to_optimality_nested = bool(sol_nested.total_value > threshold)
        else:
            to_optimality_nested = False


        # ── 3. Cut ──────────────────────────────────────────────────────
        ks = _prepare_knapsack(knapsack, "valweight")
        n = len(ks.items)
        current_best_solution_cut = QTGHotStarter(ks, depth=n, verbose=False).greedy_solution
        HotStart_cut = QTGHotStarter(ks, depth=depth, verbose=False, current_best_solution=current_best_solution_cut)
        post_QTGk_cut = HotStart_cut.partial_hot_start_QTG_k(bias=bias_in_cut, BnB=False, cut_degree=0)
        globally_good_cut = HotStart_cut.get_globally_marked_states()

        grover1_cut = Grover(post_QTGk_cut)
        cut_cf = cost_factor_cut(depth, knapsack.capacity)
        post_inner_cut = grover1_cut.get_result_inner_statistics(cost_factor=cut_cf)
        cut_inner_statistics = [grover1_cut.inner_iterations_confidence]
        cut_inner_iterations = [grover1_cut.inner_iterations]
        cut_optimal_inner_classically = [grover1_cut._iterations]

        postQTGnk_cut = HotStart_cut.partial_hot_start_QTG_nk(post_inner_cut, globally_good_cut, bias=bias_out_cut)
        grover2_cut = Grover(postQTGnk_cut)
        _, measurement_cut, _ = grover2_cut.get_result_outer_statistics(
            current_best_solution_cut, ks, optimal_inner_iterations=grover1_cut.inner_iterations,
            depth=depth, cost_factor=cut_cf)
        cut_outer_statistics = [grover2_cut.outer_iterations_confidence]
        cut_optimal_outer_classically = [grover2_cut._iterations]

        oc_cut = [cost_factor_cut(d, knapsack.capacity) for d in depths]
        cut_cost = sum(compute_inner_cost(oc_cut, cut_inner_statistics)) + \
                   sum(compute_outer_cost(oc_cut, cut_outer_statistics, cut_inner_iterations, n))

        if measurement_cut is not None:
            sol_cut = _solution_from_bitstring(ks, measurement_cut)
            to_optimality_cut = bool(sol_cut.total_value >= threshold)
        else:
            to_optimality_cut = False

        # Gate-cost tracking for single-shot runs (fixed threshold per method call)
        T_oracle = current_best_solution.total_value if current_best_solution is not None else 0
        re_nested = build_resource_estimator(_prepare_knapsack(knapsack, "value"))
        global_gatecost = sum(
            compute_global_gatecost(
                re_nested,
                global_iterations_statistics,
                [T_oracle] * len(global_iterations_statistics),
            )
        )
        nested_gatecost = sum(
            compute_inner_gatecost(
                re_nested,
                inner_iterations_statistics,
                depths,
                [T_oracle] * len(depths),
            )
        ) + sum(
            compute_outer_gatecost(
                re_nested,
                outer_iterations_statistics,
                inner_iterations,
                depths,
                [T_oracle] * len(depths),
            )
        )
        re_cut = build_resource_estimator(_prepare_knapsack(knapsack, "valweight"))
        cut_gatecost = sum(
            compute_inner_gatecost_cut(
                re_cut,
                cut_inner_statistics,
                depths,
                [T_oracle] * len(depths),
            )
        ) + sum(
            compute_outer_gatecost_cut(
                re_cut,
                cut_outer_statistics,
                cut_inner_iterations,
                depths,
                [T_oracle] * len(depths),
            )
        )


        # ── Build result ─────────────────────────────────────────────────
        return MultiMethodResult(
            instance_name=instance_name,
            knapsack_size=n,
            bias=bias,
            depth=depth,
            capweight=capweight,
            T=T,
            remaining_value_ratio=remaining_value_ratio,
            # global
            global_cost=global_cost,
            optimal_global_iterations_classically=optimal_global_iterations_classically,
            global_statistics=global_iterations_statistics,
            to_optimality_global=to_optimality_global,
            # nested
            nested_cost=nested_cost,
            optimal_inner_iterations_classically=optimal_inner_iterations_classically,
            optimal_outer_iterations_classically=optimal_outer_iterations_classically,
            optimal_inner_iterations=inner_iterations,
            inner_statistics=inner_iterations_statistics,
            outer_statistics=outer_iterations_statistics,
            optimal_nested_cost=optimal_nested_cost,
            optimal_global_cost=optimal_global_cost,
            to_optimality_nested=to_optimality_nested,
            # cut
            cut_cost=cut_cost,
            cut_inner_iterations=cut_inner_iterations,
            cut_inner_statistics=cut_inner_statistics,
            cut_outer_statistics=cut_outer_statistics,
            cut_optimal_inner_classically=cut_optimal_inner_classically,
            cut_optimal_outer_classically=cut_optimal_outer_classically,
            to_optimality_cut=to_optimality_cut,
            global_gatecost=global_gatecost,
            nested_gatecost=nested_gatecost,
            cut_gatecost=cut_gatecost,
            greedy_is_optimal=greedy_is_optimal,
        )

    def simulate_bias_sweep_capweight(self, number_of_samples: int, order: str = "value", id=None):
        """Sweep combinations of inner bias (QTG_k) and outer bias (QTG_nk) across instances,
        keyed by capweight ratio. Global always uses n/4."""
        yaml_paths = list(self.path_to_instances.rglob("*.yml")) + list(self.path_to_instances.rglob("*.yaml"))
        if not yaml_paths:
            raise FileNotFoundError(f"No YAML files found under {self.path_to_instances}")
        if number_of_samples > len(yaml_paths) or number_of_samples <= 0:
            raise ValueError(f"Requested {number_of_samples} samples but only found {len(yaml_paths)} YAML files.")
        sampled_paths = random.sample(yaml_paths, number_of_samples)

        results: list[MultiMethodBiasSweepResult] = []
        print("------------------------------")
        print(f"Bias-sweep capweight simulation on {number_of_samples} instances...")

        for yaml_path in tqdm.tqdm(sampled_paths):
            knapsack = KnapsackInstance(yaml_path)
            n = knapsack.num_items
            hot_starter = QTGHotStarter(knapsack, depth=n)
            optimal_solution = hot_starter.get_optimal_solution()
            greedy_solution = hot_starter.greedy_solution
            greedy_is_optimal = bool(greedy_solution.total_value == optimal_solution.total_value)

            if greedy_is_optimal:
                continue

            capweight_ratio = knapsack.capacity / sum(item.weight for item in knapsack.items)

            # Bias values to sweep: 0, n/10, n/7, n/4, n/2, n
            bias_values = [0, n/10, n/7, n/4, n/2, n]

            for depth_fraction in [x * 0.1 for x in range(1, 9)]:
                depth = int(depth_fraction * n)
                for inner_bias in bias_values:
                    for outer_bias in bias_values:
                        result = self._unitSim_bias_sweep(
                            knapsack, inner_bias=inner_bias, outer_bias=outer_bias,
                            depth=depth, order=order,
                            current_best_solution=greedy_solution,
                            instance_name=yaml_path.stem,
                            capweight=capweight_ratio,
                            greedy_is_optimal=greedy_is_optimal,
                        )
                        results.append(result)

        if id is None:
            raise ValueError("An 'id' must be provided to save the results.")
        result_file = self.path_to_results / f"bias_sweep_capweight_results_id{id}.csv"
        _write_results_csv(results, result_file)

    def _unitSim_bias_sweep(
        self,
        knapsack: KnapsackInstance,
        inner_bias: float,
        outer_bias: float,
        depth: int,
        order: str = "default",
        current_best_solution: Union[None, OptimizerSolution] = None,
        instance_name: str = "",
        capweight: float = -1.0,
        greedy_is_optimal: bool = False,
    ) -> MultiMethodBiasSweepResult:
        """Single-shot simulation sweeping inner/outer bias for nested (and cut).
        Global always uses n/4 as bias."""

        threshold = current_best_solution.total_value if current_best_solution is not None else 0

        # ── 1. Global (bias fixed at n/4) ────────────────────────────────
        ks = _prepare_knapsack(knapsack, "value")
        n = len(ks.items)
        global_bias = n / 4
        HotStart_global = QTGHotStarter(ks, depth=n, verbose=False, current_best_solution=current_best_solution)
        post_QTG_state_good_global = HotStart_global.hot_start(bias=global_bias)
        grover_global = Grover(post_QTG_state_good_global)
        _, measurement_global, _ = grover_global.get_result_outer_statistics(current_best_solution, ks)

        global_iterations_statistics = [grover_global.outer_iterations_confidence]
        optimal_global_iterations_classically = [grover_global._iterations]
        global_cost = sum(compute_global_cost(global_iterations_statistics, n))

        if measurement_global is not None:
            sol_global = _solution_from_bitstring(knapsack, measurement_global)
            to_optimality_global = bool(sol_global.total_value > threshold)
        else:
            to_optimality_global = False

        # ── 2. Nested ─────────────────────────────────────────────────────
        ks = _prepare_knapsack(knapsack, "value")
        n = len(ks.items)
        HotStart = QTGHotStarter(ks, depth=depth, verbose=False, current_best_solution=current_best_solution)
        post_QTGk_states_good = HotStart.partial_hot_start_QTG_k(bias=inner_bias)
        globally_good_states = HotStart.get_globally_marked_states()

        grover1 = Grover(post_QTGk_states_good)
        post_inner_grover_state = grover1.get_result_inner_statistics()
        inner_iterations_statistics = [grover1.inner_iterations_confidence]
        inner_iterations = [grover1.inner_iterations]
        optimal_inner_iterations_classically = [grover1._iterations]

        postQTGnk_states = HotStart.partial_hot_start_QTG_nk(post_inner_grover_state, globally_good_states, bias=outer_bias)
        grover2 = Grover(postQTGnk_states)
        _, measurement_nested, _ = grover2.get_result_outer_statistics(
            current_best_solution, ks, optimal_inner_iterations=grover1.inner_iterations, depth=depth)
        outer_iterations_statistics = [grover2.outer_iterations_confidence]
        optimal_outer_iterations_classically = [grover2._iterations]

        depths = [depth]
        oc = [cost_factor_nested(d) for d in depths]
        nested_cost = sum(compute_inner_cost(oc, inner_iterations_statistics)) + \
                      sum(compute_outer_cost(oc, outer_iterations_statistics, inner_iterations, n))
        optimal_global_cost = n * sum(2 * it + 1 for it in optimal_global_iterations_classically)
        optimal_nested_cost = sum(
            (2 * oi + 1) * (2 * ii * depth + n)
            for oi, ii in zip(optimal_outer_iterations_classically, optimal_inner_iterations_classically))

        if measurement_nested is not None:
            sol_nested = _solution_from_bitstring(knapsack, measurement_nested)
            to_optimality_nested = bool(sol_nested.total_value > threshold)
        else:
            to_optimality_nested = False

        # ── 3. Cut ──────────────────────────────────────────────────────
        ks = _prepare_knapsack(knapsack, "valweight")
        n = len(ks.items)
        current_best_solution_cut = QTGHotStarter(ks, depth=n).greedy_solution
        HotStart_cut = QTGHotStarter(ks, depth=depth, verbose=False, current_best_solution=current_best_solution_cut)
        post_QTGk_cut = HotStart_cut.partial_hot_start_QTG_k(bias=inner_bias, BnB=False, cut_degree=0)
        globally_good_cut = HotStart_cut.get_globally_marked_states()

        grover1_cut = Grover(post_QTGk_cut)
        cut_cf = cost_factor_cut(depth, knapsack.capacity)
        post_inner_cut = grover1_cut.get_result_inner_statistics(cost_factor=cut_cf)
        cut_inner_statistics = [grover1_cut.inner_iterations_confidence]
        cut_inner_iterations = [grover1_cut.inner_iterations]
        cut_optimal_inner_classically = [grover1_cut._iterations]

        postQTGnk_cut = HotStart_cut.partial_hot_start_QTG_nk(post_inner_cut, globally_good_cut, bias=outer_bias)
        grover2_cut = Grover(postQTGnk_cut)
        _, measurement_cut, _ = grover2_cut.get_result_outer_statistics(
            current_best_solution_cut, ks, optimal_inner_iterations=grover1_cut.inner_iterations,
            depth=depth, cost_factor=cut_cf)
        cut_outer_statistics = [grover2_cut.outer_iterations_confidence]
        cut_optimal_outer_classically = [grover2_cut._iterations]

        oc_cut = [cost_factor_cut(d, knapsack.capacity) for d in depths]
        cut_cost = sum(compute_inner_cost(oc_cut, cut_inner_statistics)) + \
                   sum(compute_outer_cost(oc_cut, cut_outer_statistics, cut_inner_iterations, n))

        if measurement_cut is not None:
            sol_cut = _solution_from_bitstring(ks, measurement_cut)
            to_optimality_cut = bool(sol_cut.total_value > threshold)
        else:
            to_optimality_cut = False

        # ── Gate-cost tracking ───────────────────────────────────────────
        T_oracle = current_best_solution.total_value if current_best_solution is not None else 0
        re_global = build_resource_estimator(_prepare_knapsack(knapsack, "value"))
        global_gatecost = sum(
            compute_global_gatecost(re_global, global_iterations_statistics,
                                    [T_oracle] * len(global_iterations_statistics))
        )
        re_nested = build_resource_estimator(_prepare_knapsack(knapsack, "value"))
        nested_gatecost = sum(
            compute_inner_gatecost(re_nested, inner_iterations_statistics, depths, [T_oracle] * len(depths))
        ) + sum(
            compute_outer_gatecost(re_nested, outer_iterations_statistics, inner_iterations, depths, [T_oracle] * len(depths))
        )
        re_cut = build_resource_estimator(_prepare_knapsack(knapsack, "valweight"))
        cut_gatecost = sum(
            compute_inner_gatecost_cut(re_cut, cut_inner_statistics, depths, [T_oracle] * len(depths))
        ) + sum(
            compute_outer_gatecost_cut(re_cut, cut_outer_statistics, cut_inner_iterations, depths, [T_oracle] * len(depths))
        )

        # ── Build result ─────────────────────────────────────────────────
        return MultiMethodBiasSweepResult(
            instance_name=instance_name,
            knapsack_size=n,
            global_bias=global_bias,
            nested_bias_inner=inner_bias,
            nested_bias_outer=outer_bias,
            cut_bias_inner=inner_bias,
            cut_bias_outer=outer_bias,
            bias_label=f"inner={inner_bias:.1f}_outer={outer_bias:.1f}",
            depth=depth,
            capweight=capweight,
            # global
            global_cost=global_cost,
            optimal_global_iterations_classically=optimal_global_iterations_classically,
            global_statistics=global_iterations_statistics,
            to_optimality_global=to_optimality_global,
            # nested
            nested_cost=nested_cost,
            optimal_inner_iterations_classically=optimal_inner_iterations_classically,
            optimal_outer_iterations_classically=optimal_outer_iterations_classically,
            optimal_inner_iterations=inner_iterations,
            inner_statistics=inner_iterations_statistics,
            outer_statistics=outer_iterations_statistics,
            optimal_nested_cost=optimal_nested_cost,
            optimal_global_cost=optimal_global_cost,
            to_optimality_nested=to_optimality_nested,
            # cut
            cut_cost=cut_cost,
            cut_inner_iterations=cut_inner_iterations,
            cut_inner_statistics=cut_inner_statistics,
            cut_outer_statistics=cut_outer_statistics,
            cut_optimal_inner_classically=cut_optimal_inner_classically,
            cut_optimal_outer_classically=cut_optimal_outer_classically,
            to_optimality_cut=to_optimality_cut,
            # gate costs
            global_gatecost=global_gatecost,
            nested_gatecost=nested_gatecost,
            cut_gatecost=cut_gatecost,
            greedy_is_optimal=greedy_is_optimal,
        )

    def simulate_fixed_cost_multi_method(self, number_of_samples: int,
                                         global_bias_factor: float,
                                         nested_inner_bias_factor: float,
                                         nested_outer_bias_factor: float,
                                         cut_inner_bias_factor: float,
                                         cut_outer_bias_factor: float,
                                         order: str = "value",
                                         id=None):
        yaml_paths = list(self.path_to_instances.rglob("*.yml")) + list(self.path_to_instances.rglob("*.yaml"))
        if not yaml_paths:
            raise FileNotFoundError(f"No YAML files found under {self.path_to_instances}")
        if number_of_samples > len(yaml_paths) or number_of_samples <= 0:
            raise ValueError(f"Requested {number_of_samples} samples but only found {len(yaml_paths)} YAML files.")
        sampled_paths = random.sample(yaml_paths, number_of_samples)

        results: list[MultiMethodApproximationResult] = []
        print("------------------------------")
        print(f"Multi-method fixed-cost approximation simulation on {number_of_samples} instances...")

        for yaml_path in tqdm.tqdm(sampled_paths):
            for i in range(1, 5):
                knapsack = KnapsackInstance(yaml_path)
                bias = global_bias_factor * knapsack.num_items
                bias_in_nested = nested_inner_bias_factor * knapsack.num_items
                bias_out_nested = nested_outer_bias_factor * knapsack.num_items
                bias_in_cut = cut_inner_bias_factor * knapsack.num_items
                bias_out_cut = cut_outer_bias_factor * knapsack.num_items
                instance_sort = InstanceFilter(knapsack)
                if not instance_sort.filter_capweight_ratio_range(0.6, 1.0):
                    print(f'Filtering instance {yaml_path.stem}. With capweight ratio {knapsack.capacity / sum(item.weight for item in knapsack.items):.2f}')
                    continue
                optimal_solution = QTGHotStarter(knapsack, depth=knapsack.num_items).get_optimal_solution()
                greedy_solution = QTGHotStarter(knapsack, depth=knapsack.num_items).greedy()
                greedy_optimal = bool(greedy_solution.total_value == optimal_solution.total_value)
                if greedy_optimal:
                    print(f'Filtering instance {yaml_path.stem}.')
                    continue
                print(f'Simulating instance: {yaml_path.stem}')
                capweight = float(knapsack.capacity / sum(item.weight for item in knapsack.items))
                n = knapsack.num_items

                for termination_exponent in [0.6,0.7, 0.8, 0.9,1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,1.8,1.9, 2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9, 3.0,3.1,3.2,3.3,3.4, 3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4, 4.5]:
                    constant = 2500
                    termination_gate_cost = int(constant * n ** termination_exponent)

                    # ── Global ──
                    solver_global = StatisticalSmartSolver(knapsack, bias=bias, iterations=50, order=order, termination_gate_threshold=termination_gate_cost)
                    global_measurement, global_statistics, optimal_global_iterations_classically, global_thresholds = solver_global.solve_global_statistics()

                    # ── Nested ──
                    smart_solver = StatisticalSmartSolver(knapsack, bias=bias, iterations=50, order=order, termination_gate_threshold=termination_gate_cost, bias_in=bias_in_nested, bias_out=bias_out_nested)
                    nested_solution, smart_inner, smart_inner_statistics, smart_outer_statistics, smart_global_statistics, depths, optimal_inner_iterations_classically, optimal_outer_iterations_classically, nested_thresholds = smart_solver.smart_solve_statistics()

                    # ── Cut ──
                    cut_solver = StatisticalSmartSolver(knapsack, bias=bias, iterations=50, order="valweight",
                                                        termination_gate_threshold=termination_gate_cost, bias_in=bias_in_cut, bias_out=bias_out_cut)
                    cut_solution, cut_inner, cut_inner_statistics, cut_outer_statistics, cut_global_statistics, cut_depths, cut_optimal_inner_classically, cut_optimal_outer_classically, cut_thresholds = cut_solver.solve_cuts_statistics()

                    # ── Gate cost calculations ──
                    re_global = build_resource_estimator(solver_global.knapsack_instance)
                    re_nested = build_resource_estimator(smart_solver.knapsack_instance)
                    re_cut = build_resource_estimator(cut_solver.knapsack_instance)

                    # Global gate cost
                    global_gatecost_list = compute_global_gatecost(re_global, global_statistics, global_thresholds)
                    global_cost = sum(global_gatecost_list)
                    
                    # Nested gate cost
                    nested_inner_gatecost_list = compute_inner_gatecost(re_nested, smart_inner_statistics, depths, nested_thresholds)
                    nested_outer_gatecost_list = compute_outer_gatecost(re_nested, smart_outer_statistics, smart_inner, depths, nested_thresholds)
                    nested_cost = sum(nested_inner_gatecost_list) + sum(nested_outer_gatecost_list)
                    nested_stepwise = [inner + outer for inner, outer in zip(nested_inner_gatecost_list, nested_outer_gatecost_list)]
                    
                    # Cut gate cost
                    cut_inner_gatecost_list = compute_inner_gatecost_cut(re_cut, cut_inner_statistics, cut_depths, cut_thresholds)
                    cut_outer_gatecost_list = compute_outer_gatecost_cut(re_cut, cut_outer_statistics, cut_inner, cut_depths, cut_thresholds)
                    cut_cost = sum(cut_inner_gatecost_list) + sum(cut_outer_gatecost_list)
                    cut_stepwise = [inner + outer for inner, outer in zip(cut_inner_gatecost_list, cut_outer_gatecost_list)]

                    if nested_cost > termination_gate_cost or cut_cost > termination_gate_cost or global_cost > termination_gate_cost:
                        raise ValueError(f"Termination gate cost of {termination_gate_cost} was exceeded by nested cost {nested_cost}, cut cost {cut_cost}, or global cost {global_cost}.")

                    to_optimality_global = bool(global_measurement.total_value >= optimal_solution.total_value)
                    to_optimality_nested = bool(nested_solution.total_value >= optimal_solution.total_value)
                    to_optimality_cut = bool(cut_solution.total_value >= optimal_solution.total_value)

                    result = MultiMethodApproximationResult(
                        instance_name=yaml_path.stem,
                        knapsack_size=n,
                        bias=bias,
                        capweight=capweight,
                        threshold_cost=termination_gate_cost,
                        termination_cost_constant=constant,
                        termination_cost_exponent=termination_exponent,
                        approximation_ratio_greedy=greedy_solution.total_value / optimal_solution.total_value,
                        greedy_is_optimal=greedy_optimal,
                        # global
                        global_cost=global_cost,
                        optimal_global_iterations_classically=optimal_global_iterations_classically,
                        global_statistics=global_statistics,
                        approximation_ratio_global=global_measurement.total_value / optimal_solution.total_value,
                        better_than_greedy_global=bool(global_measurement.total_value > greedy_solution.total_value),
                        to_optimality_global=to_optimality_global,
                        global_stepwise_cost=global_gatecost_list,
                        # nested
                        nested_cost=nested_cost,
                        optimal_inner_iterations_classically=optimal_inner_iterations_classically,
                        optimal_outer_iterations_classically=optimal_outer_iterations_classically,
                        optimal_inner_iterations=smart_inner,
                        inner_statistics=smart_inner_statistics,
                        outer_statistics=smart_outer_statistics,
                        smart_global_statistics=smart_global_statistics,
                        approximation_ratio_nested=nested_solution.total_value / optimal_solution.total_value,
                        better_than_greedy_nested=bool(nested_solution.total_value > greedy_solution.total_value),
                        to_optimality_nested=to_optimality_nested,
                        nested_stepwise_cost=nested_stepwise,
                        # cut
                        cut_cost=cut_cost,
                        cut_inner_iterations=cut_inner,
                        cut_inner_statistics=cut_inner_statistics,
                        cut_outer_statistics=cut_outer_statistics,
                        cut_optimal_inner_classically=cut_optimal_inner_classically,
                        cut_optimal_outer_classically=cut_optimal_outer_classically,
                        approximation_ratio_cut=cut_solution.total_value / optimal_solution.total_value,
                        better_than_greedy_cut=bool(cut_solution.total_value > greedy_solution.total_value),
                        to_optimality_cut=to_optimality_cut,
                        cut_stepwise_cost=cut_stepwise,
                    )
                    results.append(result)

        if id is None:
            raise ValueError("An 'id' must be provided to save the results.")
        result_file = self.path_to_results / f"fixed_cost_multi_method_results_id{id}.csv"
        _write_results_csv(results, result_file)




def _write_results_csv(results: List[MultiMethodResult], filepath: Union[str, pathlib.Path]) -> str:
    out_path = pathlib.Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if results and isinstance(results[0], MultiMethodBiasSweepResult):
        fieldnames = list(MultiMethodBiasSweepResult.__dataclass_fields__.keys())
    elif results and isinstance(results[0], MultiMethodApproximationResult):
        fieldnames = list(MultiMethodApproximationResult.__dataclass_fields__.keys())
    else:
        fieldnames = list(MultiMethodResult.__dataclass_fields__.keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return str(out_path)
