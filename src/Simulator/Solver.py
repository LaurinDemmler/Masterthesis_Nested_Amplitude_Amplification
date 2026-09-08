from datastructures.knapsack import KnapsackInstance
from generator.QTGHotStarter import QTGHotStarter
from grover.grover import Grover
from generator.GurobiSolver import _solution_from_bitstring
from Simulator.ResourceEstimator import QTGResourceEstimator
import numpy as np


# ── Cost factor functions, this is deprecated here but reflects the paper's cost counting ───────────

def cost_factor_nested(depth):
    return depth


def cost_factor_cut(depth, capacity):
    return depth + 0.33 * (np.log2(capacity) + 1)


# ── Cost aggregation helpers (kept for backward compatibility) ────────

def compute_inner_cost(cost_factors, inner_statistics):
    """Inner Grover cost per depth step: sum((1 + 2*k) * cost_factor for k in iterations)."""
    return [sum((1 + 2 * iteration) * oc for iteration in sublist)
            for oc, sublist in zip(cost_factors, inner_statistics)]

def compute_outer_cost(cost_factors, outer_statistics, inner_iterations, num_items):
    """Outer Grover cost per depth step: sum((2*j+1) * (2*inner_iters*cost_factor + n) for j in iterations)."""
    return [sum((2 * outer_iter + 1) * (2 * opt_inner * oc + num_items)
                for outer_iter in sublist)
            for sublist, opt_inner, oc in zip(outer_statistics, inner_iterations, cost_factors)]

def compute_global_cost(global_statistics, num_items):
    """Global Grover cost per step: sum(n * (2*g+1) for g in iterations)."""
    return [sum(num_items * (2 * item + 1) for item in sublist)
            for sublist in global_statistics]

def compute_budget(threshold, cost_factors, inner_statistics, outer_statistics,
                   inner_iterations, num_items, global_statistics=None):
    """Remaining budget = threshold - (inner_cost + outer_cost [+ global_cost])."""
    inner = compute_inner_cost(cost_factors, inner_statistics)
    outer = compute_outer_cost(cost_factors, outer_statistics, inner_iterations, num_items)
    if global_statistics:
        outer = outer + compute_global_cost(global_statistics, num_items)
    return threshold - (sum(inner) + sum(outer))


# ── Gate-count cost aggregation helpers ───────────────────────────────

def compute_inner_gatecost(re: QTGResourceEstimator, inner_statistics, depths, thresholds):
    """Inner Grover gate cost per QMaxSearch step using ResourceEstimator.
    For each step i, sums gatec_inner_step(g, depths[i], thresholds[i]) over all iterations g.
    """
    return [
        sum(re.gatec_inner_step(g, d, T) for g in iterations)
        for d, T, iterations in zip(depths, thresholds, inner_statistics)
    ]

def compute_outer_gatecost(re: QTGResourceEstimator, outer_statistics, inner_iterations, depths, thresholds):
    """Outer Grover gate cost per QMaxSearch step using ResourceEstimator.
    For each step i, sums gatec_outer_step(j, depths[i], inner_iterations[i], thresholds[i]).
    """
    return [
        sum(re.gatec_outer_step(j, d, ki, T) for j in iterations)
        for d, ki, T, iterations in zip(depths, inner_iterations, thresholds, outer_statistics)
    ]

def compute_global_gatecost(re: QTGResourceEstimator, global_statistics, thresholds):
    """Global Grover gate cost per QMaxSearch step.
    For each step i, sums gatec_global_step(j, thresholds[i]).
    """
    return [
        sum(re.gatec_global_step(j, T) for j in iterations)
        for T, iterations in zip(thresholds, global_statistics)
    ]

def compute_gate_budget(threshold, re: QTGResourceEstimator,
                        inner_statistics, outer_statistics, inner_iterations,
                        depths, cost_thresholds, global_statistics=None, global_thresholds=None):
    """Remaining gate-count budget = threshold - total gate cost so far."""
    inner = compute_inner_gatecost(re, inner_statistics, depths, cost_thresholds)
    outer = compute_outer_gatecost(re, outer_statistics, inner_iterations, depths, cost_thresholds)
    total = sum(inner) + sum(outer)
    if global_statistics and global_thresholds:
        total += sum(compute_global_gatecost(re, global_statistics, global_thresholds))
    return threshold - total


# ── Cut-specific gate-count cost aggregation helpers ──────────────────────

def compute_inner_gatecost_cut(re: QTGResourceEstimator, inner_statistics, depths, thresholds):
    """Inner Grover gate cost per QMaxSearch step using the cut approach.
    For each step i, sums gatec_inner_step_cut(g, depths[i], thresholds[i]) over all iterations g.
    """
    return [
        sum(re.gatec_inner_step_cut(g, d, T) for g in iterations)
        for d, T, iterations in zip(depths, thresholds, inner_statistics)
    ]

def compute_outer_gatecost_cut(re: QTGResourceEstimator, outer_statistics, inner_iterations, depths, thresholds):
    """Outer Grover gate cost per QMaxSearch step using the cut approach.
    For each step i, sums gatec_outer_step_cut(j, depths[i], inner_iterations[i], thresholds[i]).
    """
    return [
        sum(re.gatec_outer_step_cut(j, d, ki, T) for j in iterations)
        for d, ki, T, iterations in zip(depths, inner_iterations, thresholds, outer_statistics)
    ]

def compute_gate_budget_cut(threshold, re: QTGResourceEstimator,
                            inner_statistics, outer_statistics, inner_iterations,
                            depths, cost_thresholds, global_statistics=None, global_thresholds=None):
    """Remaining gate-count budget for cut approach = threshold - total gate cost so far."""
    inner = compute_inner_gatecost_cut(re, inner_statistics, depths, cost_thresholds)
    outer = compute_outer_gatecost_cut(re, outer_statistics, inner_iterations, depths, cost_thresholds)
    total = sum(inner) + sum(outer)
    if global_statistics and global_thresholds:
        total += sum(compute_global_gatecost(re, global_statistics, global_thresholds))
    return threshold - total


def build_resource_estimator(knapsack: KnapsackInstance) -> QTGResourceEstimator:
    """Convenience: build a QTGResourceEstimator from a KnapsackInstance."""
    return QTGResourceEstimator(
        weights=knapsack.weights,
        profits=knapsack.values,
        capacity=knapsack.capacity,
        profit_upper_bound=sum(knapsack.values),
    )


def find_depth_for_remaining_value(knapsack: KnapsackInstance, target_remaining_value: float) -> int:
    """Find the largest depth such that get_remaining_value(depth) >= target_remaining_value."""
    for d in range(knapsack.num_items + 1):
        if knapsack.get_remaining_value(d) < target_remaining_value:
            return max(d - 1, 0)
    return knapsack.num_items


def _prepare_knapsack(knapsack_instance: KnapsackInstance, order: str) -> KnapsackInstance:
    """Return a knapsack copy sorted according to the given order.
    'value' returns the instance as-is (already value-sorted).
    'valweight' returns a density-sorted copy."""
    if order == "valweight":
        ks = knapsack_instance._copy()
        ks.sort_items_by_density()
        return ks
    return knapsack_instance



class Solver:
    def __init__(self, knapsack_instance: KnapsackInstance, bias: float, iterations: int, order: str = "value",
                 termination_cost_threshold=None, termination_gate_threshold=None,
                 bias_in: float = None, bias_out: float = None):
        self.knapsack_instance = _prepare_knapsack(knapsack_instance, order)
        self.bias = bias
        self.bias_in = bias_in 
        self.bias_out = bias_out
        self.iterations = iterations
        self.order = order
        self.use_gate_cost = (termination_gate_threshold is not None)

        if termination_gate_threshold is not None:
            # Gate-count budget mode
            self.termination_cost_threshold = float('inf')
            self.termination_gate_threshold = termination_gate_threshold
            self.no_update_threshold = float('inf')
            self.iterations = float('inf')
            self._re = build_resource_estimator(self.knapsack_instance)
        elif termination_cost_threshold is not None:
            self.termination_cost_threshold = termination_cost_threshold
            self.termination_gate_threshold = float('inf')
            self.no_update_threshold = float('inf')
            self.iterations = float('inf')
            self._re = None
        else:
            self.termination_cost_threshold = float('inf')
            self.termination_gate_threshold = float('inf')
            self.no_update_threshold = 1
            self._re = None

    def get_remaining_value_depth(self, T: float, ratio: float = 0.6) -> int:
        if T == 0:
            raise ValueError("T must be nonzero.")
        n = self.knapsack_instance.num_items
        vals = np.array([self.knapsack_instance.get_remaining_value(d) for d in range(1, n + 1)]) / T
        depth = int(np.argmin(np.abs(vals - ratio))) + 1
        return depth
    
    def nested_solve(self):
        hot_starter = QTGHotStarter(self.knapsack_instance, depth=self.knapsack_instance.num_items)
        current_best_solution = hot_starter.greedy()
        fractional_optimum = hot_starter.get_fractional_optimum()
        #optimal_solution = hot_starter.get_optimal_solution()
        no_update_counter = 0
        iter = 0
        inner_iterations_statistics = []
        outer_iterations_statistics = []
        inner_iterations = []
        depths = []
        cost_thresholds = []
        global_iterations_statistics = []
        optimal_inner_iterations_classically = []
        optimal_outer_iterations_classically = []
        budget = self.termination_cost_threshold
        gate_budget = self.termination_gate_threshold if self.use_gate_cost else float('inf')

        while no_update_counter < self.no_update_threshold and iter < self.iterations and budget > 0 and gate_budget > 0:
            """if current_best_solution.total_value == optimal_solution.total_value:
                break"""

            iter += 1
            # Start at middle depth (max pruning), fan outward to safer shallow depths
            candidate_fractions = [0.3, 0.4, 0.2, 0.5, 0.1, 0.6]
            depth = int(candidate_fractions[(iter - 1) % len(candidate_fractions)] * self.knapsack_instance.num_items)
            depth = max(1, min(depth, self.knapsack_instance.num_items - 1))
            depths.append(depth)
            """depth = self.get_remaining_value_depth(current_best_solution.total_value, ratio=0.5)
            depths.append(depth)"""
            T_current = current_best_solution.total_value
            cost_thresholds.append(T_current)

            HotStart = QTGHotStarter(self.knapsack_instance, depth=depth, current_best_solution=current_best_solution, verbose=False)
            post_QTGk_states_good  = HotStart.partial_hot_start_QTG_k(bias=self.bias_in)
            globally_good_states = HotStart.get_globally_marked_states()
            grover1 = Grover(post_QTGk_states_good, verbose=False)

            # Inner Grover — pass gate-count params when using gate budget
            if self.use_gate_cost:
                inner_state_prep = self._re.gatec_qtg_partial(depth)
                inner_oracle = self._re.gatec_oracle_inner(depth, T_current)
                post_inner_grover_state = grover1.inner_iteration_finder(
                    remaining_budget=gate_budget,
                    gatec_state_prep=inner_state_prep, gatec_oracle=inner_oracle)
            else:
                post_inner_grover_state = grover1.inner_iteration_finder(
                    remaining_budget=budget, cost_factor=cost_factor_nested(depth))

            inner_iterations_statistics.append(grover1.inner_iterations_confidence)
            if post_inner_grover_state is None:
                break
            
            inner_iterations.append(grover1.inner_iterations)
            optimal_inner_iterations_classically.append(grover1._iterations)
            postQTGnk_states = HotStart.partial_hot_start_QTG_nk(post_inner_grover_state,globally_good_states, bias=self.bias_out)
            grover2 = Grover(postQTGnk_states, verbose=False)

            # Recompute budgets
            oc = [cost_factor_nested(d) for d in depths]
            budget = compute_budget(self.termination_cost_threshold, oc, inner_iterations_statistics, outer_iterations_statistics, inner_iterations, self.knapsack_instance.num_items)
            if self.use_gate_cost:
                gate_budget = compute_gate_budget(self.termination_gate_threshold, self._re,
                                                  inner_iterations_statistics, outer_iterations_statistics,
                                                  inner_iterations, depths, cost_thresholds)

            # Outer Grover
            if self.use_gate_cost:
                outer_state_prep = self._re.gatec_outer_state_prep(depth, grover1.inner_iterations, T_current)
                outer_oracle = self._re.gatec_oracle_full(T_current)
                found_better_solution, measurement, remaining_budget = grover2.grover_adaptive_search(
                    current_best_solution, self.knapsack_instance,
                    optimal_inner_iterations=grover1.inner_iterations, depth=depth,
                    remaining_budget=gate_budget,
                    gatec_state_prep=outer_state_prep, gatec_oracle=outer_oracle)
            else:
                found_better_solution, measurement, remaining_budget = grover2.grover_adaptive_search(
                    current_best_solution, self.knapsack_instance,
                    optimal_inner_iterations=grover1.inner_iterations, depth=depth,
                    remaining_budget=budget)

            outer_iterations_statistics.append(grover2.outer_iterations_confidence)
            optimal_outer_iterations_classically.append(grover2._iterations)
            if remaining_budget < 0:
                break

            ### Logic to accept or reject solutions and break loop.
            if found_better_solution:
                current_best_solution = _solution_from_bitstring(self.knapsack_instance, measurement)
                no_update_counter = 0
            else:
                no_update_counter += 1


            oc = [cost_factor_nested(d) for d in depths]
            budget = compute_budget(self.termination_cost_threshold, oc, inner_iterations_statistics, outer_iterations_statistics, inner_iterations, self.knapsack_instance.num_items)
            if self.use_gate_cost:
                gate_budget = compute_gate_budget(self.termination_gate_threshold, self._re,
                                                  inner_iterations_statistics, outer_iterations_statistics,
                                                  inner_iterations, depths, cost_thresholds)
        
        return (current_best_solution, inner_iterations, inner_iterations_statistics, outer_iterations_statistics, global_iterations_statistics, depths, optimal_inner_iterations_classically, optimal_outer_iterations_classically, cost_thresholds)
    
    def baseline_solve(self):
        hot_starter = QTGHotStarter(self.knapsack_instance, depth=self.knapsack_instance.num_items)
        current_best_solution = hot_starter.greedy()
        #optimal_solution = hot_starter.get_optimal_solution()
        no_update_counter = 0
        iter = 0
        global_iterations = []
        global_iterations_statistics = []
        optimal_global_iterations_classically = []
        budget = self.termination_cost_threshold
        gate_budget = self.termination_gate_threshold if self.use_gate_cost else float('inf')
        global_thresholds = []
        
        while no_update_counter < self.no_update_threshold and iter < self.iterations and budget > 0 and gate_budget > 0:
            """if current_best_solution.total_value == optimal_solution.total_value:
                break"""
            iter += 1
            T_current = current_best_solution.total_value
            global_thresholds.append(T_current)
            HotStart = QTGHotStarter(self.knapsack_instance, depth=self.knapsack_instance.num_items, current_best_solution=current_best_solution, verbose=False)
            post_QTG_state_good = HotStart.hot_start(bias=self.bias)
            grover = Grover(post_QTG_state_good, verbose=False)

            if self.use_gate_cost:
                global_state_prep = self._re.gatec_qtg_full()
                global_oracle = self._re.gatec_oracle_full(T_current)
                found_better_solution, measurement, remaining_budget = grover.grover_adaptive_search(
                    current_best_solution, self.knapsack_instance,
                    remaining_budget=gate_budget,
                    gatec_state_prep=global_state_prep, gatec_oracle=global_oracle)
            else:
                found_better_solution, measurement, remaining_budget = grover.grover_adaptive_search(
                    current_best_solution, self.knapsack_instance, remaining_budget=budget)

            global_iterations_statistics.append(grover.outer_iterations_confidence)
            if remaining_budget < 0:
                break
            optimal_global_iterations_classically.append(grover._iterations)
            if found_better_solution:
                current_best_solution = _solution_from_bitstring(self.knapsack_instance, measurement)
                no_update_counter = 0
            else:
                no_update_counter += 1
            
            budget = self.termination_cost_threshold - sum(compute_global_cost(global_iterations_statistics, self.knapsack_instance.num_items))
            if self.use_gate_cost:
                gate_budget = self.termination_gate_threshold - sum(compute_global_gatecost(self._re, global_iterations_statistics, global_thresholds))

        
        return (current_best_solution, global_iterations_statistics, optimal_global_iterations_classically, global_thresholds)
    
    
    
    def cut_solve(self):
        """Cuts are applied to the inner list of marked states"""
        hot_starter = QTGHotStarter(self.knapsack_instance, depth=self.knapsack_instance.num_items)
        current_best_solution = hot_starter.greedy()
        fractional_optimum = hot_starter.get_fractional_optimum()
        #optimal_solution = hot_starter.get_optimal_solution()
        no_update_counter = 0
        iter = 0
        inner_iterations_statistics = []
        outer_iterations_statistics = []
        inner_iterations = []
        depths = []
        cost_thresholds = []
        global_iterations_statistics = []
        optimal_inner_iterations_classically = []
        optimal_outer_iterations_classically = []
        budget = self.termination_cost_threshold
        gate_budget = self.termination_gate_threshold if self.use_gate_cost else float('inf')
        
        while no_update_counter < self.no_update_threshold and iter < self.iterations and budget > 0 and gate_budget > 0:
            """if current_best_solution.total_value == optimal_solution.total_value:
                break"""
            """if current_best_solution.total_value >= fractional_optimum.total_value*0.995:
                print(f'Solver reached solution with value {current_best_solution.total_value} close to fractional optimum {fractional_optimum.total_value}. Stopping iterations.')
                break"""

            iter += 1
            # Start at middle depth (max pruning), fan outward to safer shallow depths
            candidate_fractions = [0.4,0.5,0.3,0.6,0.2,0.7]
            depth = int(candidate_fractions[(iter - 1) % len(candidate_fractions)] * self.knapsack_instance.num_items)
            depth = max(1, min(depth, self.knapsack_instance.num_items - 1))
            depths.append(depth)
            """depth = self.get_remaining_value_depth(current_best_solution.total_value, ratio=0.5)
            depths.append(depth)"""
            T_current = current_best_solution.total_value
            cost_thresholds.append(T_current)

            HotStart = QTGHotStarter(self.knapsack_instance, depth=depth, current_best_solution=current_best_solution, verbose=False)
            post_QTGk_states_good  = HotStart.partial_hot_start_QTG_k(bias=self.bias_in, BnB=False, cut_degree=0)
            globally_good_states = HotStart.get_globally_marked_states()
            grover1 = Grover(post_QTGk_states_good, verbose=False)
            

            if self.use_gate_cost:
                inner_state_prep = self._re.gatec_qtg_partial_cut(depth)
                inner_oracle = self._re.gatec_oracle_inner_cut(depth, T_current)
                post_inner_grover_state = grover1.inner_iteration_finder(
                    remaining_budget=gate_budget,
                    gatec_state_prep=inner_state_prep, gatec_oracle=inner_oracle)
            else:
                cut_cf = cost_factor_cut(depth, self.knapsack_instance.capacity)
                post_inner_grover_state = grover1.inner_iteration_finder(
                    remaining_budget=budget, cost_factor=cut_cf)
            
            inner_iterations_statistics.append(grover1.inner_iterations_confidence)
            if post_inner_grover_state is None:
                break
            
            inner_iterations.append(grover1.inner_iterations)
            optimal_inner_iterations_classically.append(grover1._iterations)
                
            postQTGnk_states = HotStart.partial_hot_start_QTG_nk(post_inner_grover_state,globally_good_states, bias=self.bias_out)
            grover2 = Grover(postQTGnk_states, verbose=False)
            
            # Recompute budgets
            if self.use_gate_cost:
                gate_budget = compute_gate_budget_cut(self.termination_gate_threshold, self._re,
                                                      inner_iterations_statistics, outer_iterations_statistics,
                                                      inner_iterations, depths, cost_thresholds)
            else:
                oc = [cost_factor_cut(d, self.knapsack_instance.capacity) for d in depths]
                budget = compute_budget(self.termination_cost_threshold, oc, inner_iterations_statistics, outer_iterations_statistics, inner_iterations, self.knapsack_instance.num_items)
            
            # Outer Grover
            if self.use_gate_cost:
                outer_state_prep = self._re.gatec_outer_state_prep_cut(depth, grover1.inner_iterations, T_current)
                outer_oracle = self._re.gatec_oracle_full(T_current)
                found_better_solution, measurement, remaining_budget = grover2.grover_adaptive_search(
                    current_best_solution, self.knapsack_instance,
                    optimal_inner_iterations=grover1.inner_iterations, depth=depth,
                    remaining_budget=gate_budget,
                    gatec_state_prep=outer_state_prep, gatec_oracle=outer_oracle)
            else:
                cut_cf = cost_factor_cut(depth, self.knapsack_instance.capacity)
                found_better_solution, measurement, remaining_budget = grover2.grover_adaptive_search(
                    current_best_solution, self.knapsack_instance,
                    optimal_inner_iterations=grover1.inner_iterations, depth=depth,
                    remaining_budget=budget, cost_factor=cut_cf)
            
            outer_iterations_statistics.append(grover2.outer_iterations_confidence)
            optimal_outer_iterations_classically.append(grover2._iterations)
            if remaining_budget < 0:
                break

            ### Logic to accept or reject solutions and break loop.
            if found_better_solution:
                current_best_solution = _solution_from_bitstring(self.knapsack_instance, measurement)
                no_update_counter = 0
            else:
                no_update_counter += 1

            # Final budget recomputation
            if self.use_gate_cost:
                gate_budget = compute_gate_budget_cut(self.termination_gate_threshold, self._re,
                                                      inner_iterations_statistics, outer_iterations_statistics,
                                                      inner_iterations, depths, cost_thresholds)
        return (current_best_solution, inner_iterations, inner_iterations_statistics, outer_iterations_statistics, global_iterations_statistics, depths, optimal_inner_iterations_classically, optimal_outer_iterations_classically, cost_thresholds)
    