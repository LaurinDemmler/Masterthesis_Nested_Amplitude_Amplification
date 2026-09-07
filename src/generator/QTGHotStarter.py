import time
from datastructures.OptimizerSolution import OptimizerSolution
from datastructures.amplitudes import BinaryAmplitudeState
from generator.greedy import greedy_solver, fractional_greedy, frational_greedy_from_partial_bitstring
from generator.QTG import AmplitudeAssigner, AmplitudeAssignerQTGk, AmplitudeAssignerQTGnk
from generator.GurobiSolver import GurobiSolver

class QTGHotStarter:
    """Wrapper for Hot Starting using QTG.
    Expects the knapsack to already be in the desired item order (value-sorted, density-sorted, etc.).
    The greedy solution is always computed on a density-sorted copy and remapped to the knapsack's current order."""
    def __init__(self, knapsack, depth: int, verbose: bool = False, current_best_solution=None):
        if depth < 0 or depth > knapsack.num_items:
            raise ValueError(f"depth must be in [0, {knapsack.num_items}]")
        self.depth = int(depth)
        self.knapsack = knapsack
        
        # Compute greedy solution in the knapsack's current order
        if current_best_solution is not None:
            self.greedy_solution = current_best_solution
            self.greedy_threshold = current_best_solution.total_value
        else:
            self.greedy_solution = greedy_solver(knapsack)
            self.greedy_threshold = self.greedy_solution.total_value

        self.verbose = verbose
        if self.verbose:
            print(f'Greedy solution value (threshold for marked states): {self.greedy_threshold}, bitstring: {self.greedy_solution.bitstring}')
            print(f"QTG Hot Starter initialized with depth {self.depth}.")
        
    def hot_start(self, bias=0.0):
        """Hot starts using QTG. Finds all feasible states, applies QTG (with bias) and returns the post-QTG state as BinaryAmplitudeState."""
        marked_states = self.get_marked_states(self.greedy_threshold, to_depth=self.depth)
        # to debug the amplitudes for QTG uncomment this. Result will contain all feasible state amplitudes. not compatible with Grover.
        #marked_states = self.all_feasible_states()
        post_QTG_state = self.post_QTG_state(marked_states, bias=bias)
        return post_QTG_state
    
    def partial_hot_start_QTG_k(self, bias=0.0, BnB = False, cut_degree=False):
        """Hot starts using QTG from depth k to the end. Finds all feasible states, applies QTG (with bias) and returns the post-QTG state as BinaryAmplitudeState."""
        start_time = time.perf_counter()
        if BnB:
            # Single Gurobi call: enumerate partial bitstrings whose fractional
            # greedy upper bound exceeds the greedy threshold.
            marked_states = self.get_marked_states_BnB()
        elif cut_degree is not False:
            # Directly enumerate partial bitstrings whose cut upper bound
            # exceeds the greedy threshold (replaces nested criterion + filter).
            marked_states = self.get_marked_states_cut(cut_degree=cut_degree)
        else:
            marked_states = self.get_marked_states(self.greedy_threshold, to_depth=self.depth)
        if self.verbose:
            print(f"Found {len(marked_states)} QTG-k marked states to depth {self.depth} in {time.perf_counter() - start_time} seconds.")

        return AmplitudeAssignerQTGk(
            [state.bitstring for state in marked_states],
            bias=bias,
            up_to_depth=self.depth, greedy_bitstring=self.greedy_solution.bitstring
        ).get_state(self.knapsack)

    def partial_hot_start_QTG_nk(self, initial_state: BinaryAmplitudeState, globally_marked_states:list[OptimizerSolution], bias=0.0) -> BinaryAmplitudeState:
        """Hot starts using QTG from depth k to the end. Finds all feasible states, applies QTG (with bias) and returns the post-QTG state as BinaryAmplitudeState."""
        return AmplitudeAssignerQTGnk(
            initial_state,
            globally_marked_states,
            bias=bias,
            from_depth=self.depth, greedy_bitstring=self.greedy_solution.bitstring
        ).get_state(self.knapsack, initial_state=initial_state)

    def post_QTG_state(self, marked_states: list[OptimizerSolution], bias=0.0) -> BinaryAmplitudeState:
        """Applies QTG on all marked states up to self.depth with the provided bias and returns the resulting BinaryAmplitudeState."""
        bitstrings = [state.bitstring for state in marked_states]
        return AmplitudeAssigner(
            bitstrings,
            bias=bias, greedy_bitstring=self.greedy_solution.bitstring
        ).get_state(self.knapsack)

    def get_globally_marked_states(self) -> list[OptimizerSolution]:
        """Gets all states that satisfy the value inequality for the full knapsack."""
        start_time = time.perf_counter()
        result = GurobiSolver().feasible_states_with_greedy_inequality(
            self.knapsack, self.greedy_threshold, self.knapsack.num_items
        )
        end_time = time.perf_counter()
        if self.verbose:
            print(f"{len(result)} Globally marked states found in {end_time - start_time} seconds.")
        return result

    def greedy(self):
        """Finds a greedy solution for the knapsack problem."""
        return greedy_solver(self.knapsack)

    def all_feasible_states(self) -> list[OptimizerSolution]:
        """Finds all feasible states for the knapsack problem."""
        return GurobiSolver().feasible_states(self.knapsack)

    def get_marked_states(self, threshold, to_depth) -> list[OptimizerSolution]:
        """Gets all states that satisfy the value inequality at a certain depth of Grover."""
        return GurobiSolver().feasible_states_with_greedy_inequality(
            self.knapsack, threshold, to_depth
        )

    def get_optimal_solution(self) -> OptimizerSolution:
        """Gets the optimal solution for the knapsack instance using Gurobi."""
        return GurobiSolver().gurobi_optimal_solution(self.knapsack)
    
    def find_solution_with_value_T(self, T: int) -> OptimizerSolution:
        """Finds a feasible solution with value at least T using Gurobi."""
        return GurobiSolver().find_solution_with_value_T(self.knapsack, T)
    
    def get_fractional_optimum(self) -> OptimizerSolution:
        """Gets a heuristic best T value using a simple greedy approach."""
        return fractional_greedy(self.knapsack)
    
    def filter_marked_states_BnB(self, marked_states: list[OptimizerSolution]) -> list[OptimizerSolution]:
        """Filters the marked states using a Branch and Bound approach to find the most promising ones."""
        density_sorted = self.knapsack._copy()
        density_sorted.sort_items_by_density()
        return [state for state in marked_states
                if self.greedy_threshold < frational_greedy_from_partial_bitstring(self.knapsack, state, density_sorted_copy=density_sorted).total_value]
    
    def get_marked_states_BnB(self) -> list[OptimizerSolution]:
        """Return all partial bitstrings (length self.depth) whose ceiling-Dantzig
        upper bound strictly exceeds the greedy threshold.  Direct DFS enumeration."""
        return GurobiSolver().feasible_states_with_ceiling_dantzig_ub(
            self.knapsack, self.greedy_threshold, self.depth
        )

    def get_marked_states_cut(self, cut_degree: int) -> list[OptimizerSolution]:
        """Return all partial bitstrings (length self.depth) whose cut upper
        bound strictly exceeds the greedy threshold.  Uses Gurobi directly."""
        if cut_degree < 0:
            raise ValueError("cut_degree must be a non-negative integer.")
        return GurobiSolver().feasible_states_with_cut_upper_bound(
            self.knapsack, self.greedy_threshold, self.depth, cut_degree=cut_degree
        )
            

        