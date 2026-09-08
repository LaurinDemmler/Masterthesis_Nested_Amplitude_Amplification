import random
from datastructures.amplitudes import BinaryAmplitudeState
import math
from generator.GurobiSolver import _solution_from_bitstring
from scipy.stats import beta
import numpy as np


class Grover:
    """
    Grover when the provided BinaryAmplitudeState contains only the marked basis states,
    and is not normalized (it is only if there are no unmarked states at all in the subspace). We assume the remaining amplitude mass (1 - sum_good |a|^2) is uniformly
    distributed across all unmarked states.

    Notes:
      - The input state's amplitudes may sum to less than 1 (non-normalized). Let G = sum |a_g|^2 over
        the provided good states. We conceptually complete the state by assigning the remaining mass
        1 - G uniformly to the bad subspace. Thus the initial success probability used by Grover is p = G.
      - The result returned contains amplitudes only for the marked states, rescaled by Grover’s
        two-dimensional rotation factor f_marked. Bad states are implicit (uniform)
        and are not enumerated in the returned BinaryAmplitudeState.
    """

    def __init__(self, initial_state: BinaryAmplitudeState, verbose: bool = False):
        self.initial_state = initial_state
        # Marked set = exactly the states present in the initial BinaryAmplitudeState
        self._marked = list(self.initial_state._state.keys())
        # Compute success prob p from the provided good amplitudes only
        self._p = self._success_probability()
        self._iterations = self.optimal_iterations()
        self.inner_iterations = False
        self.inner_iterations_confidence = []
        self.outer_iterations_confidence = []
        self.lamda = 1.2

        if (
            not isinstance(self.initial_state.num_qubits, int)
            or self.initial_state.num_qubits <= 0
        ):
            raise ValueError("Initial state's num_qubits must be a positive integer.")
        if self._p < -1e-12 or self._p > 1.0 + 1e-12:
            raise ValueError(
                "Sum of squared amplitudes over provided good states must be in [0, 1]."
            )
        self.verbose = verbose

        if self.verbose:
            self.print_constructor_info()

    def print_constructor_info(self):
        p_good = self.get_good_amplitude(self.initial_state)
        print("Grover constructor info:")
        print(f"  Initial state num_qubits: {self.initial_state.num_qubits}")
        print(f"  Total number of marked states: {len(self._marked)}")
        print(f"  p_good (sum |a_g|^2 over provided good states): {p_good}")
        print(f"  p_bad (implicitly bad states): {1-p_good}")
        print(
            "-------------------------------------------------------------------------"
        )
        if self._iterations == 0:
            print(
                "  No Grover iterations will be performed (optimal iterations = 0) since p_success > 0.5 or p_success = 0."
            )
        else:
            print(f"  Optimal number of Grover iterations: {self._iterations}")
        print(
            "-------------------------------------------------------------------------"
        )

    def _success_probability(self) -> float:
        """
        p = sum |a_g|^2 over the provided good states.
        We assume the remaining mass (1 - p) lies uniformly in the bad subspace.
        """
        p = 0.0
        for _, a in self.initial_state._state.items():
            if a is None:
                continue
            p += abs(a) ** 2
        # Clamp to [0,1] to avoid numerical drift
        p = max(0.0, min(1.0, p))
        return p

    def get_result(self) -> BinaryAmplitudeState:
        """
        Return the marked-component of the final state after n Grover iterations using:
          |psi_n> = [sin((2n+1)θ)/sin θ] Π|psi_0> + [cos((2n+1)θ)/cos θ] (I−Π)|psi_0>,
        where sin^2 θ = p = sum |a_g|^2 over provided good states.

        The returned BinaryAmplitudeState contains only the marked states, rescaled by f_marked.
        The bad subspace is implicit and not enumerated.
        """
        success_prob = self._p
        tol = 1e-16
        out_state = BinaryAmplitudeState(self.initial_state.num_qubits)

        # Edge cases
        if success_prob <= tol:
            # No marked amplitude initially: marked components remain unchanged
            for s, a in self.initial_state._state.items():
                if a is None:
                    continue
                out_state.set_amplitude(s, a)
            return out_state

        if 1.0 - success_prob <= tol:
            # Entire state lies in the marked subspace: global phase (-1)^n on marked amplitudes
            phase = (-1.0) ** self._iterations
            for s, a in self.initial_state._state.items():
                if a is None:
                    continue
                out_state.set_amplitude(s, phase * a)
            return out_state

        # General case
        sin_theta = math.sqrt(success_prob)
        cos_theta = math.sqrt(1.0 - success_prob)
        theta = math.asin(sin_theta)
        angle = (2 * self._iterations + 1) * theta

        f_marked = math.sin(angle) / sin_theta

        # Rescale marked amplitudes
        for s, a in self.initial_state._state.items():
            if a is None:
                continue
            a = complex(a)
            out_state.set_amplitude(s, f_marked * a)
        """print(
            f"Total amplitude of good states after {self._iterations} Grover iterations: {out_state.total_probability()}"
        )"""
        return out_state

    def optimal_iterations(self) -> int:
        """n* ≈ floor(pi/(4θ) - 1/2) with sin^2 θ = p."""
        p = self._p
        tol = 1e-16
        if p <= tol or 1.0 - p <= tol:
            return 0
        theta = math.asin(math.sqrt(p))
        return max(0, int(round(math.pi / (4.0 * theta) - 0.5)))

    def get_good_amplitude(self, state: BinaryAmplitudeState) -> float:
        """
        Return p_good for the conceptual normalized state with respect to self._marked. Sum probabilities over marked and unmarked support present in 'state'.
        """
        tol = 1e-16
        p_good = 0.0
        p_bad_explicit = 0.0
        for s, a in state._state.items():
            if a is None:
                continue
            prob = abs(a) ** 2
            if s in self._marked:
                p_good += prob
            else:
                p_bad_explicit += prob

        return float(p_good)
    
    def set_amplitudes_after_grover(self, grover_iterations, out_state):
        if self._p <= 0.0:
            f_marked = 0.0
        else:
            theta = math.asin(math.sqrt(self._p))
            f_marked = math.sin((2 * grover_iterations + 1) * theta) / math.sqrt(
                    self._p
                )

        for bitstring, amp in self.initial_state._state.items():
            if amp is None:
                continue
            out_state.set_amplitude(bitstring, f_marked * complex(amp))


    def inner_iteration_finder(
        self, remaining_budget=None, cost_factor=None,
        gatec_state_prep=None, gatec_oracle=None,
    ) -> BinaryAmplitudeState:
        """
        Estimate a lower bound on the success probability via sampling, while also sampling
        Grover iteration counts (similar to Grover Adaptive Search). 

        Gate-count budgeting: if gatec_state_prep and gatec_oracle are provided,
        the per-step cost is (2g+1)*gatec_state_prep + g*gatec_oracle.
        Otherwise falls back to the old cost_factor * (2g+1).

        Returns:
            The BinaryAmplitudeState after the last applied Grover iteration.
        """
        use_gate_cost = (gatec_state_prep is not None and gatec_oracle is not None)
        upper_sampling_limit = 5 #how often do we maximally sample to try and verify a grover iteration
        lower_bound_certified = False
        grover_iterations = 0
        k = 1.0  # Initial range for GAS iteration sampling
        lower_sampling_interval = 0
        if not use_gate_cost and cost_factor is None:
            cost_factor = self.initial_state.num_qubits
        k_max = math.pi * 2 ** (self.initial_state.num_qubits / 2) / 4 #theoretically worst we could do if initial amplitude was 1/2**N
        if remaining_budget is None:
            remaining_budget = float('inf')
        else:
            k_max = float('inf')
            
        while not lower_bound_certified and k < k_max and remaining_budget > 0:
            grover_iterations = random.randrange(
                lower_sampling_interval, int(math.ceil(k))
            )
            
            # Prepare state after the sampled number of Grover iterations.
            out_state = BinaryAmplitudeState(self.initial_state.num_qubits)
            self.set_amplitudes_after_grover(grover_iterations, out_state)

            # Sample measurements from this state to certify the lower bound.
            total_samples = 0
            good_samples = 0

            while total_samples < upper_sampling_limit and not lower_bound_certified:
                measurement = out_state.sample_from()
                if use_gate_cost:
                    remaining_budget -= (2*grover_iterations + 1) * gatec_state_prep + grover_iterations * gatec_oracle
                else:
                    remaining_budget -= cost_factor * (2*grover_iterations + 1)
                if remaining_budget < 0:
                    return None
                self.inner_iterations_confidence.append(grover_iterations)

                if measurement in self._marked:
                    good_samples += 1
                total_samples += 1
                
                if total_samples == 2 and good_samples == 0:
                    #if 2/2 miss, sample a smaller interval
                    lower_sampling_interval = grover_iterations // 2
                    break
                if total_samples > 2 and total_samples - good_samples != 0:
                    #only allow 100% success
                    break
                if good_samples == upper_sampling_limit:
                    lower_bound_certified = True
                

            # Sample from exponentially increasing range
            if not lower_bound_certified:
                k = min(k_max, math.ceil(self.lamda * k))

        if not lower_bound_certified:
            # Could not certify even at max range, assuming 0 good states
            self.inner_iterations = 0
        else:
            self.inner_iterations = grover_iterations  # Keep the last grover_iterations used for outer statistics

        return out_state



    def grover_adaptive_search(
        self,
        current_best_solution,
        knapsack_instance,
        optimal_inner_iterations: int = None,
        depth: int = None,
        remaining_budget=None,
        cost_factor=None,
        gatec_state_prep=None,
        gatec_oracle=None,
    ) -> tuple[bool, str | None]:
        """
        Grover Adaptive Search with sampling over Grover iterations. As soon as
        an improved solution is found, abort and return it.

        Gate-count budgeting: if gatec_state_prep and gatec_oracle are provided,
        the per-step cost is (2j+1)*gatec_state_prep + j*gatec_oracle.
        Otherwise falls back to the old cost formula.
        """
        use_gate_cost = (gatec_state_prep is not None and gatec_oracle is not None)
        vals = (optimal_inner_iterations, depth)
        if not use_gate_cost and not (all(v is None for v in vals) or all(v is not None for v in vals)):
            raise ValueError("global_upper_sampling_bounds, optimal_inner_iterations, and depth must be either all None or all not None.")
        if not use_gate_cost and cost_factor is None:
            cost_factor = depth
        
        found_better_solution = False
        measurement = None
        k = 1
        iteration_count = 0
        if remaining_budget is None:
            remaining_budget = float('inf')
            max_iterations = np.log(knapsack_instance.num_items**8)/np.log(self.lamda) # stale code but technically here one could set a max iter count instead of a budget
        else:
            max_iterations = float('inf')

        while iteration_count < max_iterations and not found_better_solution and remaining_budget > 0:
            if optimal_inner_iterations is not None:
                # a favourable sampling range for nested, also works with normal GAS sampling
                k = max(
                        1,
                        (knapsack_instance.num_items * (2 * int(math.ceil(self.lamda**iteration_count + 1)))) / (2 * (2 * optimal_inner_iterations * depth + knapsack_instance.num_items)) - 0.5
                    )
            iteration_count += 1 
            grover_iterations = random.randrange(0, int(math.ceil(k)))
            if use_gate_cost:
                remaining_budget -= (2*grover_iterations + 1) * gatec_state_prep + grover_iterations * gatec_oracle
            else:
                remaining_budget -= (2*grover_iterations + 1) * (2*optimal_inner_iterations*cost_factor + knapsack_instance.num_items) if optimal_inner_iterations is not None else knapsack_instance.num_items*(2*grover_iterations + 1)
            if remaining_budget < 0:
                break
            self.outer_iterations_confidence.append(grover_iterations)
            
            out_state = BinaryAmplitudeState(self.initial_state.num_qubits)
            self.set_amplitudes_after_grover(grover_iterations, out_state)
            measurement = out_state.sample_from()
            if measurement is None:
                # Expand search range if sampling fails to produce a measurement
                k = self.lamda * k
                continue

            intermediate_solution = _solution_from_bitstring(
                knapsack_instance, measurement
            )

            if intermediate_solution.total_value <= current_best_solution.total_value:
                k = self.lamda * k
            else:
                found_better_solution = True
                # if better solution is found measure again to see whether it can be improved with the same iterations.
                measurement_new = out_state.sample_from()
                
                if use_gate_cost:
                    remaining_budget -= (2*grover_iterations + 1) * gatec_state_prep + grover_iterations * gatec_oracle
                else:
                    remaining_budget -= (2*grover_iterations + 1) * (2*optimal_inner_iterations*cost_factor + knapsack_instance.num_items) if optimal_inner_iterations is not None else knapsack_instance.num_items*(2*grover_iterations + 1)
                if remaining_budget < 0:
                    break
                self.outer_iterations_confidence.append(grover_iterations)
                if measurement_new is None:
                    pass
                else:
                    intermediate_solution_new = _solution_from_bitstring(
                    knapsack_instance, measurement_new
                    )
                    if intermediate_solution_new.total_value > intermediate_solution.total_value:
                        #print(f"Found better solution with value {intermediate_solution_new.total_value} compared to {intermediate_solution.total_value} in the same iteration, resampling to verify")
                        measurement = measurement_new
                        intermediate_solution = intermediate_solution_new
                break
        return found_better_solution, measurement, remaining_budget
