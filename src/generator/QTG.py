import math
from datastructures.OptimizerSolution import OptimizerSolution
from datastructures.amplitudes import BinaryAmplitudeState
import numpy as np
from collections import Counter

class AmplitudeAssigner:
    """Class for assigning amplitudes to quantum states based on their values after QTG.
    Needs a list of feasible states. Bias refers to Hadaamard bias, 0.0 recovers normal Hadamard."""
    def __init__(self, list_of_states:list, bias=0.0, greedy_bitstring: str | None = None):
        self.number_of_states = len(list_of_states)
        self.states = list_of_states
        self.bias = bias
        # 0 recorvers normal Hadamard
        self.take_amplitude = np.sqrt(1.0 / (self.bias + 2.0))
        self.skip_amplitude = np.sqrt((self.bias + 1.0) / (self.bias + 2.0))
        self.amplitudes = []
        if greedy_bitstring is not None:
            self.greedy_bitstring = greedy_bitstring
        else:
            raise ValueError("greedy_bitstring must be provided.")

    def single_state_amplitude(self, state, knapsack):
        """Calculate amplitude for a single basis state based on knapsack constraints after QTG.
        Items are assumed to already be in the correct order (the Solver pre-sorts the knapsack)."""
        amplitude = 1
        capacity = knapsack.capacity
        for i, bit in enumerate(state):
            weight = knapsack.items[i].weight
            if capacity - weight >= 0 and bit == '1' and self.greedy_bitstring[i] == '1':
                amplitude *= float(self.skip_amplitude)
                capacity -= weight
            elif capacity - weight >= 0 and bit == '1' and self.greedy_bitstring[i] == '0':
                amplitude *= float(self.take_amplitude)
                capacity -= weight
            elif capacity - weight >= 0 and bit == '0' and self.greedy_bitstring[i] == '1':
                amplitude *= float(self.take_amplitude)
            elif capacity - weight >= 0 and bit == '0' and self.greedy_bitstring[i] == '0':
                amplitude *= float(self.skip_amplitude)
            elif capacity - weight < 0 and bit == '1':
                raise ValueError("State exceeds knapsack capacity.")
            elif capacity - weight < 0 and bit == '0':
                # bypass gate (no amplitude change)
                continue
        return amplitude
    
    def get_state(self, knapsack):
        """Generate BinaryAmplitudeState after passing through full QTG circuit."""
        self.amplitudes.clear()
        for state in self.states:
            amp = self.single_state_amplitude(state, knapsack)
            self.amplitudes.append(amp)
        return BinaryAmplitudeState(num_qubits=len(self.greedy_bitstring),
                                    amplitudes={state: amp for state, amp in zip(self.states, self.amplitudes)})
        
class AmplitudeAssignerQTGk(AmplitudeAssigner):
    """Class for assigning amplitudes to a subset of quantum states based on their values after QTG to depth k.
    Inherits from AmplitudeAssigner. Bias refers to Hadaamard bias, 0.0 recovers normal Hadamard."""
    def __init__(self, list_of_states:list, bias=0.0, up_to_depth=None, greedy_bitstring: str | None = None):
        super().__init__(list_of_states, bias, greedy_bitstring)
        self.up_to_depth = up_to_depth

    def single_state_amplitude(self, state: str, knapsack) -> float:
        """Prefix-only amplitude up to self.up_to_depth (k).
        Items are assumed to already be in the correct order (the Solver pre-sorts the knapsack)."""
        amplitude = 1.0
        capacity = knapsack.capacity
        for i, bit in enumerate(state):
            if i >= self.up_to_depth:
                raise ValueError("State length exceeds up_to_depth.")
            weight = knapsack.items[i].weight
            if capacity - weight >= 0 and bit == '1' and self.greedy_bitstring[i] == '1':
                amplitude *= float(self.skip_amplitude)
                capacity -= weight
            elif capacity - weight >= 0 and bit == '0' and self.greedy_bitstring[i] == '0':
                amplitude *= float(self.skip_amplitude)
            elif capacity - weight >= 0 and bit == '1' and self.greedy_bitstring[i] == '0':
                amplitude *= float(self.take_amplitude)
                capacity -= weight
            elif capacity - weight >= 0 and bit == '0' and self.greedy_bitstring[i] == '1':
                amplitude *= float(self.take_amplitude)
            elif capacity - weight < 0 and bit == '1':
                raise ValueError("State exceeds knapsack capacity within prefix.")
            else:
                # capacity - weight < 0 and bit == '0' -> bypass gate
                continue
        return float(amplitude)

    def get_state(self, knapsack: object) -> BinaryAmplitudeState:
        """Build a good-only inner state of length k (prefixes2)."""
        amplitudes = {}
        for s in self.states:
            if len(s) > self.up_to_depth:
                raise ValueError("State length exceeds up_to_depth.")
            a_pref = self.single_state_amplitude(s, knapsack)  # amplitude of prefix after QTGk
            amplitudes[s] = a_pref
        return BinaryAmplitudeState(
            num_qubits=self.up_to_depth,
            amplitudes=amplitudes
        )
    
class AmplitudeAssignerQTGnk(AmplitudeAssigner):
    """QTG from a depth k to the end n (suffix-only) on full-length labels.
    - from_depth = k: capacity reduced by the prefix s[:k] first, then apply gates for indices k..n-1."""
    def __init__(self, state: BinaryAmplitudeState, globally_marked_states: list[OptimizerSolution], bias: float = 0.0, from_depth: int | None = None, greedy_bitstring: str | None = None):
        # Use internal _state keys as in the original implementation to maintain behavior, but we also need the starting amplitudes here
        super().__init__([bs for bs in state._state.keys()], bias, greedy_bitstring)
        if from_depth is None or from_depth < 0:
            raise ValueError("from_depth must be a non-negative integer.")
        self.from_depth = int(from_depth)
        self._marked = [sol.bitstring for sol in globally_marked_states]

    def single_state_amplitude(self, state: str, amplitude: float, knapsack) -> float:
        """Suffix-only amplitude from self.from_depth onward, using leftover capacity after the prefix.
        Items are assumed to already be in the correct order (the Solver pre-sorts the knapsack)."""
        capacity = knapsack.capacity
        # Subtract prefix weights
        for i in range(self.from_depth):
            if state[i] == '1':
                capacity -= knapsack.items[i].weight
        if capacity < 0:
            raise ValueError("Prefix exceeds knapsack capacity; it shouldn't.")
        # Apply suffix gates
        for i in range(self.from_depth, len(state)):
            bit = state[i]
            weight = knapsack.items[i].weight
            if capacity - weight >= 0 and bit == '1' and self.greedy_bitstring[i] == '1':
                amplitude *= float(self.skip_amplitude)
                capacity -= weight
            elif capacity - weight >= 0 and bit == '0' and self.greedy_bitstring[i] == '0':
                amplitude *= float(self.skip_amplitude)
            elif capacity - weight >= 0 and bit == '1' and self.greedy_bitstring[i] == '0':
                amplitude *= float(self.take_amplitude)
                capacity -= weight
            elif capacity - weight >= 0 and bit == '0' and self.greedy_bitstring[i] == '1':
                amplitude *= float(self.take_amplitude)
            elif capacity - weight < 0 and bit == '1':
                amplitude = None
                break
            else:
                # capacity - weight < 0 and bit == '0'
                continue
        return amplitude

    def get_state(self, knapsack: object, initial_state: BinaryAmplitudeState) -> BinaryAmplitudeState:
        """Applies QTG from depth k to n on the provided initial_state. """
        final_state = BinaryAmplitudeState(num_qubits=knapsack.num_items)
        marked = 0
        for bitstring in self._marked:
            marked += 1
            partial_amplitude = initial_state.get_amplitude(bitstring[0:self.from_depth])
            if partial_amplitude is False:
                print(f"Missing prefix amplitude for bitstring {bitstring}. Amplitude {initial_state.get_amplitude(bitstring[0:self.from_depth])}. This is marked state number {marked} out of {len(self._marked)}. Instancename: {knapsack.name}")
                """raise ValueError(f"Missing prefix amplitude for bitstring {bitstring}. Amplitude {initial_state.get_amplitude(bitstring[0:self.from_depth])}. This is marked state number {marked} out of {len(self._marked)}. Instancename: {knapsack.name}")"""
                continue
            amp_post = self.single_state_amplitude(bitstring, partial_amplitude, knapsack)
            final_state.set_amplitude(bitstring, amp_post)
        return final_state