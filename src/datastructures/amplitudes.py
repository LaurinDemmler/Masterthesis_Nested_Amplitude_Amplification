
import matplotlib.pyplot as plt
import numpy as np

class BinaryAmplitudeState:
    def __init__(self, num_qubits, amplitudes=None):
        """
        amplitudes: optional dict mapping bitstring -> amplitude (complex or castable to complex)
                    e.g., {"000": 1.0, "111": 0.0}
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("num_qubits must be a positive integer.")
        self.num_qubits = num_qubits
        
        self._state = {}  # sparse mapping: bitstring -> complex amplitude

        if amplitudes:
            for bitstr, amp in amplitudes.items():
                self._validate_bitstring(bitstr)
                if amp is None:
                    self._state[bitstr] = None
                else:
                    self._state[bitstr] = complex(amp)

    def _validate_bitstring(self, bitstr):
        if not isinstance(bitstr, str):
            raise ValueError("Bitstring must be a string.")
        if len(bitstr) != self.num_qubits:
            raise ValueError(f"Bitstring length must be {self.num_qubits}.")
        if any(c not in "01" for c in bitstr):
            raise ValueError("Bitstring may only contain '0' or '1'.")

    def set_amplitude(self, bitstring, amplitude):
        """Set amplitude for a basis state; overwrites if it exists."""
        self._validate_bitstring(bitstring)
        self._state[bitstring] = complex(amplitude)

    def get_amplitude(self, bitstring):
        """Get amplitude for a basis state; returns False if not present."""
        self._validate_bitstring(bitstring)
        return self._state.get(bitstring, False)

    def is_normalized(self, tol=1e-12):
        """
        Check if sum of |amplitude|^2 equals 1 within tolerance.
        Returns True if normalized, False otherwise.
        """
        norm_sq = sum(abs(a) ** 2 for a in self._state.values())
        return abs(norm_sq - 1.0) <= tol

    def probability(self, bitstring):
        """
        Return probability of a single basis state: |amplitude|^2.
        If the state is not stored, probability is 0.0.
        """
        a = self.get_amplitude(bitstring)
        return float(abs(a) ** 2)
    
    def delete_entry(self, bitstring):
        """Delete the entry for a basis state if it exists."""
        self._validate_bitstring(bitstring)
        if bitstring in self._state:
            del self._state[bitstring]
            
    def copy(self):
        """Return a deep copy of the BinaryAmplitudeState."""
        return BinaryAmplitudeState(
            num_qubits=self.num_qubits,
            amplitudes={bs: amp for bs, amp in self._state.items()}
        )
    
    def plot_state(self):
        """Plot the amplitude distribution as a bar chart with no x-axis labels.
        ONLY USE FOR SMALL NUMBER OF BITSTRINGS."""

        bitstrings = list(self._state.keys())
        amplitudes = [self._state[bs] for bs in bitstrings]
        probabilities = [abs(a) ** 2 if a is not None else 0.0 for a in amplitudes]

        plt.figure(figsize=(10, 6))
        plt.bar(bitstrings, probabilities)
        plt.xlabel('Bitstrings')
        plt.ylabel('Probability')
        plt.title('Amplitude State Probability Distribution')
        plt.tight_layout()
        plt.show()


    def plot_top_k(self, k=50):
        """
        Plot the top-k states by probability and aggregate the rest into an 'OTHER' bar.
        """
        items = [(bs, 0.0 if a is None else float(abs(a)**2))
                 for bs, a in self._state.items()]
        if not items:
            print("State is empty.")
            return
        items.sort(key=lambda kv: kv[1], reverse=True)
        top = items[:k]
        other_sum = sum(p for _, p in items[k:])
        labels = [s for s, _ in top] + (["OTHER"] if other_sum > 0 else [])
        values = [p for _, p in top] + ([other_sum] if other_sum > 0 else [])

        plt.figure(figsize=(12, 6))
        x = np.arange(len(labels))
        plt.bar(x, values, color=["tab:blue"] * len(top) + (["tab:orange"] if other_sum > 0 else []))
        plt.ylabel("Probability")
        plt.title(f"Top-{k} states (others aggregated)")
        # Show only a few labels; rotate to improve readability
        if len(labels) <= 40:
            plt.xticks(x, labels, rotation=90, fontsize=8)
        else:
            # Sparse labeling
            tick_idx = np.linspace(0, len(labels)-1, num=min(40, len(labels)), dtype=int)
            plt.xticks(tick_idx, [labels[i] for i in tick_idx], rotation=90, fontsize=8)
        plt.tight_layout()
        plt.show()
        
    def is_entangled_between(self, k, tol=1e-12):
        """
        Check if the pure state is entangled across the bipartition between qubit k and k+1.
        
        Interpretation:
        - The bitstring is split as [0..k-1] | [k..n-1].
        - Left subsystem has k qubits; right subsystem has n-k qubits.
        - Returns True if the Schmidt rank > 1 (i.e., entangled), False otherwise.
        
        Parameters:
            k (int): the cut index (0 < k < num_qubits).
            tol (float): numerical tolerance used to decide nonzero singular values.
                         This is used relatively to the largest singular value.
        
        Raises:
            ValueError: if k is not an integer in the valid range.
        """
        # Validate k
        if not isinstance(k, int):
            raise ValueError("k must be an integer.")
        if k <= 0 or k >= self.num_qubits:
            raise ValueError(f"k must satisfy 0 < k < {self.num_qubits}.")

        # Dimensions of the bipartition
        d_left = 1 << k
        d_right = 1 << (self.num_qubits - k)

        # Build the coefficient matrix M of shape (2^k, 2^(n-k))
        M = np.zeros((d_left, d_right), dtype=np.complex128)
        for bitstr, amp in self._state.items():
            if amp is None:
                continue
            left = bitstr[:k]
            right = bitstr[k:]
            i = int(left, 2)
            j = int(right, 2)
            M[i, j] = complex(amp)

        # Handle zero vector (no nonzero amplitudes)
        if not np.any(np.abs(M) > tol):
            return False  # treat as not entangled

        # Compute singular values; Schmidt rank = number of nonzero singular values
        s = np.linalg.svd(M, compute_uv=False)
        print(s)
        # Use a relative threshold to the largest singular value for scale invariance
        max_s = s[0]
        if max_s <= tol:
            # All singular values are (near) zero
            return False
        rank = int(np.sum(s > max_s * tol))

        return rank > 1
    
    def sample_from(self):
        """
        Sample a basis state according to the probability distribution defined by the amplitudes.
        Returns a bitstring.
        """
        bitstrings = list(self._state.keys())
        probabilities = [0.0 if a is None else abs(a) ** 2 for a in self._state.values()]
        total_prob = sum(probabilities)
        bad_prob = 1.0 - total_prob
        sample_array = bitstrings + [None]
        probabilities.append(bad_prob)
        #if any amplitudes in probabilities are negative make them 0
        probabilities = [max(0.0, p) for p in probabilities]
        return np.random.choice(sample_array, p=probabilities)
    def total_probability(self):
        """Return the total amplitude (sum of absolute values) of all stored states."""
        return sum(abs(a)**2 for a in self._state.values() if a is not None)