import math
from typing import List


# ── Helpers ───────────────────────────────────────────────────────────────

def numbits(x: int) -> int:
    """Number of bits needed to represent non-negative integer x.
    numbits(x) = ceil(log2(x + 1)), with numbits(0) = 1."""
    if x <= 0:
        return 1
    return math.ceil(math.log2(x + 1))


def lso(x: int) -> int:
    """Position (1-indexed from LSB) of the least significant one in x.
    Returns 0 for x == 0 (convention: no '1' bit)."""
    if x == 0:
        return 0
    pos = 1
    while (x >> (pos - 1)) & 1 == 0:
        pos += 1
    return pos


def bit_i(x: int, i: int) -> int:
    """Get the i-th bit (1-indexed from LSB) of integer x."""
    return (x >> (i - 1)) & 1


# ── QFT (Section 4.5.2) ──────────────────────────────────────────────────

def gatec_qft(register_bits: int) -> int:
    """Gate count for QFT on a register of size `register_bits`."""
    return register_bits * (register_bits + 1) // 2


# ── Direct addition/subtraction (Section 4.5.3 / 4.5.4) ─────────────────

def gatec_direct(value: int, register_bits: int) -> int:
    """Gate count for direct addition/subtraction of `value` to/from
    a QFT-ed register of size `register_bits`.
    Eq. (AdditionGates) / (SubtractionGates):
      gatec_DIRECT = 3 * (register_bits - LSO(value)) + 1
    """
    if value == 0:
        return 0
    return 3 * (register_bits - lso(value)) + 1


# ── Capacity comparison >= w_m (Section 4.5.5) ───────────────────────────

def gatec_comparison_geq(w_m: int, numbits_c: int) -> int:
    """Gate count for the multi-controlled Hadamard implementing >= w_m.
    Returns min of two implementation strategies.
    Eq. (FirstStrategyCapacityComparisonGates), (SecondStrategyCapacityComparisonGates), (CapacityComparisonGates).
    """
    nb_w = numbits(w_m)
    # Strategy 1: apply H iff tilde_c > w_m - 1
    strategy1 = sum(
        (1 - bit_i(w_m - 1, i)) * (2 * (numbits_c - i) + 1)
        for i in range(1, nb_w + 1)
        if numbits_c - i >= 0
    )
    # Strategy 2: apply H unconditionally, then inverse iff tilde_c < w_m
    strategy2 = 1 + sum(
        bit_i(w_m, i) * (2 * (numbits_c - i) + 1)
        for i in range(1, nb_w + 1)
        if numbits_c - i >= 0
    )
    return min(strategy1, strategy2)


# ── Profit comparison > T  (phase oracle) (Section 4.5.5) ────────────────

def gatec_comparison_gt(T: int, numbits_P: int) -> int:
    """Gate count for the phase oracle P > T.
    Eq. (FirstStrategyProfitComparisonGates), (SecondStrategyProfitComparisonGates), (ProfitComparisonGates).
    """
    if T < 0:
        return 0
    nb_T = numbits(T)
    strategy1 = sum(
        (1 - bit_i(T, i)) * (2 * (numbits_P - i) + 1)
        for i in range(1, nb_T + 1)
        if numbits_P - i >= 0
    )
    strategy2 = 1 + sum(
        bit_i(T + 1, i) * (2 * (numbits_P - i) + 1)
        for i in range(1, nb_T + 1)
        if numbits_P - i >= 0
    )
    return min(strategy1, strategy2)


# ── Zero reflection signflip_0 (Section 4.5.5) ──────────────────────────

def gatec_signflip0(n: int) -> int:
    """Gate count for the reflection signflip_0 on n path-register qubits.
    Eq. (ZeroComparisonGates): 2n - 1.
    """
    return max(2 * n - 1, 0)


# ── QTG (Section 4.5.6) ──────────────────────────────────────────────────

def gatec_qtg(weights: List[int], profits: List[int],
              numbits_c: int, numbits_P: int,
              skip_last_sub: bool = True) -> int:
    """Gate count for a QTG block processing the given items.

    Parameters
    ----------
    weights : list[int]
        Item weights in processing order.
    profits : list[int]
        Item profits in processing order.
    numbits_c : int
        Number of qubits in the capacity register (for the full problem).
    numbits_P : int
        Number of qubits in the profit register (for the full problem).
    skip_last_sub : bool
        True  -> last item has addition only, no subtraction (last block / full QTG).
        False -> all items have both addition and subtraction (intermediate block).
    """
    n = len(weights)
    if n == 0:
        return 0

    # Line 1: comparisons + QFTs
    comp_gates = sum(gatec_comparison_geq(w, numbits_c) for w in weights)
    qft_profit_gates = 2 * gatec_qft(numbits_P)

    if skip_last_sub:
        n_sub = n - 1
    else:
        n_sub = n

    qft_capacity_gates = 2 * n_sub * gatec_qft(numbits_c)

    # Line 2: merged copy/uncopy of ancilla qubits for parallelised arithmetic
    copy_gates = 0
    for m in range(n_sub):
        copy_gates += 2 * (max(numbits_P, numbits_c) - min(lso(profits[m]), lso(weights[m])))
    if skip_last_sub:
        # Last item: only addition (no subtraction) → copy for profit only
        copy_gates += 2 * (numbits_P - lso(profits[-1]))
    # If not skip_last_sub, all items are already counted in the loop above

    # Line 3: actual rotation gates (additions + subtractions)
    rotation_gates = 0
    for m in range(n_sub):
        rotation_gates += numbits_P - lso(profits[m]) + numbits_c - lso(weights[m]) + 2
    if skip_last_sub:
        # Last item: addition only
        rotation_gates += numbits_P - lso(profits[-1]) + 1
    # If not skip_last_sub, all items already counted

    return comp_gates + qft_profit_gates + qft_capacity_gates + copy_gates + rotation_gates


# ── QSearch / QMaxSearch cost for a sequence of iterations ────────────────

def gatec_qsearch_step(j: int, gatec_state_prep: int, gatec_oracle: int) -> int:
    """Gate count for one QSearch step at Grover power j:
    (2j+1) * gatec_state_prep + j * gatec_oracle.
    Eq. (QSearchGates).
    """
    return (2 * j + 1) * gatec_state_prep + j * gatec_oracle


# ── Cut upper-bound register build (Exact-rational approach) ───────────────

def gatec_cut_ub_build(numbits_P: int, numbits_c: int, numbits_UB: int,
                       p: int, q: int) -> int:
    """Gate count for building the upper-bound register U = q*P + p*tilde_c.

    Parameters
    ----------
    numbits_P : int
        Number of bits in the profit register P.
    numbits_c : int
        Number of bits in the capacity register tilde_c.
    numbits_UB : int
        Number of bits in the UB register.
    p : int
        The numerator of rho = p/q (value of best remaining item).
    q : int
        The denominator of rho = p/q (weight of best remaining item).

    Returns
    -------
    int
        Total gate count = 2*QFT(numbits_UB) + G_A + G_B,
        where:
          G_A = 3 * sum over i=1..numbits_P of max(0, numbits_UB - LSO(q) - i + 2)
          G_B = 3 * sum over i=1..numbits_c of max(0, numbits_UB - LSO(p) - i + 2)
    """
    # QFT wrap (forward and inverse)
    g_qft = 2 * gatec_qft(numbits_UB)

    # Block A: add q*P bitwise, sum over all bits
    lso_q = lso(q)
    g_A = 3 * sum(max(0, numbits_UB - lso_q - i + 2) for i in range(1, numbits_P + 1))

    # Block B: add p*tilde_c bitwise, sum over all bits
    lso_p = lso(p)
    g_B = 3 * sum(max(0, numbits_UB - lso_p - i + 2) for i in range(1, numbits_c + 1))

    return g_qft + g_A + g_B


# ── High-level cost computation for full solving approaches ──────────────

class QTGResourceEstimator:
    """Precomputes QTG gate counts for a knapsack instance and provides
    cost-per-iteration functions for the inner/outer/global Grover loops.

    Parameters
    ----------
    weights : list[int]
        All item weights, in processing order.
    profits : list[int]
        All item profits, in processing order.
    capacity : int
        Knapsack capacity.
    profit_upper_bound : int
        Upper bound on the optimal profit (e.g. sum of all profits or fractional optimum).
    """

    def __init__(self, weights: List[int], profits: List[int],
                 capacity: int, profit_upper_bound: int):
        self.weights = weights
        self.profits = profits
        self.n = len(weights)
        self.capacity = capacity
        self.numbits_c = numbits(capacity)
        self.numbits_P = numbits(profit_upper_bound)

    # ── Precomputed QTG gate counts ──────────────────────────────────

    def gatec_qtg_full(self) -> int:
        """Gate count for the full QTG on all n items (skip last subtraction)."""
        return gatec_qtg(self.weights, self.profits,
                         self.numbits_c, self.numbits_P, skip_last_sub=True)

    def gatec_qtg_partial(self, depth: int) -> int:
        """Gate count for partial QTG on first `depth` items.
        Does NOT skip last subtraction (more items follow)."""
        return gatec_qtg(self.weights[:depth], self.profits[:depth],
                         self.numbits_c, self.numbits_P, skip_last_sub=False)

    def gatec_qtg_remaining(self, depth: int) -> int:
        """Gate count for QTG on remaining items [depth:n] (skip last sub)."""
        return gatec_qtg(self.weights[depth:], self.profits[depth:],
                         self.numbits_c, self.numbits_P, skip_last_sub=True)

    # ── Oracle gate counts ───────────────────────────────────────────

    def gatec_oracle_full(self, T) -> int:
        """Gate count for signflip_0 + phase oracle (> T) — used by outer/global Grover."""
        return gatec_signflip0(self.n) + gatec_comparison_gt(int(T), self.numbits_P)

    def gatec_oracle_inner(self, depth: int, T) -> int:
        """Gate count for the inner Grover's oracle at partial depth.
        signflip_0 on depth qubits + comparison > T on profit register."""
        return gatec_signflip0(depth) + gatec_comparison_gt(int(T), self.numbits_P)

    # ── Per-iteration cost functions ─────────────────────────────────

    def gatec_inner_step(self, g: int, depth: int, T: int) -> int:
        """Gate cost of one inner QSearch step at Grover power g.
        (2g+1) * gatec_QTG_k + g * gatec_oracle_inner
        """
        return gatec_qsearch_step(
            g,
            self.gatec_qtg_partial(depth),
            self.gatec_oracle_inner(depth, T),
        )

    def gatec_outer_state_prep(self, depth: int, inner_iters: int, T: int) -> int:
        """Gate count for one application of the outer state preparation:
        A_outer = QTG_k · InnerGrover^{inner_iters} · QTG_{n-k}

        = (2*inner_iters + 1)*QTG_k + inner_iters*oracle_inner + QTG_{n-k}
        """
        return ((2 * inner_iters + 1) * self.gatec_qtg_partial(depth)
                + inner_iters * self.gatec_oracle_inner(depth, T)
                + self.gatec_qtg_remaining(depth))

    def gatec_outer_step(self, j: int, depth: int, inner_iters: int, T: int) -> int:
        """Gate cost of one outer QSearch step at Grover power j.
        (2j+1)*gatec_A_outer + j*(gatec_signflip0(n) + gatec_gt_T)
        """
        return gatec_qsearch_step(
            j,
            self.gatec_outer_state_prep(depth, inner_iters, T),
            self.gatec_oracle_full(T),
        )

    def gatec_global_step(self, j: int, T: int) -> int:
        """Gate cost of one global QSearch step at Grover power j.
        (2j+1)*gatec_QTG_full + j*(gatec_signflip0 + gatec_gt_T)
        """
        return gatec_qsearch_step(
            j,
            self.gatec_qtg_full(),
            self.gatec_oracle_full(T),
        )

    # ── Cut-specific methods (exact-rational upper-bound register) ──────────

    def _rho_pq(self, depth: int) -> tuple:
        """Compute p and q for the exact-rational rho = p/q = max_{j > depth} v_j / w_j.

        Returns
        -------
        tuple[int, int]
            (p, q) where p = v_j*, q = w_j*, and j* = argmax_{j > depth} v_j / w_j.
        """
        if depth <= 0:
            raise ValueError(f"Cut depth must be > 0, got depth={depth}")
        if depth >= self.n:
            raise ValueError(f"Cut depth must be < n={self.n}, got depth={depth} (no remaining items)")

        best_j = depth
        best_density = self.profits[depth] / self.weights[depth]

        for j in range(depth + 1, self.n):
            density = self.profits[j] / self.weights[j]
            if density > best_density:
                best_density = density
                best_j = j

        return (self.profits[best_j], self.weights[best_j])

    def _numbits_UB(self, p: int, q: int) -> int:
        """Compute the bit width of the UB register holding U = q*P + p*tilde_c.

        Parameters
        ----------
        p : int
            Numerator of rho.
        q : int
            Denominator of rho.

        Returns
        -------
        int
            numbits(q * sum_of_profits + p * capacity)
        """
        sum_profits = sum(self.profits)
        max_UB = q * sum_profits + p * self.capacity
        return numbits(max_UB)

    def gatec_qtg_partial_cut(self, depth: int) -> int:
        """Gate count for partial QTG on first `depth` items plus UB register build.

        Returns: gatec_qtg_partial(depth) + gatec_cut_ub_build(...)
        """
        if depth <= 0:
            raise ValueError(f"Cut depth must be > 0, got depth={depth}")
        
        p, q = self._rho_pq(depth)
        numbits_UB = self._numbits_UB(p, q)
        ub_build = gatec_cut_ub_build(self.numbits_P, self.numbits_c, numbits_UB, p, q)

        return self.gatec_qtg_partial(depth) + ub_build

    def gatec_oracle_inner_cut(self, depth: int, T) -> int:
        """Gate count for the inner Grover's oracle with the cut (UB register).

        Compares U > q*T on an UB register of width numbits_UB.
        """
        if depth <= 0:
            raise ValueError(f"Cut depth must be > 0, got depth={depth}")

        p, q = self._rho_pq(depth)
        numbits_UB = self._numbits_UB(p, q)

        return gatec_signflip0(depth) + gatec_comparison_gt(q * int(T), numbits_UB)

    def gatec_inner_step_cut(self, g: int, depth: int, T: int) -> int:
        """Gate cost of one inner QSearch step at Grover power g with the cut.

        (2g+1) * gatec_qtg_partial_cut(depth) + g * gatec_oracle_inner_cut(depth, T)
        """
        return gatec_qsearch_step(
            g,
            self.gatec_qtg_partial_cut(depth),
            self.gatec_oracle_inner_cut(depth, T),
        )

    def gatec_outer_state_prep_cut(self, depth: int, inner_iters: int, T: int) -> int:
        """Gate count for one application of the outer state preparation with cut:

        A_outer^cut = QTG_k^cut · InnerGrover^{inner_iters} · QTG_{n-k}

        = (2*inner_iters + 1)*QTG_k^cut + inner_iters*oracle_inner^cut + QTG_{n-k}
        """
        return ((2 * inner_iters + 1) * self.gatec_qtg_partial_cut(depth)
                + inner_iters * self.gatec_oracle_inner_cut(depth, T)
                + self.gatec_qtg_remaining(depth))

    def gatec_outer_step_cut(self, j: int, depth: int, inner_iters: int, T: int) -> int:
        """Gate cost of one outer QSearch step at Grover power j with the cut.

        (2j+1)*gatec_outer_state_prep_cut(...) + j*gatec_oracle_full(T)
        """
        return gatec_qsearch_step(
            j,
            self.gatec_outer_state_prep_cut(depth, inner_iters, T),
            self.gatec_oracle_full(T),
        )
