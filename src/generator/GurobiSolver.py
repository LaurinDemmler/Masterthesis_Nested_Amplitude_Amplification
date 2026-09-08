import gurobipy as gp
from gurobipy import GRB
from typing import Optional
from datastructures.OptimizerSolution import OptimizerSolution

class GurobiSolver:
    """Gurobi solver for 0/1 knapsack problem."""
    def __init__(self, verbose: bool = False, time_limit: Optional[float] = 20):
        self.verbose = verbose
        self.time_limit = time_limit

    def _set_common_params(self, m: gp.Model) -> None:
        if not self.verbose:
            m.Params.OutputFlag = 0
        if self.time_limit is not None:
            m.Params.TimeLimit = self.time_limit

    def _bitstring_from_vars(self, x, n, use_pool_value: bool = False, vals=None):
        # Fast path: use bulk-fetched values if provided
        if vals is not None:
            return "".join("1" if vals[i] > 0.5 else "0" for i in range(n))
        if use_pool_value:
            return "".join("1" if x[i].Xn > 0.5 else "0" for i in range(n))
        else:
            return "".join("1" if x[i].X > 0.5 else "0" for i in range(n))


    def gurobi_optimal_solution(self, knapsackInstance, time_limit: Optional[float] = None) -> OptimizerSolution:
        """
        Optimal 0/1 knapsack:
          maximize sum(values[i] * x[i])
          subject to sum(weights[i] * x[i]) <= capacity
        Returns: OptimizerSolution
        """

        if len(knapsackInstance.weights) != knapsackInstance.num_items or len(knapsackInstance.values) != knapsackInstance.num_items:
            raise ValueError("weights and values must have length n")

        m = gp.Model("knapsack_opt")
        self._set_common_params(m)

        x = m.addVars(knapsackInstance.num_items, vtype=GRB.BINARY, name="x")
        m.addConstr(
            gp.quicksum(knapsackInstance.weights[i] * x[i] for i in range(knapsackInstance.num_items)) <= knapsackInstance.capacity,
            name="capacity"
        )
        m.setObjective(gp.quicksum(knapsackInstance.values[i] * x[i] for i in range(knapsackInstance.num_items)), GRB.MAXIMIZE)

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimization did not reach optimality (status={m.Status}).")

        # Bulk-fetch values once, then build bitstring
        n = knapsackInstance.num_items
        var_list = [x[i] for i in range(n)]
        vals = m.getAttr(GRB.Attr.X, var_list)
        bitstr = self._bitstring_from_vars(x, n, use_pool_value=False, vals=vals)

        return _solution_from_bitstring(knapsackInstance, bitstr)

    def feasible_states_with_greedy_inequality(
        self,
        knapsackInstance,
        greedyThreshold: float,
        searchDepth: int,
        max_solutions: int = 200000,
        time_limit: Optional[float] = None
    ) -> list[OptimizerSolution]:
        """
        Feasibility search with:
          sum(w_i x_i) <= capacity
          sum_{i < searchDepth} v_i x_i >= greedyThreshold - sum_{i >= searchDepth} v_i
        Returns a list of OptimizerSolution.
        """
        depth = max(0, min(searchDepth, knapsackInstance.num_items))
        m = gp.Model("basis_states")
        self._set_common_params(m)

        x = m.addVars(depth, vtype=GRB.BINARY, name="x")
        m.addConstr(
            gp.quicksum(knapsackInstance.weights[i] * x[i] for i in range(depth)) <= knapsackInstance.capacity,
            name="capacity"
        )
        rhs = sum(knapsackInstance.values[i] for i in range(depth, knapsackInstance.num_items))
        m.addConstr(
            gp.quicksum(knapsackInstance.values[i] * x[i] for i in range(depth)) >= greedyThreshold - rhs+1,
            name="GreedyThreshold"
        )

        # Feasibility objective and pool
        m.setObjective(0.0, GRB.MINIMIZE)
        m.Params.PoolSearchMode = 2
        m.Params.PoolSolutions = max_solutions

        m.optimize()
        if m.SolCount == 0:
            pass
            #print(f'No feasible solution for instance {knapsackInstance.name} at depth {searchDepth} (total depth {knapsackInstance.num_items}) with threshold {greedyThreshold}.')
            #raise RuntimeError("No feasible solution found for the given greedy threshold and depth.")

        n = depth
        var_list = [x[i] for i in range(n)]

        solutions = []
        # Enumerate solutions with bulk attribute retrieval
        for k in range(min(m.SolCount, max_solutions)):
            m.Params.SolutionNumber = k
            vals = m.getAttr(GRB.Attr.Xn, var_list)
            bitstr = self._bitstring_from_vars(x, n, use_pool_value=True, vals=vals)
            solutions.append(bitstr)
        return [_solution_from_bitstring(knapsackInstance, s) for s in solutions]

    def feasible_states(
        self,
        knapsackInstance,
        time_limit: Optional[float] = None
    ) -> OptimizerSolution:
        """
        Gives all feasible states that satisfy the knapsack constraint.
        Returns a list of OptimizerSolution.
        """
        m = gp.Model("feasible_solutions")
        self._set_common_params(m)
        # Focus on feasibility enumeration
        m.Params.PoolSearchMode = 2
        m.Params.PoolSolutions = 1000000

        x = m.addVars(knapsackInstance.num_items, vtype=GRB.BINARY, name="x")
        m.addConstr(
            gp.quicksum(knapsackInstance.weights[i] * x[i] for i in range(knapsackInstance.num_items)) <= knapsackInstance.capacity,
            name="capacity"
        )
        m.setObjective(0.0, GRB.MINIMIZE)

        m.optimize()
        if m.SolCount == 0:
            raise RuntimeError("No feasible solutions found.")

        n = knapsackInstance.num_items
        var_list = [x[i] for i in range(n)]

        solutions = []
        # Enumerate solutions with bulk attribute retrieval
        for k in range(min(m.SolCount, 1000000)):
            m.Params.SolutionNumber = k
            vals = m.getAttr(GRB.Attr.Xn, var_list)
            bitstr = self._bitstring_from_vars(x, n, use_pool_value=True, vals=vals)
            solutions.append(bitstr)

        return [_solution_from_bitstring(knapsackInstance, s) for s in solutions]
    

    def feasible_states_with_cut_upper_bound(
        self,
        knapsackInstance,
        greedyThreshold: float,
        searchDepth: int,
        cut_degree: int = 0,
        max_solutions: int = 200000,
    ) -> list[OptimizerSolution]:
        """
        Directly enumerate all partial bitstrings (length searchDepth) whose
        cut upper bound strictly exceeds greedyThreshold.

        For cut_degree=0 the upper bound of a partial assignment is:
            UB = partial_value + remaining_capacity * best_remaining_efficiency
        where
            remaining_capacity = capacity - partial_weight
            best_remaining_efficiency = max(v_j / w_j for j >= searchDepth)

        The constraint  UB > greedyThreshold  is linearised as:
            sum((v_i - w_i * rho) * x_i) >= greedyThreshold - C * rho + eps
        with rho = best_remaining_efficiency.

        Uses direct DFS enumeration with pruning instead of Gurobi's
        solution pool for better performance.
        """
        if cut_degree != 0:
            raise NotImplementedError(f"Cut degree {cut_degree} is not yet implemented.")

        depth = max(0, min(searchDepth, knapsackInstance.num_items))

        # Best efficiency among remaining (undecided) items
        rho = 0.0
        for i in range(depth, knapsackInstance.num_items):
            w = knapsackInstance.weights[i]
            v = knapsackInstance.values[i]
            if w > 0:
                rho = max(rho, v / w)
            else:
                raise ValueError(f"Item {i} has non-positive weight {w}, which is not supported.")

        cap = knapsackInstance.capacity
        weights = knapsackInstance.weights

        # Precompute cut-bound coefficients: sum(c_i * x_i) >= rhs
        c = [knapsackInstance.values[i] - rho * weights[i] for i in range(depth)]
        rhs = greedyThreshold - rho * cap + 1e-6 # epsilon for strict inequality

        # suffix_pos[i] = sum of max(0, c[j]) for j in [i, depth)
        # Upper bound on cut-bound contribution from remaining items
        suffix_pos = [0.0] * (depth + 1)
        for i in range(depth - 1, -1, -1):
            suffix_pos[i] = suffix_pos[i + 1] + max(0.0, c[i])

        # Early exit: even all positive-coefficient items can't reach rhs
        if suffix_pos[0] < rhs:
            return []

        solutions = []
        limit = max_solutions

        def dfs(idx, pw, pc, bits):
            if len(solutions) >= limit:
                return
            if idx == depth:
                if pc >= rhs:
                    solutions.append(format(bits, f'0{depth}b'))
                return
            remaining = suffix_pos[idx + 1]
            # Try including item idx
            nw = pw + weights[idx]
            if nw <= cap:
                nc = pc + c[idx]
                if nc + remaining >= rhs:
                    dfs(idx + 1, nw, nc, (bits << 1) | 1)
            # Try excluding item idx
            if pc + remaining >= rhs:
                dfs(idx + 1, pw, pc, bits << 1)

        dfs(0, 0, 0.0, 0)

        return [_solution_from_bitstring(knapsackInstance, s) for s in solutions]

    def find_solution_with_value_T(
        self,
        knapsackInstance,
        T: float,
        allow_fallback: bool = True,
        time_limit: Optional[float] = None
    ) -> OptimizerSolution:
        """
        Tries to find a solution with total value exactly T.
        If not found and allow_fallback=True, returns a solution whose total value is closest to T.
        Returns: OptimizerSolution
        """
        # First: exact match attempt
        m = gp.Model("find_value_T_exact")
        self._set_common_params(m)

        x = m.addVars(knapsackInstance.num_items, vtype=GRB.BINARY, name="x")
        m.addConstr(
            gp.quicksum(knapsackInstance.weights[i] * x[i] for i in range(knapsackInstance.num_items)) <= knapsackInstance.capacity,
            name="capacity"
        )
        m.addConstr(
            gp.quicksum(knapsackInstance.values[i] * x[i] for i in range(knapsackInstance.num_items)) == T,
            name="value_T"
        )
        m.setObjective(0.0, GRB.MINIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            n = knapsackInstance.num_items
            var_list = [x[i] for i in range(n)]
            vals = m.getAttr(GRB.Attr.X, var_list)
            bitstr = self._bitstring_from_vars(x, n, use_pool_value=False, vals=vals)
            return _solution_from_bitstring(knapsackInstance, bitstr)

        if not allow_fallback:
            raise RuntimeError(f"No solution found with total value T={T} (status={m.Status}).")

        # Fallback: minimize |sum(values*x) - T|
        mf = gp.Model("find_value_T_closest")
        self._set_common_params(mf)

        xf = mf.addVars(knapsackInstance.num_items, vtype=GRB.BINARY, name="x")
        sum_val = gp.quicksum(knapsackInstance.values[i] * xf[i] for i in range(knapsackInstance.num_items))
        mf.addConstr(
            gp.quicksum(knapsackInstance.weights[i] * xf[i] for i in range(knapsackInstance.num_items)) <= knapsackInstance.capacity,
            name="capacity"
        )

        diff = mf.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="diff")
        mf.addConstr(diff >= sum_val - T, name="diff_pos")
        mf.addConstr(diff >= T - sum_val, name="diff_neg")


        mf.ModelSense = GRB.MINIMIZE
        mf.setObjective(diff)

        mf.optimize()
        if mf.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Fallback search failed (status={mf.Status}).")

        n = knapsackInstance.num_items
        var_list = [xf[i] for i in range(n)]
        vals = mf.getAttr(GRB.Attr.X, var_list)
        bitstr = self._bitstring_from_vars(xf, n, use_pool_value=False, vals=vals)
        return _solution_from_bitstring(knapsackInstance, bitstr)

    
def _solution_from_bitstring(knapsackInstance, bitstr: str) -> OptimizerSolution:
        """Takes Gurobi bitstring output and converts to OptimizerSolution."""
        selected_items = [knapsackInstance.items[i] for i, b in enumerate(bitstr) if b == "1"]
        total_value = float(sum(it.value for it in selected_items))
        total_weight = float(sum(it.weight for it in selected_items))
        return OptimizerSolution(
            items=selected_items,
            bitstring=bitstr,
            total_value=total_value,
            total_weight=total_weight
        )