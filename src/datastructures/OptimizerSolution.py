from dataclasses import dataclass
@dataclass
class OptimizerSolution:
    """Data structure for all Gurobi Outputs."""
    items: list
    bitstring: str
    total_value: float
    total_weight: float