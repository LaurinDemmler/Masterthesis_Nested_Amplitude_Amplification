from datastructures.OptimizerSolution import OptimizerSolution

def greedy_solver(knapsack_instance):
    """Greedy solver for 0/1 knapsack problem.
    Compute on a density-sorted copy, but output in the input knapsack's current order.
    Works with any item ordering (value-sorted, density-sorted, etc.).
    """
    greedy_copy = knapsack_instance._copy()
    greedy_copy.sort_items_by_density()

    total_value = 0
    total_weight = 0
    selected_ids = set()

    for item in greedy_copy.items:
        if total_weight + item.weight <= greedy_copy.capacity:
            selected_ids.add(item.id)
            total_weight += item.weight
            total_value += item.value

    # Build bitstring in the input knapsack's current order (using item IDs)
    bitstring = "".join(
        "1" if knapsack_instance.items[i].id in selected_ids else "0"
        for i in range(knapsack_instance.num_items)
    )

    selected_items = [
        knapsack_instance.items[i]
        for i in range(knapsack_instance.num_items)
        if knapsack_instance.items[i].id in selected_ids
    ]

    return OptimizerSolution(
        items=selected_items,
        bitstring=bitstring,
        total_value=total_value,
        total_weight=total_weight
    )
    
def fractional_greedy(knapsack_instance):
    """Fractional greedy solver for 0/1 knapsack problem.
    Compute on a density-sorted copy, but output in the input knapsack's current order.
    Works with any item ordering.
    """
    greedy_copy = knapsack_instance._copy()
    greedy_copy.sort_items_by_density()

    total_value = 0
    total_weight = 0
    selected_ids = set()

    for item in greedy_copy.items:
        if total_weight + item.weight <= greedy_copy.capacity:
            selected_ids.add(item.id)
            total_weight += item.weight
            total_value += item.value
        else:
            remaining_capacity = greedy_copy.capacity - total_weight
            fraction = remaining_capacity / item.weight
            total_value += item.value * fraction
            total_weight += item.weight * fraction
            break

    # Build bitstring in the input knapsack's current order
    bitstring = "".join(
        "1" if knapsack_instance.items[i].id in selected_ids else "0"
        for i in range(knapsack_instance.num_items)
    )

    selected_items = [
        knapsack_instance.items[i]
        for i in range(knapsack_instance.num_items)
        if knapsack_instance.items[i].id in selected_ids
    ]

    return OptimizerSolution(
        items=selected_items,
        bitstring=bitstring,
        total_value=total_value,
        total_weight=total_weight
    )
    
def frational_greedy_from_partial_bitstring(knapsack_instance, partial_solution: OptimizerSolution, density_sorted_copy=None):
    """Fractional greedy solver that starts from a given partial bitstring.
    Works with any item ordering. The items that are added greedily are always density-sorted, but the output bitstring is in the input knapsack's current order."""
    if density_sorted_copy is None:
        greedy_copy = knapsack_instance._copy()
        greedy_copy.sort_items_by_density()
    else:
        greedy_copy = density_sorted_copy

    total_value = 0
    total_weight = 0
    selected_ids = set()
    looked_at_ids = set()

    for i, bit in enumerate(partial_solution.bitstring):
        item = knapsack_instance.items[i]
        looked_at_ids.add(item.id)
        if bit == "1":
            selected_ids.add(item.id)
            total_weight += item.weight
            total_value += item.value

    for item in greedy_copy.items:
        if item.id in looked_at_ids:
            continue  # Skip already selected items
        if total_weight + item.weight <= greedy_copy.capacity:
            selected_ids.add(item.id)
            total_weight += item.weight
            total_value += item.value
        else:
            # Take fractional part
            remaining_capacity = greedy_copy.capacity - total_weight
            fraction = remaining_capacity / item.weight
            total_value += item.value * fraction
            total_weight += item.weight * fraction
            break  # Knapsack is full

    # Build bitstring in the input knapsack's current order
    bitstring = "".join(
        "1" if knapsack_instance.items[i].id in selected_ids else "0"
        for i in range(knapsack_instance.num_items)
    )

    selected_items = [
        knapsack_instance.items[i]
        for i in range(knapsack_instance.num_items)
        if knapsack_instance.items[i].id in selected_ids
    ]

    return OptimizerSolution(
        items=selected_items,
        bitstring=bitstring,
        total_value=total_value,
        total_weight=total_weight
    )
    
def upper_bound_cut_from_partial_bitstring(knapsack_instance, partial_solution: OptimizerSolution, cut_degree: int):
    """Computes an upper bound on the total value of any solution that can be reached from the given partial bitstring.
    Upper bound = partial_value + remaining_capacity * best_efficiency_of_remaining_items.
    """
    # Items already decided in the partial bitstring (both 0s and 1s up to depth)
    depth = len(partial_solution.bitstring)
    looked_at_ids = set(knapsack_instance.items[i].id for i in range(depth))

    remaining_capacity = knapsack_instance.capacity - partial_solution.total_weight

    if remaining_capacity <= 0:
        return partial_solution.total_value

    # Find the most efficient remaining (undecided) item
    best_efficiency = 0
    for item in knapsack_instance.items:
        if item.id not in looked_at_ids:
            efficiency = item.value / item.weight if item.weight > 0 else 0
            best_efficiency = max(best_efficiency, efficiency)

    return partial_solution.total_value + remaining_capacity * best_efficiency