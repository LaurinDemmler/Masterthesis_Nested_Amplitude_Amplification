from pathlib import Path
from datastructures.knapsack import KnapsackInstance


class InstanceFilter:
    def __init__(self, knapsack_instance: KnapsackInstance):
        self.knapsack_instance = knapsack_instance
    
    def filter_for_good_instances(self, T, depth) -> bool:
        take = True
        min_cap_weight_ratio = 0.6
        if self.knapsack_instance.get_capweight_ratio() <= min_cap_weight_ratio:
            take = False
        if self.knapsack_instance.get_remaining_value(depth) >= T:
            take = False
        return take
    
    def filter_for_good_instances_capweight_only(self, min_cap_weight_ratio: float) -> bool:
        take = True
        if self.knapsack_instance.get_capweight_ratio() <= min_cap_weight_ratio:
            take = False
        return take
    def filter_for_small_instances_capweight_only(self, max_cap_weight_ratio: float) -> bool:
        take = True
        if self.knapsack_instance.get_capweight_ratio() > max_cap_weight_ratio:
            take = False
        return take
    
    def filter_for_small_instances(self, max_num_items: int) -> bool:
        return self.knapsack_instance.num_items <= max_num_items
    
    def filter_instance_range(self, min_num_items: int, max_num_items: int) -> bool:
        return min_num_items <= self.knapsack_instance.num_items <= max_num_items
    
    def filter_for_small_capweight_ratio(self, max_cap_weight_ratio: float, T, depth) -> bool:
        take = True
        if self.knapsack_instance.get_capweight_ratio() > max_cap_weight_ratio:
            take = False
        if self.knapsack_instance.get_remaining_value(depth) >= T:
            #exclude RVR>1
            take = False
        return take
    
    def filter_capweight_ratio_range(self, min_cap_weight_ratio: float, max_cap_weight_ratio: float) -> bool:
        take = True
        if self.knapsack_instance.get_capweight_ratio() < min_cap_weight_ratio or self.knapsack_instance.get_capweight_ratio() > max_cap_weight_ratio:
            take = False
        return take