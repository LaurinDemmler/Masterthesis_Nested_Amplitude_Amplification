import yaml # type: ignore
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

@dataclass
class Item:
    id: int
    weight: int
    value: int

class KnapsackInstance:
    """
    Items are ALWAYS sorted by value after loading.
    """
    def __init__(self, config_path: str | Path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            raise ValueError(f"Configuration file at {config_path} invalid.")

        self.metadata = cfg.get('metadata', {})
        number_knapsack = cfg.get('number_knapsack', None)
        self.capacity = int(cfg.get('capacity', None))
        self.name = cfg.get('experiment_name', None)
        self.items = [
            Item(
                id=None,
                weight=int(row['weight']),
                value=int(row['value'])
            )
            for i, row in enumerate(cfg.get('items', []))
        ]
        self.sort_items_by_value()
        self.assign_ids_by_rank()
        self.num_items = len(self.items)
        self.ids = [it.id for it in self.items]
        self.weights = [it.weight for it in self.items]
        self.values = [it.value for it in self.items]


    def get_item_by_id(self, item_id: int):
        """Return item with given id, or None if not found."""
        return next((it for it in self.items if it.id == item_id), None)
    
    def get_properties_dict(self) -> Dict[str, Any]:
        """Return instance properties as a dictionary."""
        return {
            'metadata': self.metadata,
            'capacity': self.capacity,
            'items': [it.__dict__ for it in self.items]
        }
        
    def assign_ids_by_rank(self) -> None:    
        """Assign item ids based on current order in self.items. Used to sort by density"""
        for new_id, it in enumerate(self.items):
            it.id = new_id

    def _sync_lists(self) -> None:
        """Synchronize ids, weights, values lists with current items order."""
        self.ids = [it.id for it in self.items]
        self.weights = [it.weight for it in self.items]
        self.values = [it.value for it in self.items]

    def sort_items_by_density(self) -> None:
        """Sort items in-place by value/weight density in descending order."""
        self.items.sort(key=lambda it: it.value / it.weight if it.weight > 0 else 0, reverse=True)
        self._sync_lists()
    
    def sort_items_by_value(self) -> None:
        """Sort items in-place by value in descending order."""
        self.items.sort(key=lambda it: it.value, reverse=True)

    def is_sorted_by_density(self) -> bool:
        """Check if items are sorted by value/weight density in descending order."""
        return all((self.items[i].value / self.items[i].weight if self.items[i].weight > 0 else 0) >=
                   (self.items[i + 1].value / self.items[i + 1].weight if self.items[i + 1].weight > 0 else 0)
                   for i in range(len(self.items) - 1))
        
    def is_sorted_by_value(self) -> bool:
        """Check if items are sorted by value in descending order."""
        return all(self.items[i].value >= self.items[i + 1].value for i in range(len(self.items) - 1))
    
    def _copy(self) -> 'KnapsackInstance':
        """Create a deep copy of the KnapsackInstance."""
        new_instance = KnapsackInstance.__new__(KnapsackInstance)
        new_instance.metadata = self.metadata.copy()
        new_instance.capacity = self.capacity
        new_instance.name = self.name
        new_instance.items = [Item(id=it.id, weight=it.weight, value=it.value) for it in self.items]
        new_instance.num_items = self.num_items
        new_instance.ids = self.ids.copy()
        new_instance.weights = self.weights.copy()
        new_instance.values = self.values.copy()
        return new_instance
    
    def get_remaining_value(self, depth):
        """Calculate total value of items not included up to given depth."""
        if depth < 0 or depth > self.num_items:
            raise ValueError(f"Depth {depth} is out of bounds for number of items {self.num_items}.")
        return sum(it.value for it in self.items[depth:])
    
    def get_capweight_ratio(self):
        """Calculate capacity to total weight ratio."""
        total_weight = sum(it.weight for it in self.items)
        if total_weight == 0:
            return 0
        return self.capacity / total_weight
