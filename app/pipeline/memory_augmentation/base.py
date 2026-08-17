from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from app.dataset import DataItem


class MemoryAugmentor(ABC):
    """Base class for memory augmentation strategies."""

    name: str = ""

    def __init__(self, strategy_config: Optional[Dict[str, Any]] = None):
        self.strategy_config = strategy_config or {}

    def supports(self, data_item: DataItem) -> bool:
        """Whether the strategy should run for this data item."""
        return True

    @abstractmethod
    def augment(self, data_item: DataItem) -> None:
        """Mutate data_item in-place with memory fields."""
        raise NotImplementedError
