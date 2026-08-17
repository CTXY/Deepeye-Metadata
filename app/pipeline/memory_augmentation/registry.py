from importlib import import_module
from typing import Dict, List, Tuple

from app.config import config

from .base import MemoryAugmentor


STRATEGY_REGISTRY: Dict[str, Tuple[str, str]] = {
    "historical_qa": (".historical_qa", "HistoricalQAMemoryAugmentor"),
    "context_graph": (".context_graph", "ContextGraphMemoryAugmentor"),
    "mapping_based": (".context_graph", "ContextGraphMemoryAugmentor"),
    "amem_retrieval": (".amem_retrieval", "AMemRetrievalMemoryAugmentor"),
}


def get_augmentors_from_config() -> List[MemoryAugmentor]:
    augmentors: List[MemoryAugmentor] = []
    for strategy_name in config.memory_augmentation_config.strategies:
        strategy_spec = STRATEGY_REGISTRY.get(strategy_name)
        if strategy_spec is None:
            raise ValueError(f"Unknown memory augmentation strategy: {strategy_name}")
        module_name, class_name = strategy_spec
        strategy_cls = getattr(import_module(module_name, __package__), class_name)
        strategy_config = getattr(config.memory_augmentation_config, strategy_name, {})
        augmentors.append(strategy_cls(strategy_config=strategy_config))
    return augmentors
