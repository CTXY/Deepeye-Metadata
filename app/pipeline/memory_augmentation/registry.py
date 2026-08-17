from typing import Dict, Type, List

from app.config import config

from .base import MemoryAugmentor
from .historical_qa import HistoricalQAMemoryAugmentor
from .context_graph import ContextGraphMemoryAugmentor
from .amem_retrieval import AMemRetrievalMemoryAugmentor


STRATEGY_REGISTRY: Dict[str, Type[MemoryAugmentor]] = {
    "historical_qa": HistoricalQAMemoryAugmentor,
    "context_graph": ContextGraphMemoryAugmentor,
    "mapping_based": ContextGraphMemoryAugmentor,
    "amem_retrieval": AMemRetrievalMemoryAugmentor,
}


def get_augmentors_from_config() -> List[MemoryAugmentor]:
    augmentors: List[MemoryAugmentor] = []
    for strategy_name in config.memory_augmentation_config.strategies:
        strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
        if strategy_cls is None:
            raise ValueError(f"Unknown memory augmentation strategy: {strategy_name}")
        strategy_config = getattr(config.memory_augmentation_config, strategy_name, {})
        augmentors.append(strategy_cls(strategy_config=strategy_config))
    return augmentors
