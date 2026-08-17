import time
from pathlib import Path

from app.config import config
from app.dataset import BaseDataset, load_dataset, save_dataset
from app.logger import logger

from .registry import get_augmentors_from_config


class MemoryAugmentationRunner:
    _dataset: BaseDataset = None

    def __init__(self):
        logger.info(f"Loading dataset from {config.schema_linking_config.save_path}")
        self._dataset = load_dataset(config.schema_linking_config.save_path)
        self._augmentors = get_augmentors_from_config()

    def _augment_one(self, data_item) -> None:
        start_time = time.time()

        # Ensure backward compatibility with older pickled DataItem objects
        memory_metadata = getattr(data_item, "memory_metadata", None)
        if memory_metadata is None:
            memory_metadata = {}
            setattr(data_item, "memory_metadata", memory_metadata)
        memory_metadata.setdefault(
            "enabled_strategies",
            [augmentor.name for augmentor in self._augmentors],
        )

        for augmentor in self._augmentors:
            if not augmentor.supports(data_item):
                continue
            augmentor.augment(data_item)

        # Update timing and LLM-cost fields defensively in case they don’t exist
        memory_augmentation_time = time.time() - start_time
        setattr(data_item, "memory_augmentation_time", memory_augmentation_time)

        _zero_llm_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        memory_augmentation_llm_cost = getattr(data_item, "memory_augmentation_llm_cost", None) or dict(
            _zero_llm_cost
        )
        setattr(data_item, "memory_augmentation_llm_cost", memory_augmentation_llm_cost)

        total_time = getattr(data_item, "total_time", None) or 0.0
        setattr(data_item, "total_time", total_time + memory_augmentation_time)

        current_total_llm_cost = getattr(data_item, "total_llm_cost", None) or dict(_zero_llm_cost)
        updated_total_llm_cost = {
            "prompt_tokens": current_total_llm_cost.get("prompt_tokens", 0)
            + memory_augmentation_llm_cost.get("prompt_tokens", 0),
            "completion_tokens": current_total_llm_cost.get("completion_tokens", 0)
            + memory_augmentation_llm_cost.get("completion_tokens", 0),
            "total_tokens": current_total_llm_cost.get("total_tokens", 0)
            + memory_augmentation_llm_cost.get("total_tokens", 0),
        }
        setattr(data_item, "total_llm_cost", updated_total_llm_cost)

    def run(self):
        logger.info("Starting memory augmentation step...")
        for data_item in self._dataset:
            self._augment_one(data_item)
        logger.info("Memory augmentation step completed")
        self.save_result()

    def save_result(self):
        save_path = config.memory_augmentation_config.save_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_dataset(self._dataset, save_path)
        logger.info(f"Saved memory augmentation result to {save_path}")
