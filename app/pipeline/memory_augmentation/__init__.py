from typing import Any


__all__ = ["MemoryAugmentationRunner"]


def __getattr__(name: str) -> Any:
    if name != "MemoryAugmentationRunner":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .runner import MemoryAugmentationRunner

    globals()[name] = MemoryAugmentationRunner
    return MemoryAugmentationRunner
