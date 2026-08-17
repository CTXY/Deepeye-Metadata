from importlib import import_module
from typing import Any


_RUNNER_MODULES = {
    "ValueRetrievalRunner": ".value_retrieval",
    "SchemaLinkingRunner": ".schema_linking",
    "MemoryAugmentationRunner": ".memory_augmentation",
    "SQLGenerationRunner": ".sql_generation",
    "SQLRevisionRunner": ".sql_revision",
    "BRSelectionRunner": ".sql_selection",
}

__all__ = list(_RUNNER_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _RUNNER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
