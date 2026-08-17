"""
Run the optional Mapping Analysis pipeline step.

Loads dataset from augmented_data_retrieval or schema_linking, enriches each item with
mapping_linked_tables_and_columns, database_schema_after_mapping, and mapping_hint from
mapping_decisions.jsonl, and saves to mapping_analysis save_path.
"""

import sys

sys.path.append(".")
from runner._config_cli import configure_from_cli

configure_from_cli()

from app.logger import logger
from app.pipeline import MemoryAugmentationRunner

if __name__ == "__main__":
    logger.warning(
        "run_mapping_analysis.py is deprecated. "
        "Using MemoryAugmentationRunner instead; enable context_graph strategy in memory_augmentation config."
    )
    runner = MemoryAugmentationRunner()
    runner.run()
