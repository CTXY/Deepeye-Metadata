"""
Mapping Analysis pipeline step.

Loads the current dataset (augmented or schema_linking), fills mapping_* fields for each
data_item from offline_memory_augmentation.jsonl (db_id + question_id), and saves to mapping_analysis save_path.
Optional step; SQL generation/revision/selection load from this output when use_caf_mapping.
"""

from pathlib import Path

from app.dataset import load_dataset, save_dataset, BaseDataset
from app.config import config
from app.logger import logger
from .loader import ensure_mapping_data


class MappingAnalysisRunner:
    """
    Optional pipeline step: enrich dataset with mapping_linked_tables_and_columns,
    database_schema_after_mapping, and mapping_hint from offline context-graph memory jsonl.
    """

    _dataset: BaseDataset = None

    def __init__(self):
        # Load from same source as SQL generation without mapping: augmented if exists else schema_linking
        if (
            Path(config.augmented_data_retrieval_config.save_path).exists()
            and config.augmented_data_retrieval_config.use_augmented_data
        ):
            logger.info(f"Loading dataset from {config.augmented_data_retrieval_config.save_path}")
            self._dataset = load_dataset(config.augmented_data_retrieval_config.save_path)
        else:
            logger.info(f"Loading dataset from {config.schema_linking_config.save_path}")
            self._dataset = load_dataset(config.schema_linking_config.save_path)
        path = config.mapping_analysis_config.mapping_decisions_path
        if path is None or path == "":
            from .loader import DEFAULT_MAPPING_DECISIONS_PATH
            self._mapping_path = str(DEFAULT_MAPPING_DECISIONS_PATH)
        else:
            self._mapping_path = path
        if not Path(self._mapping_path).exists():
            logger.warning(f"Mapping decisions file not found: {self._mapping_path}. Mapping fields will remain empty.")

    def run(self):
        logger.info("Starting mapping analysis step...")
        for data_item in self._dataset:
            ensure_mapping_data(data_item, self._mapping_path)
        logger.info("Mapping analysis step completed")
        self.save_result()

    def save_result(self):
        save_path = config.mapping_analysis_config.save_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_dataset(self._dataset, save_path)
        logger.info(f"Saved mapping analysis result to {save_path}")
