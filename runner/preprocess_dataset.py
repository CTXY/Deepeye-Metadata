import sys
sys.path.append(".")
from runner._config_cli import configure_from_cli

configure_from_cli()

from app.dataset import DatasetFactory, save_dataset, load_dataset
from app.config import DatasetConfig, config
from app.logger import logger


def preprocess_dataset(dataset_config: DatasetConfig):
    logger.info(f"Preprocessing dataset: {dataset_config.type} {dataset_config.split}")
    if dataset_config.database_whitelist:
        logger.info(f"Applying database whitelist: {dataset_config.database_whitelist}")
    if dataset_config.question_id_whitelist:
        logger.info(f"Applying question_id whitelist with {len(dataset_config.question_id_whitelist)} items")
    dataset = DatasetFactory.get_dataset(dataset_config)
    logger.info(f"Dataset loaded: {len(dataset)} items")
    save_dataset(dataset, dataset_config.save_path)
    logger.info(f"Dataset saved: {dataset_config.save_path}")


if __name__ == "__main__":
    preprocess_dataset(config.dataset_config)
