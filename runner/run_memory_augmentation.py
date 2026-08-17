import sys

sys.path.append(".")
from runner._config_cli import configure_from_cli

configure_from_cli()

from app.pipeline import MemoryAugmentationRunner


if __name__ == "__main__":
    runner = MemoryAugmentationRunner()
    runner.run()
