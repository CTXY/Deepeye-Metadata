import sys
sys.path.append(".")
from runner._config_cli import configure_from_cli

configure_from_cli()

from app.pipeline.augmented_data_retrieval import AugmentedDataRetrievalRunner

def main():
    runner = AugmentedDataRetrievalRunner()
    runner.run()

if __name__ == "__main__":
    main()
