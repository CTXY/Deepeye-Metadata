import sys
sys.path.append(".")
from app.pipeline.augmented_data_retrieval import AugmentedDataRetrievalRunner

def main():
    runner = AugmentedDataRetrievalRunner()
    runner.run()

if __name__ == "__main__":
    main()
