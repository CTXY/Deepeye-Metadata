"""
Script to mine representation ambiguity clusters from database.

This script identifies columns that represent the same real-world entity
but in different physical storage formats (e.g., ID vs Name vs Code).

The process has two steps:
1. Data-Driven Dependency Mining: Find 1-to-1 relationships between columns
2. LLM Semantic Verification: Verify if these pairs represent the same entity
"""

import logging
from pathlib import Path
from caf.config import CAFConfig
from caf.config.global_config import initialize_global_config
from caf.memory.stores.semantic import SemanticMemoryStore
from caf.memory.stores.similarity_cluster import SimilarityClusterStore
from caf.memory.generators import RepresentationAmbiguityMiner

# Set up logging to see debug information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Initialize global config manager with the config file
initialize_global_config("config/caf_config.yaml")

config = CAFConfig.from_file("config/caf_config.yaml")

semantic_store = SemanticMemoryStore(config.memory)
cluster_store = SimilarityClusterStore(
    Path(
        config.memory.get("similarity", {}).get(
            "storage_path", "./memory/similarity_clusters"
        )
    )
)

# Check if data is loaded
semantic_store.bind_database("california_schools")

miner = RepresentationAmbiguityMiner(
    semantic_store=semantic_store,
    cluster_store=cluster_store,
    memory_config=config.memory,
    raw_config=config._raw_data,  # Pass raw config to access top-level llm section
)

clusters = miner.mine_and_save_clusters(database_id="california_schools")

print(f"\nFound {len(clusters)} representation ambiguity clusters")
for i, cluster in enumerate(clusters):
    print(
        f"Cluster {i+1}: {cluster.cluster_id} with {len(cluster.elements)} elements"
    )
    print(f"  Methods: {cluster.methods}")
    for elem in cluster.elements[:10]:  # Show first 10 elements
        print(f"  - {elem.table_name}.{elem.column_name}")
    if len(cluster.elements) > 10:
        print(f"  ... and {len(cluster.elements) - 10} more")


