"""
Script to mine pseudo-query collision clusters from database.

This script identifies columns that are likely to be confused during question-schema matching
by generating natural language queries for each column and checking if those queries
incorrectly retrieve other columns.
"""

import logging
from pathlib import Path
from caf.config import CAFConfig
from caf.config.global_config import initialize_global_config
from caf.memory.stores.semantic import SemanticMemoryStore
from caf.memory.stores.similarity_cluster import SimilarityClusterStore
from caf.memory.generators import PseudoQueryCollisionMiner

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
database_id = "california_schools"  # Change this to your database ID
semantic_store.bind_database(database_id)

miner = PseudoQueryCollisionMiner(
    semantic_store=semantic_store,
    cluster_store=cluster_store,
    memory_config=config.memory,
    raw_config=config._raw_data,  # Pass raw config to access top-level llm section
)

clusters = miner.mine_and_save_clusters(database_id=database_id)

print(f"\nFound {len(clusters)} pseudo-query collision clusters")
for i, cluster in enumerate(clusters):
    print(
        f"Cluster {i+1}: {cluster.cluster_id} with {len(cluster.elements)} elements"
    )
    print(f"  Methods: {cluster.methods}")
    if cluster.semantic_score_min is not None and cluster.semantic_score_max is not None:
        print(
            f"  Collision Scores: min={cluster.semantic_score_min:.3f}, "
            f"max={cluster.semantic_score_max:.3f}, "
            f"avg={cluster.semantic_score_avg:.3f}"
        )
    if cluster.collision_info:
        print(f"  Collision Details: {len(cluster.collision_info)} collisions")
        # Show first 3 collision examples
        for j, collision in enumerate(cluster.collision_info[:3], 1):
            print(
                f"    [{j}] Query: '{collision.trigger_query}' "
                f"({collision.query_type})"
            )
            print(
                f"        {collision.source_column_id} -> {collision.distractor_column_id} "
                f"(score: {collision.collision_score:.3f}, type: {collision.collision_type})"
            )
        if len(cluster.collision_info) > 3:
            print(f"    ... and {len(cluster.collision_info) - 3} more collisions")
    for elem in cluster.elements[:5]:  # Show first 5 elements
        print(f"  - {elem.table_name}.{elem.column_name}")
    if len(cluster.elements) > 5:
        print(f"  ... and {len(cluster.elements) - 5} more")

