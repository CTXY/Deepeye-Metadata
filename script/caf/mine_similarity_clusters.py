import logging
from pathlib import Path
from caf.config import CAFConfig
from caf.memory.stores.semantic import SemanticMemoryStore
from caf.memory.stores.similarity_cluster import SimilarityClusterStore
from caf.memory.generators import SimilarityClusterMiner

# Set up logging to see debug information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

config = CAFConfig.from_file("config/caf_config.yaml")

semantic_store = SemanticMemoryStore(config.memory)
cluster_store = SimilarityClusterStore(
    Path(config.memory.get("similarity", {}).get("storage_path", "./memory/similarity_clusters"))
)

# Check if data is loaded
semantic_store.bind_database("california_schools")

miner = SimilarityClusterMiner(
    semantic_store=semantic_store,
    cluster_store=cluster_store,
    memory_config=config.memory,
)

clusters = miner.mine_and_save_clusters(database_id="california_schools")

print(f"\nFound {len(clusters)} clusters")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {cluster.cluster_id} with {len(cluster.elements)} elements")
    for elem in cluster.elements[:5]:  # Show first 5 elements
        print(f"  - {elem.table_name}.{elem.column_name}")
    if len(cluster.elements) > 5:
        print(f"  ... and {len(cluster.elements) - 5} more")