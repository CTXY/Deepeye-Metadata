"""
Script to mine value overlap pairs from database.

This script identifies columns with high value overlap (Jaccard similarity >= 0.6)
by directly querying the database and comparing value sets.

NOTE: This script now outputs pairs instead of clusters.
      Use analyze_ambiguous_pairs.py for full analysis including LLM.
"""

import logging
from pathlib import Path
from caf.config import CAFConfig
from caf.memory.stores.semantic import SemanticMemoryStore
from caf.memory.stores.ambiguous_pair import AmbiguousPairStore
from caf.memory.generators import ValueOverlapClusterMiner

# Set up logging to see debug information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

config = CAFConfig.from_file("config/caf_config.yaml")

semantic_store = SemanticMemoryStore(config.memory)
pair_store = AmbiguousPairStore(
    Path(
        config.memory.get("similarity", {}).get(
            "storage_path", "./memory/ambiguous_pairs"
        )
    )
)


# Check if data is loaded
semantic_store.bind_database("california_schools")

miner = ValueOverlapClusterMiner(
    semantic_store=semantic_store,
    pair_store=pair_store,
    memory_config=config.memory,
)

pairs = miner.mine_and_save_pairs(database_id="california_schools")

print(f"\nFound {len(pairs)} value overlap pairs")
for i, pair in enumerate(pairs):
    print(
        f"Pair {i+1}: {pair.pair_id}"
    )
    print(f"  Column A: {pair.column_a.table_name}.{pair.column_a.column_name}")
    print(f"  Column B: {pair.column_b.table_name}.{pair.column_b.column_name}")
    print(f"  Methods: {pair.discovery_methods}")
    if pair.value_jaccard is not None:
        print(f"  Jaccard: {pair.value_jaccard:.3f}")

