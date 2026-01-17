#!/usr/bin/env python3
"""
Analyze Ambiguous Pairs Script

This script analyzes ambiguous column pairs discovered by miners and generates
detailed DiffProfiles for disambiguation.

Usage:
    python script/caf/analyze_ambiguous_pairs.py --database-id <db_id> [OPTIONS]

Examples:
    # Analyze all pairs for a database
    python script/caf/analyze_ambiguous_pairs.py --database-id california_schools

    # Skip data content analysis (faster, LLM only)
    python script/caf/analyze_ambiguous_pairs.py --database-id california_schools --no-data-content

    # Use more workers for parallel analysis
    python script/caf/analyze_ambiguous_pairs.py --database-id california_schools --workers 8
"""

import argparse
import logging
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from caf.config.loader import CAFConfig
from caf.config.global_config import initialize_global_config, get_llm_config
from caf.memory.stores.semantic import SemanticMemoryStore
from caf.memory.stores.ambiguous_pair import AmbiguousPairStore
from caf.memory.generators.pseudo_query_collision_miner import PseudoQueryCollisionMiner
from caf.memory.generators.value_overlap_cluster_miner import ValueOverlapClusterMiner
from caf.memory.analyzers.data_content_analyzer import DataContentAnalyzer
from caf.memory.analyzers.semantic_intent_analyzer import SemanticIntentAnalyzer
from caf.memory.analyzers.ambiguity_analyzer import AmbiguityAnalyzer
from caf.llm.client import create_llm_client, LLMConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_database_mapping(mapping_path: Path) -> dict:
    """Load database ID to path mapping."""
    if not mapping_path.exists():
        logger.warning("Database mapping not found at %s", mapping_path)
        return {}

    try:
        with mapping_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        # Reverse mapping: path -> id to id -> path
        db_mapping = {}
        for path, db_id in raw.items():
            if db_id:
                db_mapping[str(db_id)] = str(path)
        return db_mapping
    except Exception as e:
        logger.error("Failed to load database mapping: %s", e)
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ambiguous column pairs for a database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --database-id california_schools
  %(prog)s --database-id california_schools --no-data-content --workers 8
  %(prog)s --database-id california_schools --verbose
        """
    )

    # Required arguments
    parser.add_argument(
        '--database-id',
        type=str,
        required=True,
        help="Database ID to analyze"
    )

    # Optional arguments
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent.parent.parent / "config" / "caf_config.yaml",
        help="Path to CAF config file (default: ./config/caf_config.yaml)"
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help="Number of parallel workers for analysis (default: 4)"
    )

    parser.add_argument(
        '--no-data-content',
        action='store_true',
        help="Skip data content analysis (only LLM semantic analysis)"
    )

    parser.add_argument(
        '--no-semantic-intent',
        action='store_true',
        help="Skip semantic intent analysis (only data content analysis)"
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Suppress all output except errors"
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Validate config
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    try:
        # Load configuration
        logger.info("Loading CAF configuration...")
        config = CAFConfig.from_file(args.config)
        initialize_global_config(args.config)

        # Initialize stores
        logger.info("Initializing stores...")
        semantic_store = SemanticMemoryStore(config.memory)

        # Get similarity storage path
        similarity_path = Path(
            config.memory.get("similarity", {}).get(
                "storage_path", "./memory/ambiguous_pairs"
            )
        )
        pair_store = AmbiguousPairStore(similarity_path)

        # Initialize miners
        logger.info("Initializing miners...")
        pseudo_query_miner = PseudoQueryCollisionMiner(
            semantic_store=semantic_store,
            pair_store=pair_store,
            memory_config=config.memory,
            raw_config=config.raw_config,
        )

        value_overlap_miner = ValueOverlapClusterMiner(
            semantic_store=semantic_store,
            pair_store=pair_store,
            memory_config=config.memory,
        )

        # Load database mapping
        logger.info("Loading database mapping...")
        mapping_path = Path(
            config.memory.get("database_mapping_path", "./memory/database_mapping.json")
        )
        database_mapping = load_database_mapping(mapping_path)

        if args.database_id not in database_mapping:
            logger.warning(
                "Database %s not found in mapping. Data content analysis may fail.",
                args.database_id
            )

        # Initialize analyzers
        logger.info("Initializing analyzers...")

        # Data content analyzer
        data_content_analyzer = DataContentAnalyzer(
            database_mapping=database_mapping,
            config={
                "constraint_sample_size": 10000,
                "sensitivity_sample_size": 10,
                "min_common_values": 5,
            }
        )

        # Semantic intent analyzer (LLM)
        try:
            llm_config = get_llm_config()
            llm_client = create_llm_client(LLMConfig(
                provider=llm_config.provider,
                model_name=llm_config.model_name,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                timeout=llm_config.timeout,
            ))
        except Exception as e:
            logger.error("Failed to initialize LLM client: %s", e)
            sys.exit(1)

        semantic_intent_analyzer = SemanticIntentAnalyzer(
            llm_client=llm_client,
            semantic_store=semantic_store,
            config={
                "temperature": 0.3,
                "max_retries": 2,
            }
        )

        # Unified ambiguity analyzer
        ambiguity_analyzer = AmbiguityAnalyzer(
            pair_store=pair_store,
            pseudo_query_miner=pseudo_query_miner,
            value_overlap_miner=value_overlap_miner,
            data_content_analyzer=data_content_analyzer,
            semantic_intent_analyzer=semantic_intent_analyzer,
            config={
                "num_workers": args.workers,
                "enable_data_content": not args.no_data_content,
                "enable_semantic_intent": not args.no_semantic_intent,
            }
        )

        # Run analysis
        logger.info("Starting analysis for database: %s", args.database_id)
        stats = ambiguity_analyzer.analyze_database(args.database_id)

        # Display results
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info("Database: %s", stats["database_id"])
        logger.info("Total pairs: %d", stats["total_pairs"])
        logger.info("Successfully analyzed: %d", stats["analyzed_pairs"])
        logger.info("Failed: %d", stats["failed_pairs"])
        logger.info("\nDiscovery methods:")
        for method, count in stats.get("discovery_method_counts", {}).items():
            logger.info("  - %s: %d", method, count)
        logger.info("="*80 + "\n")

        # Display storage location
        pair_file = pair_store._pairs_file_for_db(args.database_id)
        logger.info("Results saved to: %s", pair_file)

        sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Analysis cancelled by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            logger.exception("Full error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()

