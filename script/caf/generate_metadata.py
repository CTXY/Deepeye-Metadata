#!/usr/bin/env python3
"""
Semantic Memory Metadata Generation Script

This script provides one-click automatic metadata generation for databases.
It analyzes database DDL, profiles data, and uses LLM to generate semantic descriptions.

Usage:
    python script/caf/generate_metadata.py --database-path /path/to/database.db [OPTIONS]

Examples:
    # Basic generation
    python script/caf/generate_metadata.py --database-path /home/yangchenyu/DeepEye-SQL-Metadata/dataset/bird/databases/dev_databases/california_schools/california_schools.sqlite --verbose
    
    # With custom database ID
    python script/caf/generate_metadata.py --database-path ./data/test.db --database-id my_test_db
    
    # Disable LLM analysis (only DDL and profiling)
    python script/caf/generate_metadata.py --database-path /home/yangchenyu/DeepEye-SQL-Metadata/dataset/bird/databases/dev_databases/california_schools/california_schools.sqlite --llm --ddl --profiling --verbose
    
    python script/caf/generate_metadata.py --database-path /home/yangchenyu/DeepEye-SQL-Metadata/dataset/bird/databases/dev_databases/card_games/card_games.sqlite --join-path-discovery
    
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from caf.config.loader import CAFConfig
from caf.system import CAFSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic metadata for database automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --database-path ./data/test.db
  %(prog)s --database-path ./data/test.db --database-id custom_db --verbose
  %(prog)s --database-path ./data/test.db --no-llm --force
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--database-path',
        type=str,
        required=True,
        help="Path to database file to analyze"
    )
    
    # Optional arguments
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent.parent.parent / "config" / "caf_config.yaml",
        help="Path to CAF config file (default: ./config/caf_config.yaml)"
    )
    
    # Generation options (new additive flags)
    parser.add_argument(
        '--ddl',
        action='store_true',
        help="Enable DDL analysis. If any of --ddl/--profiling/--llm are provided, only the provided ones are enabled."
    )
    
    parser.add_argument(
        '--profiling',
        action='store_true',
        help="Enable data profiling. If any of --ddl/--profiling/--llm are provided, only the provided ones are enabled."
    )
    
    parser.add_argument(
        '--llm',
        action='store_true',
        help="Enable LLM analysis. If any of --ddl/--profiling/--llm are provided, only the provided ones are enabled."
    )
    
    parser.add_argument(
        '--join-path-discovery',
        action='store_true',
        help="Enable join path discovery (discover non-FK relationships). If not specified, uses config default."
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force regenerate all metadata (ignore existing data)"
    )
    
    # Output options
    parser.add_argument(
        '--output-summary',
        type=Path,
        help="Save generation summary to JSON file"
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
    
    # Validate arguments
    database_path = Path(args.database_path)
    if not database_path.exists():
        logger.error(f"Database file not found: {database_path}")
        sys.exit(1)
    
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Load configuration
        logger.info("Loading CAF configuration...")
        config = CAFConfig.from_file(args.config)
        
        # Initialize CAF System (global config is automatically initialized internally)
        logger.info("Initializing CAF system...")
        caf_system = CAFSystem(config)
        
        # Determine which features to enable
        # If any of --ddl/--profiling/--llm are provided, only those are enabled
        # Otherwise, use config defaults (which default to True in generate_metadata)
        enable_ddl = args.ddl if args.ddl else None
        enable_profiling = args.profiling if args.profiling else None
        enable_llm = args.llm if args.llm else None
        
        # Join path discovery: use argument if provided, otherwise use config default (None = config default)
        enable_join_path_discovery = args.join_path_discovery if args.join_path_discovery else None
        
        # Generate metadata using CAFSystem.generate_metadata
        logger.info("=== Starting Metadata Generation ===")
        logger.info(f"Database: {database_path}")
        logger.info(f"Configuration: DDL={'Yes' if enable_ddl else ('No' if enable_ddl is False else 'Config default')}, "
                   f"Profiling={'Yes' if enable_profiling else ('No' if enable_profiling is False else 'Config default')}, "
                   f"LLM={'Yes' if enable_llm else ('No' if enable_llm is False else 'Config default')}, "
                   f"JoinPathDiscovery={'Yes' if enable_join_path_discovery else ('No' if enable_join_path_discovery is False else 'Config default')}")
        
        caf_system.generate_metadata(
            database_path=str(database_path),
            enable_ddl_analysis=enable_ddl,
            enable_profiling=enable_profiling,
            enable_llm_analysis=enable_llm,
            enable_join_path_discovery=enable_join_path_discovery,
            force_regenerate=args.force,
        )
        
        # Note: CAFSystem.generate_metadata already displays a summary internally
        # If output_summary was requested, we cannot provide it since the method returns None
        # The summary is only displayed in logs
        if args.output_summary:
            logger.warning(f"--output-summary is not supported when using CAFSystem.generate_metadata. Summary is displayed in logs only.")
        
        logger.info("✅ Metadata generation completed!")
        sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Generation cancelled by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Metadata generation failed: {e}")
        if args.verbose:
            logger.exception("Full error details:")
        sys.exit(1)

if __name__ == "__main__":
    main()
