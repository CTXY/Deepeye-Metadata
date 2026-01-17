#!/usr/bin/env python3
"""
Example script demonstrating how to use BIRD metadata import functionality.

This script shows three ways to import/generate metadata:
1. Import BIRD metadata only
2. Import BIRD metadata + generate additional metadata (combined)
3. Import BIRD then generate separately (manual two-step)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from caf.system import CAFSystem
from caf.config.loader import CAFConfig

def main():
    # Configuration
    config_path = project_root / "config" / "caf_config.yaml"
    database_path = project_root / "dataset" / "bird" / "dev" / "dev_databases" / "financial" / "financial.sqlite"
    bird_data_dir = project_root / "dataset" / "bird"
    
    # Load config
    config = CAFConfig.from_file(config_path)
    
    # Initialize CAF system
    caf_system = CAFSystem(config)
    
    print("="*80)
    print("BIRD Metadata Import Examples")
    print("="*80)
    print(f"Database: {database_path.name}")
    print(f"BIRD Data: {bird_data_dir}")
    print("="*80)
    
    # Example 1: Import BIRD metadata only
    print("\n### Example 1: Import BIRD metadata only ###\n")
    
    result1 = caf_system.import_bird_metadata(
        database_path=str(database_path),
        bird_data_dir=str(bird_data_dir),
        force_regenerate=False,
        continue_on_error=True
    )
    
    print(f"\nResult: {result1['success']}")
    if result1['errors']:
        print(f"Errors: {result1['errors']}")
    

if __name__ == "__main__":
    main()












