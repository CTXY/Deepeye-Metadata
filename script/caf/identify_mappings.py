#!/usr/bin/env python3
"""
Test script for Identify Mappings functionality

This script tests the identify_mapping function which:
- Analyzes a natural language query to extract search terms
- Retrieves mappings for each term from specified memory types
- Returns comprehensive mapping results with intent analysis

Usage:
    Modify the CONFIG section below and run: python identify_mappings.py
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

import caf

# ================================================================================
# CONFIG - Modify these values to test different queries and databases
# ================================================================================
CONFIG = {
    # The natural language query to analyze
    "query": "Please list the zip code of all the charter schools in Fresno County Office of Education.",
    
    # The database ID to connect to
    "database_id": "california_schools",
    
    # Logging level (DEBUG, INFO, WARNING, ERROR)
    "log_level": "INFO"
}

# ================================================================================
# EXAMPLE QUERIES - Uncomment and modify the query above to test different scenarios
# ================================================================================
# "What is the ratio of merged Unified School District schools in Orange County?"
# "Show me all transactions with amounts greater than 1000"
# "Find customers from California who made purchases"
# "List all products in the electronics category"

# Configure logging
logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_identify_mappings():
    """Run identify_mapping using CONFIG settings"""
    
    logger.info("="*60)
    logger.info("Testing Identify Mappings")
    logger.info("="*60)
    
    # Get configuration
    query_text = CONFIG["query"]
    database_id = CONFIG["database_id"]
    memory_types = CONFIG.get("memory_types")
    
    logger.info(f"Query: {query_text}")
    logger.info(f"Database: {database_id}")
    logger.info(f"Memory Types: {memory_types if memory_types else 'default (semantic)'}")
    logger.info("-" * 60)
    
    # Initialize CAF system
    logger.info("Initializing CAF system...")
    caf_system = caf.initialize(config_path="config/caf_config.yaml")
    logger.info("✓ CAF system initialized successfully")
    
    # Bind database (no session needed for identify_mapping)
    logger.info(f"Binding to database...")
    caf_system.bind_database(database_id)
    logger.info("✓ Database bound")
    
    # Execute identify_mapping
    logger.info("Executing identify_mapping...")
    
    result = caf_system.identify_mapping(query=query_text)
    
    # Display results
    logger.info("="*60)
    logger.info("Mapping Results")
    logger.info("="*60)
    
    # Display extracted terms
    terms = result.get("terms", [])
    print(f"\nExtracted {len(terms)} search terms:")
    print("-" * 50)
    for i, term in enumerate(terms, 1):
        print(f"{i}. {term}")
    
    # Display intent analysis
    intent_analysis = result.get("intent_analysis", {})
    entity_groups = intent_analysis.get("entity_groups", [])
    if entity_groups:
        print(f"\nIntent Analysis - {len(entity_groups)} entity groups:")
        print("-" * 50)
        for i, eg in enumerate(entity_groups, 1):
            print(f"{i}. Full Phrase: {eg.get('full_phrase', 'N/A')}")
            print(f"   Base Phrase: {eg.get('base_phrase', 'N/A')}")
            constraints = eg.get('constraints', [])
            if constraints:
                print(f"   Constraints: {', '.join(constraints)}")
            print()
    
    # # Display mappings
    # mappings = result.get("mappings", {})
    # print(f"\nMappings for {len(mappings)} terms:")
    # print("=" * 60)
    
    # for term, term_mappings in mappings.items():
    #     print(f"\nTerm: '{term}'")
    #     print("-" * 50)
        
    #     for memory_type, response in term_mappings.items():
    #         items = response.items if hasattr(response, 'items') else []
    #         print(f"  Memory Type: {memory_type}")
            
    #         for j, item in enumerate(items[:5], 1):  # Show top 5 results
    #             content = item.content
    #             item_type = content.get('item_type', 'unknown')
    #             metadata = content.get('metadata', {})
                
    #             if item_type == 'table':
    #                 table_name = metadata.get('table_name', 'unknown')
    #                 print(f"    {j}. Table: {table_name} (score: {item.score:.3f})")
    #             elif item_type == 'column':
    #                 table_name = metadata.get('table_name', 'unknown')
    #                 column_name = metadata.get('column_name', 'unknown')
    #                 print(f"    {j}. Column: {table_name}.{column_name} (score: {item.score:.3f})")
    #             else:
    #                 print(f"    {j}. {item_type}: {content} (score: {item.score:.3f})")
            
    #         if len(items) > 5:
    #             print(f"    ... and {len(items) - 5} more results")
    
    # Cleanup
    caf_system.cleanup()
    logger.info("✓ Identify mappings completed successfully!")
    return True

def main():
    """Main function"""
    try:
        success = run_identify_mappings()
        if success:
            logger.info("Test completed successfully!")
        else:
            logger.error("Test failed!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
