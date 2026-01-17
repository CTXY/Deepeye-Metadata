#!/usr/bin/env python3
"""
Test script for Semantic Memory Search functionality

This script tests the new semantic search capabilities including:
- Multi-signal retrieval (keyword, semantic, value, domain)  
- LLM-based refinement
- Index building and caching

Usage:
    Modify the CONFIG section below and run: python semantic_search.py
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
    # The semantic search query to test
    "query": "Please list the zip code of all the charter schools in Fresno County Office of Education.",
    
    # The database ID to connect to
    "database_id": "california_schools",
    
    # Maximum number of results to return
    "limit": 20,
    
    # External knowledge/evidence for SQL generation (optional)
    # This will be used when generating initial SQL if generated_sql is not provided
    "evidence": 'Charter schools refers to `Charter School (Y/N)` = 1 in the table fprm',  # Set to a string with external knowledge if available
    
    # Whether to return results grouped by query term
    "return_per_term": False,
    
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

def run_semantic_search():
    """Run semantic search using CONFIG settings"""
    
    logger.info("="*60)
    logger.info("Testing Semantic Memory Search")
    logger.info("="*60)
    
    # Get configuration
    query_text = CONFIG["query"]
    database_id = CONFIG["database_id"]
    limit = CONFIG["limit"]
    evidence = CONFIG.get("evidence")
    return_per_term = CONFIG.get("return_per_term", False)
    
    logger.info(f"Query: {query_text}")
    logger.info(f"Database: {database_id}")
    logger.info(f"Limit: {limit}")
    logger.info(f"Return per term: {return_per_term}")
    if evidence:
        logger.info(f"Evidence: {evidence[:100]}..." if len(evidence) > 100 else f"Evidence: {evidence}")
    logger.info("-" * 60)
    
    # Initialize CAF system
    logger.info("Initializing CAF system...")
    caf_system = caf.initialize(config_path="config/caf_config.yaml")
    logger.info("✓ CAF system initialized successfully")
    
    # Bind database (no session needed for retrieve_memory)
    logger.info(f"Binding to database...")
    caf_system.bind_database(database_id)
    logger.info("✓ Database bound")
    
    # Execute search
    logger.info("Executing semantic search...")
    
    # Prepare context with evidence if available
    context = None
    if evidence:
        context = {"evidence": evidence}
    
    response = caf_system.retrieve_memory(
        memory_type="semantic",
        query_content=query_text,
        context=context,
        limit=limit,
        return_per_term=return_per_term
    )
    
    # Display results
    logger.info("="*60)
    logger.info("Search Results")
    logger.info("="*60)
    
    if return_per_term:
        # Handle per-term grouped results: Tuple[PerTermMemoryResponse, List[JoinRelationship]]
        per_term_response, join_relationships = response
        
        print("\n" + "="*60)
        print("Results grouped by query term:")
        print("="*60)
        
        for query_term_result in per_term_response.query_term_results:
            query_term = query_term_result.query_term
            total_items = len(query_term_result.column_items) + len(query_term_result.term_items)
            print(f"\n--- Term: '{query_term}' ({total_items} results) ---")
            
            # Display column items
            if query_term_result.column_items:
                print(f"  Columns ({len(query_term_result.column_items)}):")
                for i, col_item in enumerate(query_term_result.column_items, 1):
                    print(f"    {i}. {col_item.schema_ref} (table: {col_item.table_name}, column: {col_item.column_name})")
                    if col_item.value_matches:
                        matched_values = col_item.value_matches.get('matched_values', [])
                        if matched_values:
                            print(f"       Matched values: {matched_values[:3]}{'...' if len(matched_values) > 3 else ''}")
            
            # Display term items
            if query_term_result.term_items:
                print(f"  Terms ({len(query_term_result.term_items)}):")
                for i, term_item in enumerate(query_term_result.term_items, 1):
                    print(f"    {i}. {term_item.term_name} (score: {term_item.score:.3f})")
                    if term_item.definition:
                        def_preview = term_item.definition[:60] + "..." if len(term_item.definition) > 60 else term_item.definition
                        print(f"       Definition: {def_preview}")
        
        # Display summary
        print(f"\n--- Summary ---")
        print(f"Total tables: {len(per_term_response.all_tables)}")
        print(f"Total schemas: {len(per_term_response.all_schemas)}")
        
        # Display join relationships
        if join_relationships:
            print("\n" + "="*60)
            print(f"Join Relationships ({len(join_relationships)} found):")
            print("="*60)
            for i, join in enumerate(join_relationships, 1):
                print(f"{i}. {join.table1}.{join.column1} -> {join.table2}.{join.column2} "
                      f"(type: {join.join_type}, confidence: {join.confidence:.3f})")
    else:
        # Handle flat list results: FlatMemoryResponse
        print(f"\nFound {len(response.column_items)} column items and {len(response.term_items)} term items:")
        print("-" * 50)
        
        # Display column items
        if response.column_items:
            print(f"\nColumn Items ({len(response.column_items)}):")
            for i, col_item in enumerate(response.column_items, 1):
                print(f"{i}. {col_item.schema_ref}")
                print(f"   Table: {col_item.table_name}, Column: {col_item.column_name}")
                if col_item.column_metadata:
                    desc = col_item.column_metadata.get('description', 'N/A')
                    if desc and desc != 'N/A':
                        desc_preview = desc[:80] + "..." if len(str(desc)) > 80 else desc
                        print(f"   Description: {desc_preview}")
                if col_item.value_matches:
                    matched_values = col_item.value_matches.get('matched_values', [])
                    if matched_values:
                        print(f"   Matched values: {matched_values[:5]}{'...' if len(matched_values) > 5 else ''}")
                print()
        
        # Display term items
        if response.term_items:
            print(f"\nTerm Items ({len(response.term_items)}):")
            for i, term_item in enumerate(response.term_items, 1):
                print(f"{i}. {term_item.term_name} (score: {term_item.score:.3f})")
                if term_item.definition:
                    def_preview = term_item.definition[:80] + "..." if len(term_item.definition) > 80 else term_item.definition
                    print(f"   Definition: {def_preview}")
                if term_item.formula:
                    print(f"   Formula: {term_item.formula}")
                if term_item.related_columns:
                    print(f"   Related columns: {', '.join(term_item.related_columns[:5])}{'...' if len(term_item.related_columns) > 5 else ''}")
                print()
        
        # Display summary
        print(f"\n--- Summary ---")
        print(f"Total tables: {len(response.all_tables)}")
        print(f"Total schemas: {len(response.all_schemas)}")
        print(f"Query time: {response.query_time_ms}ms")
        
        # Display join relationships
        if response.join_relationships:
            print("\n" + "="*60)
            print(f"Join Relationships ({len(response.join_relationships)} found):")
            print("="*60)
            for i, join in enumerate(response.join_relationships, 1):
                print(f"{i}. {join.table1}.{join.column1} -> {join.table2}.{join.column2} "
                      f"(type: {join.join_type}, confidence: {join.confidence:.3f})")
    
    print("\n" + "="*60)
    print("Full Response Object:")
    print("="*60)
    print(response)
    # Cleanup
    caf_system.cleanup()
    logger.info("✓ Search completed successfully!")
    return True

def main():
    """Main function"""
    try:
        success = run_semantic_search()
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

