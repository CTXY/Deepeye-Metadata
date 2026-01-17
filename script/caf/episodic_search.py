#!/usr/bin/env python3
"""
Enhanced Episodic Memory Search Test Script

This script tests the enhanced episodic memory search capabilities with:
- DAIL-SQL inspired multi-layer retrieval strategy
- NLQ + SQL skeleton similarity for same database
- SQL skeleton similarity only for cross-database queries
- Ground truth data exclusion

Usage:
    python script/caf/episodic_search.py --question "In which mailing street address can you find the school that has the lowest average score in reading? Also give the school's name." --database "california_schools" --evidence "Chartered schools refer to Charter = 1 in the table schools; Full name refers to first name, last name"
    python script/caf/episodic_search.py \
        --question "How many cards of legalities whose status is restricted are found in a starter deck?" \
        --database "card_games" \
        --sql "SELECT COUNT(*) \nFROM cards c \nINNER JOIN legalities l ON c.uuid = l.uuid \nWHERE c.isStarter = 1 \nAND l.status = 'Restricted'" \
        --limit 20
    python script/caf/episodic_search.py -q "How many students?" -d "financial" --limit 10
"""

import sys
import logging
import argparse
import string
import json
from typing import Dict, Any, List, Tuple

import caf
from caf.memory.types import MemoryType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def test_enhanced_episodic_search(question: str, database_id: str, evidence: str = None, generated_sql: str = None, limit: int = 5):
    """Test enhanced episodic memory search with given question and database_id"""
    
    logger.info("="*80)
    logger.info("Enhanced Episodic Memory Search Test")
    logger.info("="*80)
    
    # Initialize CAF system
    logger.info("Initializing CAF system...")
    try:
        caf_system = caf.initialize(config_path="config/caf_config.yaml")
        logger.info("✓ CAF system initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize CAF system: {e}")
        return False
    
    # Start session and bind database
    logger.info(f"Starting session and binding to database: {database_id}")
    session_id = f"test_enhanced_search_{database_id}_{hash(question) % 10000}"
    caf_system.start_session(session_id, database_id)
    logger.info("✓ Session started and database bound")

    logger.info(f"Question: {question}")
    logger.info(f"Database: {database_id}")
    logger.info(f"Generated SQL: {generated_sql if generated_sql else 'Not provided (will auto-generate)'}")
    logger.info(f"Limit: {limit}")
    logger.info("-" * 80)
    

    # Test episodic search
    logger.info("Executing episodic search...")

    
    response = caf_system.read_memory(
        memory_type=MemoryType.EPISODIC,
        query_content=question,
        context={'evidence': evidence} if evidence else None,
        generated_sql=generated_sql,
        limit=limit
    )
    
    # Organize and display results grouped by same db and cross db
    print("\n" + "="*80)
    print(f"ENHANCED SEARCH RESULTS ({len(response.items)} found)")
    print("="*80)
    
    if response.items:
        # Normalize question for grouping (same logic as episodic store)
        def normalize_question(q: str) -> str:
            """Normalize question for grouping"""
            if not q:
                return ""
            normalized = ' '.join(q.strip().split()).lower()
            # Remove trailing punctuation
            while normalized and normalized[-1] in string.punctuation:
                normalized = normalized[:-1]
            return normalized
        
        # Separate items into same db and cross db groups
        same_db_items = []
        cross_db_items = []
        
        for item in response.items:
            content = item.content
            item_database_id = content.get('database_id', '')
            if item_database_id == database_id:
                same_db_items.append(item)
            else:
                cross_db_items.append(item)
        
        # Helper function to display items grouped by question
        def display_items_grouped(items: List, db_type: str):
            """Display items grouped by normalized question"""
            if not items:
                return
            
            # Group items by normalized question
            question_groups = {}
            for item in items:
                content = item.content
                user_query = content.get('user_query', '')
                normalized_q = normalize_question(user_query)
                
                if normalized_q not in question_groups:
                    question_groups[normalized_q] = []
                question_groups[normalized_q].append(item)
            
            # Display results grouped by question
            for group_idx, (normalized_q, group_items) in enumerate(question_groups.items(), 1):
                # Use the first item's original question as display
                display_question = group_items[0].content.get('user_query', normalized_q)
                
                print(f"\n{'='*80}")
                print(f"Question Group {group_idx}: {display_question}")
                print(f"{'='*80}")
                print(f"Total SQL variants: {len(group_items)}")
                print()
                
                # Sort items by score (descending)
                sorted_items = sorted(group_items, key=lambda x: x.score, reverse=True)
                
                for sql_idx, item in enumerate(sorted_items, 1):
                    content = item.content
                    metadata_raw = content.get('metadata', {})
                    
                    # Parse metadata if it's a JSON string
                    metadata = metadata_raw
                    if isinstance(metadata_raw, str):
                        try:
                            metadata = json.loads(metadata_raw)
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    elif not isinstance(metadata_raw, dict):
                        metadata = {}
                    
                    # Check if this is an insight record
                    is_insight = False
                    if isinstance(metadata, dict):
                        is_insight = 'comparative_insight' in metadata or 'original_sqls' in metadata
                    
                    # Get SQL: for insight records, use incorrect SQL from original_sqls
                    if is_insight and isinstance(metadata, dict):
                        original_sqls = metadata.get('original_sqls', {})
                        if isinstance(original_sqls, dict):
                            generated_sql = original_sqls.get('incorrect', content.get('generated_sql', 'N/A'))
                        else:
                            generated_sql = content.get('generated_sql', 'N/A')
                    else:
                        generated_sql = content.get('generated_sql', 'N/A')
                    
                    label = content.get('label', None)
                    
                    # Determine SQL status
                    if label is True:
                        status = "✓ CORRECT"
                    elif label is False:
                        status = "✗ ERROR"
                    else:
                        status = "? UNKNOWN"
                    
                    # Add insight indicator
                    insight_indicator = " [INSIGHT]" if is_insight else ""
                    
                    print(f"\n  --- SQL Variant {sql_idx} ({status}){insight_indicator} ---")
                    print(f"  Generated SQL: {generated_sql}")
                    print(f"  Final Score: {item.score:.4f}")
                    
                    # Enhanced retrieval metadata
                    if 'retrieval_type' in content:
                        print(f"  Retrieval Type: {content['retrieval_type']}")
                    if 'nlq_similarity' in content:
                        print(f"  NLQ Similarity: {content['nlq_similarity']:.4f}")
                    if 'sql_skeleton_similarity' in content:
                        print(f"  SQL Skeleton Similarity: {content['sql_skeleton_similarity']:.4f}")
                    
                    print(f"  Database: {content.get('database_id', 'N/A')}")
                    
                    # Display comparative insight for insight records
                    if is_insight and isinstance(metadata, dict):
                        comparative_insight = metadata.get('comparative_insight')
                        if isinstance(comparative_insight, dict):
                            print(f"  --- Comparative Insight ---")
                            if 'pred_logic' in comparative_insight:
                                print(f"  Pred Logic: {comparative_insight['pred_logic']}")
                            if 'gold_logic' in comparative_insight:
                                print(f"  Gold Logic: {comparative_insight['gold_logic']}")
                            if 'key_difference' in comparative_insight:
                                print(f"  Key Difference: {comparative_insight['key_difference']}")
                    
                    if content.get('error_types'):
                        print(f"  Error Types: {content.get('error_types', 'N/A')}")
                    if content.get('feedback_text'):
                        print(f"  Feedback: {content.get('feedback_text', 'N/A')}")
                    

        
        # Display same db results
        print("\n" + "="*80)
        print(f"SAME DATABASE RESULTS ({len(same_db_items)} found)")
        print(f"Target Database: {database_id}")
        print("="*80)
        display_items_grouped(same_db_items, "same_db")
        
        # Display cross db results
        print("\n" + "="*80)
        print(f"CROSS DATABASE RESULTS ({len(cross_db_items)} found)")
        print(f"Target Database: {database_id}")
        print("="*80)
        display_items_grouped(cross_db_items, "cross_db")
            
    else:
        print("No results found")

    # Cleanup
    logger.info("\nCleaning up...")
    caf_system.cleanup()
    
    return True



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Episodic Memory Search Test Script",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--question", "-q",
        required=True,
        help="Natural language question to search for"
    )
    
    parser.add_argument(
        "--database", "-d", 
        required=True,
        help="Database ID to search in"
    )

    parser.add_argument(
        "--evidence", "-e", 
        required=False,
        default='',
        help="Evidence to help generate the SQL query"
    )
    
    parser.add_argument(
        "--sql", "-s",
        default=None,
        help="Generated SQL query (optional, will auto-generate if not provided)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 5)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)
    
    logger.info("Enhanced Episodic Memory Search Test")
    logger.info(f"Question: {args.question}")
    logger.info(f"Database: {args.database}")
    logger.info(f"SQL: {args.sql if args.sql else 'Not provided (will auto-generate)'}")
    logger.info(f"Limit: {args.limit}")
    
    success = test_enhanced_episodic_search(
        question=args.question,
        evidence=args.evidence,
        database_id=args.database, 
        generated_sql=args.sql,
        limit=args.limit
    )
    
    if success:
        logger.info("Enhanced episodic search test completed successfully!")
        sys.exit(0)
    else:
        logger.error("Enhanced episodic search test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
