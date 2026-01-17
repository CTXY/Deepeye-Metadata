"""
Test script for Guidance Memory Store

This script tests the new guidance memory functionality:
1. Load insights from insights.jsonl
2. Query with generated SQLs
3. Retrieve top-k relevant insights
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from caf.system import CAFSystem
from caf.config.loader import load_config


def test_guidance_memory():
    """Test guidance memory with sample SQLs"""
    
    print("=" * 80)
    print("Testing Guidance Memory Store")
    print("=" * 80)
    
    # 1. Initialize CAF system
    print("\n1. Initializing CAF system...")
    config = load_config()
    caf = CAFSystem(config)
    print("✓ CAF system initialized")
    
    # 2. Bind database
    print("\n2. Binding database...")
    database_id = "financial"  # Use any database ID
    caf.bind_database(database_id)
    print(f"✓ Database bound: {database_id}")
    
    # 3. Test with single SQL
    print("\n3. Testing with single SQL...")
    print("-" * 80)
    
    test_sql_1 = """
    SELECT movies.movie_title, movies.movie_popularity 
    FROM ratings 
    JOIN movies ON ratings.movie_id = movies.movie_id 
    WHERE ratings.rating_score = 5
    """
    
    print(f"Generated SQL:\n{test_sql_1.strip()}\n")
    
    response = caf.retrieve_sql_guidance(
        generated_sqls=test_sql_1,
        top_k=3
    )
    
    print(f"✓ Retrieved {len(response.items)} insights")
    print(f"  Query time: {response.query_time_ms}ms")
    print(f"  Total insights searched: {response.total_insights_searched}")
    print(f"  Total SQLs processed: {response.total_sqls_processed}")
    
    print("\nTop insights:")
    for i, item in enumerate(response.items, 1):
        print(f"\n  [{i}] Insight ID: {item.insight_id}")
        print(f"      Relevance Score: {item.relevance_score:.4f}")
        print(f"      Intent: {item.guidance.get('intent', 'N/A')}")
        print(f"      Advice: {item.guidance.get('actionable_advice', 'N/A')[:100]}...")
        
        # Show SQL risk atoms
        risk_atoms = item.retrieval_key.get('sql_risk_atoms', [])
        print(f"      SQL Risk Atoms: {', '.join(risk_atoms)}")
    
    # 4. Test with multiple SQLs
    print("\n" + "=" * 80)
    print("4. Testing with multiple SQLs...")
    print("-" * 80)
    
    test_sqls = [
        "SELECT col FROM table WHERE col = (SELECT MAX(col) FROM table)",
        "SELECT col1, col2 FROM t1 JOIN t2 WHERE condition ORDER BY col DESC LIMIT 1",
        "SELECT DISTINCT col FROM table GROUP BY col ORDER BY COUNT(col) DESC"
    ]
    
    print(f"Number of SQLs: {len(test_sqls)}\n")
    for i, sql in enumerate(test_sqls, 1):
        print(f"  SQL {i}: {sql[:70]}...")
    
    response = caf.retrieve_sql_guidance(
        generated_sqls=test_sqls,
        top_k=5
    )
    
    print(f"\n✓ Retrieved {len(response.items)} insights (Union Top-K)")
    print(f"  Query time: {response.query_time_ms}ms")
    print(f"  Total SQLs processed: {response.total_sqls_processed}")
    
    print("\nTop 5 insights:")
    for i, item in enumerate(response.items, 1):
        print(f"\n  [{i}] {item.insight_id} - Score: {item.relevance_score:.4f}")
        print(f"      Intent: {item.guidance.get('intent', 'N/A')}")
        strategy_incorrect = item.guidance.get('strategy_incorrect', {})
        print(f"      Incorrect Pattern: {strategy_incorrect.get('pattern', 'N/A')[:80]}...")
    
    # 5. Test with SQL that should match specific patterns
    print("\n" + "=" * 80)
    print("5. Testing with pattern-specific SQLs...")
    print("-" * 80)
    
    # Test SQL with ORDER BY + LIMIT pattern
    test_sql_orderby = """
    SELECT movie_title 
    FROM movies 
    ORDER BY movie_release_year ASC 
    LIMIT 1
    """
    
    print("Testing ORDER BY + LIMIT pattern (should match NULL handling insights):")
    print(f"SQL: {test_sql_orderby.strip()}\n")
    
    response = caf.retrieve_sql_guidance(
        generated_sqls=test_sql_orderby,
        top_k=3
    )
    
    print(f"✓ Retrieved {len(response.items)} insights")
    for i, item in enumerate(response.items, 1):
        print(f"\n  [{i}] {item.insight_id} - Score: {item.relevance_score:.4f}")
        print(f"      Intent: {item.guidance.get('intent', 'N/A')}")
        print(f"      Advice: {item.guidance.get('actionable_advice', 'N/A')[:120]}...")
    
    # 6. Summary
    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)


def test_guidance_store_stats():
    """Test guidance store statistics"""
    
    print("\n" + "=" * 80)
    print("Guidance Store Statistics")
    print("=" * 80)
    
    config = load_config()
    caf = CAFSystem(config)
    caf.bind_database("test_db")
    
    # Get guidance store
    from caf.memory.types import MemoryType
    guidance_store = caf._memory_base.get_memory_store(MemoryType.GUIDANCE)
    
    stats = guidance_store.get_stats()
    
    print(f"\nStore Statistics:")
    print(f"  Total insights: {stats['total_insights']}")
    print(f"  Insights path: {stats['insights_path']}")
    print(f"  Schema info available: {stats['schema_info_available']}")
    print(f"  Current database: {stats['current_database_id']}")
    
    print("\n✓ Statistics retrieved successfully")


if __name__ == "__main__":
    try:
        test_guidance_memory()
        test_guidance_store_stats()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
