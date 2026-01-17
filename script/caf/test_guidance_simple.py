"""
Simple test for Guidance Memory Store (without full CAF system initialization)

This test directly tests the GuidanceMemoryStore without requiring
full CAF system initialization, avoiding CUDA/PyTorch dependency issues.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_guidance_store_basic():
    """Basic test of guidance store functionality"""
    
    print("=" * 80)
    print("Basic Guidance Store Test")
    print("=" * 80)
    
    # 1. Import and initialize store
    print("\n1. Importing GuidanceMemoryStore...")
    from caf.memory.stores.guidance import GuidanceMemoryStore
    from caf.memory.types import MemoryQuery, MemoryType
    
    print("✓ Import successful")
    
    # 2. Create store with config
    print("\n2. Initializing store...")
    config = {
        'guidance': {
            'insights_path': '/home/yangchenyu/DeepEye-SQL-Metadata/output/error_analysis/damo/refined_insights.jsonl'
        }
    }
    
    store = GuidanceMemoryStore(config)
    print(f"✓ Store initialized with {len(store.insights)} insights")
    
    # 3. Check some insights
    print("\n3. Checking loaded insights...")
    if store.insights:
        sample_insight = store.insights[0]
        print(f"   Sample insight ID: {sample_insight.get('insight_id')}")
        print(f"   Has retrieval_key: {'retrieval_key' in sample_insight}")
        print(f"   Has guidance: {'guidance' in sample_insight}")
        
        if 'retrieval_key' in sample_insight:
            sql_risk_atoms = sample_insight['retrieval_key'].get('sql_risk_atoms', [])
            print(f"   SQL risk atoms: {sql_risk_atoms}")
    
    # 4. Test search without schema info (simple version)
    print("\n4. Testing search without schema info...")
    
    test_sql = """
    SELECT movies.movie_title, movies.movie_popularity 
    FROM ratings 
    JOIN movies ON ratings.movie_id = movies.movie_id 
    WHERE ratings.rating_score = 5
    """
    
    query = MemoryQuery(
        memory_type=MemoryType.GUIDANCE,
        query_content="",
        context={
            'generated_sqls': [test_sql],
            'top_k': 3
        }
    )
    
    response = store.search(query)
    
    print(f"✓ Search completed")
    print(f"  Retrieved items: {len(response.items)}")
    print(f"  Query time: {response.query_time_ms}ms")
    print(f"  Total insights searched: {response.total_insights_searched}")
    
    # 5. Show top results with detailed information
    print("\n5. Top results (with detailed guidance):")
    for i, item in enumerate(response.items, 1):
        print(f"\n{'='*70}")
        print(f"   [{i}] Insight ID: {item.insight_id}")
        print(f"       Relevance Score: {item.relevance_score:.4f}")
        print(f"{'='*70}")
        
        # Intent
        print(f"\n   📌 Intent:")
        print(f"       {item.guidance.get('intent', 'N/A')}")
        
        # Strategy comparison
        strategy_incorrect = item.guidance.get('strategy_incorrect', {})
        strategy_correct = item.guidance.get('strategy_correct', {})
        
        print(f"\n   ❌ Incorrect Pattern:")
        print(f"       Pattern: {strategy_incorrect.get('pattern', 'N/A')}")
        print(f"       Implication: {strategy_incorrect.get('implication', 'N/A')}")
        
        print(f"\n   ✅ Correct Pattern:")
        print(f"       Pattern: {strategy_correct.get('pattern', 'N/A')}")
        print(f"       Implication: {strategy_correct.get('implication', 'N/A')}")
        
        # Actionable advice
        advice = item.guidance.get('actionable_advice', 'N/A')
        print(f"\n   💡 Actionable Advice:")
        print(f"       {advice}")
        
        # SQL examples
        if item.qualified_incorrect_sql:
            print(f"\n   📝 Example - Incorrect SQL:")
            print(f"       {item.qualified_incorrect_sql}")
        
        if item.qualified_correct_sql:
            print(f"\n   📝 Example - Correct SQL:")
            print(f"       {item.qualified_correct_sql}")
        
        # SQL risk atoms
        risk_atoms = item.retrieval_key.get('sql_risk_atoms', [])
        if risk_atoms:
            print(f"\n   🔍 SQL Risk Atoms: {', '.join(risk_atoms)}")
    
    # 6. Test with multiple SQLs
    print("\n" + "=" * 80)
    print("6. Testing with multiple SQLs (Union Top-K)...")
    
    test_sqls = [
        "SELECT * FROM table WHERE col = (SELECT MAX(col) FROM table)",
        "SELECT col FROM table ORDER BY col DESC LIMIT 1",
        "SELECT DISTINCT col FROM table GROUP BY col"
    ]
    
    query = MemoryQuery(
        memory_type=MemoryType.GUIDANCE,
        query_content="",
        context={
            'generated_sqls': test_sqls,
            'top_k': 5
        }
    )
    
    response = store.search(query)
    
    print(f"✓ Search completed")
    print(f"  Retrieved items: {len(response.items)}")
    print(f"  Query time: {response.query_time_ms}ms")
    print(f"  SQLs processed: {response.total_sqls_processed}")
    
    print("\n   Top 5 results (with detailed guidance):")
    for i, item in enumerate(response.items, 1):
        print(f"\n{'='*70}")
        print(f"   [{i}] Insight ID: {item.insight_id}")
        print(f"       Relevance Score: {item.relevance_score:.4f}")
        print(f"{'='*70}")
        
        # Intent
        print(f"\n   📌 Intent:")
        print(f"       {item.guidance.get('intent', 'N/A')}")
        
        # Strategy comparison
        strategy_incorrect = item.guidance.get('strategy_incorrect', {})
        strategy_correct = item.guidance.get('strategy_correct', {})
        
        print(f"\n   ❌ Incorrect Pattern:")
        print(f"       {strategy_incorrect.get('pattern', 'N/A')}")
        
        print(f"\n   ✅ Correct Pattern:")
        print(f"       {strategy_correct.get('pattern', 'N/A')}")
        
        # Actionable advice
        advice = item.guidance.get('actionable_advice', 'N/A')
        print(f"\n   💡 Actionable Advice:")
        print(f"       {advice}")
        
        # SQL examples
        if item.qualified_incorrect_sql:
            print(f"\n   📝 Example - Incorrect SQL:")
            print(f"       {item.qualified_incorrect_sql}")
        
        if item.qualified_correct_sql:
            print(f"\n   📝 Example - Correct SQL:")
            print(f"       {item.qualified_correct_sql}")
    
    # 7. Test statistics
    print("\n" + "=" * 80)
    print("7. Store statistics:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)




if __name__ == "__main__":
    try:
        test_guidance_store_basic()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
