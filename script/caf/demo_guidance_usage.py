"""
Demonstration of Guidance Memory Store Usage

This script demonstrates how to use the guidance memory functionality
without requiring full CAF system initialization.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def demo_guidance_retrieval():
    """Demonstrate guidance retrieval"""
    
    print("=" * 80)
    print("Guidance Memory Store - Usage Demonstration")
    print("=" * 80)
    
    from caf.memory.stores.guidance import GuidanceMemoryStore
    from caf.memory.types import MemoryQuery, MemoryType
    
    # 1. Initialize store
    print("\n[1] Initializing Guidance Store...")
    config = {
        'guidance': {
            'insights_path': '/home/yangchenyu/DeepEye-SQL-Metadata/output/error_analysis/damo/insights.jsonl'
        }
    }
    
    store = GuidanceMemoryStore(config)
    print(f"    ✓ Loaded {len(store.insights)} insights")
    
    # 2. Example 1: Retrieve guidance for SQL with potential JOIN issue
    print("\n" + "=" * 80)
    print("[2] Example 1: SQL with Potential JOIN + DISTINCT Issue")
    print("=" * 80)
    
    sql_example_1 = """
    SELECT movies.movie_title, movies.movie_popularity 
    FROM ratings 
    JOIN movies ON ratings.movie_id = movies.movie_id 
    WHERE ratings.rating_score = 5
    """
    
    print(f"\nGenerated SQL:")
    print(f"  {' '.join(sql_example_1.split())}")
    
    query = MemoryQuery(
        memory_type=MemoryType.GUIDANCE,
        query_content="",
        context={'generated_sqls': [sql_example_1], 'top_k': 3}
    )
    
    response = store.search(query)
    
    print(f"\n📊 Results: {len(response.items)} insights (in {response.query_time_ms}ms)")
    
    for i, item in enumerate(response.items, 1):
        print(f"\n{'─'*60}")
        print(f"Insight #{i}: {item.insight_id} (Score: {item.relevance_score:.4f})")
        print(f"{'─'*60}")
        print(f"📌 Intent: {item.guidance['intent']}")
        print(f"\n❌ Common Mistake:")
        print(f"   {item.guidance['strategy_incorrect']['pattern']}")
        print(f"   → {item.guidance['strategy_incorrect']['implication']}")
        print(f"\n✅ Better Approach:")
        print(f"   {item.guidance['strategy_correct']['pattern']}")
        print(f"   → {item.guidance['strategy_correct']['implication']}")
        print(f"\n💡 Key Advice:")
        print(f"   {item.guidance['actionable_advice']}")
    
    # 3. Example 2: SQL with ORDER BY + LIMIT (NULL handling)
    print("\n\n" + "=" * 80)
    print("[3] Example 2: SQL with ORDER BY + LIMIT (NULL Handling)")
    print("=" * 80)
    
    sql_example_2 = """
    SELECT movie_title, release_year 
    FROM movies 
    ORDER BY release_year ASC 
    LIMIT 1
    """
    
    print(f"\nGenerated SQL:")
    print(f"  {' '.join(sql_example_2.split())}")
    print(f"\nPotential Issue: May return NULL values if they exist")
    
    query = MemoryQuery(
        memory_type=MemoryType.GUIDANCE,
        query_content="",
        context={'generated_sqls': [sql_example_2], 'top_k': 2}
    )
    
    response = store.search(query)
    
    print(f"\n📊 Results: {len(response.items)} insights")
    
    for i, item in enumerate(response.items, 1):
        print(f"\n[{i}] {item.insight_id} - Score: {item.relevance_score:.4f}")
        print(f"    Intent: {item.guidance['intent']}")
        print(f"    Advice: {item.guidance['actionable_advice']}")
    
    # 4. Example 3: Multiple SQLs with Union Top-K
    print("\n\n" + "=" * 80)
    print("[4] Example 3: Multiple SQLs with Union Top-K Aggregation")
    print("=" * 80)
    
    multiple_sqls = [
        "SELECT * FROM table WHERE col = (SELECT MAX(col) FROM table)",
        "SELECT col FROM table ORDER BY col DESC LIMIT 1",
        "SELECT col, COUNT(*) FROM table GROUP BY col ORDER BY COUNT(*) DESC"
    ]
    
    print(f"\nAnalyzing {len(multiple_sqls)} candidate SQLs...")
    for i, sql in enumerate(multiple_sqls, 1):
        print(f"  SQL {i}: {' '.join(sql.split())[:70]}...")
    
    query = MemoryQuery(
        memory_type=MemoryType.GUIDANCE,
        query_content="",
        context={'generated_sqls': multiple_sqls, 'top_k': 5}
    )
    
    response = store.search(query)
    
    print(f"\n📊 Aggregated Results: {len(response.items)} unique insights")
    print(f"   Query time: {response.query_time_ms}ms")
    print(f"   SQLs processed: {response.total_sqls_processed}")
    
    print(f"\nTop 5 Insights (Union Top-K across all SQLs):")
    for i, item in enumerate(response.items, 1):
        print(f"\n  [{i}] {item.insight_id} - Score: {item.relevance_score:.4f}")
        print(f"      {item.guidance['intent']}")
        print(f"      {item.guidance['actionable_advice'][:80]}...")
    
    # 5. Show statistics
    print("\n\n" + "=" * 80)
    print("[5] Store Statistics")
    print("=" * 80)
    
    stats = store.get_stats()
    print(f"\n  Total Insights: {stats['total_insights']}")
    print(f"  Insights Path: {stats['insights_path']}")
    print(f"  Schema Available: {stats['schema_info_available']}")
    
    # 6. Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("""
✓ Demonstration completed!

Key Features:
  • SQL Skeleton-based similarity matching
  • Keyword matching with SQL risk atoms
  • Union Top-K aggregation for multiple SQLs
  • Database-independent guidance patterns
  • Scoring: 70% skeleton similarity + 30% keyword match

Use Cases:
  1. Post-generation SQL review
  2. Multi-candidate SQL ranking with guidance
  3. Pattern-specific error detection
  4. Learning from historical mistakes

Integration:
  # Via CAF System API (recommended)
  response = caf.retrieve_sql_guidance(
      generated_sqls=["SELECT ...", "SELECT ..."],
      top_k=5
  )
  
  # Direct store usage (for advanced scenarios)
  store = GuidanceMemoryStore(config)
  query = MemoryQuery(...)
  response = store.search(query)
""")
    
    print("=" * 80)


def show_insight_examples():
    """Show examples of insights from the file"""
    
    print("\n" + "=" * 80)
    print("Sample Insights from Store")
    print("=" * 80)
    
    insights_path = Path('/home/yangchenyu/DeepEye-SQL-Metadata/output/error_analysis/damo/insights.jsonl')
    
    # Load first 3 insights as examples
    insights = []
    with open(insights_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            if line.strip():
                insights.append(json.loads(line))
    
    for i, insight in enumerate(insights, 1):
        print(f"\n{'='*60}")
        print(f"Sample Insight #{i}: {insight['insight_id']}")
        print(f"{'='*60}")
        
        print(f"\n📌 Intent: {insight['guidance']['intent']}")
        
        print(f"\n🔍 SQL Risk Atoms:")
        risk_atoms = insight['retrieval_key']['sql_risk_atoms']
        print(f"   {', '.join(risk_atoms)}")
        
        print(f"\n❌ Incorrect Strategy:")
        inc = insight['guidance']['strategy_incorrect']
        print(f"   Pattern: {inc['pattern']}")
        print(f"   Impact:  {inc['implication'][:100]}...")
        
        print(f"\n✅ Correct Strategy:")
        cor = insight['guidance']['strategy_correct']
        print(f"   Pattern: {cor['pattern']}")
        print(f"   Impact:  {cor['implication'][:100]}...")
        
        print(f"\n💡 Actionable Advice:")
        print(f"   {insight['guidance']['actionable_advice']}")
        
        print(f"\n📊 Verification:")
        print(f"   Success Rate: {insight['verification_success_rate']:.1%}")
        print(f"   Source Questions: {insight['source_question_ids']}")


if __name__ == "__main__":
    try:
        show_insight_examples()
        demo_guidance_retrieval()
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)










