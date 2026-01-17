#!/usr/bin/env python3
"""
Test script for ambiguous pairs functionality.

This script tests the basic functionality of the ambiguous pairs system:
1. Store/load pairs
2. Query pairs
3. Basic deduplication
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from caf.memory.stores.ambiguous_pair import AmbiguousPairStore
from caf.memory.types import (
    AmbiguousPair,
    DBElementRef,
    DiffProfile,
    DataContentProfile,
    SemanticIntentProfile,
)

def test_basic_operations():
    """Test basic store operations."""
    print("Testing basic operations...")
    
    # Create store
    store = AmbiguousPairStore(Path("./test_memory/ambiguous_pairs"))
    store.bind_database("test_db")
    
    # Create test pairs
    pair1 = AmbiguousPair(
        pair_id="test_pair_001",
        database_id="test_db",
        column_a=DBElementRef(table_name="orders", column_name="order_date"),
        column_b=DBElementRef(table_name="orders", column_name="shipping_date"),
        discovery_methods=["pseudo_query_collision"],
        semantic_collision_score=0.85,
    )
    
    pair2 = AmbiguousPair(
        pair_id="test_pair_002",
        database_id="test_db",
        column_a=DBElementRef(table_name="schools", column_name="city"),
        column_b=DBElementRef(table_name="schools", column_name="district"),
        discovery_methods=["value_overlap"],
        value_jaccard=0.65,
    )
    
    # Save pairs
    store.save_pairs("test_db", [pair1, pair2])
    print(f"✓ Saved {len([pair1, pair2])} pairs")
    
    # List pairs
    pairs = store.list_pairs()
    print(f"✓ Loaded {len(pairs)} pairs")
    assert len(pairs) == 2
    
    # Get specific pair
    pair = store.get_pair("orders", "order_date", "orders", "shipping_date")
    assert pair is not None
    print(f"✓ Retrieved specific pair: {pair.pair_id}")
    
    # Get pairs for column
    pairs_for_city = store.get_pairs_for_column("schools", "city")
    assert len(pairs_for_city) == 1
    print(f"✓ Found {len(pairs_for_city)} pairs containing 'city'")
    
    # Statistics
    stats = store.get_statistics()
    print(f"✓ Statistics: {stats}")
    
    print("\n✅ Basic operations test passed!\n")


def test_deduplication():
    """Test deduplication and merging."""
    print("Testing deduplication...")
    
    store = AmbiguousPairStore(Path("./test_memory/ambiguous_pairs"))
    store.bind_database("test_dedup")
    
    # Create duplicate pairs with different discovery methods
    pair1 = AmbiguousPair(
        pair_id="test_dedup_001",
        database_id="test_dedup",
        column_a=DBElementRef(table_name="table1", column_name="col_a"),
        column_b=DBElementRef(table_name="table1", column_name="col_b"),
        discovery_methods=["pseudo_query_collision"],
        semantic_collision_score=0.80,
    )
    
    pair2 = AmbiguousPair(
        pair_id="test_dedup_002",
        database_id="test_dedup",
        column_a=DBElementRef(table_name="table1", column_name="col_a"),
        column_b=DBElementRef(table_name="table1", column_name="col_b"),
        discovery_methods=["value_overlap"],
        value_jaccard=0.70,
    )
    
    # Save first pair
    store.save_pairs("test_dedup", [pair1])
    
    # Append second pair (should be deduplicated)
    store.append_pairs("test_dedup", [pair2], deduplicate=True)
    
    # Check result
    pairs = store.list_pairs()
    print(f"✓ After deduplication: {len(pairs)} pairs (expected: 1)")
    assert len(pairs) == 1
    
    merged = pairs[0]
    print(f"✓ Merged methods: {merged.discovery_methods}")
    assert "pseudo_query_collision" in merged.discovery_methods
    assert "value_overlap" in merged.discovery_methods
    assert merged.semantic_collision_score == 0.80
    assert merged.value_jaccard == 0.70
    
    print("\n✅ Deduplication test passed!\n")


def test_diff_profile():
    """Test DiffProfile storage and retrieval."""
    print("Testing DiffProfile...")
    
    store = AmbiguousPairStore(Path("./test_memory/ambiguous_pairs"))
    store.bind_database("test_profile")
    
    # Create pair with full DiffProfile
    data_profile = DataContentProfile(
        set_relationship="overlapping",
        jaccard_similarity=0.45,
        constraint_rule="A <= B",
        sensitivity_type="high_sensitivity",
        avg_result_overlap=0.12,
    )
    
    semantic_profile = SemanticIntentProfile(
        semantic_nuance="A is when order was placed; B is when order was shipped",
        scenario_a="Show me orders placed last week",
        scenario_b="Show me orders shipped last week",
        trigger_keywords_a=["placed", "ordered", "bought"],
        trigger_keywords_b=["shipped", "delivered", "sent"],
    )
    
    diff_profile = DiffProfile(
        data_content_profile=data_profile,
        semantic_intent_profile=semantic_profile,
        guidance_rule="Use A for sales analysis; Use B for delivery performance",
    )
    
    pair = AmbiguousPair(
        pair_id="test_profile_001",
        database_id="test_profile",
        column_a=DBElementRef(table_name="orders", column_name="order_date"),
        column_b=DBElementRef(table_name="orders", column_name="shipping_date"),
        discovery_methods=["pseudo_query_collision"],
        semantic_collision_score=0.85,
        diff_profile=diff_profile,
    )
    
    # Save and reload
    store.save_pairs("test_profile", [pair])
    pairs = store.list_pairs()
    
    assert len(pairs) == 1
    loaded_pair = pairs[0]
    
    # Verify DiffProfile
    assert loaded_pair.diff_profile is not None
    assert loaded_pair.diff_profile.data_content_profile is not None
    assert loaded_pair.diff_profile.semantic_intent_profile is not None
    assert loaded_pair.diff_profile.guidance_rule == diff_profile.guidance_rule
    
    print(f"✓ DiffProfile saved and loaded successfully")
    print(f"✓ Guidance rule: {loaded_pair.diff_profile.guidance_rule}")
    
    print("\n✅ DiffProfile test passed!\n")


def test_sorted_pair_key():
    """Test canonical pair key generation."""
    print("Testing sorted pair key...")
    
    pair1 = AmbiguousPair(
        pair_id="test_1",
        database_id="test",
        column_a=DBElementRef(table_name="t1", column_name="a"),
        column_b=DBElementRef(table_name="t1", column_name="b"),
        discovery_methods=["test"],
    )
    
    pair2 = AmbiguousPair(
        pair_id="test_2",
        database_id="test",
        column_a=DBElementRef(table_name="t1", column_name="b"),
        column_b=DBElementRef(table_name="t1", column_name="a"),
        discovery_methods=["test"],
    )
    
    key1 = pair1.get_sorted_pair_key()
    key2 = pair2.get_sorted_pair_key()
    
    print(f"✓ Key 1: {key1}")
    print(f"✓ Key 2: {key2}")
    assert key1 == key2, "Sorted keys should be equal for (a,b) and (b,a)"
    
    print("\n✅ Sorted pair key test passed!\n")


def main():
    """Run all tests."""
    print("="*80)
    print("TESTING AMBIGUOUS PAIRS FUNCTIONALITY")
    print("="*80 + "\n")
    
    try:
        test_basic_operations()
        test_deduplication()
        test_diff_profile()
        test_sorted_pair_key()
        
        print("="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()












