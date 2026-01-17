#!/usr/bin/env python3
"""
View Ambiguous Pairs Analysis Details

This script displays detailed analysis information for ambiguous column pairs.

Usage:
    python script/view_ambiguous_pairs.py [OPTIONS]

Examples:
    # View all pairs with summary
    python script/view_ambiguous_pairs.py --database-id california_schools

    # View specific pair by index
    python script/view_ambiguous_pairs.py --database-id california_schools --index 0

    # View specific pair by column names
    python script/view_ambiguous_pairs.py --database-id california_schools --column-a "School Name" --column-b "sname"

    # Export to JSON
    python script/view_ambiguous_pairs.py --database-id california_schools --export output.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


def format_column_info(col: Dict[str, Any]) -> str:
    """Format column information."""
    if isinstance(col, dict):
        return f"{col.get('table_name', 'unknown')}.{col.get('column_name', 'unknown')}"
    return str(col)


def display_pair_summary(df: pd.DataFrame, index: int):
    """Display summary information for a pair."""
    row = df.iloc[index]
    
    print("\n" + "=" * 100)
    print(f"PAIR #{index}: {row['pair_id']}")
    print("=" * 100)
    
    print(f"\n📊 BASIC INFORMATION")
    print(f"  Database ID: {row['database_id']}")
    print(f"  Column A: {format_column_info(row['column_a'])}")
    print(f"  Column B: {format_column_info(row['column_b'])}")
    print(f"  Discovery Methods: {', '.join(row['discovery_methods'])}")
    print(f"  Semantic Collision Score: {row['semantic_collision_score']:.2f}")
    print(f"  Value Jaccard: {row['value_jaccard']:.4f}" if pd.notna(row['value_jaccard']) else f"  Value Jaccard: N/A")
    print(f"  Discovered At: {row['discovered_at']}")
    print(f"  Last Analyzed At: {row['last_analyzed_at']}")


def display_collision_details(collision_details: Any):
    """Display collision details."""
    print(f"\n🔍 COLLISION DETAILS")
    print("-" * 100)
    
    if isinstance(collision_details, str):
        collision_details = json.loads(collision_details)
    
    if not isinstance(collision_details, list):
        print("  No collision details available")
        return
    
    for i, collision in enumerate(collision_details, 1):
        print(f"\n  Collision #{i}:")
        print(f"    Trigger Query: \"{collision.get('trigger_query', 'N/A')}\"")
        print(f"    Source Column: {collision.get('source_column_id', 'N/A')}")
        print(f"    Distractor Column: {collision.get('distractor_column_id', 'N/A')}")
        print(f"    Collision Score: {collision.get('collision_score', 0):.2f}")
        print(f"    Target Score: {collision.get('target_score', 0):.2f}")
        print(f"    Query Type: {collision.get('query_type', 'N/A')}")
        print(f"    Collision Type: {collision.get('collision_type', 'N/A')}")


def display_diff_profile(diff_profile: Any):
    """Display diff profile information."""
    print(f"\n📋 DIFF PROFILE")
    print("-" * 100)
    
    if isinstance(diff_profile, str):
        diff_profile = json.loads(diff_profile)
    
    if not isinstance(diff_profile, dict):
        print("  No diff profile available")
        return
    
    # Data Content Profile
    data_profile = diff_profile.get('data_content_profile', {})
    if data_profile:
        print(f"\n  📊 Data Content Profile:")
        print(f"    Set Relationship: {data_profile.get('set_relationship', 'N/A')}")
        print(f"    Containment A in B: {data_profile.get('containment_a_in_b', 0):.4f}" if data_profile.get('containment_a_in_b') is not None else "    Containment A in B: N/A")
        print(f"    Containment B in A: {data_profile.get('containment_b_in_a', 0):.4f}" if data_profile.get('containment_b_in_a') is not None else "    Containment B in A: N/A")
        print(f"    Jaccard Similarity: {data_profile.get('jaccard_similarity', 0):.4f}" if data_profile.get('jaccard_similarity') is not None else "    Jaccard Similarity: N/A")
        print(f"    Sensitivity Type: {data_profile.get('sensitivity_type', 'N/A')}")
        print(f"    Sampled Value Count: {data_profile.get('sampled_value_count', 0)}")
        
        example_cases = data_profile.get('example_cases')
        if example_cases:
            print(f"    Example Cases:")
            for case in example_cases[:5]:  # Show first 5
                print(f"      - Value: {case.get('value', 'N/A')}")
                print(f"        Rows in A: {case.get('rows_a_count', 0)}, Rows in B: {case.get('rows_b_count', 0)}")
                print(f"        Overlap Jaccard: {case.get('overlap_jaccard', 0):.4f}")
    
    # Semantic Intent Profile
    semantic_profile = diff_profile.get('semantic_intent_profile', {})
    if semantic_profile:
        print(f"\n  🧠 Semantic Intent Profile:")
        print(f"    Entity Alignment:")
        print(f"      {semantic_profile.get('entity_alignment', 'N/A')}")
        print(f"\n    Column A Entity:")
        print(f"      {semantic_profile.get('column_a_entity', 'N/A')}")
        print(f"\n    Column B Entity:")
        print(f"      {semantic_profile.get('column_b_entity', 'N/A')}")
        print(f"\n    Semantic Nuance:")
        print(f"      {semantic_profile.get('semantic_nuance', 'N/A')}")
        print(f"\n    Scenario A (Use Column A):")
        print(f"      {semantic_profile.get('scenario_a', 'N/A')}")
        print(f"\n    Scenario B (Use Column B):")
        print(f"      {semantic_profile.get('scenario_b', 'N/A')}")
        print(f"\n    Discriminative Logic:")
        print(f"      {semantic_profile.get('discriminative_logic', 'N/A')}")
        
        keywords_a = semantic_profile.get('trigger_keywords_a', [])
        keywords_b = semantic_profile.get('trigger_keywords_b', [])
        if keywords_a:
            print(f"\n    Trigger Keywords for Column A:")
            print(f"      {', '.join(keywords_a)}")
        if keywords_b:
            print(f"\n    Trigger Keywords for Column B:")
            print(f"      {', '.join(keywords_b)}")
    
    # Guidance Rule
    guidance_rule = diff_profile.get('guidance_rule')
    if guidance_rule:
        print(f"\n  💡 Guidance Rule:")
        print(f"    {guidance_rule}")
    
    # Metadata
    print(f"\n  📅 Analysis Metadata:")
    print(f"    Timestamp: {diff_profile.get('analysis_timestamp', 'N/A')}")
    print(f"    Version: {diff_profile.get('analysis_version', 'N/A')}")


def display_full_pair_details(df: pd.DataFrame, index: int):
    """Display full details for a specific pair."""
    display_pair_summary(df, index)
    
    row = df.iloc[index]
    
    # Collision details
    display_collision_details(row['collision_details'])
    
    # Diff profile
    display_diff_profile(row['diff_profile'])
    
    print("\n" + "=" * 100 + "\n")


def list_all_pairs(df: pd.DataFrame):
    """List all pairs with summary."""
    print("\n" + "=" * 100)
    print(f"ALL AMBIGUOUS PAIRS ({len(df)} total)")
    print("=" * 100)
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        col_a = format_column_info(row['column_a'])
        col_b = format_column_info(row['column_b'])
        score = row['semantic_collision_score']
        
        print(f"\n[{idx:3d}] Score: {score:8.2f} | {col_a:40s} <-> {col_b:40s}")
        print(f"     Pair ID: {row['pair_id']}")


def find_pair_by_columns(df: pd.DataFrame, col_a_name: str, col_b_name: str) -> Optional[int]:
    """Find pair index by column names."""
    for idx in range(len(df)):
        row = df.iloc[idx]
        col_a = row['column_a']
        col_b = row['column_b']
        
        col_a_str = col_a.get('column_name', '') if isinstance(col_a, dict) else str(col_a)
        col_b_str = col_b.get('column_name', '') if isinstance(col_b, dict) else str(col_b)
        
        if (col_a_name.lower() in col_a_str.lower() or col_a_str.lower() in col_a_name.lower()) and \
           (col_b_name.lower() in col_b_str.lower() or col_b_str.lower() in col_b_name.lower()):
            return idx
    
    return None


def export_to_json(df: pd.DataFrame, output_path: Path):
    """Export all pairs to JSON file."""
    data = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        collision = row['collision_details']
        if isinstance(collision, str):
            collision = json.loads(collision)
        
        diff = row['diff_profile']
        if isinstance(diff, str):
            diff = json.loads(diff)
        
        data.append({
            'pair_id': row['pair_id'],
            'database_id': row['database_id'],
            'column_a': row['column_a'],
            'column_b': row['column_b'],
            'discovery_methods': row['discovery_methods'],
            'semantic_collision_score': float(row['semantic_collision_score']) if pd.notna(row['semantic_collision_score']) else None,
            'value_jaccard': float(row['value_jaccard']) if pd.notna(row['value_jaccard']) else None,
            'collision_details': collision,
            'diff_profile': diff,
            'discovered_at': str(row['discovered_at']),
            'last_analyzed_at': str(row['last_analyzed_at']),
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Exported {len(data)} pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="View detailed analysis information for ambiguous column pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --database-id california_schools
  %(prog)s --database-id california_schools --index 0
  %(prog)s --database-id california_schools --column-a "School Name" --column-b "sname"
  %(prog)s --database-id california_schools --export output.json
        """
    )
    
    parser.add_argument(
        '--database-id',
        type=str,
        required=True,
        help="Database ID (e.g., california_schools)"
    )
    
    parser.add_argument(
        '--memory-dir',
        type=Path,
        default=Path("memory/ambiguous_pairs"),
        help="Path to ambiguous pairs memory directory (default: memory/ambiguous_pairs)"
    )
    
    parser.add_argument(
        '--index',
        type=int,
        help="Display specific pair by index"
    )
    
    parser.add_argument(
        '--column-a',
        type=str,
        help="Find pair by column A name (partial match)"
    )
    
    parser.add_argument(
        '--column-b',
        type=str,
        help="Find pair by column B name (partial match, requires --column-a)"
    )
    
    parser.add_argument(
        '--export',
        type=Path,
        help="Export all pairs to JSON file"
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help="List all pairs with summary"
    )
    
    args = parser.parse_args()
    
    # Load parquet file
    parquet_file = args.memory_dir / f"ambiguous_pairs_{args.database_id}.parquet"
    
    if not parquet_file.exists():
        print(f"❌ Error: File not found: {parquet_file}")
        sys.exit(1)
    
    print(f"📂 Loading: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"✅ Loaded {len(df)} pairs\n")
    
    # Export mode
    if args.export:
        export_to_json(df, args.export)
        return
    
    # List mode
    if args.list or (args.index is None and args.column_a is None):
        list_all_pairs(df)
        return
    
    # Find specific pair
    pair_index = None
    
    if args.index is not None:
        if args.index < 0 or args.index >= len(df):
            print(f"❌ Error: Index {args.index} out of range (0-{len(df)-1})")
            sys.exit(1)
        pair_index = args.index
    
    elif args.column_a and args.column_b:
        pair_index = find_pair_by_columns(df, args.column_a, args.column_b)
        if pair_index is None:
            print(f"❌ Error: No pair found matching column A: '{args.column_a}', column B: '{args.column_b}'")
            sys.exit(1)
    
    elif args.column_a:
        print("⚠️  Warning: --column-a requires --column-b. Searching for partial matches...")
        for idx in range(len(df)):
            row = df.iloc[idx]
            col_a = row['column_a']
            col_a_str = col_a.get('column_name', '') if isinstance(col_a, dict) else str(col_a)
            if args.column_a.lower() in col_a_str.lower():
                pair_index = idx
                break
        if pair_index is None:
            print(f"❌ Error: No pair found with column A containing '{args.column_a}'")
            sys.exit(1)
    
    # Display details
    if pair_index is not None:
        display_full_pair_details(df, pair_index)


if __name__ == "__main__":
    main()











