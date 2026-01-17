#!/usr/bin/env python3
"""
Script to view data from sub_dev_metadata_guidance.pkl file.

This script allows you to view:
- question
- final_selected_sql
- schema_metadata
- join_relationships
- sql_guidance_items

Usage:
    python script/view_metadata_guidance.py [--file PATH] [--index INDEX] [--all] [--output OUTPUT]
"""

import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import Optional, List, Any


def format_dict(data: Any, indent: int = 2) -> str:
    """Format dictionary for pretty printing"""
    return json.dumps(data, indent=indent, ensure_ascii=False)


def get_item_attr(item: Any, attr_name: str, default: Any = None) -> Any:
    """Get attribute from item, handling both dict and object access"""
    if hasattr(item, attr_name):
        return getattr(item, attr_name)
    elif isinstance(item, dict):
        return item.get(attr_name, default)
    else:
        return default


def view_item(item: Any, index: int, show_all: bool = False):
    """View a single data item"""
    print("=" * 80)
    print(f"Item Index: {index}")
    question_id = get_item_attr(item, "question_id", "N/A")
    print(f"Question ID: {question_id}")
    print("=" * 80)
    
    # Question
    print("\n[Question]")
    print("-" * 80)
    question = get_item_attr(item, "question", "(Not available)")
    print(question)
    
    # Final Selected SQL
    print("\n[Final Selected SQL]")
    print("-" * 80)
    final_sql = get_item_attr(item, "final_selected_sql")
    if final_sql:
        print(final_sql)
    else:
        print("(Not available)")
    
    # Schema Metadata
    print("\n[Schema Metadata]")
    print("-" * 80)
    schema_metadata = get_item_attr(item, "schema_metadata")
    if schema_metadata:
        if show_all:
            print(format_dict(schema_metadata))
        else:
            print(f"Total entries: {len(schema_metadata)}")
            # Show first few entries
            items_list = list(schema_metadata.items()) if isinstance(schema_metadata, dict) else []
            for i, (key, value) in enumerate(items_list[:5]):
                print(f"\n  {key}:")
                print(f"    {format_dict(value, indent=4)}")
            if len(items_list) > 5:
                print(f"\n  ... and {len(items_list) - 5} more entries")
    else:
        print("(Not available)")
    
    # Join Relationships
    print("\n[Join Relationships]")
    print("-" * 80)
    join_relationships = get_item_attr(item, "join_relationships")
    if join_relationships:
        rels_list = list(join_relationships) if isinstance(join_relationships, (list, tuple)) else []
        if show_all:
            for i, rel in enumerate(rels_list):
                print(f"\nRelationship {i+1}:")
                print(format_dict(rel, indent=2))
        else:
            print(f"Total relationships: {len(rels_list)}")
            for i, rel in enumerate(rels_list[:3]):
                print(f"\nRelationship {i+1}:")
                print(format_dict(rel, indent=2))
            if len(rels_list) > 3:
                print(f"\n  ... and {len(rels_list) - 3} more relationships")
    else:
        print("(Not available)")
    
    # SQL Guidance Items
    print("\n[SQL Guidance Items]")
    print("-" * 80)
    sql_guidance_items = get_item_attr(item, "sql_guidance_items")
    if sql_guidance_items:
        guidance_list = list(sql_guidance_items) if isinstance(sql_guidance_items, (list, tuple)) else []
        if show_all:
            for i, guidance in enumerate(guidance_list):
                print(f"\nGuidance Item {i+1}:")
                print(format_dict(guidance, indent=2))
        else:
            print(f"Total guidance items: {len(guidance_list)}")
            for i, guidance in enumerate(guidance_list[:3]):
                print(f"\nGuidance Item {i+1}:")
                print(format_dict(guidance, indent=2))
            if len(guidance_list) > 3:
                print(f"\n  ... and {len(guidance_list) - 3} more guidance items")
    else:
        print("(Not available)")
    
    print("\n" + "=" * 80 + "\n")


def main():
    project_root = Path(__file__).parent.parent
    
    parser = argparse.ArgumentParser(
        description="View data from sub_dev_metadata_guidance.pkl file"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="workspace/sql_generation/bird/sub_dev_metadata_guidance.pkl",
        help="Path to the pickle file (default: workspace/sql_generation/bird/sub_dev_metadata_guidance.pkl)"
    )
    parser.add_argument(
        "--index",
        type=int,
        help="View a specific item by index (0-based)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all items (if --index is not specified) or show full details (if --index is specified)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path to save results (JSON format)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics only"
    )
    
    args = parser.parse_args()
    
    # Resolve file path
    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = project_root / file_path
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1
    
    print(f"Loading dataset from: {file_path}")
    
    # Try using dill if available (better at handling complex pickles)
    try:
        import dill
        with open(file_path, "rb") as f:
            dataset = dill.load(f)
        print("Loaded using dill")
    except ImportError:
        # Fall back to regular pickle with path setup
        import sys
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        
        # Try to fix pydantic import issue temporarily
        import pydantic
        if not hasattr(pydantic, 'model_validator'):
            # For pydantic v1, add a dummy model_validator
            def model_validator(*args, **kwargs):
                def decorator(func):
                    return func
                return decorator
            pydantic.model_validator = model_validator
        
        with open(file_path, "rb") as f:
            dataset = pickle.load(f)
        print("Loaded using pickle")
    
    # Handle both BaseDataset objects and lists
    if hasattr(dataset, "_data"):
        data_items = dataset._data
    elif hasattr(dataset, "__iter__") and not isinstance(dataset, (str, bytes)):
        data_items = list(dataset)
    else:
        data_items = [dataset] if not isinstance(dataset, list) else dataset
    
    total_items = len(data_items)
    print(f"Loaded {total_items} items")
    
    if args.summary:
        # Show summary statistics
        print("=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        print(f"Total items: {total_items}")
        
        items_with_sql = sum(1 for item in data_items if get_item_attr(item, "final_selected_sql"))
        items_with_metadata = sum(1 for item in data_items if get_item_attr(item, "schema_metadata"))
        items_with_joins = sum(1 for item in data_items if get_item_attr(item, "join_relationships"))
        items_with_guidance = sum(1 for item in data_items if get_item_attr(item, "sql_guidance_items"))
        
        print(f"\nItems with final_selected_sql: {items_with_sql} ({items_with_sql/total_items*100:.1f}%)")
        print(f"Items with schema_metadata: {items_with_metadata} ({items_with_metadata/total_items*100:.1f}%)")
        print(f"Items with join_relationships: {items_with_joins} ({items_with_joins/total_items*100:.1f}%)")
        print(f"Items with sql_guidance_items: {items_with_guidance} ({items_with_guidance/total_items*100:.1f}%)")
        
        if items_with_metadata:
            total_metadata_entries = sum(
                len(get_item_attr(item, "schema_metadata", {})) 
                for item in data_items 
                if get_item_attr(item, "schema_metadata")
            )
            avg_metadata = total_metadata_entries / items_with_metadata
            print(f"\nAverage schema_metadata entries per item: {avg_metadata:.1f}")
        
        if items_with_joins:
            total_joins = sum(
                len(get_item_attr(item, "join_relationships", [])) 
                for item in data_items 
                if get_item_attr(item, "join_relationships")
            )
            avg_joins = total_joins / items_with_joins
            print(f"Average join_relationships per item: {avg_joins:.1f}")
        
        if items_with_guidance:
            total_guidance = sum(
                len(get_item_attr(item, "sql_guidance_items", [])) 
                for item in data_items 
                if get_item_attr(item, "sql_guidance_items")
            )
            avg_guidance = total_guidance / items_with_guidance
            print(f"Average sql_guidance_items per item: {avg_guidance:.1f}")
        
        return 0
    
    # Prepare output data if output file is specified
    output_data = []
    
    if args.index is not None:
        if args.index < 0 or args.index >= total_items:
            print(f"Error: Index {args.index} is out of range (0-{total_items-1})", file=sys.stderr)
            return 1
        
        item = data_items[args.index]
        view_item(item, args.index, show_all=args.all)
        
        if args.output:
            output_data.append({
                "index": args.index,
                "question_id": get_item_attr(item, "question_id"),
                "question": get_item_attr(item, "question"),
                "final_selected_sql": get_item_attr(item, "final_selected_sql"),
                "schema_metadata": get_item_attr(item, "schema_metadata"),
                "join_relationships": get_item_attr(item, "join_relationships"),
                "sql_guidance_items": get_item_attr(item, "sql_guidance_items")
            })
    else:
        # Show all items
        for i, item in enumerate(data_items):
            view_item(item, i, show_all=args.all)
            
            if args.output:
                output_data.append({
                    "index": i,
                    "question_id": get_item_attr(item, "question_id"),
                    "question": get_item_attr(item, "question"),
                    "final_selected_sql": get_item_attr(item, "final_selected_sql"),
                    "schema_metadata": get_item_attr(item, "schema_metadata"),
                    "join_relationships": get_item_attr(item, "join_relationships"),
                    "sql_guidance_items": get_item_attr(item, "sql_guidance_items")
                })
    
    # Save to output file if specified
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Output saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
