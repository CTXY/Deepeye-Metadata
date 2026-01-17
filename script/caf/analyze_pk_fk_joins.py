"""
Script to analyze PK-FK relationships vs actual JOINs in BIRD dataset.

This script:
1. Extracts predefined PK-FK relationships from SQLite databases
2. Extracts JOIN relationships from SQL queries in dev.json
3. Compares them to find JOINs not covered by PK-FK relationships
"""

import json
import sqlite3
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import sqlglot
from sqlglot import expressions


def get_pk_fk_relationships(db_path: str) -> List[Dict]:
    """
    Extract PK-FK relationships from a SQLite database.
    
    Returns:
        List of dicts with structure:
        {
            'from_table': str,
            'from_column': str,
            'to_table': str,
            'to_column': str
        }
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    pk_fk_relationships = []
    
    for table in tables:
        # Get foreign key information for this table
        cursor.execute(f"PRAGMA foreign_key_list('{table}')")
        fk_info = cursor.fetchall()
        
        for fk in fk_info:
            # fk format: (id, seq, table, from, to, on_update, on_delete, match)
            pk_fk_relationships.append({
                'from_table': normalize_table_name(table),
                'from_column': normalize_table_name(fk[3]),  # from column
                'to_table': normalize_table_name(fk[2]),     # referenced table
                'to_column': normalize_table_name(fk[4])     # referenced column (to)
            })
    
    conn.close()
    return pk_fk_relationships


def normalize_table_name(name: str) -> str:
    """Normalize table name by removing quotes and converting to lowercase."""
    if not name:
        return ""
    # Remove quotes and whitespace
    name = name.strip().strip('"').strip("'").strip('`')
    return name.lower()


def extract_table_aliases(sql: str) -> Dict[str, str]:
    """
    Extract table aliases from SQL query using sqlglot AST parsing.
    Returns dict mapping normalized_alias -> actual_table_name.
    For derived tables/subqueries, maps alias -> alias (since we can't determine actual table).
    """
    table_aliases = {}
    
    try:
        # Parse SQL with sqlglot
        ast = sqlglot.parse_one(sql, dialect='sqlite', error_level=sqlglot.ErrorLevel.IGNORE)
        if not ast:
            return {}
        
        # Walk through AST to find all table aliases
        for node in ast.walk():
            if isinstance(node, expressions.Table):
                # Simple table with optional alias
                table_name = str(node.name)
                table_name_normalized = normalize_table_name(table_name)
                
                if node.alias:
                    alias = str(node.alias)
                    alias_normalized = normalize_table_name(alias)
                    # Map alias to actual table name
                    table_aliases[alias_normalized] = table_name_normalized
                
                # Also map table name to itself
                table_aliases[table_name_normalized] = table_name_normalized
            
            elif isinstance(node, (expressions.Subquery, expressions.CTE)):
                # Derived table or CTE with alias
                if node.alias:
                    alias = str(node.alias)
                    alias_normalized = normalize_table_name(alias)
                    # For subqueries, we can't determine the actual table
                    # Map alias to itself (we'll skip these in join extraction)
                    table_aliases[alias_normalized] = 'subquery'
    
    except Exception as e:
        # If parsing fails, return empty dict
        print(f"Warning: Failed to parse SQL for alias extraction: {e}")
        return {}
    
    return table_aliases


def resolve_table_name(identifier: str, table_aliases: Dict[str, str]) -> Optional[str]:
    """
    Resolve an identifier (could be alias or table name) to actual table name.
    Returns None if it's a subquery or can't be resolved.
    """
    identifier_normalized = normalize_table_name(identifier)
    resolved = table_aliases.get(identifier_normalized, identifier_normalized)
    
    # Return None if it's a subquery
    if resolved == 'subquery':
        return None
    
    return resolved


def extract_join_conditions_from_sql(sql: str) -> List[Dict]:
    """
    Extract JOIN conditions from SQL query using sqlglot AST parsing.
    
    Returns:
        List of dicts with structure:
        {
            'left_table': str,
            'left_column': str,
            'right_table': str,
            'right_column': str,
            'join_type': str
        }
    """
    joins = []
    
    # Extract table aliases first
    table_aliases = extract_table_aliases(sql)
    
    try:
        # Parse SQL with sqlglot
        ast = sqlglot.parse_one(sql, dialect='sqlite', error_level=sqlglot.ErrorLevel.IGNORE)
        if not ast:
            return []
        
        # Find all SELECT nodes (main query and subqueries)
        select_nodes = list(ast.find_all(expressions.Select))
        if not select_nodes:
            # If no SELECT found, try to use the root if it's a Select
            if isinstance(ast, expressions.Select):
                select_nodes = [ast]
            else:
                return []
        
        for select_node in select_nodes:
            # Extract joins from JOIN clauses
            # JOIN nodes are stored in select_node.args['joins']
            join_nodes = select_node.args.get('joins', [])
            
            for join_node in join_nodes:
                if not isinstance(join_node, expressions.Join):
                    continue
                
                join_type = join_node.args.get('kind', 'INNER')
                join_type_str = str(join_type).upper() if join_type else 'INNER'
                
                # Extract ON condition
                on_condition = join_node.args.get('on')
                if not on_condition:
                    continue
                
                # Extract join conditions from ON clause
                join_conditions = _extract_join_conditions_from_expression(
                    on_condition, table_aliases
                )
                
                for condition in join_conditions:
                    if condition:
                        condition['join_type'] = join_type_str
                        joins.append(condition)
            
            # Also check WHERE clause for implicit joins (table1.col = table2.col)
            where_clause = select_node.args.get('where')
            if where_clause:
                where_joins = _extract_join_conditions_from_expression(
                    where_clause, table_aliases
                )
                for condition in where_joins:
                    if condition:
                        condition['join_type'] = 'IMPLICIT'
                        joins.append(condition)
    
    except Exception as e:
        print(f"Warning: Failed to parse SQL for join extraction: {e}")
        return []
    
    return joins


def _extract_join_conditions_from_expression(
    expr: expressions.Expression, 
    table_aliases: Dict[str, str]
) -> List[Optional[Dict]]:
    """
    Extract join conditions from an expression (ON clause or WHERE clause).
    Returns list of join condition dicts or None for non-join conditions.
    """
    conditions = []
    
    # Handle equality expressions (column1 = column2)
    if isinstance(expr, expressions.EQ):
        condition = _extract_equality_join_condition(expr, table_aliases)
        if condition:
            return [condition]
    
    # Handle AND/OR expressions (recursively process both sides)
    elif isinstance(expr, (expressions.And, expressions.Or)):
        left_conditions = _extract_join_conditions_from_expression(
            expr.left, table_aliases
        )
        right_conditions = _extract_join_conditions_from_expression(
            expr.right, table_aliases
        )
        return left_conditions + right_conditions
    
    # For other expression types, return empty list
    return []


def _extract_equality_join_condition(
    eq_expr: expressions.EQ,
    table_aliases: Dict[str, str]
) -> Optional[Dict]:
    """
    Extract join condition from an equality expression if both sides are columns.
    Returns None if it's not a join condition (e.g., column = literal).
    """
    left = eq_expr.left
    right = eq_expr.right
    
    # Both sides must be Column expressions
    if not (isinstance(left, expressions.Column) and isinstance(right, expressions.Column)):
        return None
    
    # Extract left side
    left_column = str(left.name)
    left_table_ref = str(left.table) if left.table else None
    
    # Extract right side
    right_column = str(right.name)
    right_table_ref = str(right.table) if right.table else None
    
    # Both must have table references
    if not left_table_ref or not right_table_ref:
        return None
    
    # Resolve table names (handle aliases)
    left_table = resolve_table_name(left_table_ref, table_aliases)
    right_table = resolve_table_name(right_table_ref, table_aliases)
    
    # Skip if either couldn't be resolved (e.g., subquery)
    if not left_table or not right_table:
        return None
    
    # Skip if both sides are the same (self-join on same column, likely not a real join condition)
    if left_table == right_table and left_column == right_column:
        return None
    
    # Normalize column names
    left_column_normalized = normalize_table_name(left_column)
    right_column_normalized = normalize_table_name(right_column)
    
    return {
        'left_table': left_table,
        'left_column': left_column_normalized,
        'right_table': right_table,
        'right_column': right_column_normalized,
        'join_type': 'UNKNOWN'  # Will be set by caller
    }


def join_to_tuple(join: Dict) -> Tuple:
    """
    Convert a join relationship to a normalized tuple for comparison.
    Normalizes by sorting tables alphabetically to handle bidirectional joins.
    """
    left = (join['left_table'], join['left_column'])
    right = (join['right_table'], join['right_column'])
    
    # Sort to make comparison order-independent
    if left < right:
        return (left[0], left[1], right[0], right[1])
    else:
        return (right[0], right[1], left[0], left[1])


def pk_fk_to_tuple(pk_fk: Dict) -> Tuple:
    """Convert a PK-FK relationship to a normalized tuple for comparison."""
    left = (pk_fk['from_table'], pk_fk['from_column'])
    right = (pk_fk['to_table'], pk_fk['to_column'])
    
    # Sort to make comparison order-independent
    if left < right:
        return (left[0], left[1], right[0], right[1])
    else:
        return (right[0], right[1], left[0], left[1])


def main():
    # Configuration
    base_dir = Path('/home/yangchenyu/DeepEye-SQL-Metadata/dataset/bird')
    db_dir = base_dir / 'databases' / 'dev_databases'
    dev_json_path = base_dir / 'dev' / 'dev.json'
    output_dir = Path('/home/yangchenyu/DeepEye-SQL-Metadata/output/pk_fk_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dev.json
    with open(dev_json_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    # Get all database directories
    db_dirs = [d for d in db_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # Results storage
    results = {}
    all_stats = {
        'total_databases': 0,
        'total_predefined_pk_fks': 0,
        'total_queries': 0,
        'total_joins_in_queries': 0,
        'total_joins_not_in_pk_fk': 0,
        'queries_with_non_pk_fk_joins': 0
    }
    
    non_pk_fk_joins_detail = []
    
    # Process each database
    for db_folder in sorted(db_dirs):
        db_name = db_folder.name
        db_file = db_folder / f"{db_name}.sqlite"
        
        if not db_file.exists():
            print(f"Warning: Database file not found: {db_file}")
            continue
        
        print(f"\nProcessing database: {db_name}")
        
        # Extract PK-FK relationships
        pk_fk_relationships = get_pk_fk_relationships(str(db_file))
        pk_fk_tuples = set([pk_fk_to_tuple(pk_fk) for pk_fk in pk_fk_relationships])
        
        print(f"  Found {len(pk_fk_relationships)} PK-FK relationships")
        
        # Get queries for this database
        db_queries = [q for q in dev_data if q['db_id'] == db_name]
        
        # Extract JOINs from queries
        all_joins = []
        query_details = []
        
        for query in db_queries:
            sql = query['SQL']
            question_id = query['question_id']
            
            joins = extract_join_conditions_from_sql(sql)
            
            if joins:
                all_joins.extend(joins)
                query_details.append({
                    'question_id': question_id,
                    'sql': sql,
                    'joins': joins
                })
        
        # Convert joins to tuples for comparison
        join_tuples = [join_to_tuple(j) for j in all_joins]
        unique_join_tuples = set(join_tuples)
        
        print(f"  Found {len(db_queries)} queries with {len(all_joins)} total joins ({len(unique_join_tuples)} unique)")
        
        # Find joins not in PK-FK
        non_pk_fk_joins = unique_join_tuples - pk_fk_tuples
        
        print(f"  Joins NOT in PK-FK: {len(non_pk_fk_joins)}")
        
        # Count queries with non-PK-FK joins
        queries_with_non_pk_fk = 0
        for qd in query_details:
            query_join_tuples = set([join_to_tuple(j) for j in qd['joins']])
            if query_join_tuples & non_pk_fk_joins:  # Intersection
                queries_with_non_pk_fk += 1
                
                # Add to detail list
                non_pk_fk_joins_in_query = query_join_tuples & non_pk_fk_joins
                non_pk_fk_joins_detail.append({
                    'db_id': db_name,
                    'question_id': qd['question_id'],
                    'sql': qd['sql'],
                    'non_pk_fk_joins': [
                        {
                            'table1': j[0],
                            'column1': j[1],
                            'table2': j[2],
                            'column2': j[3]
                        } for j in non_pk_fk_joins_in_query
                    ]
                })
        
        # Store results for this database
        results[db_name] = {
            'pk_fk_relationships': [
                {
                    'from_table': r['from_table'],
                    'from_column': r['from_column'],
                    'to_table': r['to_table'],
                    'to_column': r['to_column']
                } for r in pk_fk_relationships
            ],
            'unique_joins_in_queries': [
                {
                    'table1': j[0],
                    'column1': j[1],
                    'table2': j[2],
                    'column2': j[3]
                } for j in unique_join_tuples
            ],
            'joins_not_in_pk_fk': [
                {
                    'table1': j[0],
                    'column1': j[1],
                    'table2': j[2],
                    'column2': j[3]
                } for j in non_pk_fk_joins
            ],
            'statistics': {
                'num_pk_fk_relationships': len(pk_fk_relationships),
                'num_queries': len(db_queries),
                'num_total_joins': len(all_joins),
                'num_unique_joins': len(unique_join_tuples),
                'num_joins_not_in_pk_fk': len(non_pk_fk_joins),
                'num_queries_with_non_pk_fk_joins': queries_with_non_pk_fk,
                'percentage_non_pk_fk_joins': round(len(non_pk_fk_joins) / len(unique_join_tuples) * 100, 2) if unique_join_tuples else 0
            }
        }
        
        # Update overall stats
        all_stats['total_databases'] += 1
        all_stats['total_predefined_pk_fks'] += len(pk_fk_relationships)
        all_stats['total_queries'] += len(db_queries)
        all_stats['total_joins_in_queries'] += len(all_joins)
        all_stats['total_joins_not_in_pk_fk'] += len(non_pk_fk_joins)
        all_stats['queries_with_non_pk_fk_joins'] += queries_with_non_pk_fk
    
    # Calculate final percentages
    if all_stats['total_joins_in_queries'] > 0:
        all_stats['percentage_non_pk_fk_joins'] = round(
            all_stats['total_joins_not_in_pk_fk'] / all_stats['total_joins_in_queries'] * 100, 2
        )
    
    # Save results
    output_file = output_dir / 'pk_fk_join_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'overall_statistics': all_stats,
            'per_database_results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total databases analyzed: {all_stats['total_databases']}")
    print(f"Total predefined PK-FK relationships: {all_stats['total_predefined_pk_fks']}")
    print(f"Total queries: {all_stats['total_queries']}")
    print(f"Total JOIN occurrences in queries: {all_stats['total_joins_in_queries']}")
    print(f"Total unique JOINs not in PK-FK: {all_stats['total_joins_not_in_pk_fk']}")
    print(f"Percentage: {all_stats.get('percentage_non_pk_fk_joins', 0):.2f}%")
    print(f"Queries with non-PK-FK joins: {all_stats['queries_with_non_pk_fk_joins']}")
    print(f"\nResults saved to: {output_file}")
    
    # Save detailed non-PK-FK joins
    detail_file = output_dir / 'non_pk_fk_joins_detail.json'
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(non_pk_fk_joins_detail, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed non-PK-FK joins saved to: {detail_file}")
    
    # Create a summary report
    report_file = output_dir / 'analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PK-FK vs JOIN Relationship Analysis Report\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total databases analyzed: {all_stats['total_databases']}\n")
        f.write(f"Total predefined PK-FK relationships: {all_stats['total_predefined_pk_fks']}\n")
        f.write(f"Total queries: {all_stats['total_queries']}\n")
        f.write(f"Total JOIN occurrences: {all_stats['total_joins_in_queries']}\n")
        f.write(f"Unique JOINs not in PK-FK: {all_stats['total_joins_not_in_pk_fk']}\n")
        f.write(f"Percentage: {all_stats.get('percentage_non_pk_fk_joins', 0):.2f}%\n")
        f.write(f"Queries with non-PK-FK joins: {all_stats['queries_with_non_pk_fk_joins']}\n\n")
        
        f.write("\nPER-DATABASE BREAKDOWN\n")
        f.write("-"*80 + "\n")
        
        for db_name in sorted(results.keys()):
            db_result = results[db_name]
            stats = db_result['statistics']
            
            f.write(f"\nDatabase: {db_name}\n")
            f.write(f"  PK-FK relationships: {stats['num_pk_fk_relationships']}\n")
            f.write(f"  Queries: {stats['num_queries']}\n")
            f.write(f"  Unique joins: {stats['num_unique_joins']}\n")
            f.write(f"  Joins not in PK-FK: {stats['num_joins_not_in_pk_fk']}\n")
            f.write(f"  Percentage: {stats['percentage_non_pk_fk_joins']:.2f}%\n")
            f.write(f"  Queries with non-PK-FK joins: {stats['num_queries_with_non_pk_fk_joins']}\n")
            
            if db_result['joins_not_in_pk_fk']:
                f.write(f"  Non-PK-FK joins:\n")
                for join in db_result['joins_not_in_pk_fk']:
                    f.write(f"    - {join['table1']}.{join['column1']} = {join['table2']}.{join['column2']}\n")
    
    print(f"Summary report saved to: {report_file}")


if __name__ == '__main__':
    main()
