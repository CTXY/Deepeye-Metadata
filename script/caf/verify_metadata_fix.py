#!/usr/bin/env python3
"""
验证metadata保存修复的脚本

用法:
    python script/caf/verify_metadata_fix.py california_schools
"""
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any

def verify_metadata_consistency(database_id: str) -> Dict[str, Any]:
    """验证metadata的一致性"""
    memory_dir = Path(f"memory/semantic_memory/{database_id}")
    
    results = {
        'database_id': database_id,
        'status': 'unknown',
        'issues': [],
        'stats': {}
    }
    
    # 1. 检查文件是否存在
    column_file = memory_dir / "column.pkl"
    versions_file = memory_dir / "field_versions.pkl"
    
    if not column_file.exists():
        results['status'] = 'error'
        results['issues'].append(f"Column file not found: {column_file}")
        return results
    
    if not versions_file.exists():
        results['status'] = 'warning'
        results['issues'].append(f"Field versions file not found: {versions_file}")
        # Continue without versions check
    
    # 2. 读取数据
    column_df = pd.read_pickle(column_file)
    
    # 3. 统计主表数据
    total_columns = len(column_df)
    with_description = column_df['description'].notna().sum()
    with_pattern = column_df['pattern_description'].notna().sum()
    
    results['stats'] = {
        'total_columns': total_columns,
        'with_description': with_description,
        'with_pattern_description': with_pattern,
        'description_percentage': (with_description / total_columns * 100) if total_columns > 0 else 0,
        'pattern_percentage': (with_pattern / total_columns * 100) if total_columns > 0 else 0
    }
    
    # 4. 检查版本表一致性
    if versions_file.exists():
        versions_df = pd.read_pickle(versions_file)
        column_versions = versions_df[versions_df['metadata_type'] == 'column']
        
        # 统计版本表中的description
        desc_versions = column_versions[column_versions['field_name'] == 'description']
        version_columns = set()
        for _, row in desc_versions.iterrows():
            if pd.notna(row.get('table_name')) and pd.notna(row.get('column_name')):
                version_columns.add(f"{row['table_name']}.{row['column_name']}")
        
        # 统计主表中的description
        main_columns = set()
        for _, row in column_df[column_df['description'].notna()].iterrows():
            main_columns.add(f"{row['table_name']}.{row['column_name']}")
        
        # 找出不一致
        in_versions_not_main = version_columns - main_columns
        
        results['stats']['version_descriptions'] = len(version_columns)
        results['stats']['missing_in_main'] = len(in_versions_not_main)
        
        if in_versions_not_main:
            results['issues'].append(
                f"Found {len(in_versions_not_main)} columns with description in versions "
                f"but not in main table (data loss!)"
            )
    
    # 5. 判断状态
    if not results['issues']:
        if results['stats']['description_percentage'] > 80:
            results['status'] = 'excellent'
        elif results['stats']['description_percentage'] > 50:
            results['status'] = 'good'
        else:
            results['status'] = 'needs_improvement'
    else:
        results['status'] = 'has_issues'
    
    return results


def print_results(results: Dict[str, Any]):
    """打印验证结果"""
    print("=" * 80)
    print(f"Metadata Verification Results for: {results['database_id']}")
    print("=" * 80)
    
    # 状态指示器
    status_icons = {
        'excellent': '🌟',
        'good': '✅',
        'needs_improvement': '⚠️',
        'has_issues': '❌',
        'error': '💥',
        'warning': '⚠️',
        'unknown': '❓'
    }
    
    icon = status_icons.get(results['status'], '❓')
    print(f"\nOverall Status: {icon} {results['status'].upper()}")
    
    # 统计信息
    if results['stats']:
        print("\n📊 Statistics:")
        stats = results['stats']
        print(f"  Total columns: {stats.get('total_columns', 0)}")
        print(f"  With description: {stats.get('with_description', 0)} "
              f"({stats.get('description_percentage', 0):.1f}%)")
        print(f"  With pattern_description: {stats.get('with_pattern_description', 0)} "
              f"({stats.get('pattern_percentage', 0):.1f}%)")
        
        if 'version_descriptions' in stats:
            print(f"\n  Version table descriptions: {stats['version_descriptions']}")
            print(f"  Missing in main table: {stats.get('missing_in_main', 0)}")
    
    # 问题列表
    if results['issues']:
        print(f"\n⚠️ Issues Found ({len(results['issues'])}):")
        for i, issue in enumerate(results['issues'], 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✅ No issues found!")
    
    # 建议
    print("\n💡 Recommendations:")
    if results['status'] == 'excellent':
        print("  🎉 Excellent! Metadata is well-populated.")
    elif results['status'] == 'good':
        print("  👍 Good coverage. Consider generating metadata for remaining columns.")
    elif results['status'] == 'needs_improvement':
        print("  ⚠️ Low coverage detected. Recommend running:")
        print(f"     python script/caf/generate_metadata.py --database {results['database_id']} --force")
    elif results['status'] == 'has_issues':
        print("  ❌ Data consistency issues detected!")
        print("  🔧 This was a known bug that has been fixed.")
        print("  📝 Please regenerate metadata with the fixed code:")
        print(f"     python script/caf/generate_metadata.py --database {results['database_id']} --force")
    elif results['status'] == 'error':
        print("  💥 Critical errors found. Check file paths and permissions.")
    
    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_metadata_fix.py <database_id>")
        print("Example: python verify_metadata_fix.py california_schools")
        sys.exit(1)
    
    database_id = sys.argv[1]
    
    print(f"\n🔍 Verifying metadata for database: {database_id}\n")
    
    results = verify_metadata_consistency(database_id)
    print_results(results)
    
    # Exit code based on status
    if results['status'] in ['excellent', 'good']:
        sys.exit(0)
    elif results['status'] == 'needs_improvement':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()








