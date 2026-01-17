#!/usr/bin/env python3
"""
PKL文件导出为CSV脚本

用于将semantic_memory中的pkl文件导出为CSV格式，方便查看和分析

Usage:
    python scripts/export_pkl_to_csv.py [--database DATABASE_ID] [--type METADATA_TYPE] [--path FILE_PATH] [--output OUTPUT_PATH]

Examples:
    # 导出california_schools数据库的column metadata为CSV
    python script/caf/export_pkl_to_csv.py --database california_schools --type term
    
    # 导出指定文件
    python scripts/export_pkl_to_csv.py --path /home/yangchenyu/Text2SQL/memory/semantic_memory/california_schools/column.pkl
    
    # 指定输出路径
    python scripts/export_pkl_to_csv.py --database california_schools --type column --output output/columns.csv
    
    # 导出所有数据库的所有文件
    python scripts/export_pkl_to_csv.py --export-all --output-dir output/
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import json

def export_pkl_to_csv(file_path: Path, output_path: Optional[Path] = None, 
                     include_index: bool = False, encoding: str = 'utf-8'):
    """将pkl文件导出为CSV"""
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    try:
        # 加载DataFrame
        df = pd.read_pickle(file_path)
        
        # 确定输出路径
        if output_path is None:
            output_path = file_path.with_suffix('.csv')
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 导出为CSV
        df.to_csv(output_path, index=include_index, encoding=encoding)
        
        print(f"✅ 成功导出: {file_path} -> {output_path}")
        print(f"📊 数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"💾 输出文件大小: {output_path.stat().st_size / 1024:.2f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False

def export_all_pkl_files(semantic_memory_dir: Path, output_dir: Path, 
                         include_index: bool = False, encoding: str = 'utf-8'):
    """导出所有pkl文件为CSV"""
    if not semantic_memory_dir.exists():
        print("❌ semantic_memory目录不存在")
        return False
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_count = 0
    failed_count = 0
    
    print(f"🗂️  开始导出所有pkl文件到: {output_dir}")
    print("=" * 60)
    
    # 遍历所有数据库目录
    for db_dir in semantic_memory_dir.iterdir():
        if not db_dir.is_dir():
            continue
            
        print(f"\n📁 处理数据库: {db_dir.name}")
        
        # 创建数据库子目录
        db_output_dir = output_dir / db_dir.name
        db_output_dir.mkdir(exist_ok=True)
        
        # 处理该数据库下的所有pkl文件
        pkl_files = list(db_dir.glob("*.pkl"))
        for pkl_file in pkl_files:
            output_path = db_output_dir / f"{pkl_file.stem}.csv"
            
            if export_pkl_to_csv(pkl_file, output_path, include_index, encoding):
                exported_count += 1
            else:
                failed_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 导出完成:")
    print(f"  ✅ 成功: {exported_count} 个文件")
    print(f"  ❌ 失败: {failed_count} 个文件")
    
    return exported_count > 0

def create_summary_report(semantic_memory_dir: Path, output_dir: Path):
    """创建数据摘要报告"""
    summary_data = {
        "databases": {},
        "total_files": 0,
        "total_rows": 0,
        "export_timestamp": pd.Timestamp.now().isoformat()
    }
    
    for db_dir in semantic_memory_dir.iterdir():
        if not db_dir.is_dir():
            continue
            
        db_name = db_dir.name
        db_data = {
            "files": {},
            "total_rows": 0
        }
        
        pkl_files = list(db_dir.glob("*.pkl"))
        for pkl_file in pkl_files:
            try:
                df = pd.read_pickle(pkl_file)
                file_info = {
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "file_size_kb": pkl_file.stat().st_size / 1024,
                    "column_names": list(df.columns)
                }
                db_data["files"][pkl_file.stem] = file_info
                db_data["total_rows"] += df.shape[0]
                summary_data["total_rows"] += df.shape[0]
            except Exception as e:
                db_data["files"][pkl_file.stem] = {"error": str(e)}
        
        summary_data["databases"][db_name] = db_data
        summary_data["total_files"] += len(pkl_files)
    
    # 保存摘要报告
    summary_path = output_dir / "export_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"📋 摘要报告已保存: {summary_path}")
    return summary_data

def main():
    parser = argparse.ArgumentParser(description="将semantic_memory中的pkl文件导出为CSV格式")
    parser.add_argument(
        '--database', '-d',
        type=str,
        help="数据库ID (例如: california_schools)"
    )
    parser.add_argument(
        '--type', '-t',
        type=str,
        choices=['database', 'table', 'column', 'relationship', 'term'],
        help="metadata类型"
    )
    parser.add_argument(
        '--path', '-p',
        type=Path,
        help="直接指定pkl文件路径"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help="输出CSV文件路径"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help="批量导出时的输出目录"
    )
    parser.add_argument(
        '--export-all',
        action='store_true',
        help="导出所有数据库的所有pkl文件"
    )
    parser.add_argument(
        '--include-index',
        action='store_true',
        help="在CSV中包含行索引"
    )
    parser.add_argument(
        '--encoding',
        type=str,
        default='utf-8',
        help="CSV文件编码 (默认: utf-8)"
    )
    parser.add_argument(
        '--create-summary',
        action='store_true',
        help="创建数据摘要报告"
    )
    parser.add_argument(
        '--semantic-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / "memory" / "semantic_memory",
        help="semantic_memory目录路径 (默认: ./memory/semantic_memory)"
    )
    
    args = parser.parse_args()
    
    # 导出所有文件
    if args.export_all:
        output_dir = args.output_dir or Path("output")
        success = export_all_pkl_files(args.semantic_dir, output_dir, 
                                     args.include_index, args.encoding)
        
        if args.create_summary:
            create_summary_report(args.semantic_dir, output_dir)
        
        if not success:
            sys.exit(1)
        return
    
    # 确定要导出的文件
    file_path = None
    
    if args.path:
        # 直接指定文件路径
        file_path = args.path
    elif args.database and args.type:
        # 通过数据库ID和类型指定
        file_path = args.semantic_dir / args.database / f"{args.type}.pkl"
    else:
        print("❌ 请指定要导出的文件:")
        print("  方式1: --path /path/to/file.pkl")
        print("  方式2: --database DATABASE_ID --type METADATA_TYPE")
        print("  方式3: --export-all (导出所有文件)")
        sys.exit(1)
    
    # 导出文件
    success = export_pkl_to_csv(file_path, args.output, 
                               args.include_index, args.encoding)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
