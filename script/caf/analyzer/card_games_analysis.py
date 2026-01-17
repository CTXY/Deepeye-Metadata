#!/usr/bin/env python3
"""
Card Games 数据库专用分析脚本

专门分析 card_games 数据库中 setCode 字段的关系，特别关注 'OGW' 等特定集合。

运行方式：
conda activate deepeye
python script/caf/analyzer/card_games_analysis.py --set-code OGW
python script/caf/analyzer/card_games_analysis.py --full-analysis

Author: Generated for DeepEye-SQL-Metadata project
"""

import argparse
import os
import sys
import sqlite3
from typing import Dict, Any, List

# 兼容从项目根目录运行
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from script.caf.analyzer.database_field_analyzer import (
    analyze_cross_table_fields,
    get_table_schema_info,
    quick_field_overview
)


def analyze_specific_set_code(db_path: str, set_code: str = "OGW"):
    """分析特定 setCode 的详细情况"""
    print(f"\n" + "="*80)
    print(f"详细分析: setCode = '{set_code}'")
    print("="*80)
    
    conn = sqlite3.connect(db_path)
    
    try:
        # 1. 基本统计
        print(f"\n📊 '{set_code}' 基本统计:")
        
        # set_translations 中的情况
        query1 = "SELECT COUNT(*) FROM set_translations WHERE setCode = ?"
        count1 = conn.execute(query1, (set_code,)).fetchone()[0]
        print(f"  在 set_translations 中出现: {count1} 行")
        
        if count1 > 0:
            # 查看翻译信息
            trans_query = """
            SELECT language, translation 
            FROM set_translations 
            WHERE setCode = ? 
            ORDER BY language
            """
            translations = conn.execute(trans_query, (set_code,)).fetchall()
            print(f"  支持的语言数量: {len(translations)}")
            for lang, trans in translations[:5]:  # 显示前5种语言
                print(f"    {lang}: {trans}")
        
        # cards 中的情况
        query2 = "SELECT COUNT(*) FROM cards WHERE setCode = ?"
        count2 = conn.execute(query2, (set_code,)).fetchone()[0]
        print(f"  在 cards 中出现: {count2} 行")
        
        if count2 > 0:
            # 查看卡牌类型分布
            type_query = """
            SELECT type, COUNT(*) as count 
            FROM cards 
            WHERE setCode = ? 
            GROUP BY type 
            ORDER BY count DESC
            LIMIT 5
            """
            card_types = conn.execute(type_query, (set_code,)).fetchall()
            print(f"  主要卡牌类型:")
            for card_type, count in card_types:
                print(f"    {card_type}: {count} 张")
        
        # 2. 连接分析
        print(f"\n🔗 连接分析:")
        join_query = """
        SELECT COUNT(*) 
        FROM set_translations st 
        INNER JOIN cards c ON st.setCode = c.setCode 
        WHERE st.setCode = ?
        """
        join_count = conn.execute(join_query, (set_code,)).fetchone()[0]
        print(f"  成功连接的行数: {join_count}")
        
        if count1 > 0 and count2 > 0:
            expected_joins = count1 * count2
            print(f"  预期连接行数: {count1} × {count2} = {expected_joins}")
            print(f"  连接效率: {join_count / expected_joins:.1%}")
        
        # 3. 数据完整性检查
        print(f"\n🔍 数据完整性检查:")
        
        # 检查集合信息是否在 sets 表中
        sets_query = "SELECT COUNT(*) FROM sets WHERE code = ?"
        sets_count = conn.execute(sets_query, (set_code,)).fetchone()[0]
        print(f"  在 sets 主表中存在: {'是' if sets_count > 0 else '否'}")
        
        if sets_count > 0:
            set_info_query = """
            SELECT name, releaseDate, type 
            FROM sets 
            WHERE code = ?
            """
            set_info = conn.execute(set_info_query, (set_code,)).fetchone()
            if set_info:
                name, release_date, set_type = set_info
                print(f"  集合名称: {name}")
                print(f"  发布日期: {release_date}")
                print(f"  集合类型: {set_type}")
        
        # 4. 质量问题识别
        print(f"\n⚠️  潜在问题:")
        issues = []
        
        if count1 == 0:
            issues.append(f"'{set_code}' 在 set_translations 中不存在，可能缺少翻译信息")
        
        if count2 == 0:
            issues.append(f"'{set_code}' 在 cards 中不存在，可能是无效的集合代码")
        
        if sets_count == 0:
            issues.append(f"'{set_code}' 在 sets 主表中不存在，数据完整性有问题")
        
        if not issues:
            issues.append("未发现明显的数据质量问题")
        
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
    finally:
        conn.close()


def full_setcode_analysis(db_path: str):
    """全面的 setCode 关系分析"""
    print("\n" + "="*80)
    print("全面 setCode 关系分析")
    print("="*80)
    
    # 1. 跨表关系分析
    print("\n🔗 跨表关系分析:")
    result = analyze_cross_table_fields(
        db_path=db_path,
        table_a="set_translations",
        field_a="setCode",
        table_b="cards",
        field_b="setCode"
    )
    
    print(f"关系类型: {result['relationship_type']}")
    
    print("\n关键发现:")
    for finding in result["key_findings"]:
        print(f"  • {finding}")
    
    print("\n使用建议:")
    for rec in result["usage_recommendations"]:
        print(f"  • {rec}")
    
    # 2. 覆盖率详细分析
    print(f"\n📈 覆盖率详细分析:")
    
    conn = sqlite3.connect(db_path)
    try:
        # 找出 set_translations 中有但 cards 中没有的 setCode
        missing_in_cards_query = """
        SELECT st.setCode, COUNT(*) as trans_count
        FROM set_translations st
        LEFT JOIN cards c ON st.setCode = c.setCode
        WHERE c.setCode IS NULL
        GROUP BY st.setCode
        ORDER BY trans_count DESC
        LIMIT 10
        """
        missing_in_cards = conn.execute(missing_in_cards_query).fetchall()
        
        if missing_in_cards:
            print(f"  set_translations 中有但 cards 中没有的 setCode (前10个):")
            for set_code, count in missing_in_cards:
                print(f"    {set_code}: {count} 条翻译")
        
        # 找出 cards 中最常见但 set_translations 中没有翻译的 setCode
        missing_translations_query = """
        SELECT c.setCode, COUNT(*) as card_count
        FROM cards c
        LEFT JOIN set_translations st ON c.setCode = st.setCode
        WHERE st.setCode IS NULL
        GROUP BY c.setCode
        ORDER BY card_count DESC
        LIMIT 10
        """
        missing_translations = conn.execute(missing_translations_query).fetchall()
        
        if missing_translations:
            print(f"\n  cards 中有但缺少翻译的 setCode (前10个):")
            for set_code, count in missing_translations:
                print(f"    {set_code}: {count} 张卡牌")
        
    finally:
        conn.close()
    
    # 3. 推荐的改进建议
    print(f"\n💡 改进建议:")
    raw_data = result.get("raw_data", {})
    
    if "cross_table_join_analysis" in raw_data:
        join_analysis = raw_data["cross_table_join_analysis"]
        
        if not join_analysis["can_be_foreign_key"]:
            print("  • 考虑在 cards 表中添加外键约束到 sets.code")
        
        if join_analysis["information_loss_ratio"] > 0.3:
            print("  • 建议使用 LEFT JOIN 来保留更多卡牌信息")
        
        if len(join_analysis["unmatched_values_a"]) > 0:
            print("  • 检查 set_translations 中的孤立 setCode，可能需要清理")
        
        if len(join_analysis["unmatched_values_b"]) > 0:
            print("  • 为缺少翻译的 setCode 添加翻译信息")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Card Games 数据库 setCode 分析")
    parser.add_argument(
        "--set-code",
        type=str,
        help="分析特定的 setCode，例如 'OGW'"
    )
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="执行全面的 setCode 关系分析"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="/home/yangchenyu/DeepEye-SQL-Metadata/data/bird/dev/dev_databases/card_games/card_games.sqlite",
        help="数据库文件路径"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db_path):
        print(f"❌ 数据库文件不存在: {args.db_path}")
        return 1
    
    print("🃏 Card Games 数据库 setCode 分析工具")
    print("="*80)
    
    if args.set_code:
        analyze_specific_set_code(args.db_path, args.set_code)
    
    if args.full_analysis:
        full_setcode_analysis(args.db_path)
    
    # 如果没有指定任何参数，默认分析 OGW
    if not args.set_code and not args.full_analysis:
        print("未指定分析参数，默认分析 setCode = 'OGW'")
        analyze_specific_set_code(args.db_path, "OGW")
    
    print("\n" + "="*80)
    print("✅ 分析完成!")
    return 0


if __name__ == "__main__":
    exit(main())
