#!/usr/bin/env python3
"""
新版查询差异分析器测试脚本

测试改进后的3个核心函数和新报告系统。
使用Card Games数据库验证"Query 1 vs Query 2"的分析能力。

Author: Generated for DeepEye-SQL-Metadata project
Date: 2025-12
"""

import os
import sys
from pathlib import Path

# 添加路径
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from llm_query_difference_interface import (
    check_field_uniqueness,
    analyze_query_difference, 
    generate_query_strategy_report,
    get_database_table_info,
    suggest_query_analysis_targets,
    complete_query_difference_workflow
)


def test_new_analyzer_system():
    """
    测试新版分析器系统
    
    重点验证用户关心的核心场景：
    1. SELECT * FROM cards WHERE setCode = 'OGW'
    2. SELECT * FROM cards JOIN set_translations ON ... WHERE setCode = 'OGW'
    """
    
    # Card Games数据库路径
    db_path = "/home/yangchenyu/DeepEye-SQL-Metadata/dataset/bird/databases/dev_databases/card_games/card_games.sqlite"
    
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        return
    
    print("🚀 测试新版查询差异分析器")
    print("=" * 60)
    
    # 测试1: 数据库基本信息
    print("\n📋 测试1: 获取数据库基本信息")
    db_info = get_database_table_info(db_path)
    if "error" in db_info:
        print(f"❌ 失败: {db_info['error']}")
        return
        
    print(f"✅ 发现 {db_info['total_tables']} 个表:")
    for table in db_info['tables']:
        print(f"   - {table['name']}")
    
    # 测试2: 字段唯一性检查
    print("\n🔍 测试2: 字段唯一性检查")
    
    # 检查 cards.setCode 是否适合作为JOIN key
    print("\n检查 cards.setCode:")
    cards_uniqueness = check_field_uniqueness(db_path, "cards", "setCode")
    if "error" in cards_uniqueness:
        print(f"❌ 失败: {cards_uniqueness['error']}")
    else:
        print(f"  ✅ 是否适合作为JOIN key: {'是' if cards_uniqueness['can_be_join_key'] else '否'}")
        print(f"  ✅ 是否完全唯一: {'是' if cards_uniqueness['is_unique'] else '否'}")
        print(f"  ✅ 重复率: {cards_uniqueness['duplication_rate']*100:.2f}%")
        print(f"  ✅ 唯一值数量: {cards_uniqueness['unique_values']}")
    
    # 检查 set_translations.setCode
    print("\n检查 set_translations.setCode:")
    translations_uniqueness = check_field_uniqueness(db_path, "set_translations", "setCode")
    if "error" in translations_uniqueness:
        print(f"❌ 失败: {translations_uniqueness['error']}")
    else:
        print(f"  ✅ 是否适合作为JOIN key: {'是' if translations_uniqueness['can_be_join_key'] else '否'}")
        print(f"  ✅ 是否完全唯一: {'是' if translations_uniqueness['is_unique'] else '否'}")
        print(f"  ✅ 重复率: {translations_uniqueness['duplication_rate']*100:.2f}%")
        print(f"  ✅ 唯一值数量: {translations_uniqueness['unique_values']}")
    
    # 测试3: 核心查询差异分析  
    print("\n🔥 测试3: 查询差异分析 (cards.setCode vs set_translations.setCode)")
    
    analysis_result = analyze_query_difference(
        db_path,
        "cards", "setCode",           # Query 1: SELECT ... FROM cards WHERE setCode = 'OGW'
        "set_translations", "setCode"  # Query 2: SELECT ... FROM cards JOIN set_translations WHERE setCode = 'OGW'
    )
    
    if "error" in analysis_result:
        print(f"❌ 分析失败: {analysis_result['error']}")
        return
    
    print("✅ 查询差异分析完成！关键发现:")
    
    # 显示关键指标
    join_mapping = analysis_result['join_mapping']
    print(f"  🔍 最大扇出: {join_mapping['max_fan_out']}")
    print(f"  📊 匹配率: {join_mapping['match_ratio']*100:.1f}%")
    print(f"  📏 关系类型: {join_mapping['mapping_type']}")
    print(f"  ⚠️  膨胀风险: {join_mapping['fan_out_risk']}")
    print(f"  ⚠️  过滤风险: {join_mapping['filtering_risk']}")
    print(f"  🎯 预计JOIN结果行数: {join_mapping['estimated_result_rows']}")
    
    # 数据完整性
    completeness = analysis_result['data_completeness']
    print(f"  📋 数据完整性: {completeness['completeness_ratio']*100:.1f}%")
    print(f"  🔤 缺失值数量: {completeness['missing_in_b_count']}")
    
    # 测试4: 生成结构化报告
    print("\n📊 测试4: 生成查询策略报告")
    
    report_result = generate_query_strategy_report(
        analysis_result,
        main_table_name="卡牌表(cards)", 
        join_table_name="系列翻译表(set_translations)",
        save_to_temp=True,
        output_format="all"
    )
    
    if "error" in report_result:
        print(f"❌ 报告生成失败: {report_result['error']}")
        return
    
    print("✅ 结构化报告生成成功！")
    
    # 显示核心诊断结果
    diagnosis = report_result['report']['executive_diagnosis']
    print(f"  🏆 最终结论: {diagnosis['final_conclusion']}")
    print(f"  💡 快速总结: {diagnosis['quick_summary']}")
    
    # 显示建议
    advice = report_result['report']['actionable_advice']
    if advice['priority_recommendations']:
        rec = advice['priority_recommendations'][0]
        print(f"  🎯 主要建议: {rec['strategy']}")
        print(f"  📝 原因: {rec['reason']}")
    
    # 显示保存的文件
    if 'saved_files' in report_result:
        files = report_result['saved_files']
        print(f"\n📁 报告已保存到temp文件夹:")
        print(f"  - JSON格式: {os.path.basename(files.get('json', ''))}")
        print(f"  - Markdown格式: {os.path.basename(files.get('markdown', ''))}")
        print(f"  - 文本摘要: {os.path.basename(files.get('text_summary', ''))}")
    
    # 测试5: 完整工作流
    print("\n🎬 测试5: 完整工作流")
    
    workflow_result = complete_query_difference_workflow(
        db_path,
        "cards", "setCode",
        "set_translations", "setCode",
        "卡牌表", "系列翻译表",
        save_report=True
    )
    
    if "error" in workflow_result:
        print(f"❌ 工作流失败: {workflow_result['error']}")
    else:
        print("✅ 完整工作流执行成功！")
        summary = workflow_result['summary'] 
        print(f"  🎯 分析对象: {summary['main_field']} vs {summary['join_field']}")
        print(f"  🏆 结论: {summary['conclusion']}")
        print(f"  ⚠️  关键风险: 膨胀({summary['key_risks']['fan_out']}) / 过滤({summary['key_risks']['filtering']})")
    
    print("\n" + "=" * 60)
    print("🎉 测试完成！新版分析器运行正常")
    

if __name__ == "__main__":
    test_new_analyzer_system()
