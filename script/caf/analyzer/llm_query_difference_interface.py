#!/usr/bin/env python3
"""
LLM查询差异分析接口 (LLM Query Difference Analysis Interface)

专为LLM设计的简化接口，专门解决"两个SQL查询为什么结果不同"的问题。
整合新的3个核心函数和改进的报告系统。

主要函数：
1. check_field_uniqueness() - 检查字段是否适合作为JOIN key
2. analyze_query_difference() - 分析两种查询方式的差异
3. generate_query_strategy_report() - 生成决策报告

Author: Generated for DeepEye-SQL-Metadata project
Date: 2025-12
"""

import os
from typing import Dict, Any, List, Optional
import json

try:
    from .query_difference_analyzer import QueryDifferenceAnalyzer, quick_query_difference_analysis
    from .query_strategy_report_generator import QueryStrategyReportGenerator, generate_query_strategy_report_from_analysis
except ImportError:
    # 如果相对导入失败，尝试直接导入
    from query_difference_analyzer import QueryDifferenceAnalyzer, quick_query_difference_analysis
    from query_strategy_report_generator import QueryStrategyReportGenerator, generate_query_strategy_report_from_analysis


def check_field_uniqueness(db_path: str, table: str, column: str) -> Dict[str, Any]:
    """
    检查字段是否适合作为JOIN key
    
    这个函数判断字段是否具有成为主键或唯一键的潜质，
    帮助LLM理解该字段是否适合用于JOIN操作。
    
    Args:
        db_path: 数据库文件路径
        table: 表名
        column: 字段名
    
    Returns:
        字段唯一性分析结果，关键指标：
        - is_unique: 是否完全唯一
        - can_be_primary_key: 能否作为主键  
        - can_be_join_key: 能否作为JOIN的"One"端
        - duplication_rate: 重复数据比例
        
    Example:
        result = check_field_uniqueness("db.sqlite", "cards", "setCode")
        if result['can_be_join_key']:
            print("该字段适合作为JOIN的主导端")
        else:
            print(f"字段重复率: {result['duplication_rate']*100:.1f}%")
    """
    if not os.path.exists(db_path):
        return {"error": f"数据库文件不存在: {db_path}"}
    
    try:
        with QueryDifferenceAnalyzer(db_path) as analyzer:
            return analyzer.check_unique_constraint(table, column)
    except Exception as e:
        return {"error": f"字段唯一性检查失败: {str(e)}"}


def analyze_query_difference(
    db_path: str,
    main_table: str, main_column: str,      # Query 1: SELECT ... FROM main_table WHERE main_column = 'value'
    join_table: str, join_column: str       # Query 2: SELECT ... FROM main_table JOIN join_table WHERE join_column = 'value'
) -> Dict[str, Any]:
    """
    分析两种查询方式的核心差异（核心函数）
    
    专门回答："为什么单表查询和JOIN查询结果不同？"
    
    这个函数执行完整的查询差异分析，重点关注：
    - Fan-out Risk: JOIN后行数是否会变多（数据重复）
    - Filtering Risk: JOIN后是否会丢失数据
    
    Args:
        db_path: 数据库文件路径
        main_table: 主查询表名（通常是包含主要数据的表）
        main_column: 主查询字段名
        join_table: JOIN表名（通常是过滤或补充信息的表）
        join_column: JOIN字段名
    
    Returns:
        完整的查询差异分析结果，包含：
        - left_field_uniqueness: 主表字段唯一性分析
        - right_field_uniqueness: JOIN表字段唯一性分析
        - join_mapping: JOIN映射关系分析（关键：膨胀和丢失风险）
        - data_completeness: 数据完整性比较
        
    Example:
        # 分析cards表单查 vs JOIN set_translations的差异
        result = analyze_query_difference(
            "database.sqlite",
            "cards", "setCode",           # Query 1: SELECT ... FROM cards WHERE setCode = 'OGW'
            "set_translations", "setCode"  # Query 2: SELECT ... FROM cards JOIN set_translations WHERE setCode = 'OGW'
        )
        
        # 检查关键风险
        join_analysis = result['join_mapping']
        print(f"膨胀风险: {join_analysis['fan_out_risk']}")
        print(f"过滤风险: {join_analysis['filtering_risk']}")
        print(f"预计JOIN结果行数: {join_analysis['estimated_result_rows']}")
    """
    if not os.path.exists(db_path):
        return {"error": f"数据库文件不存在: {db_path}"}
    
    try:
        return quick_query_difference_analysis(db_path, main_table, main_column, join_table, join_column)
    except Exception as e:
        return {"error": f"查询差异分析失败: {str(e)}"}


def generate_query_strategy_report(
    analysis_result: Dict[str, Any],
    main_table_name: str = "主表",
    join_table_name: str = "JOIN表",
    save_to_temp: bool = True,
    output_format: str = "all"  # "dict", "json", "markdown", "all"
) -> Dict[str, Any]:
    """
    生成查询策略决策报告
    
    将查询差异分析结果转换为易于理解的决策报告，
    直接回答"我应该使用哪种查询方式？"
    
    Args:
        analysis_result: analyze_query_difference的结果
        main_table_name: 主表显示名称
        join_table_name: JOIN表显示名称
        save_to_temp: 是否保存到temp文件夹
        output_format: 输出格式
        
    Returns:
        结构化报告，包含：
        - executive_diagnosis: 核心差异诊断
        - scenario_simulation: 场景化模拟
        - field_relationship_map: 字段关系图谱
        - actionable_advice: 具体的开发指导
        
    Example:
        analysis = analyze_query_difference(db_path, "cards", "setCode", "set_translations", "setCode")
        report = generate_query_strategy_report(analysis, "cards", "set_translations")
        
        # 获取关键建议
        diagnosis = report['report']['executive_diagnosis']
        print(f"最终结论: {diagnosis['final_conclusion']}")
        
        advice = report['report']['actionable_advice']
        for rec in advice['priority_recommendations']:
            print(f"建议: {rec['strategy']} - {rec['reason']}")
    """
    if "error" in analysis_result:
        return analysis_result
    
    try:
        temp_dir = "/home/yangchenyu/DeepEye-SQL-Metadata/script/caf/analyzer/temp"
        
        result = generate_query_strategy_report_from_analysis(
            analysis_result=analysis_result,
            output_dir=temp_dir if save_to_temp else None,
            table_a_name=main_table_name,
            table_b_name=join_table_name,
            save_to_file=save_to_temp
        )
        
        # 根据输出格式返回相应内容
        if output_format == "dict":
            return result['report']
        elif output_format == "json":
            if save_to_temp and 'saved_files' in result:
                return {"json_content": result['report'], "json_file": result['saved_files']['json']}
            else:
                return {"json_content": result['report']}
        elif output_format == "markdown":
            if save_to_temp and 'saved_files' in result:
                return {"markdown_file": result['saved_files']['markdown']}
            else:
                # 生成临时markdown内容
                generator = QueryStrategyReportGenerator()
                from .query_strategy_report_generator import QueryStrategyReport
                report_obj = QueryStrategyReport(**result['report'])
                return {"markdown_content": generator._format_markdown_report(report_obj)}
        else:  # "all"
            return result
            
    except Exception as e:
        return {"error": f"报告生成失败: {str(e)}"}


def get_database_table_info(db_path: str, table_name: Optional[str] = None) -> Dict[str, Any]:
    """
    获取数据库表信息（简化版）
    
    帮助LLM了解数据库结构，识别潜在的分析目标。
    
    Args:
        db_path: 数据库文件路径
        table_name: 表名（可选）
        
    Returns:
        数据库表信息
    """
    if not os.path.exists(db_path):
        return {"error": f"数据库文件不存在: {db_path}"}
    
    try:
        import sqlite3
        import pandas as pd
        
        conn = sqlite3.connect(db_path)
        
        if table_name:
            # 获取特定表信息
            pragma_query = f"PRAGMA table_info('{table_name}')"
            columns_info = pd.read_sql_query(pragma_query, conn)
            
            if len(columns_info) == 0:
                conn.close()
                return {"error": f"表 '{table_name}' 不存在"}
            
            count_query = f"SELECT COUNT(*) as row_count FROM `{table_name}`"
            row_count = pd.read_sql_query(count_query, conn).iloc[0]['row_count']
            
            conn.close()
            return {
                "table_name": table_name,
                "row_count": int(row_count),
                "columns": [{"name": col['name'], "type": col['type']} for _, col in columns_info.iterrows()]
            }
        
        else:
            # 获取所有表列表
            tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            tables = pd.read_sql_query(tables_query, conn)['name'].tolist()
            conn.close()
            
            return {
                "database_path": db_path,
                "tables": [{"name": table} for table in tables],
                "total_tables": len(tables)
            }
            
    except Exception as e:
        return {"error": f"获取数据库信息失败: {str(e)}"}


def suggest_query_analysis_targets(db_path: str, max_suggestions: int = 5) -> Dict[str, Any]:
    """
    建议潜在的查询差异分析目标
    
    基于字段名相似性，建议可能值得进行查询差异分析的字段对。
    
    Args:
        db_path: 数据库文件路径
        max_suggestions: 最大建议数量
        
    Returns:
        建议的分析目标列表
    """
    if not os.path.exists(db_path):
        return {"error": f"数据库文件不存在: {db_path}"}
    
    try:
        # 获取数据库结构
        schema_info = get_database_table_info(db_path)
        if "error" in schema_info:
            return schema_info
        
        suggestions = []
        
        # 简单的启发式规则：寻找同名字段
        tables = schema_info["tables"]
        for i, table_a in enumerate(tables):
            table_a_info = get_database_table_info(db_path, table_a["name"])
            if "error" in table_a_info:
                continue
                
            for j, table_b in enumerate(tables):
                if i >= j:  # 避免重复比较
                    continue
                    
                table_b_info = get_database_table_info(db_path, table_b["name"])
                if "error" in table_b_info:
                    continue
                
                # 寻找同名字段
                cols_a = {col["name"] for col in table_a_info["columns"]}
                cols_b = {col["name"] for col in table_b_info["columns"]}
                
                common_columns = cols_a & cols_b
                
                for common_col in common_columns:
                    if len(suggestions) >= max_suggestions:
                        break
                        
                    # 过滤掉一些常见但不重要的字段
                    if common_col.lower() in ['id', 'created_at', 'updated_at']:
                        continue
                    
                    suggestions.append({
                        "main_table": table_a["name"],
                        "main_column": common_col,
                        "join_table": table_b["name"], 
                        "join_column": common_col,
                        "reason": f"两表都有字段 '{common_col}'，可能存在关联关系",
                        "analysis_query_1": f"SELECT ... FROM {table_a['name']} WHERE {common_col} = 'value'",
                        "analysis_query_2": f"SELECT ... FROM {table_a['name']} JOIN {table_b['name']} ON ... WHERE {common_col} = 'value'"
                    })
        
        return {
            "database_path": db_path,
            "suggestions_count": len(suggestions),
            "suggestions": suggestions
        }
        
    except Exception as e:
        return {"error": f"建议生成失败: {str(e)}"}


# 便利函数：一站式查询差异分析工作流
def complete_query_difference_workflow(
    db_path: str,
    main_table: str, main_column: str,
    join_table: str, join_column: str,
    main_table_display_name: Optional[str] = None,
    join_table_display_name: Optional[str] = None,
    save_report: bool = True
) -> Dict[str, Any]:
    """
    完整的查询差异分析工作流
    
    一次调用完成所有分析步骤：
    1. 字段唯一性检查
    2. 查询差异分析
    3. 策略报告生成
    
    最适合LLM快速获取完整分析结果。
    
    Args:
        db_path: 数据库文件路径
        main_table: 主查询表名
        main_column: 主查询字段名
        join_table: JOIN表名  
        join_column: JOIN字段名
        main_table_display_name: 主表显示名称
        join_table_display_name: JOIN表显示名称
        save_report: 是否保存报告到temp文件夹
        
    Returns:
        完整分析结果，包含原始数据和结构化报告
        
    Example:
        # 一次性完成cards vs set_translations的查询差异分析
        result = complete_query_difference_workflow(
            "database.sqlite",
            "cards", "setCode",
            "set_translations", "setCode",
            "卡牌表", "系列翻译表"
        )
        
        # 快速查看结论
        print(result['report']['executive_diagnosis']['final_conclusion'])
        
        # 查看保存的报告文件
        if result.get('saved_files'):
            print(f"报告已保存: {result['saved_files']['markdown']}")
    """
    # 执行查询差异分析
    analysis_result = analyze_query_difference(db_path, main_table, main_column, join_table, join_column)
    
    if "error" in analysis_result:
        return analysis_result
    
    # 生成报告
    display_name_a = main_table_display_name or main_table
    display_name_b = join_table_display_name or join_table
    
    report_result = generate_query_strategy_report(
        analysis_result,
        main_table_name=display_name_a,
        join_table_name=display_name_b,
        save_to_temp=save_report,
        output_format="all"
    )
    
    if "error" in report_result:
        return {
            "analysis_data": analysis_result,
            "report_error": report_result["error"]
        }
    
    return {
        "analysis_data": analysis_result,
        "report": report_result["report"],
        "saved_files": report_result.get("saved_files", {}),
        "summary": {
            "main_field": f"{main_table}.{main_column}",
            "join_field": f"{join_table}.{join_column}", 
            "conclusion": report_result["report"]["executive_diagnosis"]["final_conclusion"],
            "key_risks": {
                "fan_out": analysis_result["join_mapping"]["fan_out_risk"],
                "filtering": analysis_result["join_mapping"]["filtering_risk"]
            }
        }
    }
