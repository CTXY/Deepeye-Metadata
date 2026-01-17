#!/usr/bin/env python3
"""
查询差异分析器 (Query Difference Analyzer)

专门用于回答"为什么两个SQL查询结果不同"的问题。
基于用户提供的核心需求，实现3个精炼的函数：

1. check_unique_constraint() - 判断字段是否具有唯一性
2. analyze_join_mapping() - 分析JOIN的膨胀和丢失风险  
3. compare_data_completeness() - 比较数据完整性

重点关注：
- Fan-out Risk (膨胀风险): JOIN后行数变多
- Filtering Risk (过滤风险): JOIN后某些数据查不到

Author: Generated for DeepEye-SQL-Metadata project
Date: 2025-12
"""

import sqlite3
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class UniqueConstraintResult:
    """唯一性约束检查结果"""
    is_unique: bool  # 是否完全唯一
    null_count: int  # 空值数量
    duplication_rate: float  # 重复数据比例
    total_rows: int
    unique_values: int
    can_be_primary_key: bool  # 能否作为主键
    can_be_join_key: bool  # 能否作为JOIN的"One"端


@dataclass  
class JoinMappingResult:
    """JOIN映射分析结果"""
    max_fan_out: int  # 最大扇出：Left表一个值在Right表最多对应几行
    match_ratio: float  # 覆盖率：Left表中的值在Right表中的覆盖百分比
    mapping_type: str  # 关系类型: "1:1", "1:N", "N:1", "N:N", "no_match"
    fan_out_risk: str  # 膨胀风险等级: "无风险", "中等风险", "高风险"
    filtering_risk: str  # 过滤风险等级: "无风险", "中等风险", "高风险" 
    estimated_result_rows: int  # 预计INNER JOIN结果行数


@dataclass
class DataCompletenessResult:
    """数据完整性比较结果"""
    missing_in_b_samples: List[str]  # A有但B没有的值样本
    missing_in_b_count: int  # A有但B没有的值总数
    null_in_b_count: int  # B表中该字段为Null的行数
    completeness_ratio: float  # B表对A表的完整性比例
    data_quality_risk: str  # 数据质量风险等级


class QueryDifferenceAnalyzer:
    """查询差异分析器"""
    
    def __init__(self, db_path: str):
        """
        初始化分析器
        
        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = db_path
        self.connection = None
    
    def __enter__(self):
        self.connection = sqlite3.connect(self.db_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
    
    def check_unique_constraint(self, table: str, column: str) -> Dict[str, Any]:
        """
        判断该字段是否具有成为主键（PK）或唯一键（UK）的物理潜质
        
        目的：判断该字段是否具有成为主键（PK）或唯一键（UK）的物理潜质
        
        分析逻辑：
        - 计算 Count(*) 和 Count(DISTINCT column)
        - 如果两者相等，则为 Unique
        - 如果不等，计算重复率
        
        Args:
            table: 表名
            column: 字段名
            
        Returns:
            UniqueConstraintResult: 包含唯一性分析结果
            
        关键指标：
            - is_unique: 是否完全唯一
            - null_count: 空值数量（主键不能有空值）
            - duplication_rate: 重复数据的比例
            - can_be_primary_key: 能否作为JOIN的"One"端
        """
        try:
            # 基础统计查询
            stats_query = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT("{column}") as non_null_count,
                COUNT(DISTINCT "{column}") as unique_values
            FROM "{table}"
            """
            
            stats = pd.read_sql_query(stats_query, self.connection).iloc[0]
            total_rows = int(stats['total_rows'])
            non_null_count = int(stats['non_null_count']) 
            unique_values = int(stats['unique_values'])
            null_count = total_rows - non_null_count
            
            # 判断唯一性
            is_unique = (unique_values == non_null_count) and (null_count == 0)
            
            # 计算重复率
            if non_null_count > 0:
                duplication_rate = 1.0 - (unique_values / non_null_count)
            else:
                duplication_rate = 0.0
            
            # 判断能否作为主键/JOIN键
            can_be_primary_key = is_unique and (null_count == 0)
            can_be_join_key = (duplication_rate < 0.1)  # 重复率低于10%可考虑作为JOIN的主导端
            
            result = UniqueConstraintResult(
                is_unique=is_unique,
                null_count=null_count,
                duplication_rate=round(duplication_rate, 4),
                total_rows=total_rows,
                unique_values=unique_values,
                can_be_primary_key=can_be_primary_key,
                can_be_join_key=can_be_join_key
            )
            
            return asdict(result)
            
        except Exception as e:
            return {
                "error": f"唯一性约束检查失败 {table}.{column}: {str(e)}",
                "table": table,
                "column": column
            }
    
    def analyze_join_mapping(self, left_table: str, left_col: str, 
                           right_table: str, right_col: str) -> Dict[str, Any]:
        """
        这是最关键的函数。模拟 LEFT JOIN 和 INNER JOIN 的差异，
        直接回答"Query 1 和 Query 2 为什么不同"
        
        分析逻辑：
        - Fan-out Analysis (膨胀分析): 对于Left表中的一个值，Right表中平均有几行？最大有几行？
        - Orphan Analysis (孤儿/丢失分析): Left表中有多少比例的值，在Right表中根本找不到？
        
        Args:
            left_table: Left表名（通常是主要查询表，如cards）
            left_col: Left表字段名
            right_table: Right表名（通常是过滤表，如set_translations） 
            right_col: Right表字段名
            
        Returns:
            JoinMappingResult: 包含JOIN映射分析结果
            
        关键指标：
            - max_fan_out: Left表一个值最多在Right表对应几行（如果>1说明有膨胀风险）
            - match_ratio: Left表中的值在Right表中的覆盖率（如果<100%说明有过滤风险）
            - mapping_type: 推断关系类型 (1:1, 1:N, N:1, N:N)
        """
        try:
            # 1. 膨胀风险分析 (Fan-out Analysis)
            fan_out_query = f"""
            SELECT 
                l."{left_col}" as left_value,
                COUNT(r."{right_col}") as right_count
            FROM "{left_table}" l
            LEFT JOIN "{right_table}" r ON l."{left_col}" = r."{right_col}"
            WHERE l."{left_col}" IS NOT NULL
            GROUP BY l."{left_col}"
            """
            
            fan_out_df = pd.read_sql_query(fan_out_query, self.connection)
            
            if len(fan_out_df) == 0:
                return asdict(JoinMappingResult(
                    max_fan_out=0,
                    match_ratio=0.0,
                    mapping_type="no_match",
                    fan_out_risk="无风险",
                    filtering_risk="高风险 - 完全无匹配",
                    estimated_result_rows=0
                ))
            
            max_fan_out = int(fan_out_df['right_count'].max())
            avg_fan_out = float(fan_out_df['right_count'].mean())
            
            # 2. 过滤风险分析 (Filtering Risk Analysis) 
            # 计算Left表中有多少值在Right表中能找到
            matched_count = len(fan_out_df[fan_out_df['right_count'] > 0])
            total_left_values = len(fan_out_df)
            match_ratio = matched_count / total_left_values if total_left_values > 0 else 0.0
            
            # 3. 预计JOIN结果行数
            estimated_result_rows = int(fan_out_df['right_count'].sum())
            
            # 4. 推断映射类型
            if max_fan_out == 0:
                mapping_type = "no_match"
            elif max_fan_out == 1 and avg_fan_out <= 1.0:
                mapping_type = "1:1"  
            elif max_fan_out > 1:
                mapping_type = "1:N"  # Left表的值对应Right表多行
            else:
                mapping_type = "complex"
            
            # 5. 风险等级评估
            # 膨胀风险
            if max_fan_out <= 1:
                fan_out_risk = "无风险"
            elif max_fan_out <= 5:
                fan_out_risk = "中等风险"
            else:
                fan_out_risk = "高风险"
            
            # 过滤风险  
            if match_ratio >= 0.95:
                filtering_risk = "无风险"
            elif match_ratio >= 0.80:
                filtering_risk = "中等风险"
            else:
                filtering_risk = "高风险"
            
            result = JoinMappingResult(
                max_fan_out=max_fan_out,
                match_ratio=round(match_ratio, 4),
                mapping_type=mapping_type,
                fan_out_risk=fan_out_risk,
                filtering_risk=filtering_risk,
                estimated_result_rows=estimated_result_rows
            )
            
            return asdict(result)
            
        except Exception as e:
            return {
                "error": f"JOIN映射分析失败 {left_table}.{left_col} -> {right_table}.{right_col}: {str(e)}"
            }
    
    def compare_data_completeness(self, table_a: str, col_a: str, 
                                table_b: str, col_b: str) -> Dict[str, Any]:
        """
        检查信息完整性（针对 WHERE 子句的有效性）
        
        目的：检查信息完整性（针对WHERE子句的有效性）
        
        分析逻辑：比如setCode在A表不为空，但在B表可能是NULL
        
        Args:
            table_a: 表A名称
            col_a: 表A字段名
            table_b: 表B名称
            col_b: 表B字段名
            
        Returns:
            DataCompletenessResult: 数据完整性比较结果
            
        关键指标：
            - missing_in_b_samples: A有但B没有的值（会导致Inner Join过滤掉A的行）
            - null_in_b_count: B表中该字段为Null的行数
        """
        try:
            # 1. 找出A表有但B表没有的值
            missing_query = f"""
            SELECT DISTINCT a."{col_a}" as missing_value
            FROM "{table_a}" a
            LEFT JOIN "{table_b}" b ON a."{col_a}" = b."{col_b}"
            WHERE a."{col_a}" IS NOT NULL 
              AND b."{col_b}" IS NULL
            LIMIT 20
            """
            
            missing_df = pd.read_sql_query(missing_query, self.connection)
            missing_in_b_samples = missing_df['missing_value'].astype(str).tolist()
            
            # 2. 计算A表有但B表没有的值的总数
            missing_count_query = f"""
            SELECT COUNT(DISTINCT a."{col_a}") as missing_count
            FROM "{table_a}" a
            LEFT JOIN "{table_b}" b ON a."{col_a}" = b."{col_b}"  
            WHERE a."{col_a}" IS NOT NULL
              AND b."{col_b}" IS NULL
            """
            
            missing_count = int(pd.read_sql_query(missing_count_query, self.connection).iloc[0]['missing_count'])
            
            # 3. 计算B表中字段为NULL的行数
            null_count_query = f"""
            SELECT COUNT(*) as null_count
            FROM "{table_b}"
            WHERE "{col_b}" IS NULL
            """
            
            null_in_b_count = int(pd.read_sql_query(null_count_query, self.connection).iloc[0]['null_count'])
            
            # 4. 计算完整性比例
            total_a_values_query = f"""
            SELECT COUNT(DISTINCT "{col_a}") as total_count
            FROM "{table_a}"
            WHERE "{col_a}" IS NOT NULL
            """
            
            total_a_values = int(pd.read_sql_query(total_a_values_query, self.connection).iloc[0]['total_count'])
            
            if total_a_values > 0:
                completeness_ratio = 1.0 - (missing_count / total_a_values)
            else:
                completeness_ratio = 1.0
            
            # 5. 数据质量风险评估
            if completeness_ratio >= 0.95 and null_in_b_count == 0:
                data_quality_risk = "低风险"
            elif completeness_ratio >= 0.80 and null_in_b_count <= total_a_values * 0.05:
                data_quality_risk = "中等风险" 
            else:
                data_quality_risk = "高风险"
            
            result = DataCompletenessResult(
                missing_in_b_samples=missing_in_b_samples,
                missing_in_b_count=missing_count,
                null_in_b_count=null_in_b_count,
                completeness_ratio=round(completeness_ratio, 4),
                data_quality_risk=data_quality_risk
            )
            
            return asdict(result)
            
        except Exception as e:
            return {
                "error": f"数据完整性比较失败 {table_a}.{col_a} vs {table_b}.{col_b}: {str(e)}"
            }


def quick_query_difference_analysis(
    db_path: str,
    left_table: str, left_col: str,  # 主表：通常是Query 1中的单表
    right_table: str, right_col: str  # 过滤表：通常是Query 2中的JOIN表
) -> Dict[str, Any]:
    """
    快速执行完整的查询差异分析
    
    一次性调用所有3个核心函数，专门回答：
    "为什么 SELECT ... FROM left_table WHERE left_col = 'value'"
    "和 SELECT ... FROM left_table JOIN right_table ... WHERE right_col = 'value'"
    "的结果不同？"
    
    Args:
        db_path: 数据库路径
        left_table: 主查询表（如cards）
        left_col: 主查询字段（如cards.setCode）
        right_table: JOIN表（如set_translations）
        right_col: JOIN字段（如set_translations.setCode）
        
    Returns:
        完整的查询差异分析结果
    """
    with QueryDifferenceAnalyzer(db_path) as analyzer:
        return {
            "left_field_uniqueness": analyzer.check_unique_constraint(left_table, left_col),
            "right_field_uniqueness": analyzer.check_unique_constraint(right_table, right_col),
            "join_mapping": analyzer.analyze_join_mapping(left_table, left_col, right_table, right_col),
            "data_completeness": analyzer.compare_data_completeness(left_table, left_col, right_table, right_col),
            "analysis_metadata": {
                "analyzer_type": "query_difference_analysis",
                "version": "2.0",
                "database_path": db_path,
                "primary_query_field": f"{left_table}.{left_col}",
                "join_filter_field": f"{right_table}.{right_col}"
            }
        }








