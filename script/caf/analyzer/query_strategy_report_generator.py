#!/usr/bin/env python3
"""
查询策略评估报告生成器 (Query Strategy Report Generator)

专门为LLM生成针对"Query 1 vs Query 2"疑惑的决策报告。
基于新的3个核心函数结果，生成易理解的分析报告。

Author: Generated for DeepEye-SQL-Metadata project  
Date: 2025-12
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime


@dataclass
class QueryStrategyReport:
    """查询策略评估报告"""
    # 1. 核心差异诊断
    executive_diagnosis: Dict[str, Any]
    
    # 2. 场景化模拟 
    scenario_simulation: Dict[str, Any]
    
    # 3. 字段关系深度图谱
    field_relationship_map: Dict[str, Any]
    
    # 4. 开发指导
    actionable_advice: Dict[str, Any]
    
    # 元数据
    metadata: Dict[str, Any]


class QueryStrategyReportGenerator:
    """查询策略报告生成器"""
    
    def __init__(self, output_dir: str = "/home/yangchenyu/DeepEye-SQL-Metadata/script/caf/analyzer/temp"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_query_strategy_report(
        self,
        analysis_result: Dict[str, Any],
        table_a_name: str = "主表",
        table_b_name: str = "JOIN表"
    ) -> QueryStrategyReport:
        """
        根据查询差异分析结果生成结构化报告
        
        Args:
            analysis_result: quick_query_difference_analysis的结果
            table_a_name: 主表名称（用于报告显示）
            table_b_name: JOIN表名称（用于报告显示）
            
        Returns:
            QueryStrategyReport: 结构化的查询策略报告
        """
        
        # 提取分析数据
        left_uniqueness = analysis_result.get('left_field_uniqueness', {})
        right_uniqueness = analysis_result.get('right_field_uniqueness', {})
        join_mapping = analysis_result.get('join_mapping', {})
        data_completeness = analysis_result.get('data_completeness', {})
        metadata = analysis_result.get('analysis_metadata', {})
        
        # 1. 生成核心差异诊断
        executive_diagnosis = self._generate_executive_diagnosis(
            left_uniqueness, right_uniqueness, join_mapping, data_completeness,
            table_a_name, table_b_name
        )
        
        # 2. 生成场景化模拟
        scenario_simulation = self._generate_scenario_simulation(
            join_mapping, data_completeness, metadata, table_a_name, table_b_name
        )
        
        # 3. 生成字段关系图谱
        field_relationship_map = self._generate_field_relationship_map(
            left_uniqueness, right_uniqueness, join_mapping, metadata
        )
        
        # 4. 生成开发指导
        actionable_advice = self._generate_actionable_advice(
            executive_diagnosis, join_mapping, data_completeness, 
            table_a_name, table_b_name
        )
        
        # 组装报告
        report = QueryStrategyReport(
            executive_diagnosis=executive_diagnosis,
            scenario_simulation=scenario_simulation,
            field_relationship_map=field_relationship_map,
            actionable_advice=actionable_advice,
            metadata={
                **metadata,
                "report_generated_at": datetime.now().isoformat(),
                "report_version": "2.0",
                "table_a_name": table_a_name,
                "table_b_name": table_b_name
            }
        )
        
        return report
    
    def _generate_executive_diagnosis(
        self, 
        left_uniqueness: Dict[str, Any],
        right_uniqueness: Dict[str, Any], 
        join_mapping: Dict[str, Any],
        data_completeness: Dict[str, Any],
        table_a: str,
        table_b: str
    ) -> Dict[str, Any]:
        """生成核心差异诊断"""
        
        # 分析数据膨胀风险
        max_fan_out = join_mapping.get('max_fan_out', 0)
        fan_out_risk = join_mapping.get('fan_out_risk', '未知')
        
        if max_fan_out <= 1:
            fanout_status = "✅ 低风险"
            fanout_impact = f"{table_b}中每个值最多对应1条记录，不会导致数据重复"
        elif max_fan_out <= 5:
            fanout_status = "⚠️ 中等风险"
            fanout_impact = f"{table_b}中每个值最多对应{max_fan_out}条记录，JOIN可能导致数据重复{max_fan_out}倍"
        else:
            fanout_status = "🚨 高风险"
            fanout_impact = f"{table_b}中每个值最多对应{max_fan_out}条记录，JOIN会导致严重的数据膨胀"
        
        # 分析数据丢失风险
        match_ratio = join_mapping.get('match_ratio', 0)
        missing_count = data_completeness.get('missing_in_b_count', 0)
        
        if match_ratio >= 0.95:
            filtering_status = "✅ 低风险"
            filtering_impact = f"几乎所有{table_a}中的值都能在{table_b}找到，INNER JOIN不会丢失数据"
        elif match_ratio >= 0.80:
            filtering_status = "⚠️ 中等风险"
            filtering_impact = f"{table_a}中有{(1-match_ratio)*100:.1f}%的值在{table_b}中找不到，INNER JOIN会丢失部分数据"
        else:
            filtering_status = "🚨 高风险"
            filtering_impact = f"{table_a}中有{(1-match_ratio)*100:.1f}%的值在{table_b}中找不到，INNER JOIN会丢失大量数据"
        
        # 分析字段唯一性
        left_unique = left_uniqueness.get('is_unique', False)
        right_unique = right_uniqueness.get('is_unique', False)
        
        if left_unique and right_unique:
            uniqueness_desc = f"两个字段都是唯一的，这是理想的1:1关系"
        elif left_unique:
            uniqueness_desc = f"{table_a}字段唯一，{table_b}字段非唯一，形成1:N关系"
        elif right_unique:
            uniqueness_desc = f"{table_b}字段唯一，{table_a}字段非唯一，形成N:1关系"
        else:
            uniqueness_desc = f"两个字段都非唯一，可能形成复杂的N:N关系"
        
        # 最终结论
        has_major_difference = (max_fan_out > 1) or (match_ratio < 0.9)
        final_conclusion = "存在重大区别" if has_major_difference else "结果基本一致"
        
        return {
            "final_conclusion": final_conclusion,
            "risk_matrix": [
                {
                    "风险维度": "数据膨胀 (Fan-out)",
                    "检测结果": fanout_status + f" (最大扇出: {max_fan_out})",
                    "影响解释": fanout_impact
                },
                {
                    "风险维度": "数据丢失 (Filtering)", 
                    "检测结果": filtering_status + f" (匹配率: {match_ratio*100:.1f}%)",
                    "影响解释": filtering_impact
                },
                {
                    "风险维度": "字段唯一性检查",
                    "检测结果": f"{table_a}: {'唯一' if left_unique else '非唯一'}, {table_b}: {'唯一' if right_unique else '非唯一'}",
                    "影响解释": uniqueness_desc
                }
            ],
            "quick_summary": f"预计JOIN查询结果行数: {join_mapping.get('estimated_result_rows', 0)}，存在数据{'膨胀' if max_fan_out > 1 else '过滤'}风险"
        }
    
    def _generate_scenario_simulation(
        self, 
        join_mapping: Dict[str, Any],
        data_completeness: Dict[str, Any], 
        metadata: Dict[str, Any],
        table_a: str,
        table_b: str
    ) -> Dict[str, Any]:
        """生成场景化模拟"""
        
        primary_field = metadata.get('primary_query_field', 'table_a.field')
        join_field = metadata.get('join_filter_field', 'table_b.field')
        
        # 假设具体值进行模拟
        sample_value = "'OGW'"  # 使用用户提到的示例值
        estimated_rows = join_mapping.get('estimated_result_rows', 0)
        max_fan_out = join_mapping.get('max_fan_out', 1)
        
        # Query 1 模拟
        query1_behavior = f"仅查询{table_a}表"
        query1_prediction = f"返回{table_a}表中所有{sample_value}相关的记录"
        query1_accuracy = "✅ 准确反映主表数据"
        
        # Query 2 模拟 
        query2_behavior = f"先将{table_a}与{table_b}连接，再过滤"
        if max_fan_out > 1:
            query2_prediction = f"可能返回{estimated_rows}行（每条主表记录重复{max_fan_out}次）"
            query2_issue = f"每一条{table_a}的记录都重复出现了{max_fan_out}次"
        elif join_mapping.get('match_ratio', 0) < 1.0:
            missing_samples = data_completeness.get('missing_in_b_samples', [])
            query2_prediction = f"返回更少的结果，可能丢失数据"
            query2_issue = f"如果查询的值(如{sample_value})在{table_b}中不存在，结果将为0行"
        else:
            query2_prediction = f"返回结果与Query 1基本一致"
            query2_issue = "无明显问题"
        
        return {
            "specific_value_example": sample_value,
            "query_comparison": {
                "query_1_single_table": {
                    "行为描述": query1_behavior,
                    "结果预测": query1_prediction,
                    "准确性": query1_accuracy
                },
                "query_2_join": {
                    "行为描述": query2_behavior,
                    "结果预测": query2_prediction,
                    "潜在问题": query2_issue
                }
            },
            "data_examples": {
                "missing_in_join_table": data_completeness.get('missing_in_b_samples', [])[:5],
                "risk_explanation": f"以上值存在于{table_a}中，但在{table_b}中找不到，使用JOIN查询会丢失相关数据"
            }
        }
    
    def _generate_field_relationship_map(
        self,
        left_uniqueness: Dict[str, Any],
        right_uniqueness: Dict[str, Any],
        join_mapping: Dict[str, Any], 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成字段关系深度图谱"""
        
        mapping_type = join_mapping.get('mapping_type', 'unknown')
        max_fan_out = join_mapping.get('max_fan_out', 0)
        
        # 映射关系描述
        if mapping_type == "1:1":
            relationship_desc = "严格的一对一关系"
            join_safety = "安全，不会产生数据重复"
        elif mapping_type == "1:N":
            relationship_desc = f"一对多关系（1:{max_fan_out}）"
            join_safety = f"不安全，会产生{max_fan_out}倍数据重复"
        elif mapping_type == "no_match":
            relationship_desc = "完全无匹配关系"
            join_safety = "JOIN查询将返回空结果"
        else:
            relationship_desc = "复杂的多对多关系"
            join_safety = "需要谨慎处理，可能产生意外结果"
        
        # 连接建议
        left_can_be_key = left_uniqueness.get('can_be_join_key', False)
        right_can_be_key = right_uniqueness.get('can_be_join_key', False)
        
        if left_can_be_key and right_can_be_key:
            join_recommendation = "✅ 推荐直接使用字段JOIN"
        elif not left_can_be_key and not right_can_be_key:
            join_recommendation = "❌ 不建议直接JOIN，两个字段都不适合作为连接键"
        else:
            join_recommendation = "⚠️ 谨慎使用JOIN，需要额外的过滤条件"
        
        return {
            "mapping_relationship": mapping_type,
            "relationship_description": relationship_desc,
            "join_safety_assessment": join_safety,
            "connection_advice": {
                "recommendation": join_recommendation,
                "left_field_suitability": "适合" if left_can_be_key else "不适合",
                "right_field_suitability": "适合" if right_can_be_key else "不适合",
                "additional_requirements": self._get_join_requirements(join_mapping)
            }
        }
    
    def _get_join_requirements(self, join_mapping: Dict[str, Any]) -> List[str]:
        """获取JOIN的额外要求"""
        requirements = []
        
        max_fan_out = join_mapping.get('max_fan_out', 0)
        match_ratio = join_mapping.get('match_ratio', 0)
        
        if max_fan_out > 1:
            requirements.append(f"需要额外条件确保1:1关系，避免{max_fan_out}倍数据重复")
        
        if match_ratio < 0.9:
            requirements.append(f"需要检查数据完整性，{(1-match_ratio)*100:.1f}%的数据可能丢失")
        
        return requirements or ["当前JOIN条件已足够"]
    
    def _generate_actionable_advice(
        self,
        executive_diagnosis: Dict[str, Any],
        join_mapping: Dict[str, Any], 
        data_completeness: Dict[str, Any],
        table_a: str,
        table_b: str
    ) -> Dict[str, Any]:
        """生成开发指导"""
        
        max_fan_out = join_mapping.get('max_fan_out', 0)
        match_ratio = join_mapping.get('match_ratio', 0)
        
        recommendations = []
        
        # 基本建议
        if max_fan_out <= 1 and match_ratio >= 0.9:
            recommendations.append({
                "priority": "推荐",
                "strategy": f"两种查询方式都可以使用",
                "reason": "数据质量良好，无重大风险"
            })
        elif max_fan_out > 1:
            recommendations.append({
                "priority": "高优先级",
                "strategy": f"建议使用单表查询（Query 1）",
                "reason": f"JOIN查询会导致数据重复{max_fan_out}倍"
            })
        elif match_ratio < 0.8:
            recommendations.append({
                "priority": "高优先级", 
                "strategy": f"建议使用单表查询（Query 1）",
                "reason": f"JOIN查询会丢失{(1-match_ratio)*100:.1f}%的数据"
            })
        
        # 如果必须使用JOIN的建议
        if max_fan_out > 1:
            join_fix = f"如果必须使用JOIN，需要添加额外条件确保唯一性"
        else:
            join_fix = f"JOIN查询相对安全，但要注意数据完整性"
        
        # SQL示例
        primary_field = f"{table_a}.field"
        join_field = f"{table_b}.field"
        
        sql_examples = {
            "recommended_single_table": f"SELECT columns FROM {table_a} WHERE field = 'value'",
            "cautious_join": f"SELECT columns FROM {table_a} a LEFT JOIN {table_b} b ON a.field = b.field WHERE ...",
            "join_with_conditions": f"SELECT DISTINCT columns FROM {table_a} a JOIN {table_b} b ON a.field = b.field WHERE ..."
        }
        
        return {
            "priority_recommendations": recommendations,
            "join_usage_guidance": join_fix,
            "sql_examples": sql_examples,
            "performance_considerations": [
                "单表查询性能更好" if max_fan_out > 1 else "JOIN查询性能可接受",
                f"预计JOIN结果行数: {join_mapping.get('estimated_result_rows', 0)}"
            ],
            "data_quality_warnings": self._get_data_quality_warnings(data_completeness)
        }
    
    def _get_data_quality_warnings(self, data_completeness: Dict[str, Any]) -> List[str]:
        """获取数据质量警告"""
        warnings = []
        
        missing_count = data_completeness.get('missing_in_b_count', 0)
        null_count = data_completeness.get('null_in_b_count', 0)
        
        if missing_count > 0:
            warnings.append(f"发现{missing_count}个值在JOIN表中缺失")
        
        if null_count > 0:
            warnings.append(f"JOIN表中有{null_count}行空值")
        
        return warnings or ["数据质量良好"]
    
    def export_report(self, report: QueryStrategyReport, file_prefix: str = "query_strategy_report") -> Dict[str, str]:
        """
        导出报告到多种格式
        
        Args:
            report: 查询策略报告
            file_prefix: 文件名前缀
            
        Returns:
            导出的文件路径字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON格式
        json_path = os.path.join(self.output_dir, f"{file_prefix}_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        
        # Markdown格式
        md_path = os.path.join(self.output_dir, f"{file_prefix}_{timestamp}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self._format_markdown_report(report))
        
        # 简化文本格式
        txt_path = os.path.join(self.output_dir, f"{file_prefix}_{timestamp}_summary.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self._format_text_summary(report))
        
        return {
            "json": json_path,
            "markdown": md_path, 
            "text_summary": txt_path
        }
    
    def _format_markdown_report(self, report: QueryStrategyReport) -> str:
        """格式化Markdown报告"""
        table_a = report.metadata.get('table_a_name', '主表')
        table_b = report.metadata.get('table_b_name', 'JOIN表')
        
        md = f"""# 查询策略评估报告

## {table_a} vs {table_b}

### 1. 核心差异诊断

**最终结论**: {report.executive_diagnosis['final_conclusion']}

**快速总结**: {report.executive_diagnosis['quick_summary']}

#### 风险矩阵

| 风险维度 | 检测结果 | 影响解释 |
|:---------|:---------|:---------|
"""
        
        for risk in report.executive_diagnosis['risk_matrix']:
            md += f"| {risk['风险维度']} | {risk['检测结果']} | {risk['影响解释']} |\n"
        
        md += f"""
### 2. 场景化模拟

针对具体值 {report.scenario_simulation['specific_value_example']} 的查询：

#### Query 1 (单表查询)
- **行为**: {report.scenario_simulation['query_comparison']['query_1_single_table']['行为描述']}
- **结果**: {report.scenario_simulation['query_comparison']['query_1_single_table']['结果预测']}

#### Query 2 (JOIN查询)
- **行为**: {report.scenario_simulation['query_comparison']['query_2_join']['行为描述']}  
- **结果**: {report.scenario_simulation['query_comparison']['query_2_join']['结果预测']}
- **问题**: {report.scenario_simulation['query_comparison']['query_2_join']['潜在问题']}

### 3. 字段关系图谱

- **关系类型**: {report.field_relationship_map['relationship_description']}
- **连接安全性**: {report.field_relationship_map['join_safety_assessment']}
- **建议**: {report.field_relationship_map['connection_advice']['recommendation']}

### 4. 开发指导

#### 优先建议
"""
        
        for rec in report.actionable_advice['priority_recommendations']:
            md += f"- **{rec['priority']}**: {rec['strategy']} - {rec['reason']}\n"
        
        md += f"""
#### SQL示例
```sql
-- 推荐的单表查询
{report.actionable_advice['sql_examples']['recommended_single_table']}

-- 谨慎的JOIN查询
{report.actionable_advice['sql_examples']['cautious_join']}
```

#### 性能考虑
"""
        
        for consideration in report.actionable_advice['performance_considerations']:
            md += f"- {consideration}\n"
        
        return md
    
    def _format_text_summary(self, report: QueryStrategyReport) -> str:
        """格式化文本摘要"""
        return f"""查询策略评估摘要
==================

结论: {report.executive_diagnosis['final_conclusion']}

关键发现:
{report.executive_diagnosis['quick_summary']}

推荐策略: 
{report.actionable_advice['priority_recommendations'][0]['strategy'] if report.actionable_advice['priority_recommendations'] else '需要详细分析'}

生成时间: {report.metadata.get('report_generated_at', 'Unknown')}
"""


def generate_query_strategy_report_from_analysis(
    analysis_result: Dict[str, Any],
    output_dir: str = "/home/yangchenyu/DeepEye-SQL-Metadata/script/caf/analyzer/temp",
    table_a_name: str = "主表",
    table_b_name: str = "JOIN表",
    save_to_file: bool = True
) -> Dict[str, Any]:
    """
    便利函数：从分析结果直接生成并保存报告
    
    Args:
        analysis_result: quick_query_difference_analysis的结果
        output_dir: 输出目录
        table_a_name: 主表名称
        table_b_name: JOIN表名称  
        save_to_file: 是否保存到文件
        
    Returns:
        包含报告内容和文件路径的字典
    """
    generator = QueryStrategyReportGenerator(output_dir)
    
    report = generator.generate_query_strategy_report(
        analysis_result, table_a_name, table_b_name
    )
    
    result = {"report": asdict(report)}
    
    if save_to_file:
        file_paths = generator.export_report(report)
        result["saved_files"] = file_paths
    
    return result








