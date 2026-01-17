# SQL Intent Detector - 识别SQL中的所有计算意图

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import sqlglot
from sqlglot import expressions

logger = logging.getLogger(__name__)


class CalculationIntentType(Enum):
    """标准计算意图类型枚举"""
    # 排序相关
    ORDER_BY_ASC = "order_by_asc"
    ORDER_BY_DESC = "order_by_desc"
    
    # 分组相关
    GROUP_BY = "group_by"
    
    # 过滤相关
    HAVING = "having"
    
    # 聚合函数
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    
    # 计算相关
    DIVISION = "division"
    RATIO = "ratio"
    PERCENTAGE = "percentage"
    
    # 去重
    DISTINCT = "distinct"
    
    # 限制
    LIMIT = "limit"
    
    # 窗口函数
    WINDOW_FUNCTION = "window_function"
    PARTITION_BY = "partition_by"
    ROW_NUMBER = "row_number"
    RANK = "rank"
    DENSE_RANK = "dense_rank"
    
    # 子查询
    SUBQUERY = "subquery"
    EXISTS = "exists"
    IN_SUBQUERY = "in_subquery"
    
    # 连接
    JOIN = "join"
    INNER_JOIN = "inner_join"
    LEFT_JOIN = "left_join"
    RIGHT_JOIN = "right_join"
    FULL_JOIN = "full_join"


# 意图标签映射：将意图类型映射到用于匹配ActionGuidance的标签
INTENT_TAG_MAPPING = {
    # 排序
    CalculationIntentType.ORDER_BY_ASC: "sorting",
    CalculationIntentType.ORDER_BY_DESC: "sorting",
    
    # 分组和聚合
    CalculationIntentType.GROUP_BY: "aggregate",
    CalculationIntentType.COUNT: "aggregate",
    CalculationIntentType.SUM: "aggregate",
    CalculationIntentType.AVG: "aggregate",
    CalculationIntentType.MAX: "aggregate",
    CalculationIntentType.MIN: "aggregate",
    
    # 过滤
    CalculationIntentType.HAVING: "filter",
    
    # 计算
    CalculationIntentType.DIVISION: "calc",
    CalculationIntentType.RATIO: "calc",
    CalculationIntentType.PERCENTAGE: "calc",
    
    # 去重
    CalculationIntentType.DISTINCT: "deduplication",
    
    # 限制
    CalculationIntentType.LIMIT: "limit",
    
    # 窗口函数
    CalculationIntentType.WINDOW_FUNCTION: "window",
    CalculationIntentType.PARTITION_BY: "window",
    CalculationIntentType.ROW_NUMBER: "window",
    CalculationIntentType.RANK: "window",
    CalculationIntentType.DENSE_RANK: "window",
    
    # 子查询
    CalculationIntentType.SUBQUERY: "subquery",
    CalculationIntentType.EXISTS: "subquery",
    CalculationIntentType.IN_SUBQUERY: "subquery",
    
    # 连接
    CalculationIntentType.JOIN: "join",
    CalculationIntentType.INNER_JOIN: "join",
    CalculationIntentType.LEFT_JOIN: "join",
    CalculationIntentType.RIGHT_JOIN: "join",
    CalculationIntentType.FULL_JOIN: "join",
}


@dataclass
class CalculationIntent:
    """计算意图数据结构"""
    intent_type: CalculationIntentType  # 意图类型
    intent_tag: str  # 用于匹配ActionGuidance的标签
    details: Dict[str, Any]  # 详细信息（如排序列、分组列等）
    sql_fragment: str  # 相关的SQL片段
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'intent_type': self.intent_type.value,
            'intent_tag': self.intent_tag,
            'details': self.details,
            'sql_fragment': self.sql_fragment
        }
    
    def get_intent_set(self) -> Set[str]:
        """获取意图集合（用于匹配ActionGuidance）"""
        return {self.intent_type.value, self.intent_tag}


class SQLIntentDetector:
    """SQL计算意图识别器"""
    
    def __init__(self):
        """初始化意图识别器"""
        pass
    
    def detect_all_intents(self, sql: str) -> List[CalculationIntent]:
        """
        识别SQL中的所有计算意图
        
        Args:
            sql: SQL查询字符串
            
        Returns:
            识别出的所有计算意图列表
        """
        if not sql or not sql.strip():
            return []
        
        try:
            # 解析SQL AST
            ast = sqlglot.parse_one(sql, dialect='sqlite', error_level=sqlglot.ErrorLevel.IGNORE)
            if ast is None:
                logger.warning(f"Failed to parse SQL: {sql[:100]}...")
                return []
            
            intents = []
            
            # 识别各种意图
            intents.extend(self._detect_order_by(ast))
            intents.extend(self._detect_group_by(ast))
            intents.extend(self._detect_having(ast))
            intents.extend(self._detect_aggregate_functions(ast))
            intents.extend(self._detect_division(ast))
            intents.extend(self._detect_distinct(ast))
            intents.extend(self._detect_limit(ast))
            intents.extend(self._detect_window_functions(ast))
            intents.extend(self._detect_joins(ast))
            intents.extend(self._detect_subqueries(ast))
            
            logger.debug(f"Detected {len(intents)} calculation intents from SQL")
            return intents
            
        except Exception as e:
            logger.error(f"Error detecting intents from SQL: {e}")
            return []
    
    def _detect_order_by(self, ast) -> List[CalculationIntent]:
        """检测ORDER BY意图"""
        intents = []
        
        try:
            # 查找所有SELECT节点（包括子查询）
            for select_node in ast.find_all(expressions.Select):
                order_node = select_node.args.get('order')
                if order_node and hasattr(order_node, 'expressions') and order_node.expressions:
                    for order_expr in order_node.expressions:
                        # 检查排序方向
                        is_desc = order_expr.args.get('desc', False)
                        intent_type = CalculationIntentType.ORDER_BY_DESC if is_desc else CalculationIntentType.ORDER_BY_ASC
                        intent_tag = INTENT_TAG_MAPPING.get(intent_type, "sorting")
                        
                        # 提取排序列信息
                        expr_sql = order_expr.this.sql(dialect='sqlite') if hasattr(order_expr, 'this') else str(order_expr)
                        
                        intent = CalculationIntent(
                            intent_type=intent_type,
                            intent_tag=intent_tag,
                            details={
                                'column': expr_sql,
                                'direction': 'DESC' if is_desc else 'ASC'
                            },
                            sql_fragment=f"ORDER BY {expr_sql} {'DESC' if is_desc else 'ASC'}"
                        )
                        intents.append(intent)
        except Exception as e:
            logger.debug(f"Error detecting ORDER BY: {e}")
        
        return intents
    
    def _detect_group_by(self, ast) -> List[CalculationIntent]:
        """检测GROUP BY意图"""
        intents = []
        
        try:
            for select_node in ast.find_all(expressions.Select):
                group_node = select_node.args.get('group')
                if group_node and hasattr(group_node, 'expressions') and group_node.expressions:
                    group_exprs = [e.sql(dialect='sqlite') for e in group_node.expressions]
                    group_sql = ", ".join(group_exprs)
                    
                    intent = CalculationIntent(
                        intent_type=CalculationIntentType.GROUP_BY,
                        intent_tag=INTENT_TAG_MAPPING[CalculationIntentType.GROUP_BY],
                        details={
                            'columns': group_exprs
                        },
                        sql_fragment=f"GROUP BY {group_sql}"
                    )
                    intents.append(intent)
        except Exception as e:
            logger.debug(f"Error detecting GROUP BY: {e}")
        
        return intents
    
    def _detect_having(self, ast) -> List[CalculationIntent]:
        """检测HAVING意图"""
        intents = []
        
        try:
            for select_node in ast.find_all(expressions.Select):
                having_node = select_node.args.get('having')
                if having_node and hasattr(having_node, 'this'):
                    having_sql = having_node.this.sql(dialect='sqlite')
                    
                    intent = CalculationIntent(
                        intent_type=CalculationIntentType.HAVING,
                        intent_tag=INTENT_TAG_MAPPING[CalculationIntentType.HAVING],
                        details={
                            'condition': having_sql
                        },
                        sql_fragment=f"HAVING {having_sql}"
                    )
                    intents.append(intent)
        except Exception as e:
            logger.debug(f"Error detecting HAVING: {e}")
        
        return intents
    
    def _detect_aggregate_functions(self, ast) -> List[CalculationIntent]:
        """检测聚合函数意图（COUNT, SUM, AVG, MAX, MIN）"""
        intents = []
        
        # 聚合函数类型映射
        agg_function_map = {
            expressions.Count: CalculationIntentType.COUNT,
            expressions.Sum: CalculationIntentType.SUM,
            expressions.Avg: CalculationIntentType.AVG,
            expressions.Max: CalculationIntentType.MAX,
            expressions.Min: CalculationIntentType.MIN,
        }
        
        try:
            for select_node in ast.find_all(expressions.Select):
                # 检查SELECT列表中的聚合函数
                for expr in select_node.args.get('expressions', []):
                    for agg_type, intent_type in agg_function_map.items():
                        agg_func = expr.find(agg_type)
                        if agg_func:
                            intent_tag = INTENT_TAG_MAPPING.get(intent_type, "aggregate")
                            func_sql = agg_func.sql(dialect='sqlite')
                            
                            intent = CalculationIntent(
                                intent_type=intent_type,
                                intent_tag=intent_tag,
                                details={
                                    'function': intent_type.value,
                                    'expression': func_sql
                                },
                                sql_fragment=func_sql
                            )
                            intents.append(intent)
                            break  # 每个表达式只匹配一次
        except Exception as e:
            logger.debug(f"Error detecting aggregate functions: {e}")
        
        return intents
    
    def _detect_division(self, ast) -> List[CalculationIntent]:
        """检测除法/比率计算意图"""
        intents = []
        
        try:
            # 查找除法运算符
            for div_expr in ast.find_all(expressions.Div):
                left_sql = div_expr.this.sql(dialect='sqlite') if hasattr(div_expr, 'this') else ""
                right_sql = div_expr.expression.sql(dialect='sqlite') if hasattr(div_expr, 'expression') else ""
                div_sql = f"{left_sql} / {right_sql}"
                
                # 检查是否在ROUND函数中（可能是百分比计算）
                parent = div_expr.parent
                is_percentage = False
                if parent and isinstance(parent, expressions.Round):
                    is_percentage = True
                    intent_type = CalculationIntentType.PERCENTAGE
                else:
                    # 检查是否可能是比率计算
                    intent_type = CalculationIntentType.DIVISION
                
                intent_tag = INTENT_TAG_MAPPING.get(intent_type, "calc")
                
                intent = CalculationIntent(
                    intent_type=intent_type,
                    intent_tag=intent_tag,
                    details={
                        'numerator': left_sql,
                        'denominator': right_sql,
                        'is_percentage': is_percentage
                    },
                    sql_fragment=div_sql
                )
                intents.append(intent)
        except Exception as e:
            logger.debug(f"Error detecting division: {e}")
        
        return intents
    
    def _detect_distinct(self, ast) -> List[CalculationIntent]:
        """检测DISTINCT意图"""
        intents = []
        
        try:
            for select_node in ast.find_all(expressions.Select):
                if select_node.args.get('distinct'):
                    intent = CalculationIntent(
                        intent_type=CalculationIntentType.DISTINCT,
                        intent_tag=INTENT_TAG_MAPPING[CalculationIntentType.DISTINCT],
                        details={},
                        sql_fragment="DISTINCT"
                    )
                    intents.append(intent)
                    break  # 每个SELECT只检测一次
        except Exception as e:
            logger.debug(f"Error detecting DISTINCT: {e}")
        
        return intents
    
    def _detect_limit(self, ast) -> List[CalculationIntent]:
        """检测LIMIT意图"""
        intents = []
        
        try:
            for select_node in ast.find_all(expressions.Select):
                limit_node = select_node.args.get('limit')
                if limit_node:
                    limit_value = None
                    if hasattr(limit_node, 'this'):
                        limit_value = limit_node.this.sql(dialect='sqlite')
                    
                    intent = CalculationIntent(
                        intent_type=CalculationIntentType.LIMIT,
                        intent_tag=INTENT_TAG_MAPPING[CalculationIntentType.LIMIT],
                        details={
                            'limit_value': limit_value
                        },
                        sql_fragment=f"LIMIT {limit_value}" if limit_value else "LIMIT"
                    )
                    intents.append(intent)
        except Exception as e:
            logger.debug(f"Error detecting LIMIT: {e}")
        
        return intents
    
    def _detect_window_functions(self, ast) -> List[CalculationIntent]:
        """检测窗口函数意图"""
        intents = []
        
        try:
            # 查找窗口函数
            window_function_types = {
                expressions.RowNumber: CalculationIntentType.ROW_NUMBER,
                expressions.Rank: CalculationIntentType.RANK,
                expressions.DenseRank: CalculationIntentType.DENSE_RANK,
            }
            
            for select_node in ast.find_all(expressions.Select):
                for expr in select_node.args.get('expressions', []):
                    for func_type, intent_type in window_function_types.items():
                        func = expr.find(func_type)
                        if func:
                            intent_tag = INTENT_TAG_MAPPING.get(intent_type, "window")
                            func_sql = func.sql(dialect='sqlite')
                            
                            # 检查是否有PARTITION BY
                            has_partition = False
                            if hasattr(func, 'over') and func.over:
                                partition = func.over.args.get('partition')
                                if partition:
                                    has_partition = True
                            
                            intent = CalculationIntent(
                                intent_type=intent_type,
                                intent_tag=intent_tag,
                                details={
                                    'function': intent_type.value,
                                    'has_partition_by': has_partition
                                },
                                sql_fragment=func_sql
                            )
                            intents.append(intent)
                            break
                    
                    # 检查PARTITION BY（独立检测）
                    if hasattr(expr, 'over') and expr.over:
                        partition = expr.over.args.get('partition')
                        if partition:
                            intent = CalculationIntent(
                                intent_type=CalculationIntentType.PARTITION_BY,
                                intent_tag=INTENT_TAG_MAPPING[CalculationIntentType.PARTITION_BY],
                                details={
                                    'partition_columns': [p.sql(dialect='sqlite') for p in partition.expressions] if hasattr(partition, 'expressions') else []
                                },
                                sql_fragment=f"PARTITION BY {partition.sql(dialect='sqlite')}"
                            )
                            intents.append(intent)
        except Exception as e:
            logger.debug(f"Error detecting window functions: {e}")
        
        return intents
    
    def _detect_joins(self, ast) -> List[CalculationIntent]:
        """检测JOIN意图"""
        intents = []
        
        try:
            join_type_map = {
                'INNER': CalculationIntentType.INNER_JOIN,
                'LEFT': CalculationIntentType.LEFT_JOIN,
                'RIGHT': CalculationIntentType.RIGHT_JOIN,
                'FULL': CalculationIntentType.FULL_JOIN,
            }
            
            for join_node in ast.find_all(expressions.Join):
                join_kind = join_node.kind.upper() if hasattr(join_node, 'kind') and join_node.kind else 'INNER'
                intent_type = join_type_map.get(join_kind, CalculationIntentType.JOIN)
                intent_tag = INTENT_TAG_MAPPING.get(intent_type, "join")
                
                # 提取连接表信息
                this_table = join_node.this.sql(dialect='sqlite') if hasattr(join_node, 'this') else ""
                on_condition = join_node.on.sql(dialect='sqlite') if hasattr(join_node, 'on') else ""
                
                intent = CalculationIntent(
                    intent_type=intent_type,
                    intent_tag=intent_tag,
                    details={
                        'join_type': join_kind,
                        'table': this_table,
                        'on_condition': on_condition
                    },
                    sql_fragment=f"{join_kind} JOIN {this_table} ON {on_condition}" if on_condition else f"{join_kind} JOIN {this_table}"
                )
                intents.append(intent)
        except Exception as e:
            logger.debug(f"Error detecting JOINs: {e}")
        
        return intents
    
    def _detect_subqueries(self, ast) -> List[CalculationIntent]:
        """检测子查询意图"""
        intents = []
        
        try:
            # 检测EXISTS子查询
            for exists_expr in ast.find_all(expressions.Exists):
                intent = CalculationIntent(
                    intent_type=CalculationIntentType.EXISTS,
                    intent_tag=INTENT_TAG_MAPPING[CalculationIntentType.EXISTS],
                    details={},
                    sql_fragment="EXISTS"
                )
                intents.append(intent)
            
            # 检测IN子查询
            for in_expr in ast.find_all(expressions.In):
                if in_expr.this and hasattr(in_expr, 'expressions'):
                    # 检查expressions是否是子查询
                    for expr in in_expr.expressions:
                        if isinstance(expr, expressions.Subquery):
                            intent = CalculationIntent(
                                intent_type=CalculationIntentType.IN_SUBQUERY,
                                intent_tag=INTENT_TAG_MAPPING[CalculationIntentType.IN_SUBQUERY],
                                details={},
                                sql_fragment="IN (subquery)"
                            )
                            intents.append(intent)
                            break
            
            # 检测普通子查询（在FROM子句中）
            for subquery in ast.find_all(expressions.Subquery):
                # 避免重复计数（如果已经在EXISTS或IN中检测过）
                if not any(i.intent_type in [CalculationIntentType.EXISTS, CalculationIntentType.IN_SUBQUERY] 
                          for i in intents):
                    intent = CalculationIntent(
                        intent_type=CalculationIntentType.SUBQUERY,
                        intent_tag=INTENT_TAG_MAPPING[CalculationIntentType.SUBQUERY],
                        details={},
                        sql_fragment="(subquery)"
                    )
                    intents.append(intent)
        except Exception as e:
            logger.debug(f"Error detecting subqueries: {e}")
        
        return intents
    
    def get_intent_set_from_sql(self, sql: str) -> Set[str]:
        """
        从SQL中提取意图集合（用于匹配ActionGuidance）
        
        Args:
            sql: SQL查询字符串
            
        Returns:
            意图集合（包含intent_type和intent_tag）
        """
        intents = self.detect_all_intents(sql)
        intent_set = set()
        
        for intent in intents:
            intent_set.add(intent.intent_type.value)
            intent_set.add(intent.intent_tag)
        
        return intent_set




