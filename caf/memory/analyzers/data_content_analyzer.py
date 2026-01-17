"""
Data Content Analyzer - 数据内容维度分析器

实现 Ambiguity Discriminator Matrix 中的数据内容维度分析：
1. 集合拓扑分析 (Set Topology)
2. 逻辑约束检测 (Logical Constraints)
3. 结果敏感性/反事实分析 (Result Sensitivity / Counterfactual Analysis)
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from ..types import DataContentProfile, AmbiguousPair, DBElementRef

logger = logging.getLogger(__name__)


class DataContentAnalyzer:
    """
    数据内容维度分析器.
    
    通过统计学和集合论，量化"错选"的代价和物理规律。
    需要访问数据库执行 SQL 查询。
    """

    def __init__(
        self,
        database_mapping: Dict[str, str],  # database_id -> database_path
        config: Optional[Dict] = None,
    ):
        """
        Args:
            database_mapping: Mapping from database_id to database file path
            config: Configuration dict with optional parameters
        """
        self.database_mapping = database_mapping
        self.config = config or {}
        
        # Configuration
        self.constraint_sample_size = self.config.get("constraint_sample_size", 10000)
        self.sensitivity_sample_size = self.config.get("sensitivity_sample_size", 10)
        self.min_common_values = self.config.get("min_common_values", 5)

    def analyze_pair(
        self,
        pair: AmbiguousPair,
        database_path: Optional[str] = None,
    ) -> Optional[DataContentProfile]:
        """
        分析一对模糊字段的数据内容维度.
        
        Args:
            pair: 模糊字段对
            database_path: 数据库路径 (如果未提供，从 database_mapping 获取)
        
        Returns:
            DataContentProfile or None if analysis failed
        """
        # Resolve database path
        if database_path is None:
            database_path = self.database_mapping.get(pair.database_id)
        
        if not database_path or not Path(database_path).exists():
            logger.error(
                "Database path not found for pair %s (database_id: %s)",
                pair.pair_id,
                pair.database_id,
            )
            return None

        try:
            # Open database connection
            conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row

            # Extract column information
            table_a = pair.column_a.table_name
            col_a = pair.column_a.column_name
            table_b = pair.column_b.table_name
            col_b = pair.column_b.column_name

            logger.info(
                "Analyzing data content for pair: %s.%s <-> %s.%s",
                table_a, col_a, table_b, col_b,
            )

            # 1. Set Topology Analysis
            set_relationship, containment_a_in_b, containment_b_in_a, jaccard = (
                self._analyze_set_topology(conn, table_a, col_a, table_b, col_b)
            )

            # 2. Logical Constraints Detection
            constraint_rule, violation_rate, sample_size = (
                self._analyze_logical_constraints(conn, table_a, col_a, table_b, col_b)
            )

            # 3. Result Sensitivity / Counterfactual Analysis
            sensitivity_type, avg_overlap, sampled_count, examples = (
                self._analyze_result_sensitivity(conn, table_a, col_a, table_b, col_b)
            )

            conn.close()

            # Build DataContentProfile
            profile = DataContentProfile(
                set_relationship=set_relationship,
                containment_a_in_b=containment_a_in_b,
                containment_b_in_a=containment_b_in_a,
                jaccard_similarity=jaccard,
                constraint_rule=constraint_rule,
                constraint_violation_rate=violation_rate,
                constraint_sample_size=sample_size,
                sensitivity_type=sensitivity_type,
                avg_result_overlap=avg_overlap,
                sampled_value_count=sampled_count,
                example_cases=examples,
            )

            logger.info(
                "Data content analysis completed: relationship=%s, constraint=%s, sensitivity=%s",
                set_relationship,
                constraint_rule,
                sensitivity_type,
            )

            return profile

        except Exception as e:
            logger.error(
                "Failed to analyze data content for pair %s: %s",
                pair.pair_id,
                e,
            )
            return None

    def _analyze_set_topology(
        self,
        conn: sqlite3.Connection,
        table_a: str,
        col_a: str,
        table_b: str,
        col_b: str,
    ) -> Tuple[str, Optional[float], Optional[float], Optional[float]]:
        """
        分析集合拓扑关系：谁包含谁？
        
        Returns:
            (set_relationship, containment_a_in_b, containment_b_in_a, jaccard_similarity)
        """
        try:
            # Get distinct non-null values for both columns
            query_a = f"SELECT DISTINCT `{col_a}` as value FROM `{table_a}` WHERE `{col_a}` IS NOT NULL"
            query_b = f"SELECT DISTINCT `{col_b}` as value FROM `{table_b}` WHERE `{col_b}` IS NOT NULL"

            cursor_a = conn.execute(query_a)
            values_a = {str(row["value"]).strip().lower() for row in cursor_a if row["value"] is not None}

            cursor_b = conn.execute(query_b)
            values_b = {str(row["value"]).strip().lower() for row in cursor_b if row["value"] is not None}

            if not values_a or not values_b:
                return "unknown", None, None, None

            # Compute set operations
            intersection = values_a & values_b
            union = values_a | values_b

            # Compute metrics
            containment_a_in_b = len(intersection) / len(values_a) if len(values_a) > 0 else 0.0
            containment_b_in_a = len(intersection) / len(values_b) if len(values_b) > 0 else 0.0
            jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0

            # Determine relationship
            if containment_a_in_b > 0.95:
                relationship = "A_subset_of_B"
            elif containment_b_in_a > 0.95:
                relationship = "B_subset_of_A"
            elif jaccard < 0.1:
                relationship = "mutually_exclusive"
            else:
                relationship = "overlapping"

            logger.debug(
                "Set topology: %s (containment_a_in_b=%.3f, containment_b_in_a=%.3f, jaccard=%.3f)",
                relationship, containment_a_in_b, containment_b_in_a, jaccard,
            )

            return relationship, containment_a_in_b, containment_b_in_a, jaccard

        except Exception as e:
            logger.warning("Failed to analyze set topology: %s", e)
            return "unknown", None, None, None

    def _analyze_logical_constraints(
        self,
        conn: sqlite3.Connection,
        table_a: str,
        col_a: str,
        table_b: str,
        col_b: str,
    ) -> Tuple[Optional[str], Optional[float], Optional[int]]:
        """
        检测逻辑约束：A 和 B 之间是否存在铁律 (A <= B, A > B, etc.)
        
        仅当两列在同一张表时才进行检测。
        
        Returns:
            (constraint_rule, violation_rate, sample_size)
        """
        # Only check constraints if columns are in the same table
        if table_a != table_b:
            return None, None, None

        try:
            # Sample rows where both columns are not null
            sample_query = f"""
                SELECT `{col_a}` as val_a, `{col_b}` as val_b
                FROM `{table_a}`
                WHERE `{col_a}` IS NOT NULL AND `{col_b}` IS NOT NULL
                LIMIT {self.constraint_sample_size}
            """

            cursor = conn.execute(sample_query)
            rows = cursor.fetchall()

            if len(rows) < 10:  # Too few rows to detect constraints
                return None, None, len(rows)

            # Try to convert to numeric for comparison
            try:
                values_a = []
                values_b = []
                for row in rows:
                    try:
                        val_a = float(row["val_a"])
                        val_b = float(row["val_b"])
                        values_a.append(val_a)
                        values_b.append(val_b)
                    except (ValueError, TypeError):
                        continue

                if len(values_a) < 10:  # Not enough numeric values
                    return None, None, len(rows)

                # Check different constraint types
                violations_le = sum(1 for a, b in zip(values_a, values_b) if a > b)
                violations_ge = sum(1 for a, b in zip(values_a, values_b) if a < b)
                violations_eq = sum(1 for a, b in zip(values_a, values_b) if abs(a - b) > 1e-6)

                total = len(values_a)
                violation_rate_le = violations_le / total
                violation_rate_ge = violations_ge / total
                violation_rate_eq = violations_eq / total

                # Determine constraint (if any)
                if violation_rate_le < 0.05:  # Strict constraint: A <= B
                    return "A <= B", violation_rate_le, total
                elif violation_rate_ge < 0.05:  # Strict constraint: A >= B
                    return "A >= B", violation_rate_ge, total
                elif violation_rate_eq < 0.05:  # Strict constraint: A == B
                    return "A == B", violation_rate_eq, total
                else:
                    return None, None, total

            except Exception as e:
                logger.debug("Failed to detect numeric constraints: %s", e)
                return None, None, len(rows)

        except Exception as e:
            logger.warning("Failed to analyze logical constraints: %s", e)
            return None, None, None

    def _analyze_result_sensitivity(
        self,
        conn: sqlite3.Connection,
        table_a: str,
        col_a: str,
        table_b: str,
        col_b: str,
    ) -> Tuple[Optional[str], Optional[float], Optional[int], Optional[List[Dict[str, Any]]]]:
        """
        分析结果敏感性：在 WHERE 子句中用 A 替换 B，结果集怎么变？
        
        核心问题：如果用错了字段，会导致什么后果？
        - 收缩型 (Shrinkage): 结果剧减 -> A 是强限制条件
        - 漂移型 (Shift): 结果数量差不多，但 ID 完全不同 -> A/B 针对不同群体
        - 低敏感性 (Low Sensitivity): 结果集高度重叠 -> A/B 是同义词
        
        Returns:
            (sensitivity_type, avg_result_overlap, sampled_value_count, example_cases)
        """
        try:
            # Find common values between A and B
            common_values_query = f"""
                SELECT DISTINCT a.value
                FROM (SELECT DISTINCT `{col_a}` as value FROM `{table_a}` WHERE `{col_a}` IS NOT NULL) a
                INNER JOIN (SELECT DISTINCT `{col_b}` as value FROM `{table_b}` WHERE `{col_b}` IS NOT NULL) b
                ON a.value = b.value
                LIMIT {self.sensitivity_sample_size * 2}
            """

            cursor = conn.execute(common_values_query)
            common_values = [row["value"] for row in cursor if row["value"] is not None]

            if len(common_values) < self.min_common_values:
                logger.debug(
                    "Too few common values (%d) to analyze sensitivity",
                    len(common_values),
                )
                return "insufficient_data", None, len(common_values), None

            # Sample values for testing
            sampled_values = common_values[:self.sensitivity_sample_size]

            overlaps = []
            example_cases = []

            for val in sampled_values:
                try:
                    # Escape single quotes in value
                    val_escaped = str(val).replace("'", "''")

                    # Get row IDs/PKs for A=val
                    # Use ROWID as a fallback primary key
                    query_a = f"""
                        SELECT ROWID as id FROM `{table_a}`
                        WHERE `{col_a}` = '{val_escaped}'
                        LIMIT 1000
                    """
                    cursor_a = conn.execute(query_a)
                    rows_a = {row["id"] for row in cursor_a}

                    # Get row IDs/PKs for B=val
                    query_b = f"""
                        SELECT ROWID as id FROM `{table_b}`
                        WHERE `{col_b}` = '{val_escaped}'
                        LIMIT 1000
                    """
                    cursor_b = conn.execute(query_b)
                    rows_b = {row["id"] for row in cursor_b}

                    if not rows_a or not rows_b:
                        continue

                    # Compute Jaccard similarity of result sets
                    intersection = len(rows_a & rows_b)
                    union = len(rows_a | rows_b)
                    jaccard = intersection / union if union > 0 else 0.0

                    overlaps.append(jaccard)

                    # Store example case
                    example_cases.append({
                        "value": val,
                        "rows_a_count": len(rows_a),
                        "rows_b_count": len(rows_b),
                        "overlap_jaccard": jaccard,
                    })

                except Exception as e:
                    logger.debug("Failed to analyze sensitivity for value %s: %s", val, e)
                    continue

            if not overlaps:
                return "insufficient_data", None, 0, None

            avg_overlap = np.mean(overlaps)

            # Determine sensitivity type
            if avg_overlap > 0.9:
                sensitivity_type = "low_sensitivity"  # Synonyms/redundant (safe to swap)
            elif avg_overlap < 0.1:
                sensitivity_type = "high_sensitivity"  # Distinct semantics (dangerous to swap)
            else:
                sensitivity_type = "context_dependent"  # Partial overlap

            logger.debug(
                "Result sensitivity: %s (avg_overlap=%.3f, sampled=%d)",
                sensitivity_type, avg_overlap, len(overlaps),
            )

            # Keep only top 5 examples for storage
            example_cases = sorted(
                example_cases,
                key=lambda x: x["overlap_jaccard"],
            )[:5]

            return sensitivity_type, avg_overlap, len(overlaps), example_cases

        except Exception as e:
            logger.warning("Failed to analyze result sensitivity: %s", e)
            return None, None, None, None












