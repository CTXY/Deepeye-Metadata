"""
SQL Preprocessor - Ensures all columns have explicit table names

This module preprocesses SQL queries to:
1. Replace table aliases with actual table names
2. Add table names to columns that don't have them (based on context)
3. Ensure consistent formatting
"""

import logging
import re
import json
import sqlite3
from pathlib import Path
import sqlglot
from sqlglot import expressions
from sqlglot.optimizer.scope import build_scope
from typing import Dict, Set, Optional, Tuple, Any, List

logger = logging.getLogger(__name__)


class SQLPreprocessor:
    """
    SQL Preprocessor that ensures all columns have explicit table names.
    
    This ensures proper extraction of table information from columns
    and makes AST pruning more reliable.
    
    Supports two modes:
    1. Scope-based inference (when schema is not provided)
    2. Schema-based resolution (when schema is provided or loaded from database)
    """
    
    def __init__(self, 
                 case_sensitive: bool = False,
                 schema: Optional[Dict[str, Set[str]]] = None,
                 database_id: Optional[str] = None,
                 db_path: Optional[str] = None,
                 use_schema_cache: bool = True):
        """
        Initialize the SQL preprocessor.
        
        Args:
            case_sensitive: Whether table/column names are case sensitive
            schema: Optional schema dict mapping column_name (lowercase) to set of table names (lowercase)
            database_id: Optional database identifier for loading schema from mapping file
            db_path: Optional database file path for loading schema directly
            use_schema_cache: Whether to use/save schema cache file (default: True)
        """
        self.case_sensitive = case_sensitive
        self._provided_schema = schema
        self._schema = None
        self.database_id = database_id
        self.db_path = db_path
        self.use_schema_cache = use_schema_cache
        self._schema_loaded = False
    
    @property
    def schema(self) -> Optional[Dict[str, Set[str]]]:
        """Lazy load schema when first accessed"""
        if not self._schema_loaded:
            self._load_schema_if_needed()
        return self._schema
    
    def _load_schema_if_needed(self):
        """Load schema if database_id or db_path is provided"""
        if self._schema_loaded:
            return
        
        # Use provided schema first
        if self._provided_schema is not None:
            self._schema = self._provided_schema
            logger.debug(f"Using provided schema with {len(self._schema)} columns")
        # Otherwise try to load from database_id or db_path
        elif self.database_id or self.db_path:
            self._schema = self._get_schema(
                database_id=self.database_id,
                schema=self._provided_schema,
                db_path=self.db_path,
                use_cache=self.use_schema_cache
            )
            if self._schema:
                logger.debug(f"Loaded schema with {len(self._schema)} columns")
            else:
                logger.debug("Failed to load schema, will use scope-based inference")
        else:
            logger.debug("No schema source provided, will use scope-based inference")
        
        self._schema_loaded = True
    
    def _normalize_identifier(self, identifier: Optional[str]) -> Optional[str]:
        """Normalize identifier based on case sensitivity"""
        if identifier is None:
            return None
        if not self.case_sensitive:
            return identifier.lower()
        return identifier
    
    # =========================================================================
    # Schema Loading Methods (migrated from sql_qualifier.py)
    # =========================================================================
    
    @staticmethod
    def _get_schema_file_path(db_path: str) -> Path:
        """
        Get the schema file path for a database.
        
        Schema file is stored in the same directory as the database file,
        with name: {database_name}.schema.json
        """
        db_path_obj = Path(db_path)
        schema_file = db_path_obj.parent / f"{db_path_obj.stem}.schema.json"
        return schema_file
    
    @staticmethod
    def _load_schema_from_file(schema_file: Path) -> Optional[Dict[str, Set[str]]]:
        """Load schema from JSON file"""
        if not schema_file.exists():
            return None
        
        try:
            with schema_file.open("r", encoding="utf-8") as f:
                schema_data = json.load(f)
            
            # Convert list values back to sets
            schema = {}
            for col_name, table_list in schema_data.items():
                schema[col_name] = set(table_list)
            
            logger.debug(f"Loaded schema from {schema_file}")
            return schema
        except Exception as e:
            logger.warning(f"Failed to load schema from {schema_file}: {e}")
            return None
    
    @staticmethod
    def _save_schema_to_file(schema: Dict[str, Set[str]], schema_file: Path) -> bool:
        """Save schema to JSON file"""
        try:
            # Convert sets to lists for JSON serialization
            schema_data = {}
            for col_name, table_set in schema.items():
                schema_data[col_name] = sorted(list(table_set))
            
            # Ensure parent directory exists
            schema_file.parent.mkdir(parents=True, exist_ok=True)
            
            with schema_file.open("w", encoding="utf-8") as f:
                json.dump(schema_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved schema to {schema_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save schema to {schema_file}: {e}")
            return False
    
    @staticmethod
    def _load_schema_from_db_path(db_path: str) -> Dict[str, Set[str]]:
        """
        Load database schema from database file path.
        
        Returns:
            Dict mapping column_name (lowercase) to set of table names (lowercase) that contain it.
        """
        schema: Dict[str, Set[str]] = {}
        
        try:
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")
            table_names = [row[0] for row in cursor.fetchall()]
            
            # For each table, get its columns
            for table_name in table_names:
                cursor.execute(f"PRAGMA table_info(`{table_name}`)")
                columns = cursor.fetchall()
                
                for col_info in columns:
                    col_name = col_info[1]  # Column name is at index 1
                    col_name_lower = col_name.lower()
                    
                    if col_name_lower not in schema:
                        schema[col_name_lower] = set()
                    schema[col_name_lower].add(table_name.lower())
            
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to load schema from {db_path}: {e}")
            return {}
        
        return schema
    
    @staticmethod
    def _resolve_database_path(database_id: str) -> Optional[str]:
        """
        Resolve database_id to database_path using mapping file.
        
        Args:
            database_id: Database identifier
            
        Returns:
            Database file path if found, None otherwise
        """
        try:
            from caf.config.paths import PathConfig
            mapping_path = PathConfig.get_database_mapping_path()
            
            if not mapping_path.exists():
                logger.warning(f"Database mapping file not found at {mapping_path}")
                return None
            
            with mapping_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            # raw: {"/abs/path/to/db.sqlite": "database_id", ...}
            for abs_path, db_id in raw.items():
                if str(db_id) == str(database_id):
                    return abs_path
        except Exception as e:
            logger.error(f"Failed to load database mapping: {e}")
            return None
        
        return None
    
    def _get_schema(self,
                   database_id: Optional[str] = None,
                   schema: Optional[Dict[str, Set[str]]] = None,
                   db_path: Optional[str] = None,
                   use_cache: bool = True) -> Dict[str, Set[str]]:
        """
        Get schema from various sources with caching support.
        
        Priority:
        1. Use provided schema if available
        2. Load from cache file if exists (when use_cache=True)
        3. Load from db_path if provided
        4. Resolve database_id to db_path and load schema
        """
        if schema is not None:
            return schema
        
        if db_path is None and database_id is not None:
            db_path = self._resolve_database_path(database_id)
            if db_path is None:
                logger.warning(f"Could not resolve database path for database_id: {database_id}")
                return {}
        
        if db_path is not None:
            return self._load_schema_with_cache(db_path, use_cache=use_cache)
        
        return {}
    
    def _load_schema_with_cache(self, db_path: str, use_cache: bool = True) -> Dict[str, Set[str]]:
        """
        Load schema from database, with optional caching to file.
        
        Priority:
        1. Load from schema file if exists and use_cache is True
        2. Load from database and save to file if use_cache is True
        """
        schema_file = self._get_schema_file_path(db_path)
        
        # Try to load from cache file first
        if use_cache:
            cached_schema = self._load_schema_from_file(schema_file)
            if cached_schema is not None:
                return cached_schema
        
        # Load from database
        schema = self._load_schema_from_db_path(db_path)
        
        # Save to cache file if enabled
        if use_cache and schema:
            self._save_schema_to_file(schema, schema_file)
        
        return schema
    
    # =========================================================================
    # End of Schema Loading Methods
    # =========================================================================
    
    def preprocess(self, sql: str) -> Tuple[str, Dict[str, str], Dict[str, str]]:
        """
        Preprocess SQL to ensure all columns have explicit table names
        and return a map of derived tables/CTEs and their "SQL-Logic" format.
        
        Args:
            sql: Original SQL query
            
        Returns:
            Tuple[str, Dict[str, str], Dict[str, str]]:
            - preprocessed_sql: SQL with aliases replaced
            - named_query_map: {alias: raw_subquery_sql}
            - named_query_logic_map: {alias: sql_logic_string}
        """
        named_query_map_for_user = {}
        named_query_logic_map = {} # 新增: 存储 SQL-Logic
        
        try:
            # Preprocess SQL for parsing (fix LIKE clauses, etc.)
            preprocessed_sql = self._preprocess_sql_for_parsing(sql)
            
            # Parse SQL AST
            ast = sqlglot.parse_one(preprocessed_sql, dialect='sqlite', error_level=sqlglot.ErrorLevel.IGNORE)
            
            # 检查 AST 是否成功解析
            if not ast:
                logger.error("SQL parsing failed, returning original.")
                return sql, named_query_map_for_user, named_query_logic_map

            # ----------------------------------------------------
            # 步骤 1: 收集别名、派生表和 CTE
            table_aliases, named_query_map_for_user = self._collect_table_aliases(ast)
            logger.debug(f"Table aliases (internal use): {table_aliases}")
            logger.debug(f"Derived tables/CTEs (for user): {named_query_map_for_user}")
            # ----------------------------------------------------

            # ----------------------------------------------------
            # 步骤 2: 将原始SQL转换为 SQL-Logic (新增)
            for alias, raw_sql in named_query_map_for_user.items():
                named_query_logic_map[alias] = self._sql_to_logic(raw_sql)
            logger.debug(f"SQL Logic Map (for user): {named_query_logic_map}")
            # ----------------------------------------------------

            # 步骤 3: 找到主/第一个 SELECT 节点 (其他函数需要它)
            select_node = ast.find(expressions.Select)
            if not select_node:
                if isinstance(ast, expressions.Select):
                    select_node = ast
                else:
                    logger.warning("Could not find SELECT node in SQL, processing may be incomplete.")
                    # 即使没有 SELECT, 也返回已处理的 SQL 和 map
                    return sql, named_query_map_for_user, named_query_logic_map
            
            # 步骤 4: 构建表上下文 (使用 table_aliases)
            table_context = self._build_table_context(select_node, table_aliases)
            
            # 步骤 5: 添加表名 (使用 table_aliases, 就地修改 AST)
            self._add_table_names_to_columns(select_node, table_context, table_aliases)
            
            # 步骤 6: 替换别名 (使用 table_aliases, 就地修改 AST)
            self._replace_aliases_with_table_names(select_node, table_aliases)
            
            # 步骤 7: 移除表别名 (就地修改 AST)
            self._remove_table_aliases(select_node)
            
            # 步骤 8: 重新生成SQL
            result_sql = ast.sql(dialect='sqlite')
            
            logger.debug(f"Preprocessed SQL: {result_sql}")
            return result_sql, named_query_map_for_user, named_query_logic_map
            
        except Exception as e:
            logger.error(f"SQL preprocessing failed: {e}")
            logger.debug(f"Original SQL: {sql}")
            # 失败时返回原始SQL和（可能不完整的）maps
            return sql, named_query_map_for_user, named_query_logic_map

    def _preprocess_sql_for_parsing(self, sql: str) -> str:
        """
        Preprocess SQL to fix issues that might cause sqlglot parsing to fail.
        
        主要修复：
        1. LIKE子句中的通配符：确保LIKE '%pattern%'中的%被正确处理
        2. 其他可能的解析问题
        
        Args:
            sql: 原始SQL字符串
            
        Returns:
            预处理后的SQL字符串
        """
        # 问题：sqlglot可能将LIKE子句中的%误解析为模运算符
        # 例如：LIKE '%arena,mtgo%' 可能被解析为 LIKE % arena,mtgo %
        # 
        # 解决方案：确保LIKE子句中的字符串字面量被正确识别
        # 我们需要找到所有LIKE子句，并确保它们后面的值是字符串字面量
        
        # 模式1：LIKE后面跟着单引号字符串（可能包含%通配符）
        # LIKE '%pattern%' 或 LIKE 'pattern'
        like_single_quote_pattern = r"\bLIKE\s+'([^']*)'"
        
        def fix_like_single_quote(match):
            pattern = match.group(1)
            # 确保字符串字面量被正确引用
            return f"LIKE '{pattern}'"
        
        # 模式2：LIKE后面跟着双引号字符串（可能包含%通配符）
        # LIKE "%pattern%" 或 LIKE "pattern"
        like_double_quote_pattern = r'\bLIKE\s+"([^"]*)"'
        
        def fix_like_double_quote(match):
            pattern = match.group(1)
            # 确保字符串字面量被正确引用
            return f'LIKE "{pattern}"'
        
        # 应用修复
        sql = re.sub(like_single_quote_pattern, fix_like_single_quote, sql, flags=re.IGNORECASE)
        sql = re.sub(like_double_quote_pattern, fix_like_double_quote, sql, flags=re.IGNORECASE)
        
        # 如果LIKE后面没有引号，尝试添加引号（这种情况较少见，但可能发生）
        # LIKE %pattern% -> LIKE '%pattern%'
        like_no_quote_pattern = r"\bLIKE\s+%([^%]+)%"
        
        def fix_like_no_quote(match):
            pattern = match.group(1)
            return f"LIKE '%{pattern}%'"
        
        sql = re.sub(like_no_quote_pattern, fix_like_no_quote, sql, flags=re.IGNORECASE)
        
        return sql

    def _sql_to_logic(self, sql: str) -> str:
        """
        Converts a raw SQL string into the "SQL-Logic" fragment chain format.
        
        Args:
            sql: The raw SQL string (e.g., from a CTE or derived table).
            
        Returns:
            A string formatted according to the SQL-Logic specification.
        """
        try:
            ast = sqlglot.parse_one(sql, dialect='sqlite', error_level=sqlglot.ErrorLevel.IGNORE)
            if not ast:
                return "[Error: Failed to parse SQL for logic generation]"
            return self._build_logic_chain(ast)
        except Exception as e:
            logger.warning(f"SQL-to-Logic generation failed: {e}")
            return f"[Error: Logic generation failed: {e}]"

    def _build_logic_chain(self, node: expressions.Expression) -> str:
        """
        Recursively builds the logic chain from any sqlglot AST node.
        Dispatches to specialized builders based on node type.
        
        This is the main recursive dispatcher.
        
        Args:
            node: The AST node to convert.
            
        Returns:
            The formatted logic chain string.
        """
        if isinstance(node, expressions.Select):
            return self._build_select_logic(node)
        
        if isinstance(node, expressions.With):
            return self._build_with_logic(node)
            
        if isinstance(node, (expressions.Union, expressions.Intersect, expressions.Except)):
            return self._build_set_op_logic(node)

        if isinstance(node, expressions.Values):
            # VALUES (...) 
            return f"[{self._format_clause_expression(node)}]"

        # Fallback for other expression types (e.g., Literal, or a complex expression
        # that doesn't start with SELECT/WITH/UNION)
        return f"[{self._format_clause_expression(node)}]"
    
    def _format_clause_expression(self, node: expressions.Expression) -> str:
        """
        Formats a clause expression (e.g., a WHERE condition, a SELECT column)
        by templatizing identifiers and recursively handling subqueries.
        
        Args:
            node: The clause expression node.
            
        Returns:
            Formatted string for that clause.
        """
        
        # 1. 递归替换所有子查询
        def subquery_replacer(n: expressions.Expression) -> expressions.Expression:
            if isinstance(n, expressions.Subquery):
                # 递归调用 _build_logic_chain
                inner_logic = self._build_logic_chain(n.this)
                # 格式化为 ( ... )
                formatted = f"({inner_logic})"
                if n.alias:
                    # 别名也需要 templatize
                    formatted += f" AS {self._templatize_node(n.alias)}"
                
                # 用一个 Literal 节点替换 Subquery 节点
                # this=formatted, is_string=False 确保它被当作SQL片段注入
                return expressions.Literal(this=formatted, is_string=False)
            return n

        node_with_subqueries_replaced = node.transform(subquery_replacer, copy=True)
        
        # 2. templatize 所有剩余的标识符
        def identifier_replacer(n: expressions.Expression) -> expressions.Expression:
            if isinstance(n, expressions.Identifier):
                # e.g., "Person" -> "<Person>"
                # e.g., "name" -> "<name>"
                return expressions.Identifier(this=f"<{n.this}>", quoted=False)
            return n

        fully_templated_node = node_with_subqueries_replaced.transform(identifier_replacer)
        
        # 3. 生成最终的 SQL 片段
        return fully_templated_node.sql(dialect='sqlite')

    def _templatize_node(self, node: expressions.Expression) -> str:
        """
        Simple utility to templatize a single node (like an alias).
        """
        if isinstance(node, expressions.Identifier):
            return f"<{node.this}>"
        # 回退
        return f"<{node.sql()}>"


    def _build_with_logic(self, node: expressions.With) -> str:
        """Builds a logic chain for a WITH statement (CTEs)."""
        parts = []
        # 1. 处理所有 CTEs
        for cte in node.args.get('expressions', []):
            cte_alias = self._templatize_node(cte.alias)
            # 递归调用主分派器处理 CTE 的内部
            inner_logic = self._build_logic_chain(cte.this) 
            parts.append(f"[WITH {cte_alias} AS ( {inner_logic} )]")
        
        # 2. 处理 WITH 之后的 主查询
        # 递归调用主分派器
        main_logic = self._build_logic_chain(node.this)
        parts.append(main_logic)
        
        return " -> ".join(parts)

    def _build_set_op_logic(self, node: expressions.Expression) -> str:
        """Builds a logic chain for Set Operations (UNION, INTERSECT, EXCEPT)."""
        op_name = type(node).__name__.upper()
        
        # 递归调用主分派器处理 左侧 和 右侧
        left_logic = self._build_logic_chain(node.left)
        right_logic = self._build_logic_chain(node.right)
        
        # 格式: ( left_chain ) -> [OPERATOR] -> ( right_chain )
        return f"({left_logic}) -> [{op_name}] -> ({right_logic})"

    def _build_select_logic(self, main_select: expressions.Select) -> str:
        """Builds a logic chain for a standard SELECT statement."""
        # 这是之前 _build_logic_chain 的核心逻辑 (已移除 WITH 部分)
        parts = []
        
        # --- 2. Handle FROM ---
        from_clause = main_select.args.get('from')
        if from_clause:
            parts.append(f"[FROM {self._format_clause_expression(from_clause.this)}]")

        # --- 3. Handle JOINs ---
        for join in main_select.args.get('joins', []):
            join_type = join.args.get('kind', 'INNER').upper()
            target = self._format_clause_expression(join.this)  # Table or Subquery
            on_condition = self._format_clause_expression(join.args.get('on'))
            parts.append(f"[{join_type} JOIN {target} ON {on_condition}]")

        # --- 4. Handle WHERE ---
        where_clause = main_select.args.get('where')
        if where_clause:
            parts.append(f"[WHERE {self._format_clause_expression(where_clause.this)}]")

        # --- 5. Handle GROUP BY ---
        group_clause = main_select.args.get('group')
        if group_clause:
            parts.append(f"[GROUP BY {self._format_clause_expression(group_clause)}]")
    
        # --- 6. Handle HAVING ---
        having_clause = main_select.args.get('having')
        if having_clause:
            parts.append(f"[HAVING {self._format_clause_expression(having_clause.this)}]")

        # --- 7. Handle SELECT ---
        select_exprs = main_select.expressions
        if select_exprs:
            select_list = ", ".join(self._format_clause_expression(e) for e in select_exprs)
            
            # 检查是否有 DISTINCT
            if main_select.args.get('distinct'):
                parts.append(f"[SELECT DISTINCT {select_list}]")
            else:
                parts.append(f"[SELECT {select_list}]")

        # --- 8. Handle ORDER BY ---
        order_clause = main_select.args.get('order')
        if order_clause:
            parts.append(f"[ORDER BY {self._format_clause_expression(order_clause)}]")

        # --- 9. Handle LIMIT ---
        limit_clause = main_select.args.get('limit')
        if limit_clause:
            parts.append(f"[LIMIT {self._format_clause_expression(limit_clause.this)}]")

        return " -> ".join(parts)
    
    def _collect_table_aliases(self, ast: expressions.Expression) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Collects two types of mappings by walking the entire AST:
        1.  An 'alias map' for preprocessing: {normalized_alias -> target_name}
            - For simple tables: {'t1': 'Person'}
            - For derived tables: {'t3': 'T3'}
            - For CTEs: {'yearly_set_counts': 'yearly_set_counts'}
        2.  A 'named query map' (CTEs and Derived Tables) for user output: {defined_name -> subquery_sql}
            - {'T3': 'SELECT ...'} (Derived Table)
            - {'yearly_set_counts': 'SELECT ...'} (CTE)

        Args:
            ast: The root AST node of the parsed SQL query.

        Returns:
            Tuple[alias_map, named_query_map]
        """
        alias_map = {} # 供 _add_table_names_to_columns 内部使用
        named_query_map = {} # 供用户（调用者）使用
        
        # 从根节点遍历整个 AST
        for node in ast.walk():
            
            # 查找所有带别名的节点 (Table, Subquery, CTE)
            if not (isinstance(node, (expressions.Table, expressions.Subquery, expressions.CTE)) and node.alias):
                continue

            # str(node.alias) 适用于 TableAlias (AS x) 和 Identifier (CTE 'x')
            defined_alias = str(node.alias)
            
            # 规范化别名，用作 'alias_map' 的键
            normalized_alias_key = self._normalize_identifier(defined_alias)
            if not normalized_alias_key:
                continue

            if isinstance(node, expressions.Table):
                # 案例 1: 简单表别名 (Person AS T1)
                target_name = str(node.name)
                # 映射 't1' -> 'Person'
                alias_map[normalized_alias_key] = target_name
            
            elif isinstance(node, (expressions.Subquery, expressions.CTE)):
                # 案例 2: 派生表 ((...) AS T3)
                # 案例 3: CTE (WITH yearly_set_counts AS (...))
                
                # 1. 更新 alias_map 供内部使用
                # 映射 't3' -> 'T3'
                # 映射 'yearly_set_counts' (norm) -> 'yearly_set_counts' (def)
                alias_map[normalized_alias_key] = defined_alias 

                # 2. 更新 named_query_map 供用户使用
                try:
                    # 'node.this' 是 Subquery/CTE 内部的 Select 节点
                    subquery_sql = node.this.sql(dialect='sqlite')
                    named_query_map[defined_alias] = subquery_sql
                except Exception as e:
                    logger.error(f"Failed to extract SQL for {type(node).__name__} {defined_alias}: {e}")
                    named_query_map[defined_alias] = f"[Error extracting SQL: {e}]"
            
        return alias_map, named_query_map

    def _build_table_context(self, select_node: expressions.Select, 
                            table_aliases: Dict[str, str]) -> Dict[str, Set[str]]:
        """
        Build table context for each scope in the query.
        
        Returns:
            Dictionary mapping scope_id -> set of available table names
        """
        context = {}
        available_tables = set()
        
        # Get FROM table
        from_node = select_node.args.get('from')
        if from_node and hasattr(from_node, 'this'):
            table_expr = from_node.this
            if isinstance(table_expr, expressions.Table):
                table_name = str(table_expr.name)
                available_tables.add(self._normalize_identifier(table_name) or table_name)
        
        # Get JOIN tables
        for join_node in select_node.find_all(expressions.Join):
            if hasattr(join_node, 'this') and isinstance(join_node.this, expressions.Table):
                table_name = str(join_node.this.name)
                available_tables.add(self._normalize_identifier(table_name) or table_name)
        
        # Store context for main SELECT
        context['main'] = available_tables
        
        return context
    
    def _add_table_names_to_columns(self, select_node: expressions.Select,
                                   table_context: Dict[str, Set[str]],
                                   table_aliases: Dict[str, str]) -> None:
        """
        Add table names to columns that don't have them.
        
        Strategy:
        1. If schema is available: use schema-based resolution (more accurate)
        2. If no schema: use scope-based inference (fallback)
        """
        if self.schema:
            logger.debug("Using schema-based column resolution")
            self._add_table_names_with_schema(select_node, table_context, table_aliases)
        else:
            logger.debug("Using scope-based column resolution")
            self._add_table_names_with_scope(select_node, table_context, table_aliases)
    
    def _add_table_names_with_scope(self, select_node: expressions.Select,
                                    table_context: Dict[str, Set[str]],
                                    table_aliases: Dict[str, str]) -> None:
        """
        Add table names to columns using scope-based inference.
        
        Uses sqlglot's scope builder to infer table names for columns.
        
        Strategy:
        1. Use sqlglot's build_scope to get column-to-table mappings
        2. For columns without table names, use the inferred table from scope
        3. Replace aliases with actual table names
        """
        available_tables = table_context.get('main', set())
        
        # Build scope to get column-to-table mappings
        try:
            root_scope = build_scope(select_node)
            column_table_mapping = {}
            
            def collect_column_mappings(scope):
                """Recursively collect column mappings from scope"""
                scope_tables = {}
                for table in scope.tables:
                    table_name = str(table.alias or table.name)
                    actual_table_name = str(table.name)
                    scope_tables[table_name] = actual_table_name
                
                for column in scope.columns:
                    column_node = column.expression
                    if isinstance(column_node, expressions.Column):
                        column_id = id(column_node)
                        table_ref = None
                        
                        if column_node.table:
                            table_ref = str(column_node.table)
                            if table_ref in scope_tables:
                                table_ref = scope_tables[table_ref]
                        elif column.table:
                            table_name = str(column.table.alias or column.table.name)
                            if table_name in scope_tables:
                                table_ref = scope_tables[table_name]
                            else:
                                table_ref = str(column.table.name)
                        elif len(scope_tables) == 1:
                            table_ref = list(scope_tables.values())[0]
                        
                        if table_ref:
                            column_table_mapping[column_id] = table_ref
                
                # Recursively process child scopes
                for child in getattr(scope, "children", []):
                    collect_column_mappings(child)
            
            collect_column_mappings(root_scope)
        except Exception as e:
            logger.debug(f"Scope building failed, using fallback method: {e}")
            column_table_mapping = {}
        
        def process_column(column_node: expressions.Column) -> None:
            column_id = id(column_node)
            
            if not column_node.table:
                # Column has no table name - try to infer from scope
                if column_id in column_table_mapping:
                    inferred_table = column_table_mapping[column_id]
                    column_node.set('table', inferred_table)
                    logger.debug(f"Inferred table '{inferred_table}' for column '{column_node.name}' from scope")
                elif len(available_tables) == 1:
                    # Only one table available, use it
                    table_name = list(available_tables)[0]
                    column_node.set('table', table_name)
                    logger.debug(f"Assigned table '{table_name}' to column '{column_node.name}' (single table)")
                elif len(available_tables) > 1:
                    # Multiple tables - use first table as fallback
                    table_name = list(available_tables)[0]
                    column_node.set('table', table_name)
                    logger.warning(f"Assigned table '{table_name}' to column '{column_node.name}' (multiple tables, using first)")
            else:
                # Column has a table reference - check if it's an alias
                table_ref = str(column_node.table)
                normalized_ref = self._normalize_identifier(table_ref)
                
                # Check if it's an alias
                if table_ref in table_aliases:
                    column_node.set('table', table_aliases[table_ref])
                elif normalized_ref and normalized_ref in table_aliases:
                    # Case-insensitive match
                    column_node.set('table', table_aliases[normalized_ref])
                else:
                    # Check if it's an actual table name (normalize and check)
                    normalized_table_ref = self._normalize_identifier(table_ref)
                    if normalized_table_ref in available_tables:
                        # It's already a valid table name, keep it
                        pass
                    else:
                        # Try to find matching table (case-insensitive)
                        for table in available_tables:
                            if self._normalize_identifier(table) == normalized_table_ref:
                                column_node.set('table', table)
                                break
        
        # Process all columns in the query
        for node in select_node.walk():
            if isinstance(node, expressions.Column):
                process_column(node)
    
    def _add_table_names_with_schema(self, select_node: expressions.Select,
                                     table_context: Dict[str, Set[str]],
                                     table_aliases: Dict[str, str]) -> None:
        """
        Add table names to columns using database schema.
        
        Uses database schema to accurately resolve which table a column belongs to.
        This is more accurate than scope-based inference but requires schema information.
        
        Strategy:
        1. Extract tables used in the query
        2. For each column, resolve its table using schema
        3. Handle aliases properly
        """
        available_tables = table_context.get('main', set())
        
        # Extract tables used in query
        tables_in_query = set()
        for node in select_node.walk():
            if isinstance(node, expressions.Table):
                table_name = self._normalize_identifier(str(node.name))
                if table_name:
                    tables_in_query.add(table_name)
        
        def resolve_column_table(col_name: str, table_ref: Optional[str]) -> Optional[str]:
            """Resolve which table a column belongs to"""
            col_name_lower = self._normalize_identifier(col_name)
            
            # If table_ref is provided, resolve it (could be alias)
            if table_ref:
                table_ref_lower = self._normalize_identifier(table_ref)
                # Check if it's an alias
                if table_ref_lower in table_aliases:
                    return table_aliases[table_ref_lower]
                # Otherwise, it's the actual table name
                return table_ref_lower
            
            # No table reference provided, try to resolve from schema
            if not self.schema:
                return None
            
            possible_tables = self.schema.get(col_name_lower, set())
            
            # Filter to only tables that are actually in the query
            possible_tables = possible_tables.intersection(tables_in_query)
            
            if len(possible_tables) == 1:
                # Only one table has this column and it's in the query
                return list(possible_tables)[0]
            elif len(possible_tables) > 1:
                # Ambiguous: multiple tables have this column
                logger.debug(f"Ambiguous column '{col_name}': found in tables {possible_tables}")
                # Return the first one (could be improved with better heuristics)
                return list(possible_tables)[0]
            else:
                # Column not found in schema or not in any table in the query
                logger.debug(f"Could not resolve table for column '{col_name}'")
                return None
        
        # Process all column nodes
        for node in select_node.walk():
            if isinstance(node, expressions.Column):
                col_name = str(node.name)
                if not col_name:
                    continue
                
                # Get current table reference
                table_ref = None
                if node.table:
                    if isinstance(node.table, str):
                        table_ref = node.table
                    elif hasattr(node.table, 'name'):
                        table_ref = node.table.name
                    elif hasattr(node.table, 'this'):
                        table_ref = node.table.this if isinstance(node.table.this, str) else str(node.table.this)
                    else:
                        table_ref = str(node.table)
                
                # Resolve table name
                resolved_table = resolve_column_table(col_name, table_ref)
                
                # Set the table name on the column node
                if resolved_table:
                    node.set('table', resolved_table)
                elif not table_ref:
                    # Column has no table and we couldn't resolve it
                    # Leave it as is (unqualified)
                    logger.debug(f"Could not qualify column '{col_name}', leaving unqualified")
                else:
                    # table_ref exists but couldn't be resolved (e.g., invalid alias)
                    # Keep the original table reference
                    logger.debug(f"Could not resolve table for column '{col_name}' with table_ref '{table_ref}', keeping original")
    
    def _replace_aliases_with_table_names(self, select_node: expressions.Select,
                                         table_aliases: Dict[str, str]) -> None:
        """
        Replace all table alias references with actual table names.
        
        This is already handled in _add_table_names_to_columns, but we do it
        again here to be thorough.
        """
        for node in select_node.walk():
            if isinstance(node, expressions.Column) and node.table:
                table_ref = str(node.table)
                normalized_ref = self._normalize_identifier(table_ref)
                
                if table_ref in table_aliases:
                    node.set('table', table_aliases[table_ref])
                elif normalized_ref and normalized_ref in table_aliases:
                    node.set('table', table_aliases[normalized_ref])
    
    def _remove_table_aliases(self, select_node: expressions.Select) -> None:
        """Remove alias declarations from all table nodes"""
        for node in select_node.walk():
            if isinstance(node, expressions.Table) and node.alias:
                node.set('alias', None)
    
    # =========================================================================
    # Public API Methods
    # =========================================================================
    
    def qualify_sql(self, sql: str) -> str:
        """
        Normalize SQL by resolving table aliases and qualifying all columns.
        
        This method provides compatibility with sql_qualifier.qualify_sql().
        
        Args:
            sql: SQL query string
            
        Returns:
            Normalized SQL with:
            - All table aliases replaced with actual table names
            - All columns qualified with table names (table.column format)
            - Alias declarations (AS xxx) removed
        """
        result_sql, _, _ = self.preprocess(sql)
        return result_sql
    
    def normalize_sql_aliases(self, sql: str) -> Tuple[str, Dict[str, str]]:
        """
        将SQL中的别名替换为实际表名，并移除 "AS xxx" 别名声明
        
        Args:
            sql: 原始SQL字符串
            
        Returns:
            (normalized_sql, alias_mapping): 标准化后的SQL和别名映射字典
        """
        try:
            # 预处理：修复LIKE子句中的通配符问题
            # sqlglot可能将LIKE '%pattern%'中的%误解析为模运算符
            # 我们需要确保LIKE子句中的通配符被正确处理
            preprocessed_sql = self._preprocess_sql_for_parsing(sql)
            
            ast = sqlglot.parse_one(preprocessed_sql, dialect='sqlite')
            table_aliases = {}
            
            # 收集所有表别名映射，并移除别名
            for node in ast.walk():
                if isinstance(node, expressions.Table):
                    table_name = str(node.name)
                    if node.alias:
                        alias = str(node.alias)
                        table_aliases[alias] = table_name
                        # 移除别名
                        node.set('alias', None)
            
            # 在AST中替换所有别名引用为实际表名
            for node in ast.walk():
                if isinstance(node, expressions.Column):
                    if node.table:
                        table_ref = str(node.table)
                        if table_ref in table_aliases:
                            node.set('table', table_aliases[table_ref])
            
            # 重新生成SQL（已经移除了别名声明，并替换了别名引用）
            normalized_sql = ast.sql()
            
            logger.debug(f"SQL别名标准化: {table_aliases}")
            return normalized_sql, table_aliases
            
        except Exception as e:
            logger.error(f"SQL别名标准化失败: {e}")
            logger.debug(f"原始SQL: {sql}")
            # 如果解析失败，返回原始SQL（不进行别名标准化）
            return sql, {}
    
    def extract_sql_schema(self, sql: str) -> Dict[str, Any]:
        """
        提取SQL schema信息，包括表名、列、条件和连接
        
        使用 sqlglot.optimizer.scope 自动解析列的来源，简化代码并提高准确性。
        
        Args:
            sql: SQL查询字符串
            
        Returns:
            {
                'tables': [table_names],
                'columns': [{'table': table_name, 'column': column_name}],
                'conditions': [{'table': table_name, 'column': column_name, 'operator': op, 'value': val}],
                'joins': [join_info]
            }
        """
        try:
            # 预处理SQL，修复可能的解析问题
            preprocessed_sql = self._preprocess_sql_for_parsing(sql)
            ast = sqlglot.parse_one(preprocessed_sql, dialect='sqlite')
            
            tables = set()
            columns = []
            conditions = []
            joins = []
            table_aliases = {}
            
            # 首先收集所有表信息和别名映射
            for node in ast.walk():
                if isinstance(node, expressions.Table):
                    table_name = str(node.name)
                    # 规范化表名（根据大小写敏感性）
                    normalized_table = self._normalize_identifier(table_name)
                    tables.add(normalized_table)
                    
                    if node.alias:
                        table_aliases[str(node.alias)] = normalized_table
            
            # 使用 scope 自动解析列的来源
            column_table_mapping: Dict[int, Optional[str]] = {}
            try:
                root_scope = build_scope(ast)

                # 遍历所有 scope 来建立列到表的映射
                def collect_column_mappings(scope):
                    """递归收集列映射"""
                    # 获取当前 scope 中的表
                    scope_tables = {}
                    for table in scope.tables:
                        table_name = str(table.alias or table.name)
                        normalized_table = self._normalize_identifier(str(table.name))
                        scope_tables[table_name] = normalized_table
                        if table.alias:
                            table_aliases[table_name] = normalized_table

                    # 处理当前 scope 中的列
                    for column in scope.columns:
                        column_node = column.expression
                        column_name = str(column_node.name)

                        table_ref = None
                        if column_node.table:
                            table_ref = str(column_node.table)
                            if table_ref in table_aliases:
                                table_ref = table_aliases[table_ref]
                            elif table_ref in scope_tables:
                                table_ref = scope_tables[table_ref]
                        elif column.table:
                            table_name = str(column.table.alias or column.table.name)
                            if table_name in table_aliases:
                                table_ref = table_aliases[table_name]
                            elif table_name in scope_tables:
                                table_ref = scope_tables[table_name]
                            else:
                                table_ref = self._normalize_identifier(str(column.table.name))
                        elif len(scope_tables) == 1:
                            table_ref = list(scope_tables.values())[0]

                        column_table_mapping[id(column_node)] = table_ref

                    # 递归处理子 scope
                    for child in getattr(scope, "children", []):
                        collect_column_mappings(child)

                collect_column_mappings(root_scope)
            except Exception as scope_error:
                # 如果 scope 解析失败，回退到原来的方法
                logger.debug(f"Scope解析失败，使用回退方法: {scope_error}")
                column_table_mapping = self._build_column_table_mapping_fallback(ast, table_aliases)
            
            # 收集所有JOIN ON子句中的条件节点ID，避免将它们作为普通条件提取
            join_condition_nodes = set()
            for node in ast.walk():
                if isinstance(node, expressions.Join) and hasattr(node, 'args') and 'on' in node.args and node.args['on']:
                    on_condition = node.args['on']
                    
                    # 收集JOIN条件节点ID
                    if hasattr(on_condition, 'walk'):
                        for condition_node in on_condition.walk():
                            if isinstance(condition_node, (expressions.EQ, expressions.NEQ, expressions.GT, 
                                                         expressions.GTE, expressions.LT, expressions.LTE,
                                                         expressions.Like, expressions.In)):
                                join_condition_nodes.add(id(condition_node))
                    else:
                        # If it's a single condition node, add it directly
                        if isinstance(on_condition, (expressions.EQ, expressions.NEQ, expressions.GT, 
                                                   expressions.GTE, expressions.LT, expressions.LTE,
                                                   expressions.Like, expressions.In)):
                            join_condition_nodes.add(id(on_condition))
            
            # 提取列信息
            processed_columns = set()
            for node in ast.walk():
                if isinstance(node, expressions.Column):
                    column_name = str(node.name)
                    table_ref = column_table_mapping.get(id(node))
                    
                    # 规范化表名和列名（根据大小写敏感性）
                    normalized_table = self._normalize_identifier(table_ref)
                    normalized_column = self._normalize_identifier(column_name)
                    
                    # 避免重复添加相同的列
                    column_entry = {'table': normalized_table, 'column': normalized_column}
                    column_key = (normalized_table, normalized_column)
                    if column_key not in processed_columns:
                        columns.append(column_entry)
                        processed_columns.add(column_key)
                
                elif isinstance(node, (expressions.EQ, expressions.NEQ, expressions.GT, 
                                     expressions.GTE, expressions.LT, expressions.LTE,
                                     expressions.Like, expressions.In)):
                    # 只提取不在JOIN ON子句中的条件
                    if id(node) not in join_condition_nodes:
                        condition_info = self._extract_condition_info_v2(node, table_aliases, column_table_mapping)
                        if condition_info:
                            # 规范化条件中的表名和列名
                            if condition_info.get('table'):
                                condition_info['table'] = self._normalize_identifier(condition_info['table'])
                            if condition_info.get('column'):
                                condition_info['column'] = self._normalize_identifier(condition_info['column'])
                            conditions.append(condition_info)
                
                elif isinstance(node, expressions.Join):
                    join_info = self._extract_join_info(node, table_aliases)
                    if join_info:
                        # 规范化 JOIN 条件中的表名和列名
                        for condition in join_info.get('conditions', []):
                            if condition.get('table'):
                                condition['table'] = self._normalize_identifier(condition['table'])
                            if condition.get('column'):
                                condition['column'] = self._normalize_identifier(condition['column'])
                        joins.append(join_info)
            
            return {
                'tables': sorted(list(tables)),
                'columns': columns,
                'conditions': conditions,
                'joins': joins
            }
            
        except Exception as e:
            logger.error(f"SQLglot parsing failed: {e}")
            return {
                'tables': [],
                'columns': [],
                'conditions': [],
                'joins': []
            }
    
    def _build_column_table_mapping_fallback(self, ast, table_aliases: Dict[str, str]) -> Dict[int, Optional[str]]:
        """
        回退方法：当 scope 解析失败时，使用原来的逻辑构建列到表的映射
        
        Args:
            ast: SQL AST
            table_aliases: 表别名映射
            
        Returns:
            列节点ID到表名的映射
        """
        column_table_mapping = {}
        select_nodes = list(ast.find_all(expressions.Select))
        
        for select_node in select_nodes:
            from_tables = []
            if hasattr(select_node, 'args') and 'from' in select_node.args:
                from_clause = select_node.args['from']
                if from_clause and isinstance(from_clause, expressions.From):
                    table_expr = from_clause.this
                    if isinstance(table_expr, expressions.Table):
                        table_name = str(table_expr.name)
                        from_tables.append(table_name)
            
            # 处理SELECT列表中的列
            if hasattr(select_node, 'expressions'):
                for expr in select_node.expressions:
                    for column_node in expr.find_all(expressions.Column):
                        if id(column_node) not in column_table_mapping:
                            if column_node.table:
                                table_ref = str(column_node.table)
                                if table_ref in table_aliases:
                                    table_ref = table_aliases[table_ref]
                                column_table_mapping[id(column_node)] = table_ref
                            elif len(from_tables) == 1:
                                column_table_mapping[id(column_node)] = from_tables[0]
                            else:
                                column_table_mapping[id(column_node)] = None
            
            # 处理WHERE、ORDER BY、GROUP BY、HAVING中的列
            for part in ['where', 'order', 'group', 'having']:
                if hasattr(select_node, 'args') and part in select_node.args and select_node.args[part]:
                    for column_node in select_node.args[part].find_all(expressions.Column):
                        if id(column_node) not in column_table_mapping:
                            if column_node.table:
                                table_ref = str(column_node.table)
                                if table_ref in table_aliases:
                                    table_ref = table_aliases[table_ref]
                                column_table_mapping[id(column_node)] = table_ref
                            elif len(from_tables) == 1:
                                column_table_mapping[id(column_node)] = from_tables[0]
                            else:
                                column_table_mapping[id(column_node)] = None
            
            # 处理JOIN ON条件中的列
            for join_node in select_node.find_all(expressions.Join):
                if hasattr(join_node, 'args') and 'on' in join_node.args and join_node.args['on']:
                    on_condition = join_node.args['on']
                    if hasattr(on_condition, 'find_all'):
                        for column_node in on_condition.find_all(expressions.Column):
                            if id(column_node) not in column_table_mapping:
                                if column_node.table:
                                    table_ref = str(column_node.table)
                                    if table_ref in table_aliases:
                                        table_ref = table_aliases[table_ref]
                                    column_table_mapping[id(column_node)] = table_ref
                                else:
                                    column_table_mapping[id(column_node)] = None
        
        return column_table_mapping
    
    def _extract_condition_info_v2(self, condition_node, table_aliases: Dict[str, str], column_table_mapping: Dict[int, Optional[str]]) -> Optional[Dict[str, Any]]:
        """从条件节点提取信息（v2版本，使用column_table_mapping）"""
        try:
            left = condition_node.left
            right = condition_node.right
            operator = type(condition_node).__name__.lower()
            
            if isinstance(left, expressions.Column):
                column_name = str(left.name)
                table_ref = column_table_mapping.get(id(left))
                
                # 提取值
                value = None
                # 检查是否是子查询（Subquery）
                if isinstance(right, expressions.Subquery):
                    # 对于子查询，使用 sql() 方法生成 SQL 字符串
                    try:
                        subquery_sql = right.sql(dialect='sqlite')
                        # 对于schema提取，直接使用子查询SQL，不进行规范化
                        value = subquery_sql
                    except Exception:
                        value = str(right)
                elif hasattr(right, 'this'):
                    value = str(right.this)
                elif hasattr(right, 'name'):
                    value = str(right.name)
                else:
                    value = str(right)
                
                return {
                    'table': table_ref,
                    'column': column_name,
                    'operator': operator,
                    'value': value
                }
        except Exception as e:
            logger.debug(f"Failed to extract condition info v2: {e}")
        
        return None
    
    def _extract_join_info(self, join_node, table_aliases: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """从JOIN节点提取信息"""
        try:
            # Get join type from args
            join_kind = join_node.args.get('kind', 'INNER') if hasattr(join_node, 'args') else 'INNER'
            
            join_info = {
                'type': join_kind.lower(),
                'conditions': []
            }
            
            # Access the ON condition correctly
            if hasattr(join_node, 'args') and 'on' in join_node.args and join_node.args['on']:
                on_condition = join_node.args['on']
                
                # 对于JOIN的条件提取，我们需要一个空的table_context_map
                # 因为JOIN条件通常都有明确的表引用
                empty_context_map = {}
                
                # Handle single condition or walk through complex conditions
                if hasattr(on_condition, 'walk'):
                    for condition in on_condition.walk():
                        if isinstance(condition, (expressions.EQ, expressions.NEQ)):
                            cond_info = self._extract_condition_info(condition, table_aliases, empty_context_map)
                            if cond_info:
                                join_info['conditions'].append(cond_info)
                else:
                    # Single condition case
                    if isinstance(on_condition, (expressions.EQ, expressions.NEQ)):
                        cond_info = self._extract_condition_info(on_condition, table_aliases, empty_context_map)
                        if cond_info:
                            join_info['conditions'].append(cond_info)
            
            return join_info if join_info['conditions'] else None
            
        except Exception as e:
            logger.debug(f"Failed to extract join info: {e}")
        
        return None
    
    def _extract_condition_info(self, condition_node, table_aliases: Dict[str, str], table_context_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """从条件节点提取信息"""
        try:
            left = condition_node.left
            right = condition_node.right
            operator = type(condition_node).__name__.lower()
            
            if isinstance(left, expressions.Column):
                table_ref = None
                column_name = str(left.name)
                
                if left.table:
                    # 显式指定了表名
                    table_ref = str(left.table)
                    if table_ref in table_aliases:
                        table_ref = table_aliases[table_ref]
                else:
                    # 尝试从上下文推断表名
                    table_ref = self._infer_table_for_column(left, table_context_map)
                
                # 提取值
                value = None
                if hasattr(right, 'this'):
                    value = str(right.this)
                elif hasattr(right, 'name'):
                    value = str(right.name)
                else:
                    value = str(right)
                
                return {
                    'table': table_ref,
                    'column': column_name,
                    'operator': operator,
                    'value': value
                }
        except Exception as e:
            logger.debug(f"Failed to extract condition info: {e}")
        
        return None
    
    def _infer_table_for_column(self, column_node, table_context_map: Dict[str, str]) -> Optional[str]:
        """
        根据上下文推断列所属的表
        
        Args:
            column_node: 列节点
            table_context_map: 表上下文映射
            
        Returns:
            推断的表名，如果无法推断则返回None
        """
        # 向上遍历找到最近的SELECT节点
        parent = column_node.parent
        while parent:
            if isinstance(parent, expressions.Select):
                context_id = id(parent)
                if context_id in table_context_map:
                    return table_context_map[context_id]
                break
            parent = parent.parent
        
        return None
    
    def calculate_complexity_score(self, sql: str) -> int:
        """
        Calculate SQL complexity score based on AST structure.
        
        Uses heuristic rules to assign a complexity score:
        - Window Functions: +5 (very complex)
        - CTE (WITH clauses): +3 (multi-step logic)
        - Subqueries: +3 each (nested logic)
        - Set Operations (UNION/INTERSECT): +3 (query merging)
        - Joins: +1 each (entity relationships)
        - HAVING: +1 (post-aggregation filtering)
        
        Args:
            sql: SQL query string
            
        Returns:
            Complexity score (0-100+). Higher score means more complex.
            Returns 100 if parsing fails (treat as very complex).
        """
        try:
            # Preprocess SQL for parsing
            preprocessed_sql = self._preprocess_sql_for_parsing(sql)
            
            # Parse SQL AST
            parsed_ast = sqlglot.parse_one(preprocessed_sql, dialect='sqlite', error_level=sqlglot.ErrorLevel.IGNORE)
            
            if parsed_ast is None:
                logger.warning("Failed to parse SQL for complexity calculation, treating as very complex")
                return 100
                
        except Exception as e:
            logger.warning(f"SQL parsing failed for complexity calculation: {e}, treating as very complex")
            return 100
        
        score = 0
        
        # 1. Check highest priority complex operations (high scores)
        # Window Functions (OVER ...)
        if parsed_ast.find(expressions.Window):
            score += 5
            logger.debug("Found Window Function: +5")
        
        # CTE (WITH ...)
        if parsed_ast.find(expressions.CTE):
            score += 3
            logger.debug("Found CTE: +3")
        
        # Set Operations (UNION, INTERSECT, EXCEPT)
        set_ops = list(parsed_ast.find_all(expressions.Union))
        set_ops.extend(list(parsed_ast.find_all(expressions.Intersect)))
        set_ops.extend(list(parsed_ast.find_all(expressions.Except)))
        if set_ops:
            score += len(set_ops) * 3
            logger.debug(f"Found {len(set_ops)} Set Operation(s): +{len(set_ops) * 3}")
        
        # 2. Check subqueries
        # Find all subqueries, but exclude those inside CTE definitions (already counted)
        all_subqueries = list(parsed_ast.find_all(expressions.Subquery))
        non_cte_subqueries = []
        for subq in all_subqueries:
            # Check if this subquery is inside a CTE definition
            parent = subq.parent
            is_in_cte = False
            while parent:
                if isinstance(parent, expressions.CTE):
                    is_in_cte = True
                    break
                parent = getattr(parent, 'parent', None)
            
            if not is_in_cte:
                non_cte_subqueries.append(subq)
        
        if non_cte_subqueries:
            score += len(non_cte_subqueries) * 3
            logger.debug(f"Found {len(non_cte_subqueries)} Subquery(ies): +{len(non_cte_subqueries) * 3}")
        
        # 3. Check Joins and other medium complexity operations
        joins = list(parsed_ast.find_all(expressions.Join))
        if joins:
            score += len(joins) * 1
            logger.debug(f"Found {len(joins)} Join(s): +{len(joins)}")
        
        # HAVING clause (post-aggregation filtering)
        if parsed_ast.find(expressions.Having):
            score += 1
            logger.debug("Found HAVING: +1")
        
        logger.debug(f"SQL complexity score: {score}")
        return score

