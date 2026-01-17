# SQL Qualifier - Normalize SQL by resolving table aliases and qualifying all columns
#
# DEPRECATED: This module has been integrated into SQLPreprocessor.
# Please use caf.preprocess.sql_preprocessor.SQLPreprocessor instead.
#
# This module is kept for backward compatibility and will be removed in a future version.

import json
import logging
import sqlite3
import warnings
from pathlib import Path
from typing import Dict, Set, Optional, Tuple
import sqlglot
from sqlglot import expressions
from sqlglot.optimizer.scope import build_scope

from caf.config.paths import PathConfig

logger = logging.getLogger(__name__)

# Deprecation warning for the module
warnings.warn(
    "The sql_qualifier module is deprecated and will be removed in a future version. "
    "Please use caf.preprocess.sql_preprocessor.SQLPreprocessor instead.",
    DeprecationWarning,
    stacklevel=2
)


def get_schema_file_path(db_path: str) -> Path:
    """
    Get the schema file path for a database.
    
    Schema file is stored in the same directory as the database file,
    with name: {database_name}.schema.json
    
    Args:
        db_path: Database file path
        
    Returns:
        Path to schema file
    """
    db_path_obj = Path(db_path)
    schema_file = db_path_obj.parent / f"{db_path_obj.stem}.schema.json"
    return schema_file


def load_schema_from_file(schema_file: Path) -> Optional[Dict[str, Set[str]]]:
    """
    Load schema from JSON file.
    
    Args:
        schema_file: Path to schema JSON file
        
    Returns:
        Schema dict if file exists and is valid, None otherwise
    """
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


def save_schema_to_file(schema: Dict[str, Set[str]], schema_file: Path) -> bool:
    """
    Save schema to JSON file.
    
    Args:
        schema: Schema dict mapping column_name to set of table names
        schema_file: Path to save schema file
        
    Returns:
        True if saved successfully, False otherwise
    """
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


def load_schema_from_db_path(db_path: str) -> Dict[str, Set[str]]:
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


def load_schema_with_cache(db_path: str, use_cache: bool = True) -> Dict[str, Set[str]]:
    """
    Load schema from database, with optional caching to file.
    
    Priority:
    1. Load from schema file if exists and use_cache is True
    2. Load from database and save to file if use_cache is True
    
    Args:
        db_path: Database file path
        use_cache: Whether to use/save schema cache file
        
    Returns:
        Schema dict mapping column_name (lowercase) to set of table names (lowercase)
    """
    schema_file = get_schema_file_path(db_path)
    
    # Try to load from cache file first
    if use_cache:
        cached_schema = load_schema_from_file(schema_file)
        if cached_schema is not None:
            return cached_schema
    
    # Load from database
    schema = load_schema_from_db_path(db_path)
    
    # Save to cache file if enabled
    if use_cache and schema:
        save_schema_to_file(schema, schema_file)
    
    return schema


def resolve_database_path(database_id: str) -> Optional[str]:
    """
    Resolve database_id to database_path using mapping file.
    
    Args:
        database_id: Database identifier
        
    Returns:
        Database file path if found, None otherwise
    """
    mapping_path = PathConfig.get_database_mapping_path()
    
    if not mapping_path.exists():
        logger.warning(f"Database mapping file not found at {mapping_path}")
        return None
    
    try:
        import json
        with mapping_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        # raw: {"/abs/path/to/db.sqlite": "database_id", ...}
        for abs_path, db_id in raw.items():
            if str(db_id) == str(database_id):
                return abs_path
    except Exception as e:
        logger.error(f"Failed to load database mapping from {mapping_path}: {e}")
        return None
    
    return None


def get_schema(
    database_id: Optional[str] = None, 
    schema: Optional[Dict[str, Set[str]]] = None, 
    db_path: Optional[str] = None,
    use_cache: bool = True
) -> Dict[str, Set[str]]:
    """
    Get schema from various sources with caching support.
    
    Priority:
    1. Use provided schema if available
    2. Load from cache file if exists (when use_cache=True)
    3. Load from db_path if provided
    4. Resolve database_id to db_path and load schema
    
    Args:
        database_id: Database identifier
        schema: Pre-loaded schema dict
        db_path: Database file path
        use_cache: Whether to use/save schema cache file (default: True)
        
    Returns:
        Schema dict mapping column_name (lowercase) to set of table names (lowercase)
    """
    if schema is not None:
        return schema
    
    if db_path is None and database_id is not None:
        db_path = resolve_database_path(database_id)
        if db_path is None:
            logger.warning(f"Could not resolve database path for database_id: {database_id}")
            return {}
    
    if db_path is not None:
        return load_schema_with_cache(db_path, use_cache=use_cache)
    
    return {}


def extract_table_aliases(ast) -> Dict[str, str]:
    """
    Extract table alias mappings from SQL AST.
    
    Returns:
        Dict mapping alias (lowercase) to actual table name (lowercase)
    """
    aliases = {}
    
    for node in ast.walk():
        if isinstance(node, expressions.Table):
            table_name = str(node.name).lower()
            if node.alias:
                alias = str(node.alias).lower()
                aliases[alias] = table_name
    
    return aliases


def extract_tables_in_query(ast) -> Set[str]:
    """
    Extract all table names used in the query (including from JOINs).
    
    Returns:
        Set of table names (lowercase)
    """
    tables = set()
    
    for node in ast.walk():
        if isinstance(node, expressions.Table):
            table_name = str(node.name).lower()
            tables.add(table_name)
    
    return tables


def resolve_column_table(
    col_name: str,
    table_ref: Optional[str],
    aliases: Dict[str, str],
    schema: Dict[str, Set[str]],
    tables_in_query: Set[str]
) -> Optional[str]:
    """
    Resolve which table a column belongs to.
    
    Args:
        col_name: Column name
        table_ref: Table reference (could be alias or actual table name)
        aliases: Mapping of aliases to actual table names
        schema: Schema dict mapping column_name to set of table names
        tables_in_query: Set of table names used in the query
        
    Returns:
        Resolved table name (lowercase) if found, None otherwise
    """
    col_name_lower = col_name.lower()
    
    # If table_ref is provided, resolve it (could be alias)
    if table_ref:
        table_ref_lower = table_ref.lower()
        # Check if it's an alias
        if table_ref_lower in aliases:
            return aliases[table_ref_lower]
        # Otherwise, it's the actual table name
        return table_ref_lower
    
    # No table reference provided, try to resolve from schema
    if not schema:
        return None
    
    possible_tables = schema.get(col_name_lower, set())
    
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


def qualify_sql(
    sql: str,
    schema: Optional[Dict[str, Set[str]]] = None,
    database_id: Optional[str] = None,
    db_path: Optional[str] = None,
    use_cache: bool = True
) -> str:
    """
    DEPRECATED: Use SQLPreprocessor.qualify_sql() instead.
    
    Normalize SQL by:
    1. Resolving table aliases to actual table names
    2. Qualifying all columns to table.column format
    
    Args:
        sql: SQL query string
        schema: Optional schema dict mapping column_name (lowercase) to set of table names (lowercase).
                If not provided, will be loaded from db_path or database_id.
        database_id: Optional database identifier. Used to resolve database path if schema not provided.
        db_path: Optional database file path. Used to load schema if schema not provided.
        use_cache: Whether to use/save schema cache file (default: True)
        
    Returns:
        Normalized SQL with all columns in table.column format and aliases resolved
    """
    # Issue deprecation warning
    warnings.warn(
        "qualify_sql() is deprecated and will be removed in a future version. "
        "Please use SQLPreprocessor.qualify_sql() from caf.preprocess.sql_preprocessor instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Redirect to SQLPreprocessor for actual functionality
    try:
        from caf.preprocess.sql_preprocessor import SQLPreprocessor
        
        preprocessor = SQLPreprocessor(
            case_sensitive=False,
            schema=schema,
            database_id=database_id,
            db_path=db_path,
            use_schema_cache=use_cache
        )
        return preprocessor.qualify_sql(sql)
    except Exception as e:
        logger.error(f"Failed to use SQLPreprocessor, falling back to legacy implementation: {e}")
        # Fall back to original implementation if SQLPreprocessor fails
        
    # Legacy implementation (kept as fallback)
    try:
        # Parse SQL
        parsed = sqlglot.parse_one(sql, dialect='sqlite', error_level=sqlglot.ErrorLevel.IGNORE)
        if parsed is None:
            logger.warning(f"Failed to parse SQL: {sql[:100]}...")
            return sql
        
        # Get schema (with caching support)
        schema_dict = get_schema(database_id=database_id, schema=schema, db_path=db_path, use_cache=use_cache)
        
        # Extract table aliases
        aliases = extract_table_aliases(parsed)
        
        # Extract tables used in query
        tables_in_query = extract_tables_in_query(parsed)
        
        # Build scope for better column resolution
        try:
            scope = build_scope(parsed)
        except Exception as e:
            logger.debug(f"Scope building failed: {e}")
            scope = None
        
        # Process all column nodes
        for node in parsed.find_all(expressions.Column):
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
            resolved_table = resolve_column_table(
                col_name=col_name,
                table_ref=table_ref,
                aliases=aliases,
                schema=schema_dict,
                tables_in_query=tables_in_query
            )
            
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
        
        # Remove table aliases from FROM/JOIN clauses
        for node in parsed.walk():
            if isinstance(node, expressions.Table) and node.alias:
                node.set('alias', None)
        
        # Generate normalized SQL
        normalized_sql = parsed.sql(dialect='sqlite')
        
        return normalized_sql
        
    except Exception as e:
        logger.error(f"Error qualifying SQL: {e}")
        logger.debug(f"Original SQL: {sql}")
        return sql


def generate_schema_cache(db_path: str, force_regenerate: bool = False) -> bool:
    """
    Generate and save schema cache file for a database.
    
    Args:
        db_path: Database file path
        force_regenerate: If True, regenerate even if cache file exists
        
    Returns:
        True if schema was generated successfully, False otherwise
    """
    schema_file = get_schema_file_path(db_path)
    
    # Check if cache already exists
    if not force_regenerate and schema_file.exists():
        logger.info(f"Schema cache already exists for {db_path}, skipping")
        return True
    
    # Load schema from database
    schema = load_schema_from_db_path(db_path)
    
    if not schema:
        logger.warning(f"No schema extracted from {db_path}")
        return False
    
    # Save to cache file
    success = save_schema_to_file(schema, schema_file)
    
    if success:
        logger.info(f"Generated schema cache for {db_path} -> {schema_file}")
    
    return success


def generate_schema_caches_from_mapping(force_regenerate: bool = False) -> Dict[str, bool]:
    """
    Generate schema cache files for all databases in the mapping file.
    
    Args:
        force_regenerate: If True, regenerate even if cache files exist
        
    Returns:
        Dict mapping database_id to success status (True/False)
    """
    mapping_path = PathConfig.get_database_mapping_path()
    
    if not mapping_path.exists():
        logger.warning(f"Database mapping file not found at {mapping_path}")
        return {}
    
    try:
        with mapping_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        
        results = {}
        for abs_path, db_id in raw.items():
            if not abs_path or not Path(abs_path).exists():
                logger.warning(f"Database file not found: {abs_path}")
                results[str(db_id)] = False
                continue
            
            success = generate_schema_cache(abs_path, force_regenerate=force_regenerate)
            results[str(db_id)] = success
        
        logger.info(f"Generated schema caches for {sum(results.values())}/{len(results)} databases")
        return results
        
    except Exception as e:
        logger.error(f"Failed to generate schema caches: {e}")
        return {}

