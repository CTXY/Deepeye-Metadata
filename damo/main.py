#!/usr/bin/env python3
# Simplified NL2SQL reasoning module with CAF integration
# This is a basic example showing how to use NL2SQL with CAF cognitive framework

import argparse
import json
import logging
import string
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

# Disable httpx INFO logs to avoid verbose HTTP request logging
# This must be set before any httpx client is initialized
logging.getLogger("httpx").setLevel(logging.WARNING)


# Add project root to path for imports
# damo/ is directly under project root, so parent.parent is the project root
sys.path.append(str(Path(__file__).parent.parent))


# Import other modules
from config import NL2SQLConfig, NL2SQLMode
from data_handler import BirdDataHandler
from engines.llm_engine import LLMEngine
from evaluation import NL2SQLEvaluator

# Direct CAF import - simplified interface, no internal types needed
import caf

# Inline data models (previously in types.py)

class NL2SQLQuery(BaseModel):
    """NL2SQL query request"""
    question: str
    db_id: str
    evidence: Optional[str] = None
    ground_truth_sql: Optional[str] = None
    use_caf: bool = True
    mode: NL2SQLMode = NL2SQLMode.LLM_ONLY
    session_id: Optional[str] = None
    max_retries: int = 3
    temperature: float = 0.0

class NL2SQLResult(BaseModel):
    """NL2SQL reasoning result"""
    generated_sql: str
    execution_success: bool
    execution_result: Optional[List[Dict[str, Any]]] = None
    generation_time_ms: int
    database_id: str
    session_id: Optional[str] = None

class NL2SQLResponse(BaseModel):
    """Complete NL2SQL response"""
    success: bool
    result: Optional[NL2SQLResult] = None
    error_message: Optional[str] = None
    feedback_collected: bool = False
    feedback_details: Optional[Dict[str, Any]] = None  # 添加反馈详情
    session_id: str
    database_id: str
    timestamp: datetime


def _format_schema_for_feedback(schema) -> str:
    """Format database schema for CAF feedback context"""
    schema_parts = []
    
    # Add database identifier
    schema_parts.append(f"Database: {schema.db_id}")
    schema_parts.append("")  # Empty line for better formatting
    
    # Add table schemas with CREATE TABLE format
    for table_id, table_name in enumerate(schema.table_names):
        table_columns = []
        
        # Get columns for this table
        for col_info in schema.column_names:
            if col_info["table_id"] == table_id:
                col_name = col_info["column_name"]
                col_type = ""
                
                # Find column type
                col_index = next((i for i, c in enumerate(schema.column_names) 
                                if c == col_info), None)
                if col_index is not None and col_index < len(schema.column_types):
                    col_type = schema.column_types[col_index]
                
                # Format column with type
                column_def = f"    {col_name} {col_type.upper()}"
                
                # Add description as comment if available
                if (schema.table_descriptions and 
                    table_name in schema.table_descriptions and 
                    col_name in schema.table_descriptions[table_name]):
                    description = schema.table_descriptions[table_name][col_name]
                    column_def += f" # {description}"
                
                table_columns.append(column_def)
        
        # Format as CREATE TABLE statement
        if table_columns:
            columns_str = ',\n'.join(table_columns)
            table_def = f"CREATE TABLE {table_name}(\n{columns_str}\n)"
        else:
            table_def = f"CREATE TABLE {table_name}()"
        
        schema_parts.append(table_def)
        schema_parts.append("")  # Empty line between tables
    
    return "\n".join(schema_parts)

def _normalize_question(question: str) -> str:
    """Normalize question for grouping (same logic as episodic_search.py)"""
    if not question:
        return ""
    normalized = ' '.join(question.strip().split()).lower()
    # Remove trailing punctuation
    while normalized and normalized[-1] in string.punctuation:
        normalized = normalized[:-1]
    return normalized

def _format_episodic_memory_for_llm(memory_items: List[Any], current_database_id: str, memory_type: str = "episodic") -> str:
    """Format episodic memory items into structured text for LLM consumption, grouped by question and database"""
    
    if not memory_items:
        return f"No episodic cases found."
    
    formatted_text = f"=== EPISODIC CASES FROM HISTORICAL INTERACTIONS ===\n\n"
    formatted_text += "Below are historical NL2SQL cases that can help you:\n"
    formatted_text += "- Learn from CORRECT examples to write better SQL. PAY attention to the similar patterns, details of historical logs to the existing task.\n"
    formatted_text += "- Avoid making the same mistakes as ERROR cases\n"
    formatted_text += "- Pay attention to Comparative Insights which explain the key differences between incorrect and correct approaches\n\n"
    
    # Separate items into same db and cross db groups
    same_db_items = []
    cross_db_items = []
    
    for item in memory_items:
        content = item.content
        item_database_id = content.get('database_id', '')
        if item_database_id == current_database_id:
            same_db_items.append(item)
        else:
            cross_db_items.append(item)
    
    def format_items_grouped(items: List[Any], db_type: str) -> str:
        """Format items grouped by normalized question"""
        if not items:
            return ""
        
        section_text = ""
        
        # Group items by normalized question
        question_groups = {}
        for item in items:
            content = item.content
            user_query = content.get('user_query', '')
            normalized_q = _normalize_question(user_query)
            
            if normalized_q not in question_groups:
                question_groups[normalized_q] = []
            question_groups[normalized_q].append(item)
        
        # Format results grouped by question
        for group_idx, (normalized_q, group_items) in enumerate(question_groups.items(), 1):
            # Use the first item's original question as display
            display_question = group_items[0].content.get('user_query', normalized_q)
            
            section_text += f"\n--- Question Group {group_idx}: {display_question} ---\n"
            
            # Sort items by score (descending)
            sorted_items = sorted(group_items, key=lambda x: x.score, reverse=True)
            
            for sql_idx, item in enumerate(sorted_items, 1):
                content = item.content
                score = item.score
                
                # Parse metadata if it's a JSON string
                metadata_raw = content.get('metadata', {})
                metadata = metadata_raw
                if isinstance(metadata_raw, str):
                    try:
                        metadata = json.loads(metadata_raw)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                elif not isinstance(metadata_raw, dict):
                    metadata = {}
                
                # Check if this is an insight record
                is_insight = False
                if isinstance(metadata, dict):
                    is_insight = 'comparative_insight' in metadata or 'original_sqls' in metadata
                
                # Extract key information
                database_id = content.get('database_id', 'unknown')
                label = content.get('label', None)
                
                # Get SQL: for insight records, use incorrect SQL from original_sqls
                if is_insight and isinstance(metadata, dict):
                    original_sqls = metadata.get('original_sqls', {})
                    if isinstance(original_sqls, dict):
                        generated_sql = original_sqls.get('incorrect', content.get('generated_sql', content.get('final_sql', 'No SQL')))
                    else:
                        generated_sql = content.get('generated_sql', content.get('final_sql', 'No SQL'))
                else:
                    # Try generated_sql first, then fall back to final_sql
                    generated_sql = content.get('generated_sql', content.get('final_sql', 'No SQL'))
                
                # Determine SQL status
                if label is True:
                    status = "✓ CORRECT"
                elif label is False:
                    status = "✗ ERROR"
                else:
                    status = "? UNKNOWN"
                
                # Add insight indicator
                insight_indicator = " [INSIGHT]" if is_insight else ""
                
                # Format the SQL variant
                section_text += f"  SQL Variant {sql_idx} ({status}){insight_indicator}:\n"
                section_text += f"  Database: {database_id}\n"
                section_text += f"  Generated SQL: {generated_sql}\n"
                
                # Display comparative insight for insight records
                if is_insight and isinstance(metadata, dict):
                    comparative_insight = metadata.get('comparative_insight')
                    if isinstance(comparative_insight, dict):
                        section_text += f"  📋 Comparative Insight:\n"
                        if 'pred_logic' in comparative_insight:
                            section_text += f"    • Pred Logic: {comparative_insight['pred_logic']}\n"
                        if 'gold_logic' in comparative_insight:
                            section_text += f"    • Gold Logic: {comparative_insight['gold_logic']}\n"
                        if 'key_difference' in comparative_insight:
                            section_text += f"    • Key Difference: {comparative_insight['key_difference']}\n"
                
                # Add error types and feedback if available
                if content.get('error_types'):
                    section_text += f"  Error Types: {content.get('error_types')}\n"
                if content.get('feedback_text'):
                    section_text += f"  Feedback: {content.get('feedback_text')}\n"
                
                section_text += "\n"
            
            section_text += "-" * 60 + "\n"
        
        return section_text
    
    # Format same database results
    if same_db_items:
        formatted_text += f"🎯 SAME DATABASE CASES (Database: {current_database_id})\n"
        formatted_text += f"Found {len(same_db_items)} cases from the same database - these are most relevant:\n"
        formatted_text += "=" * 70 + "\n"
        formatted_text += format_items_grouped(same_db_items, "same_db")
        formatted_text += "\n"
    
    # Format cross database results
    if cross_db_items:
        formatted_text += f"🌐 CROSS DATABASE CASES\n"
        formatted_text += f"Found {len(cross_db_items)} cases from other databases - use these for general SQL patterns:\n"
        formatted_text += "=" * 70 + "\n"
        formatted_text += format_items_grouped(cross_db_items, "cross_db")
    
    formatted_text += "\n" + "=" * 80 + "\n\n"
    return formatted_text

def _format_semantic_memory_for_llm(memory_items: List[Any]) -> str:
    """Format semantic memory items into structured text for LLM consumption"""
    
    if not memory_items:
        return "No relevant semantic information found."
    
    formatted_text = "=== SEMANTIC KNOWLEDGE FROM DATABASE SCHEMA ===\n\n"
    formatted_text += "This section provides SUPPLEMENTARY INFORMATION about the database to help you understand:\n"
    formatted_text += "• The meaning and content of some specific tables and columns\n"
    formatted_text += "• Actual values stored in the database columns\n"
    formatted_text += "• Relationships between tables (JOIN conditions)\n"
    formatted_text += "This information MIGHT BE HELPFUL to this task.\n\n"
    
    for i, item in enumerate(memory_items, 1):
        content = item.content
        item_type = content.get('item_type', 'unknown')
        metadata = content.get('metadata', {})
        score = item.score
        
        formatted_text += f"Semantic Memory:\n"
        
        if item_type == 'anchor_selection':
            # Handle anchor selection type memory items
            
            # Format required tables
            required_tables = metadata.get('required_tables', [])
            if required_tables:
                formatted_text += "## METADATA of Tables:\n"
                tables_metadata = metadata.get('tables_metadata', {})
                for table_name in required_tables:
                    formatted_text += f"• {table_name}\n"
                    # Add table description if available
                    if table_name in tables_metadata:
                        table_info = tables_metadata[table_name]
                        description = table_info.get('description', '')
                        if description:
                            formatted_text += f"  Description: {description}\n"
                    formatted_text += "\n"
            
            # Format selected columns with descriptions and matched values
            selected_columns = metadata.get('selected_columns', [])
            if selected_columns:
                formatted_text += "## METADATA of COLUMNS:\n"
                columns_metadata = metadata.get('columns_metadata', {})
                value_matches = metadata.get('value_matches', {})
                
                for column_full_name in selected_columns:
                    formatted_text += f"• {column_full_name}\n"
                    
                    # Add column description (excluding whole_column_name and data_type as requested)
                    if column_full_name in columns_metadata:
                        col_info = columns_metadata[column_full_name]
                        description = col_info.get('description', '')
                        if description:
                            formatted_text += f"  Description: {description}\n"
                    
                    # Add column metadata (has_null information)
                    if column_full_name in columns_metadata:
                        col_info = columns_metadata[column_full_name]
                        has_null = col_info.get('has_nulls', False)
                        if has_null:
                            formatted_text += f"  Contains NULL values: Yes\n"
                    
                    # Add matched values if available
                    if column_full_name in value_matches:
                        match_info = value_matches[column_full_name]
                        values = match_info.get('values', [])
                        types = match_info.get('types', [])
                        encoding_mappings = match_info.get('encoding_mappings', {})
                        
                        if values:
                            if encoding_mappings:
                                # When encoding_mappings exists, keys are actual DB values, values are explanations
                                formatted_text += f"  Some Values in the Column:\n"
                                for actual_value, explanation in encoding_mappings.items():
                                    formatted_text += f"  '{actual_value}' → Meaning: '{explanation}'\n"
                                # formatted_text += f"  Note: Query should use the database values (keys), not the explanations\n"
                            else:
                                formatted_text += "  Some Values in the Column:\n"
                                for v in values:
                                    formatted_text += f"  - '{str(v)}'\n"
                            formatted_text += "\n"
                    
                    formatted_text += "\n"
            
            # Format value matches that might not be in selected columns
            value_matches = metadata.get('value_matches', {})
            if value_matches:
                # Only show value matches that are not already shown in selected columns
                other_matches = {k: v for k, v in value_matches.items() if k not in selected_columns}
                if other_matches:
                    formatted_text += "## OTHER VALUE MATCHES:\n"
                    for column_name, match_info in other_matches.items():
                        values = match_info.get('values', [])
                        types = match_info.get('types', [])
                        encoding_mappings = match_info.get('encoding_mappings', {})
                        
                        formatted_text += f"• {column_name}:\n"
                        
                        if encoding_mappings:
                            # When encoding_mappings exists, keys are actual DB values, values are explanations
                            formatted_text += f"  Matched Values (with encoding mappings):\n"
                            for actual_value, explanation in encoding_mappings.items():
                                formatted_text += f"    Database: '{actual_value}' → Meaning: '{explanation}'\n"
                            # formatted_text += f"  Note: Query should use the database values (keys), not the explanations\n"
                        else:
                                formatted_text += "  Values in the Column\n"
                                for v in values:
                                    formatted_text += f"  - '{str(v)}'\n"
                        
                        formatted_text += "\n"
                    formatted_text += "\n"
            
            # Format join relationships
            join_relationships = metadata.get('join_relationships', [])
            if join_relationships:
                formatted_text += "## JOIN RELATIONSHIPS:\n"
                formatted_text += "The following tables are connected by these relationships:\n\n"
                
                for i, join_rel in enumerate(join_relationships, 1):
                    # Extract join relationship information
                    table1 = getattr(join_rel, 'table1', None)
                    column1 = getattr(join_rel, 'column1', None)
                    table2 = getattr(join_rel, 'table2', None)
                    column2 = getattr(join_rel, 'column2', None)
                    join_type = getattr(join_rel, 'join_type', 'unknown')
                    
                    if table1 and column1 and table2 and column2:
                        formatted_text += f"{i}. **{table1}.{column1}** ↔ **{table2}.{column2}**\n"
                        if join_type == 'foreign_key':
                            formatted_text += f"   Relationship Type: PK-FK\n"
                
                formatted_text += "-" * 60 + "\n\n"
        
        else:
            # Handle other item types (fallback to original logic)
            if item_type == 'table':
                table_name = metadata.get('table_name', 'unknown')
                formatted_text += f"📊 TABLE: {table_name}\n"
            elif item_type == 'column':
                table_name = metadata.get('table_name', 'unknown')
                column_name = metadata.get('column_name', 'unknown')
                formatted_text += f"🔗 COLUMN: {table_name}.{column_name}\n"
                
                # Add has_null information if available
                has_null = metadata.get('has_null', False)
                if has_null:
                    formatted_text += f"  Contains NULL values: Yes\n"
            elif item_type == 'term':
                term_name = metadata.get('term_name', 'unknown')
                formatted_text += f"💡 TERM: {term_name}\n"
            else:
                formatted_text += f"❓ UNKNOWN TYPE: {item_type}\n"
            
            # Add any available description
            description = metadata.get('description', '')
            if description:
                formatted_text += f"  Description: {description}\n"

            
            formatted_text += "\n"
        
        formatted_text += "-" * 40 + "\n\n"
    
    formatted_text += "=" * 50 + "\n\n"
    return formatted_text

def _load_results_file(results_path: Optional[str], logger: logging.Logger) -> List[Dict[str, Any]]:
    """Load existing batch results to enable resume functionality."""
    if not results_path:
        return []
    
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if isinstance(data.get("results"), list):
                return data["results"]
            if isinstance(data.get("detailed_results"), list):
                return data["detailed_results"]
        elif isinstance(data, list):
            return data
        
        logger.warning(f"Results file {results_path} has unexpected structure. Starting fresh.")
    except FileNotFoundError:
        logger.warning(f"Results file not found: {results_path}. A new file will be created.")
    except Exception as e:
        logger.warning(f"Failed to load results file {results_path}: {e}")
    
    return []

def _build_progress_summary(results: List[Dict[str, Any]], total_items: int) -> Dict[str, Any]:
    """Build incremental summary statistics for progress tracking."""
    processed_items = len(results)
    successful = sum(1 for r in results if r.get("execution_success"))
    failed = processed_items - successful
    success_rate = successful / processed_items * 100 if processed_items else 0.0
    correct = sum(1 for r in results if r.get("ex_correct"))
    execution_accuracy = correct / processed_items * 100 if processed_items else 0.0
    processing_complete = processed_items >= total_items if total_items else False
    
    return {
        "total_items": total_items,
        "processed_items": processed_items,
        "successful": successful,
        "failed": failed,
        "success_rate": success_rate,
        "execution_accuracy": execution_accuracy,
        "processing_complete": processing_complete
    }

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    # Disable httpx INFO logs to avoid verbose HTTP request logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

def nl2sql_reasoning(query: NL2SQLQuery, config: NL2SQLConfig, llm_engine=None, data_handler=None, caf_system=None) -> NL2SQLResponse:
    """
    Core NL2SQL reasoning function - direct CAF usage without wrapper layers
    
    This function shows the basic workflow using CAF's native interfaces:
    1. Initialize components (data handler, LLM engine)
    2. Initialize CAF system directly
    3. Retrieve relevant context from CAF memory
    4. Generate SQL using LLM
    5. Execute SQL and validate results
    6. Collect user feedback via CAF
    7. Store experience in CAF episodic memory
    """
    start_time = time.time()
    session_id = query.session_id or str(uuid.uuid4())
    logger = logging.getLogger(__name__)
    
    # Initialize basic components (reuse existing instances if provided)
    if data_handler is None:
        data_handler = BirdDataHandler(config.data)
    if llm_engine is None:
        llm_engine = LLMEngine(config.llm)
    
    # Initialize or use provided CAF system
    if query.use_caf and config.caf.enabled:
        try:
            # If CAF system not provided, initialize it
            if caf_system is None:
                caf_system = caf.initialize(config_path=config.caf.config_path)
                logger.info(f"CAF system initialized for session: {session_id}")
            
            # Start new session for this query
            caf_system.start_session(session_id, query.db_id)
            logger.info(f"CAF session started: {session_id} for database: {query.db_id}")
        except Exception as e:
            logger.warning(f"CAF session start failed: {e}. Continuing without CAF.")
            caf_system = None
    
    try:
        # Step 1: Get memory context from CAF if enabled  
        memory_context = {}
        formatted_memory_text = ""
        
        if caf_system:
            try:
                # Retrieve episodic memory (unified - no distinction between successful/error cases)
                if config.enable_episodic and "episodic" in config.caf.memory_types:
                    logger.info("Retrieving episodic memory...")
                    
                    # Prepare context for episodic memory retrieval
                    episodic_context = {}
                    if query.evidence:
                        episodic_context['evidence'] = query.evidence
                    
                    # Retrieve episodic memory (unified call - system returns all relevant cases)
                    # Use max of success/error limits if available, otherwise default to 10
                    episodic_limit = 5
                    if hasattr(config.caf, 'episodic_success_limit') and hasattr(config.caf, 'episodic_error_limit'):
                        episodic_limit = max(config.caf.episodic_success_limit, config.caf.episodic_error_limit) or 10
                    elif hasattr(config.caf, 'episodic_limit'):
                        episodic_limit = config.caf.episodic_limit or 5
                    
                    episodic_kwargs = {
                        "memory_type": "episodic",
                        "query_content": query.question,
                        "limit": episodic_limit,
                    }
                    
                    # Add context if available
                    if episodic_context:
                        episodic_kwargs["context"] = episodic_context
                    
                    # Add similarity threshold if specified
                    if config.caf.episodic_similarity_threshold is not None:
                        episodic_kwargs["similarity_threshold"] = config.caf.episodic_similarity_threshold
                    
                    # Call read_memory (system returns all relevant cases regardless of label)
                    episodic_response = caf_system.read_memory(**episodic_kwargs)
                    
                    # Store all retrieved items
                    memory_context["episodic"] = episodic_response.items
                    
                    # Format episodic cases for LLM
                    if memory_context["episodic"]:
                        formatted_memory_text += _format_episodic_memory_for_llm(
                            memory_context["episodic"], query.db_id, "episodic"
                        )
                    
                    logger.info(f"Retrieved {len(memory_context['episodic'])} episodic cases")
                else:
                    logger.info("Episodic memory retrieval disabled by configuration")
                    memory_context["episodic"] = []
                
                # Retrieve semantic memory
                if config.enable_semantic and "semantic" in config.caf.memory_types:
                    logger.info("Retrieving semantic memory...")
                    
                    semantic_kwargs = {
                        "memory_type": "semantic",
                        "query_content": query.question
                    }
                    
                    # Add limit if specified in config
                    if config.caf.semantic_limit is not None:
                        semantic_kwargs["limit"] = config.caf.semantic_limit
                    
                    semantic_response = caf_system.read_memory(**semantic_kwargs)
                    
                    memory_context["semantic"] = semantic_response.items
                    
                    # Format semantic memory for LLM
                    if semantic_response.items:
                        formatted_memory_text += _format_semantic_memory_for_llm(semantic_response.items)
                    
                    logger.info(f"Retrieved {len(semantic_response.items)} semantic memory items")
                
                logger.info(f"Retrieved CAF memory: {list(memory_context.keys())}")
                
            except Exception as e:
                logger.warning(f"Failed to retrieve CAF memory: {e}")
                memory_context = {}
                formatted_memory_text = ""
        
        # Step 2: Get database schema
        schema = data_handler.get_database_schema(query.db_id)
        
        # Step 3: Prepare context for LLM
        context = {
            "question": query.question,
            "evidence": query.evidence,
            "database_id": query.db_id,
            "schema": schema.dict(),
            "memory": memory_context,
            "formatted_memory_text": formatted_memory_text  # Add formatted memory text for LLM
        }
        
        # Step 4: Generate SQL using LLM
        generated_sql, llm_interaction = llm_engine.generate_sql(
            question=query.question,
            schema=schema,
            context=context,
            evidence=query.evidence
        )
        
        # Step 5: Create result object
        result = NL2SQLResult(
            generated_sql=generated_sql,
            execution_success=False,
            generation_time_ms=int((time.time() - start_time) * 1000),
            database_id=query.db_id,
            session_id=session_id
        )
        
        # Step 6: Execute SQL if enabled
        if config.evaluation.enable_execution:
            print("======================EXECUTING SQL WHEN Reasoning==========================")
            success, results, error = data_handler.execute_sql(query.db_id, generated_sql)
            result.execution_success = success
            result.execution_result = results
            if not success:
                logger.warning(f"SQL execution failed: {error}")
        
        # Step 7: Collect user feedback via CAF if enabled
        feedback_collected = False
        feedback_details = None
        if caf_system and config.caf.feedback_enabled:
            try:
                # Format schema for feedback context
                formatted_schema = _format_schema_for_feedback(schema)
                
                # Use CAF's simplified feedback interface - no need to know about internal types
                feedback = caf_system.request_feedback(
                    feedback_type="sql_validation",
                    user_query=query.question,
                    generated_sql=result.generated_sql,
                    ground_truth_sql=query.ground_truth_sql,
                    db_schema=formatted_schema,
                    execution_result={
                        "database_id": query.db_id,
                        "execution_success": result.execution_success,
                        "execution_result": result.execution_result
                    }
                )
                feedback_collected = True
                
                # 保存详细的反馈信息
                feedback_details = {
                    "feedback_id": feedback.feedback_id,
                    "is_correct": feedback.is_correct,
                    "error_category": getattr(feedback, 'error_category', None),
                    "error_subcategory": getattr(feedback, 'error_subcategory', None),
                    "analysis": getattr(feedback, 'analysis', None),
                    "suggestion": getattr(feedback, 'suggestion', None),
                    "confidence_score": getattr(feedback, 'confidence_score', None),
                    "feedback_type": "sql_validation",
                    "timestamp": feedback.timestamp
                }
                
                logger.info("User feedback collected via CAF")
            except Exception as e:
                logger.warning(f"Failed to collect CAF feedback: {e}")
        
        # Step 8: Store episodic trace in CAF
        if caf_system:
            try:
                # Use CAF's simplified session finalization - no need to know about internal types
                caf_system.finalize_session(
                    user_query=query.question,
                    generated_sql=result.generated_sql,
                    final_sql=result.generated_sql,
                    execution_result=result.execution_result,
                    execution_success=result.execution_success,
                    metadata={
                        "mode": query.mode.value,
                        "evidence": query.evidence,
                        "generation_time_ms": result.generation_time_ms
                    },
                    store_episodic=False
                )
                logger.info(f"CAF session finalized: {session_id}")
            except Exception as e:
                logger.warning(f"Failed to finalize CAF session: {e}")
        
        # Return successful response
        response = NL2SQLResponse(
            success=True,
            result=result,
            feedback_collected=feedback_collected,
            feedback_details=feedback_details,  # 包含详细反馈信息
            session_id=session_id,
            database_id=query.db_id,
            timestamp=datetime.now()
        )
        
        logger.info(f"NL2SQL reasoning completed successfully in {result.generation_time_ms}ms")
        return response
        
    except Exception as e:
        logger.error(f"NL2SQL reasoning failed: {e}")
        return NL2SQLResponse(
            success=False,
            error_message=str(e),
            session_id=session_id,
            database_id=query.db_id,
            timestamp=datetime.now()
        )

def single_query_mode(args):
    """Handle single query mode"""
    # Load configuration
    if args.config:
        config = NL2SQLConfig.from_file(args.config)
    else:
        config = NL2SQLConfig.default()
        # Override with command line args
        if args.mode:
            config.mode = NL2SQLMode(args.mode)
        if args.api_key:
            config.llm.api_key = args.api_key
    
    # Override database description setting if provided
    if hasattr(args, 'use_database_description') and args.use_database_description:
        config.data.use_database_description = True
    
    # Override episodic memory setting if provided
    if hasattr(args, 'enable_episodic') and args.enable_episodic is not None:
        config.enable_episodic = args.enable_episodic
    
    # Override semantic memory setting if provided
    if hasattr(args, 'enable_semantic') and args.enable_semantic is not None:
        config.enable_semantic = args.enable_semantic
    
    # Override log level for debug reasoning if requested
    log_level = config.log_level
    setup_logging(log_level, config.log_file)
    logger = logging.getLogger(__name__)
    
    # Create query
    query = NL2SQLQuery(
        question=args.question,
        db_id=args.db_id,
        evidence=args.evidence,
        ground_truth_sql=getattr(args, 'ground_truth_sql', None),
        use_caf=args.use_caf,
        mode=config.mode,
        session_id=args.session_id
    )
    
    logger.info(f"Processing query: {args.question}")
    logger.info(f"Database: {args.db_id}")
    
    # Initialize CAF system for single query processing
    caf_system = None
    if query.use_caf and config.caf.enabled:
        try:
            caf_system = caf.initialize(config_path=config.caf.config_path)
            logger.info("CAF system initialized for single query processing")
        except Exception as e:
            logger.warning(f"CAF initialization failed: {e}. Continuing without CAF.")
            caf_system = None
    
    # Execute reasoning - direct function call instead of reasoner class
    response = nl2sql_reasoning(query, config, caf_system=caf_system)
    
    # Print results
    print("\n" + "="*60)
    print("NL2SQL REASONING RESULTS")
    print("="*60)
    
    if response.success:
        result = response.result
        print(f"Question: {args.question}")
        print(f"Database: {args.db_id}")
        print(f"Generated SQL: {result.generated_sql}")
        print(f"Execution Success: {result.execution_success}")
        print(f"Generation Time: {result.generation_time_ms}ms")
        
        if result.execution_result:
            print(f"Query Results: {len(result.execution_result)} rows")
            if args.show_results:
                for i, row in enumerate(result.execution_result[:5]):
                    print(f"  Row {i+1}: {row}")
                if len(result.execution_result) > 5:
                    print(f"  ... and {len(result.execution_result) - 5} more rows")
        
        print(f"\nFeedback Collected: {response.feedback_collected}")
        
        # 显示详细的反馈信息
        if response.feedback_collected and response.feedback_details:
            print("\n=== Detailed Feedback ===")
            feedback = response.feedback_details
            print(f"Feedback ID: {feedback.get('feedback_id', 'N/A')}")
            print(f"SQL Correctness: {'✓ Correct' if feedback.get('is_correct') else '✗ Incorrect'}")
            
            if not feedback.get('is_correct'):
                print(f"Error Category: {feedback.get('error_category', 'N/A')}")
                print(f"Error Subcategory: {feedback.get('error_subcategory', 'N/A')}")
                print(f"Analysis: {feedback.get('analysis', 'N/A')}")
                if feedback.get('suggestion'):
                    print(f"Suggestion: {feedback.get('suggestion')}")
            else:
                print(f"Analysis: {feedback.get('analysis', 'SQL query is correct')}")
            
            if feedback.get('confidence_score'):
                print(f"Confidence: {feedback.get('confidence_score'):.2f}")
            
            print(f"Timestamp: {feedback.get('timestamp', 'N/A')}")
            print("========================")
        
    else:
        print(f"ERROR: {response.error_message}")
    
    print("="*60)
    
    # Save results if requested
    if args.output:
        output_data = {
            "query": query.dict(),
            "response": response.dict()
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")

def batch_mode(args):
    """Handle batch processing mode"""
    # Load configuration
    if args.config:
        config = NL2SQLConfig.from_file(args.config)
    else:
        config = NL2SQLConfig.default()
        if args.mode:
            config.mode = NL2SQLMode(args.mode)
        if args.api_key:
            config.llm.api_key = args.api_key
    
    # Override database description setting if provided
    if hasattr(args, 'use_database_description') and args.use_database_description:
        config.data.use_database_description = True
    
    # Override episodic memory setting if provided
    if hasattr(args, 'enable_episodic') and args.enable_episodic is not None:
        config.enable_episodic = args.enable_episodic
    
    # Override semantic memory setting if provided
    if hasattr(args, 'enable_semantic') and args.enable_semantic is not None:
        config.enable_semantic = args.enable_semantic
    
    # Disable CAF if requested
    if hasattr(args, 'no_caf') and args.no_caf:
        config.caf.enabled = False
        logger = logging.getLogger(__name__)
        logger.info("CAF disabled as requested")
    
    # Override log level for debug reasoning if requested
    log_level = config.log_level
    setup_logging(log_level, config.log_file)
    logger = logging.getLogger(__name__)
    
    resume_results_path = getattr(args, 'resume_results', None)
    results = _load_results_file(resume_results_path, logger)
    question_index_map = {}
    questions_to_rerun = set()
    
    for idx, record in enumerate(results):
        question_text = (record.get("question") or "").strip()
        if not question_text:
            continue
        question_index_map[question_text] = idx
        if not record.get("execution_success"):
            questions_to_rerun.add(question_text)
    
    if resume_results_path and results:
        logger.info(f"Loaded {len(results)} existing results from {resume_results_path}")
    
    # Create shared components (optimization: create once, reuse for all queries)
    data_handler = BirdDataHandler(config.data)
    llm_engine = LLMEngine(config.llm)
    logger.info("Initialized shared LLM engine and data handler for batch processing")
    
    # Initialize CAF system once for batch processing
    caf_system = None
    if config.caf.enabled and not (hasattr(args, 'no_caf') and args.no_caf):
        try:
            caf_system = caf.initialize(config_path=config.caf.config_path)
            logger.info("CAF system initialized for batch processing")
        except Exception as e:
            logger.warning(f"CAF initialization failed: {e}. Continuing without CAF.")
            caf_system = None
    
    # Load dataset
    logger.info(f"Loading {args.split} dataset...")
    dataset = data_handler.load_dataset(args.split)
    logger.info(f"Loaded {len(dataset)} items from {args.split} dataset")
    
    # Filter by databases if specified
    if hasattr(args, 'databases') and args.databases:
        original_count = len(dataset)
        dataset = [item for item in dataset if item.db_id in args.databases]
        logger.info(f"Filtered dataset by databases {args.databases}: {original_count} -> {len(dataset)} items")
        
        if len(dataset) == 0:
            logger.error(f"No data found for databases: {args.databases}")
            return
    
    # Take percentage from the end if specified
    if hasattr(args, 'percentage') and args.percentage:
        if args.percentage == 100:
            logger.info(f"Percentage is 100%, processing all {len(dataset)} items")
        elif args.percentage < 100:
            original_count = len(dataset)
            
            # Group by database to ensure we take percentage from each database
            db_groups = {}
            for item in dataset:
                if item.db_id not in db_groups:
                    db_groups[item.db_id] = []
                db_groups[item.db_id].append(item)
            
            # Take last percentage from each database
            filtered_dataset = []
            for db_id, db_items in db_groups.items():
                count_to_take = int(len(db_items) * args.percentage / 100)
                if count_to_take == 0 and len(db_items) > 0:
                    count_to_take = 1  # Take at least 1 item if there are any
                
                # Take from the end (last percentage)
                last_items = db_items[-count_to_take:] if count_to_take > 0 else []
                filtered_dataset.extend(last_items)
                logger.info(f"Database {db_id}: taking last {count_to_take}/{len(db_items)} items ({args.percentage}%)")
            
            dataset = filtered_dataset
            logger.info(f"Filtered dataset by percentage ({args.percentage}% from end): {original_count} -> {len(dataset)} items")
        else:
            logger.warning(f"Invalid percentage value: {args.percentage}. Must be between 1 and 100. Processing all items.")

    logger.info(f"Final dataset size before processing: {len(dataset)} items")
    
    if hasattr(args, 'limit') and args.limit:
        dataset = dataset[:args.limit]
        logger.info(f"Processing first {args.limit} items")
    
    # Prepare output file for incremental saving
    output_file = args.output or f"nl2sql_results_{args.split}.json"
    
    # Process queries with incremental saving
    start_time = time.time()  # Track processing start time

    evaluator = NL2SQLEvaluator(data_handler)
    
    # Initialize counter for tracking correct predictions
    total_correct = 0

    for i, item in enumerate(dataset):
        logger.info(f"Processing item {i+1}/{len(dataset)}: {item.question_id}")
        
        question_key = item.question.strip()
        print(f"Question key: {question_key}")

        if question_key in ["What are the categories of the top 2 oldest events?", "Calculate the average age of people who have apps installed but are not active on their devices."]:
            continue
        
        if question_key in question_index_map and question_key not in questions_to_rerun:
            logger.debug(f"Skipping already successful question: {item.question_id}")
            continue
        
        query = NL2SQLQuery(
            question=item.question,
            db_id=item.db_id,
            evidence=item.evidence,
            ground_truth_sql=item.sql,
            use_caf=(not hasattr(args, 'no_caf') or not args.no_caf) and config.caf.enabled,
            mode=config.mode,
            session_id=f"batch_{args.split}_{item.question_id}"
        )
        
        try:
            # Use direct function call with shared components (optimization)
            response = nl2sql_reasoning(query, config, llm_engine=llm_engine, data_handler=data_handler, caf_system=caf_system)
            
            # Immediate evaluation of generated SQL if we have ground truth
            ex_correct = None
            
            if response.success and item.sql:

                try:
                    ex_correct = evaluator._compare_execution_results(
                            generated_sql=response.result.generated_sql,
                            ground_truth_sql=item.sql,
                            db_id=item.db_id,
                            question_id=item.question_id
                        )
                        
                    
                    # Print immediate evaluation result
                    status = "✅ CORRECT" if ex_correct else "❌ INCORRECT"
                    print(f"  {status} - {item.question_id} (DB: {item.db_id})")
                    if not ex_correct:
                        print(f"    Generated: {response.result.generated_sql[:100]}...")
                        print(f"    Ground Truth: {item.sql[:100]}...")
                    
                    # Track correct predictions for final summary
                    if ex_correct:
                        total_correct += 1
                        
                except Exception as e:
                    print(f"  ❌ EVAL ERROR - {item.question_id}: {e}")
                    ex_correct = False
                
            result_data = {
                "question_id": item.question_id,
                "db_id": item.db_id,
                "question": item.question,
                "evidence": item.evidence or "",
                "ground_truth_sql": item.sql,
                "generated_sql": response.result.generated_sql if response.success else "",
                "execution_success": response.result.execution_success if response.success else False,
                # "execution_result": response.result.execution_result if response.success else None,  # Removed to save space
                "ex_correct": ex_correct,  # Add immediate EX evaluation result
                "generation_time_ms": response.result.generation_time_ms if response.success else 0,
                "error_message": response.error_message if not response.success else None,
                "caf_used": query.use_caf,  # Record whether CAF was used
                "memory_used": query.use_caf and config.caf.enabled  # Record whether memory was used
            }
            
            if question_key in question_index_map:
                results[question_index_map[question_key]] = result_data
            else:
                results.append(result_data)
                question_index_map[question_key] = len(results) - 1
            
            if result_data.get("execution_success"):
                questions_to_rerun.discard(question_key)
            else:
                questions_to_rerun.add(question_key)

        except Exception as e:
            logger.error(f"Failed to process item {item.question_id}: {e}")
            result_data = {
                "question_id": item.question_id,
                "db_id": item.db_id,
                "question": item.question,
                "evidence": item.evidence or "",
                "ground_truth_sql": item.sql,
                "generated_sql": "",
                "execution_success": False,
                # "execution_result": None,  # Removed to save space
                "ex_correct": False,
                "generation_time_ms": 0,
                "error_message": str(e),
                "caf_used": query.use_caf,
                "memory_used": query.use_caf and config.caf.enabled
            }
            if question_key in question_index_map:
                results[question_index_map[question_key]] = result_data
            else:
                results.append(result_data)
                question_index_map[question_key] = len(results) - 1
            
            questions_to_rerun.add(question_key)
        
        # Incremental save after each item processing
        try:
            # Create intermediate output data with current progress
            progress_summary = _build_progress_summary(results, len(dataset))
            current_output_data = {
                "results": results,
                "summary": progress_summary
            }
            
            with open(output_file, 'w') as f:
                json.dump(current_output_data, f, indent=2, default=str)
            
            logger.debug(f"Incremental save completed for item {i+1}/{len(dataset)}")
            
        except Exception as e:
            logger.error(f"Failed to save incremental results: {e}")
    
    # Print summary
    successful_count = sum(1 for r in results if r.get('execution_success'))
    processed_count = len(results)
    failed_count = processed_count - successful_count
    success_rate = successful_count / processed_count * 100 if processed_count else 0.0
    
    print(f"\nBatch Processing Summary:")
    print(f"Total items: {len(dataset)}")
    print(f"Processed items: {processed_count}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Calculate final Execution Accuracy (EX) based on already computed results
    # No need to recompute since we already have ex_correct for each item
    has_ground_truth = any(result.get('ground_truth_sql') for result in results)
    if has_ground_truth:
        try:
            # Create evaluation summary without recomputing
            ex_total_count = sum(1 for result in results if result.get('ex_correct') is not None)
            ex_correct_count = sum(1 for result in results if result.get('ex_correct'))
            ex_accuracy = ex_correct_count / ex_total_count if ex_total_count > 0 else 0.0
            
            ex_evaluation = {
                "ex_accuracy": ex_accuracy,
                "total": ex_total_count,
                "correct": ex_correct_count,
                "incorrect": ex_total_count - ex_correct_count,
                "accuracy_percentage": ex_accuracy * 100,
            }
            
            # Print evaluation summary using existing function
            evaluator.print_evaluation_summary(ex_evaluation)
            
            # Add evaluation results to output
            evaluation_summary = {
                "execution_accuracy": ex_evaluation,
                "timestamp": datetime.now().isoformat(),
                "evaluation_type": "EX (Execution Accuracy)"
            }
        except Exception as e:
            logger.error(f"Failed to calculate execution accuracy: {e}")
            evaluation_summary = {"error": f"EX calculation failed: {e}"}
    else:
        evaluation_summary = {"note": "No ground truth SQL available for EX evaluation"}
    
    # Calculate final statistics
    total_time = time.time() - start_time if 'start_time' in locals() else 0
    
    # Group results by database for statistics
    db_stats = {}
    for result in results:
        db_id = result['db_id']
        if db_id not in db_stats:
            db_stats[db_id] = {'total': 0, 'success': 0, 'execution_success': 0}
        
        db_stats[db_id]['total'] += 1
        
        if result.get('execution_success'):
            db_stats[db_id]['execution_success'] += 1
    
    # Calculate EX statistics
    ex_correct_count = sum(1 for result in results if result.get('ex_correct'))
    ex_total_count = sum(1 for result in results if result.get('ex_correct') is not None)
    successful_total = sum(1 for result in results if result.get('execution_success'))
    
    # Prepare final output in the same format as process_filtered_dataset.py
    final_output_data = {
        "experiment_info": {
            "description": "DAMO NL2SQL reasoning batch processing",
            "target_databases": getattr(args, 'databases', []) or list(db_stats.keys()),
            "data_split": args.split,
            "caf_enabled": config.caf.enabled,
            "memory_enabled": config.caf.enabled,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_processing_time_seconds": total_time,
            "percentage_filter": getattr(args, 'percentage', 100)
        },
        "summary_statistics": {
            "total_processed": len(results),
            "total_successful": successful_total,
            "total_failed": len(results) - successful_total,
            "overall_success_rate": successful_total / len(results) * 100 if len(results) > 0 else 0,
            "average_generation_time_ms": sum(r.get('generation_time_ms', 0) for r in results if r.get('execution_success')) / max(successful_total, 1),
            "ex_correct": ex_correct_count,
            "ex_total": ex_total_count,
            "ex_accuracy": ex_correct_count / ex_total_count * 100 if ex_total_count > 0 else 0
        },
        "database_statistics": db_stats,
        "detailed_results": results
    }
    
    # Add execution accuracy evaluation if available
    if evaluation_summary and "execution_accuracy" in evaluation_summary:
        final_output_data["execution_accuracy"] = evaluation_summary["execution_accuracy"]
    elif evaluation_summary:
        final_output_data["execution_accuracy"] = evaluation_summary
    
    with open(output_file, 'w') as f:
        json.dump(final_output_data, f, indent=2, default=str)
    
    print(f"Final results with evaluation saved to: {output_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="NL2SQL Reasoning Module with CAF Integration")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single query mode
    single_parser = subparsers.add_parser("query", help="Process single query")
    single_parser.add_argument("question", type=str, help="Natural language question")
    single_parser.add_argument("db_id", type=str, help="Database identifier")
    
    # Global arguments for single query
    single_parser.add_argument("--config", type=str, help="Path to configuration file")
    single_parser.add_argument("--mode", type=str, choices=["llm_only"],
                              help="Reasoning mode (only llm_only supported)")
    single_parser.add_argument("--api-key", type=str, help="LLM API key")
    single_parser.add_argument("--use-caf", action="store_true", default=True,
                              help="Use CAF cognitive basis")
    single_parser.add_argument("--no-caf", dest="use_caf", action="store_false",
                              help="Disable CAF cognitive basis")
    single_parser.add_argument("--use-database-description", action="store_true", default=False,
                              help="Include database description files with column and value descriptions")
    single_parser.add_argument("--enable-episodic", action="store_true", default=None,
                              help="Enable retrieval of episodic memory")
    single_parser.add_argument("--disable-episodic", dest="enable_episodic", action="store_false",
                              help="Disable retrieval of episodic memory")
    single_parser.add_argument("--enable-semantic", action="store_true", default=None,
                              help="Enable retrieval of semantic memory")
    single_parser.add_argument("--disable-semantic", dest="enable_semantic", action="store_false",
                              help="Disable retrieval of semantic memory")
    
    # Query-specific arguments
    single_parser.add_argument("--evidence", type=str, help="External knowledge/evidence")
    single_parser.add_argument("--ground-truth-sql", type=str, help="Ground truth SQL for validation/evaluation")
    single_parser.add_argument("--session-id", type=str, help="Session identifier")
    single_parser.add_argument("--output", type=str, help="Output file path")
    single_parser.add_argument("--show-results", action="store_true",
                              help="Show query execution results")
    single_parser.add_argument("--show-steps", action="store_true",
                              help="Show reasoning steps")
    
    # Batch processing mode
    batch_parser = subparsers.add_parser("batch", help="Process dataset in batch")
    batch_parser.add_argument("split", type=str, choices=["train", "dev"],
                             help="Dataset split to process")
    
    # Global arguments for batch processing
    batch_parser.add_argument("--config", type=str, help="Path to configuration file")
    batch_parser.add_argument("--mode", type=str, choices=["llm_only"],
                             help="Reasoning mode (only llm_only supported)")
    batch_parser.add_argument("--api-key", type=str, help="LLM API key")
    batch_parser.add_argument("--use-caf", action="store_true", default=True,
                             help="Use CAF cognitive basis")
    batch_parser.add_argument("--no-caf", action="store_true", default=False,
                             help="Disable CAF cognitive basis")
    batch_parser.add_argument("--use-database-description", action="store_true", default=False,
                             help="Include database description files with column and value descriptions")
    batch_parser.add_argument("--enable-episodic", action="store_true", default=None,
                             help="Enable retrieval of episodic memory")
    batch_parser.add_argument("--disable-episodic", dest="enable_episodic", action="store_false",
                             help="Disable retrieval of episodic memory")
    batch_parser.add_argument("--enable-semantic", action="store_true", default=None,
                             help="Enable retrieval of semantic memory")
    batch_parser.add_argument("--disable-semantic", dest="enable_semantic", action="store_false",
                             help="Disable retrieval of semantic memory")
    
    # Batch-specific arguments
    batch_parser.add_argument("--databases", nargs='+', type=str,
                             help="Filter by specific database IDs")
    batch_parser.add_argument("--percentage", type=int, 
                             help="Take percentage from the end of each database (e.g., 50 for last 50%%)")
    batch_parser.add_argument("--limit", type=int, help="Limit number of items to process")
    batch_parser.add_argument("--output", type=str, help="Output file path")
    batch_parser.add_argument("--debug-reasoning", action="store_true",
                             help="Enable debug mode to show LLM prompts and responses")
    batch_parser.add_argument("--resume-results", type=str,
                             help="Resume processing using an existing results file")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == "query":
        single_query_mode(args)
    elif args.command == "batch":
        batch_mode(args)

if __name__ == "__main__":
    main()
