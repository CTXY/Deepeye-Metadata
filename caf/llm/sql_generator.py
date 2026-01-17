# CAF SQL Generator - Generate initial SQL from natural language queries
# This module provides SQL generation capability for CAF system
# Based on DAMO's LLMEngine approach, but implemented as a standalone CAF module

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .client import BaseLLMClient, LLMConfig, create_llm_client
from ..prompts.sql_generation import get_comment_prompt, get_cot_prompt

logger = logging.getLogger(__name__)

@dataclass
class DatabaseSchema:
    """Simplified database schema representation for SQL generation"""
    database_id: str
    table_names: List[str]
    columns: Dict[str, List[Dict[str, Any]]]  # table_name -> [{"column_name": str, "data_type": str, ...}]
    table_descriptions: Optional[Dict[str, Dict[str, str]]] = None  # table_name -> {column_name: description}
    
    def to_create_statements(self) -> str:
        """Convert schema to CREATE TABLE statements"""
        statements = []
        for table_name in self.table_names:
            columns = self.columns.get(table_name, [])
            if not columns:
                continue
            
            column_defs = []
            for col in columns:
                col_name = col.get("column_name", "")
                col_type = str(col.get("data_type", "TEXT"))
                is_pk = col.get("is_primary_key", False)
                
                col_def = f"    {col_name} {col_type.upper()}"
                if is_pk:
                    col_def += " PRIMARY KEY"
                
                # Add description as comment if available
                if self.table_descriptions and table_name in self.table_descriptions:
                    if col_name in self.table_descriptions[table_name]:
                        description = self.table_descriptions[table_name][col_name]
                        col_def += f", # {description}"
                    else:
                        col_def += ","
                else:
                    col_def += ","
                
                column_defs.append(col_def)
            
            if column_defs:
                create_stmt = f"CREATE TABLE {table_name}\n(\n" + "\n".join(column_defs) + "\n)"
                statements.append(create_stmt)
        
        return "\n\n".join(statements)

class SQLGenerator:
    """
    SQL Generator for CAF system
    
    Generates initial SQL queries from natural language questions.
    This is used when generated_sql is not provided to read_memory.
    """
    
    def __init__(self, llm_config: Optional[LLMConfig] = None, llm_client: Optional[BaseLLMClient] = None):
        """
        Initialize SQL generator
        
        Args:
            llm_config: LLM configuration (used if llm_client is None)
            llm_client: Pre-configured LLM client (takes precedence over llm_config)
        """
        if llm_client:
            self.llm_client = llm_client
        elif llm_config:
            self.llm_client = create_llm_client(llm_config)
        else:
            raise ValueError("Either llm_config or llm_client must be provided")
        
        logger.info(f"SQLGenerator initialized with LLM: {self.llm_client.model_name}")
    
    def generate_sql(self, 
                     question: str,
                     schema: DatabaseSchema,
                     evidence: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate SQL query from natural language question
        
        Args:
            question: Natural language question
            schema: Database schema information
            evidence: External knowledge/evidence (optional)
            context: Additional context (optional)
            
        Returns:
            Generated SQL query string
        """
        try:
            # Build prompt
            prompt = self._build_prompt(question, schema, evidence, context)
            
            logger.debug(f"Generating SQL for question: {question[:100]}...")
            
            # Call LLM
            response = self.llm_client.call(prompt)
            
            # Extract SQL from response
            sql = self._extract_sql_from_response(response)
            
            # Clean and validate SQL
            sql = self._clean_generated_sql(sql)
            
            if not sql:
                raise ValueError("Empty SQL generated")
            
            return sql
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise
    
    def _build_prompt(self, 
                     question: str, 
                     schema: DatabaseSchema,
                     evidence: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build complete prompt for LLM including schema, context, and question
        
        Based on DAMO-ConvAI's prompt structure
        """
        prompt_parts = []
        
        # Add schema information
        schema_prompt = schema.to_create_statements()
        prompt_parts.append(schema_prompt)
        
        # Add memory context if available
        if context and context.get("formatted_memory_text"):
            formatted_memory_text = context["formatted_memory_text"]
            if formatted_memory_text and formatted_memory_text.strip():
                prompt_parts.append(f"\n-- Previous Experience and Relevant Knowledge:")
                prompt_parts.append(formatted_memory_text)
        
        # Add evidence and question
        comment_prompt = get_comment_prompt(question, evidence)
        prompt_parts.append(comment_prompt)
        
        # Add chain of thought prompt
        cot_prompt = get_cot_prompt()
        prompt_parts.append(cot_prompt)
        
        # Add SQL start
        prompt_parts.append("SELECT ")
        
        return "\n\n".join(prompt_parts)
    
    
    def _extract_sql_from_response(self, response: str) -> str:
        """
        Extract SQL from LLM response
        
        Handles both completion and chat completion responses, including:
        - SQL in code blocks (```sql ... ```)
        - SQL after explanatory text
        - Direct SQL responses
        """
        if not response:
            return ""
        
        # First, try to extract SQL from code blocks
        # Look for SQL code blocks
        sql_block_pattern = r'```sql\s*\n(.*?)\n```'
        sql_blocks = re.findall(sql_block_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if sql_blocks:
            # Use the first SQL block found
            sql = sql_blocks[0].strip()
            logger.debug("Extracted SQL from code block")
        else:
            # Look for SQL code blocks without language specification
            code_block_pattern = r'```\s*\n(.*?)\n```'
            code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
            
            # Check if any code block contains SELECT
            sql = None
            for block in code_blocks:
                if block.strip().upper().startswith('SELECT'):
                    sql = block.strip()
                    logger.debug("Extracted SQL from generic code block")
                    break
            
            if not sql:
                # Try to find SQL after common patterns
                patterns = [
                    r'Here is the SQL query[:\s]*\n(.*?)(?:\n\n|\n--|\nExplanation|\n[A-Z]|\Z)',
                    r'SQL query[:\s]*\n(.*?)(?:\n\n|\n--|\nExplanation|\n[A-Z]|\Z)',
                    r'Query[:\s]*\n(.*?)(?:\n\n|\n--|\nExplanation|\n[A-Z]|\Z)',
                    r'SELECT\s+(.*?)(?:\n\n|\n--|\nExplanation|\n[A-Z]|\Z)'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
                    if matches:
                        sql_candidate = matches[0].strip()
                        if not sql_candidate.upper().startswith('SELECT'):
                            sql_candidate = 'SELECT ' + sql_candidate
                        sql = sql_candidate
                        logger.debug(f"Extracted SQL using pattern: {pattern[:20]}...")
                        break
                
                if not sql:
                    # Last resort: look for any line starting with SELECT
                    lines = response.split('\n')
                    sql_lines = []
                    found_select = False
                    
                    for line in lines:
                        line_stripped = line.strip()
                        if line_stripped.upper().startswith('SELECT'):
                            found_select = True
                            sql_lines.append(line)
                        elif found_select:
                            # Continue collecting lines until we hit explanatory text
                            if (line_stripped.startswith('--') or 
                                line_stripped.startswith('#') or
                                line_stripped.startswith('Explanation') or
                                (line_stripped and not any(c in line_stripped for c in ['SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'JOIN', 'UNION', '(', ')', ',', '=', '>', '<', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'AS', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']))):
                                break
                            sql_lines.append(line)
                    
                    if sql_lines:
                        sql = '\n'.join(sql_lines).strip()
                        logger.debug("Extracted SQL from lines starting with SELECT")
        
        if not sql:
            logger.warning("Could not extract SQL from response")
            return ""
        
        # Clean up the SQL
        sql = self._clean_generated_sql(sql)
        
        return sql
    
    def _clean_generated_sql(self, sql: str) -> str:
        """
        Clean and validate generated SQL
        
        - Remove trailing semicolons
        - Handle column names with special characters
        - Basic SQL validation
        """
        if not sql:
            return ""
        
        # Remove leading/trailing whitespace
        sql = sql.strip()
        
        # Remove trailing semicolon
        if sql.endswith(';'):
            sql = sql[:-1].strip()
        
        return sql

