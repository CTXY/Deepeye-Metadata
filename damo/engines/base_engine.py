# Base engine for NL2SQL reasoning

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from data_handler import DatabaseSchema

class BaseEngine(ABC):
    """
    Abstract base class for NL2SQL reasoning engines
    
    All reasoning engines (LLM, fine-tuned, etc.) should inherit from this class
    and implement the generate_sql method.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the engine with configuration
        
        Args:
            config: Engine-specific configuration
        """
        self.config = config
    
    @abstractmethod
    def generate_sql(self, question: str, schema: DatabaseSchema, 
                    context: Dict[str, Any], evidence: Optional[str] = None) -> str:
        """
        Generate SQL query from natural language question
        
        Args:
            question: Natural language question
            schema: Database schema information
            context: Additional context including memory, sample data, etc.
            evidence: External knowledge/evidence
            
        Returns:
            Generated SQL query as string
        """
        pass
    
    def _format_schema_for_prompt(self, schema: DatabaseSchema, include_samples: bool = True) -> str:
        """
        Format database schema for prompt inclusion
        
        Args:
            schema: Database schema
            include_samples: Whether to include sample data
            
        Returns:
            Formatted schema string
        """
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
    
    def _format_memory_context(self, memory_context: Dict[str, Any]) -> str:
        """
        Format memory context for prompt inclusion
        
        Args:
            memory_context: Retrieved memory context
            
        Returns:
            Formatted memory string
        """
        if not memory_context:
            return ""
        
        memory_parts = []
        
        for memory_type, items in memory_context.items():
            if items:
                memory_parts.append(f"\n{memory_type.title()} Memory:")
                for item in items[:3]:  # Limit to top 3 items
                    content = item.content if hasattr(item, 'content') else str(item)
                    memory_parts.append(f"- {content}")
        
        return "\n".join(memory_parts) if memory_parts else ""
    
    def _clean_generated_sql(self, sql: str) -> str:
        """
        Clean and format generated SQL
        
        Args:
            sql: Raw generated SQL
            
        Returns:
            Cleaned SQL
        """
        if not sql:
            return ""
        
        # Remove common prefixes
        sql = sql.strip()
        
        # Remove markdown code blocks
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1]) if len(lines) > 2 else sql
        
        # Remove trailing semicolon and whitespace
        sql = sql.rstrip("; \n\t")
        
        # Ensure it starts with SELECT (basic validation)
        if not sql.upper().strip().startswith("SELECT"):
            # Try to find SELECT in the text
            sql_upper = sql.upper()
            select_idx = sql_upper.find("SELECT")
            if select_idx != -1:
                sql = sql[select_idx:]
        
        return sql.strip()
