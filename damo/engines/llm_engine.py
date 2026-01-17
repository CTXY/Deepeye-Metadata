# LLM-based NL2SQL reasoning engine

from openai import OpenAI
import logging
import time
import json
from typing import Dict, Any, Optional, List

from .base_engine import BaseEngine
from data_handler import DatabaseSchema
from config import LLMConfig

logger = logging.getLogger(__name__)

class LLMEngine(BaseEngine):
    """
    LLM-based NL2SQL reasoning engine
    
    Based on DAMO-ConvAI's LLM approach, adapted for CAF integration
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.config = config
        
        # Set up OpenAI client
        if config.provider == "openai":
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url if config.base_url else None
            )
            if config.base_url:
                logger.info(f"Using custom OpenAI API base URL: {config.base_url}")
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        
        logger.info(f"LLMEngine initialized with {config.provider} model: {config.model_name}")
    
    def generate_sql(self, question: str, schema: DatabaseSchema, 
                    context: Dict[str, Any], evidence: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
        """
        Generate SQL using LLM with schema and context
        
        Args:
            question: Natural language question
            schema: Database schema
            context: Additional context including memory and sample data
            evidence: External knowledge/evidence
            
        Returns:
            Tuple of (Generated SQL query, LLM interaction info)
        """
        # Build prompt
        prompt = self._build_prompt(question, schema, context, evidence)
        
        print("================================================")
        print(prompt)
        print("================================================")
        
        # Call LLM
        try:
            response = self._call_llm(prompt)
            sql = self._extract_sql_from_response(response)
            
            # Clean and validate SQL
            sql = self._clean_generated_sql(sql)
            
            if not sql:
                raise ValueError("Empty SQL generated")
            
            # Record LLM interaction information
            llm_interaction = {
                "input_prompt": prompt,
                "raw_output": response,
                "extracted_sql": sql,
                "prompt_length": len(prompt),
                "response_length": len(response) if response else 0
            }

            print("================================================")
            print(response)
            print("================================================")
        
            return sql, llm_interaction
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise
    
    def _build_prompt(self, question: str, schema: DatabaseSchema, 
                     context: Dict[str, Any], evidence: Optional[str] = None) -> str:
        """
        Build complete prompt for LLM including schema, context, and examples
        
        Based on DAMO-ConvAI's prompt structure
        """
        prompt_parts = []
        
        # Add schema information
        schema_prompt = self._generate_schema_prompt(schema, context.get("sample_data", {}))
        prompt_parts.append(schema_prompt)
        
        # Add memory context if available
        formatted_memory_text = context.get("formatted_memory_text", "")
        if formatted_memory_text and formatted_memory_text.strip():
            prompt_parts.append(f"\n-- CONTEXTUAL KNOWLEDGE AND HISTORICAL EXPERIENCE --")
            prompt_parts.append("The following sections provide additional context to help you write accurate SQL queries:\n")
            prompt_parts.append(formatted_memory_text)
        
        # Add evidence and question
        comment_prompt = self._generate_comment_prompt(question, evidence)
        prompt_parts.append(comment_prompt)
        
        # Add chain of thought prompt
        cot_prompt = self._generate_cot_prompt()
        prompt_parts.append(cot_prompt)
        
        # Add SQL start
        prompt_parts.append("SELECT ")
        
        return "\n\n".join(prompt_parts)
    
    def _generate_schema_prompt(self, schema: DatabaseSchema, sample_data: Dict[str, List[Dict]] = None) -> str:
        """
        Generate schema prompt with CREATE statements and sample data
        
        Based on DAMO-ConvAI's schema generation
        """
        schema_parts = []
        
        for table_name in schema.table_names:
            # Build CREATE TABLE statement
            create_statement = self._build_create_statement(table_name, schema)
            
            # Add sample data if available
            if sample_data and table_name in sample_data:
                sample_rows = sample_data[table_name]
                if sample_rows:
                    sample_comment = self._format_sample_data(table_name, sample_rows)
                    create_statement += f"\n{sample_comment}"
            
            schema_parts.append(create_statement)
        
        return "\n\n".join(schema_parts)
    
    def _build_create_statement(self, table_name: str, schema: DatabaseSchema) -> str:
        """Build CREATE TABLE statement for a table"""
        columns = []
        
        # Get columns for this table
        table_id = schema.table_names.index(table_name)
        
        for i, col_info in enumerate(schema.column_names):
            if col_info["table_id"] == table_id:
                col_name = col_info["column_name"]
                col_type = schema.column_types[i] if i < len(schema.column_types) else "TEXT"
                
                # Check if primary key
                is_pk = any(pk["column_id"] == i for pk in schema.primary_keys)
                pk_constraint = " PRIMARY KEY" if is_pk else ""
                
                # Build column definition with type and constraints
                column_def = f"    {col_name} {col_type.upper()}{pk_constraint}"
                
                # Add description as comment if available
                if (schema.table_descriptions and 
                    table_name in schema.table_descriptions and 
                    col_name in schema.table_descriptions[table_name]):
                    description = schema.table_descriptions[table_name][col_name]
                    column_def += f", # {description}"
                else:
                    column_def += f","
                
                columns.append(column_def)
        
        create_stmt = f"CREATE TABLE {table_name}\n(\n" + "\n".join(columns) + "\n)"
        return create_stmt
    
    def _format_sample_data(self, table_name: str, sample_rows: List[Dict], num_rows: int = 3) -> str:
        """Format sample data as SQL comment"""
        if not sample_rows:
            return ""
        
        # Handle reserved table names
        safe_table_name = f"`{table_name}`" if table_name.lower() in ['order', 'by', 'group'] else table_name
        
        # Get column names and format rows
        if sample_rows:
            columns = list(sample_rows[0].keys())
            rows_text = []
            
            for row in sample_rows[:num_rows]:
                row_values = [str(row.get(col, 'NULL')) for col in columns]
                rows_text.append(" ".join(f"{val:>15}" for val in row_values))
            
            header = " ".join(f"{col:>15}" for col in columns)
            sample_text = header + "\n" + "\n".join(rows_text)
            
            comment = f"/* \n{num_rows} example rows: \nSELECT * FROM {safe_table_name} LIMIT {num_rows}; \n{sample_text} \n*/"
            return comment
        
        return ""
    
    def _generate_comment_prompt(self, question: str, evidence: Optional[str] = None) -> str:
        """
        Generate comment prompt with question and evidence
        
        Based on DAMO-ConvAI's comment generation
        """
        if evidence and evidence.strip():
            question_prompt = f"-- Question: {question}"
            knowledge_prompt = f"-- Hint (i.e., Evidence) (CRITICAL - READ CAREFULLY): {evidence}"
            # pattern_prompt = "-- IMPORTANT: The Hint above is provided by the user and is of great significance for SQL writing. You MUST carefully read and strictly follow all requirements in the user-provided hint. Using valid SQLite and strictly adhering to the hint requirements, answer the following questions for the tables provided above."
            pattern_prompt = """-- CRITICAL: The Hint above is provided by the user and is of GREAT SIGNIFICANCE for SQL writing. 
-- You MUST carefully read and STRICTLY FOLLOW ALL requirements in the user-provided hint.
-- IMPORTANT: If the hint contains mapping relationships (e.g., "X -> table.column = 'value'"), you MUST use EXACTLY the table and column names specified in the hint. DO NOT use similar columns or alternative table names - use ONLY what is explicitly specified in the hint mapping.
-- Using valid SQLite and strictly adhering to the hint requirements, answer the following questions for the tables provided above."""
            
            return f"{knowledge_prompt}\n{pattern_prompt}\n{question_prompt}"
        else:
            question_prompt = f"-- Question: {question}"
            pattern_prompt = "-- Using valid SQLite, answer the following questions for the tables provided above."
            return f"{pattern_prompt}\n{question_prompt}"
    
    def _generate_cot_prompt(self) -> str:
        """Generate chain of thought prompt"""
        return """Generate the SQL after thinking step by step:

IMPORTANT: Please provide your response in the following format:
1. First, explain your reasoning step by step
2. Then provide the final SQL query enclosed in a code block like this:
Your response MUST follow exactly:
```json
{
  "analysis": "<short step-by-step reasoning>",
  "sql": "<single SQLite query only>"
}
```

CRITICAL REQUIREMENTS:
- DO NOT OUTPUT UNNECESSARY FIELDS: Only include columns in your SELECT clause that are specifically requested by the user. Do not add extra columns, even if they might seem related or useful. If the user asks for specific information, provide ONLY that information - nothing more, nothing less.
- Use backticks (`) for ALL COLUMN NAMES in the SQL query, e.g., `column name`, `School Name`, `free meal count (k-12)`
- This is mandatory for SQLite compatibility when column names contain spaces, special characters, or parentheses
- Ensure the SQL is valid SQLite syntax
- Only return the executable SQL query in the code block

Example of correct column name usage:
- `School Name` (not School Name)
- `free meal count (k-12)` (not "free meal count (k-12)")
- T1.`enrollment (k-12)` (not T1.enrollment (k-12))
"""
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API with the constructed prompt
        
        Based on DAMO-ConvAI's OpenAI integration
        """
        try:
            if self.config.provider == "openai":
                # Use ChatCompletion API for GPT models  
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    timeout=self.config.timeout
                )
                return response.choices[0].message.content
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
                
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
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
        
        # Prefer structured JSON extraction if available
        json_sql = self._extract_sql_from_json_response(response)
        if json_sql:
            return self._clean_generated_sql(json_sql)
        
        # Fallback: extract SQL from code blocks / patterns
        import re
        
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

    def _extract_sql_from_json_response(self, response: str) -> Optional[str]:
        """Attempt to parse SQL from a JSON-formatted answer."""
        if not response:
            return None
        
        candidates = [response.strip()]
        
        # Look for explicit ```json blocks
        import re
        json_block_pattern = r'```json\s*(.*?)```'
        json_blocks = re.findall(json_block_pattern, response, re.DOTALL | re.IGNORECASE)
        candidates.extend(block.strip() for block in json_blocks)
        
        for candidate in candidates:
            if not candidate:
                continue
            
            stripped = candidate.strip()
            
            # Remove leading/trailing triple backticks if still present
            if stripped.startswith("```") and stripped.endswith("```"):
                stripped = stripped[3:-3].strip()
            
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            
            if isinstance(parsed, dict):
                sql_value = parsed.get("sql") or parsed.get("final_sql")
                if isinstance(sql_value, str) and sql_value.strip():
                    return sql_value.strip()
        
        return None
    
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
    