"""
Metadata Generation Prompts

Prompts used for generating database, table, and column metadata.
"""

import json


def get_database_analysis_prompt(database_id: str, table_info: list) -> str:
    """
    Build prompt for database-level analysis
    
    Args:
        database_id: Database identifier
        table_info: List of table information strings
        
    Returns:
        Formatted database analysis prompt
    """
    return f"""
Analyze the database structure below. Provide a concise description and a business domain.

Database: {database_id}
Tables:
{chr(10).join(table_info)}

Respond in a valid JSON format with keys "description" and "domain".
- "description": A 1-2 sentence summary of the database's business purpose.
- "domain": The most specific business domain (e.g., "E-commerce", "Healthcare").

JSON Analysis:"""


def get_table_analysis_prompt(db_context_line: str, table_prompts: list) -> str:
    """
    Build prompt for batch table analysis
    
    Args:
        db_context_line: Database context description
        table_prompts: List of formatted table prompts
        
    Returns:
        Formatted table analysis prompt
    """
    return f"""
Analyze the following tables from a database. {db_context_line if db_context_line else ''}
For each table, provide a clear 1-2 sentence business description of its purpose.

Tables to analyze:
---
{chr(10).join(['---'+chr(10)+p for p in table_prompts])}
---

Respond with a single JSON object where keys are the table names and values are their descriptions.
Example: {{"table_one": "Describes customers...", "table_two": "Tracks product inventory..."}}

JSON Output:"""


def get_column_analysis_prompt(
    table_md: str,
    instructions: list,
    output_schema: dict
) -> str:
    """
    Build prompt for column analysis
    
    Args:
        table_md: Markdown table with sample data
        instructions: List of analysis instructions
        output_schema: Expected output schema as dict
        
    Returns:
        Formatted column analysis prompt
    """
    return f"""**Role:** Data Analyst

**Task:** Analyze table columns based on the requirements below.

**Data:**

{table_md}

**Requirements:**

{chr(10).join(instructions)}

**Output Format:**

Return a JSON object with this exact schema for each column:

{json.dumps(output_schema, indent=2)}

**JSON Output:**

"""


def get_query_generation_prompt(
    table_name: str,
    column_name: str,
    description: str
) -> str:
    """
    Build prompt for generating search phrases for a column
    
    Args:
        table_name: Name of the table
        column_name: Name of the column
        description: Column description
        
    Returns:
        Formatted query generation prompt
    """
    return f"""You are generating search phrases (keywords/phrases) that users might type when searching for a specific database column.

**Target Column Information:**
- Column Name: {column_name}
- Column Description: {description or "No description available"}

**CRITICAL REQUIREMENT:** 
Generate SEARCH PHRASES (keywords/phrases), NOT complete questions. These should be short phrases that users would type in a search box. Focus on the COLUMN itself, not the table or database context.

**Task:** Generate exactly 3 short search phrases (1-4 words each) that users might type when looking for THIS COLUMN's data:

1. **Exact Match #1**: A direct phrase about THIS COLUMN (e.g., for a "dname" column with description "district segment": "district" or "district segment")
2. **Exact Match #2**: Another direct phrase about THIS COLUMN (e.g., for a "dname" column: "district name" or "school district")
3. **Concept-Based**: An abstract or conceptual phrase using layman terms that relates to THIS COLUMN's semantic meaning (e.g., for a "city" column: "location" or "where")

**Important constraints:**
- Generate PHRASES/KEYWORDS, NOT complete questions (e.g., "district" NOT "What is the district?")
- Focus on the COLUMN itself, not the table name or database context
- Do NOT mention the exact column name if it is a technical code (like "cds", "id", "dname", etc.)
- Use natural, search-friendly phrases
- Keep phrases short (1-4 words typically)
- Each phrase should help users find THIS COLUMN's data specifically
- For Exact Match phrases, use words from the description or related terms (e.g., for "dname" with description "district segment", use "district", "district segment", "district name")

**Output format (JSON array only, no other text):**
["phrase1", "phrase2", "phrase3"]

Where phrase1 and phrase2 are Exact Match, and phrase3 is Concept-Based."""



