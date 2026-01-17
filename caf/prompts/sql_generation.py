"""
SQL Generation Prompts

Prompts used for generating SQL queries from natural language questions.
"""


def get_comment_prompt(question: str, evidence: str = None) -> str:
    """
    Generate comment prompt with question and evidence
    
    Based on DAMO-ConvAI's comment generation
    
    Args:
        question: Natural language question
        evidence: External knowledge/evidence (optional)
        
    Returns:
        Formatted comment prompt string
    """
    if evidence and evidence.strip():
        question_prompt = f"-- Question: {question}"
        knowledge_prompt = f"-- Hint (i.e., Evidence) (CRITICAL - READ CAREFULLY): {evidence}"
        pattern_prompt = """-- CRITICAL: The Hint above is provided by the user and is of GREAT SIGNIFICANCE for SQL writing. 
-- You MUST carefully read and STRICTLY FOLLOW ALL requirements in the user-provided hint.
-- IMPORTANT: If the hint contains mapping relationships (e.g., "X -> table.column = 'value'"), you MUST use EXACTLY the table and column names specified in the hint. DO NOT use similar columns or alternative table names - use ONLY what is explicitly specified in the hint mapping.
-- Using valid SQLite and strictly adhering to the hint requirements, answer the above question."""
        return f"{knowledge_prompt}\n{pattern_prompt}\n{question_prompt}"
    else:
        question_prompt = f"-- Question: {question}"
        pattern_prompt = "-- Using valid SQLite, answer the above question."
        return f"{pattern_prompt}\n{question_prompt}"


def get_cot_prompt() -> str:
    """
    Generate chain of thought prompt for SQL generation
    
    Returns:
        Chain of thought prompt string
    """
    return """Generate the SQL after thinking step by step:

IMPORTANT: Please provide your response in the following format:
1. First, explain your reasoning step by step
2. Then provide the final SQL query enclosed in a code block like this:

```sql
[YOUR SQL QUERY HERE]
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



