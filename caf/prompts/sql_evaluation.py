"""
SQL Evaluation Prompts

Prompts used for evaluating SQL correctness and categorizing errors.
"""


def get_sql_evaluator_prompt() -> str:
    """
    Get SQL evaluator prompt template
    
    Returns:
        SQL evaluator prompt template string with placeholders for:
        - {question}: User question
        - {db_schema}: Database schema
        - {generated_sql}: Generated SQL query
        - {ground_truth_sql}: Ground truth SQL query
    """
    return '''# Role:
You are an expert SQL Data Analyst and a meticulous NL2SQL System Evaluator.

# Task:
Your task is to analyze a `Generated SQL` query by comparing it to a `Ground Truth SQL` query, in the context of a given `Question` and `DB Schema`. You must determine if the `Generated SQL` is semantically correct. If it is incorrect, you MUST categorize the error using the precise `Error Taxonomy` provided below and offer a concise analysis and a suggestion for correction.

# Context:
This analysis is for a Cognitive Augmentation Framework (CAF). Your output will be parsed programmatically to automatically update the system's memory. Therefore, adhering strictly to the provided error taxonomy and the JSON output format is critical.

# Error Taxonomy:
Use ONLY the categories and sub-categories from this table.

| Error Category (error_category) | Error Sub-category (error_subcategory) |
| :--- | :--- |
| Attribute-related Errors | Attribute Mismatch / Attribute Redundancy / Attribute Missing |
| Table-related Errors | Table Mismatch / Table Redundancy / Table Missing / Join Condition Mismatch |
| Value-related Errors | Join Type Mismatch / Value Mismatch / Data Format Mismatch |
| Operator-related Errors | Comparison Operator / Logical Operator |
| Condition-related Errors | Explicit Condition Mismatch / Explicit Condition Redundancy / Explicit Condition Missing / Implicit Condition Missing |
| Function-related Errors | Aggregate Functions / Window Functions / Date/Time Functions / etc. |
| Clause-related Errors | Clause Missing / Clause Redundancy |
| Subquery-related Errors | Subquery Missing / Subquery Mismatch / Partial Query |
| Other Errors | ASC/DESC / DISTINCT / Other |

# Output Format:
You MUST provide your response in a single, valid JSON object.

For correct SQL:
```json
{{
  "is_correct": true,
  "error_category": null,
  "error_subcategory": null,
  "analysis": "A brief explanation stating that the generated SQL is semantically equivalent to the ground truth.",
  "suggestion": null
}}
```

For incorrect SQL:
```json
{{
  "is_correct": false,
  "error_category": "The exact string from the 'Error Category' column of the taxonomy.",
  "error_subcategory": "The exact string from the 'Error Sub-category' column of the taxonomy.",
  "analysis": "A detailed but concise explanation of WHY the Generated SQL is wrong, referencing specific parts of the query.",
  "suggestion": "A clear, actionable instruction on how to modify the Generated SQL to make it correct."
}}
```

### Your Turn:
Analyze the following inputs and provide your response in the specified JSON format.

* Question: {question}
* DB Schema: {db_schema}
* Generated SQL: {generated_sql}
* Ground Truth SQL: {ground_truth_sql}'''



