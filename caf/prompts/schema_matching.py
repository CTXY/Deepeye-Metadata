"""
Schema Matching Prompts

Prompts used for matching user queries to database schemas.
"""


def get_schema_matching_prompt(
    user_question: str,
    target_phrase: str,
    candidates_text: str,
    top_k: int
) -> str:
    """
    Build LLM prompt for schema matching
    
    Args:
        user_question: User's original question
        target_phrase: Current phrase being matched
        candidates_text: Formatted text of candidate schemas
        top_k: Number of schemas to select
        
    Returns:
        Formatted schema matching prompt
    """
    return f"""You are a schema matching expert. Your task is to select the most relevant database schemas (columns) for a specific phrase extracted from a user question.

**Context - User Question:**
{user_question}

**Target Phrase:**
"{target_phrase}"

**Task:**
Select the top {top_k} most relevant schemas (table.column format) that best match the target phrase "{target_phrase}" in the context of the user question.

**Candidate Schemas:**
{candidates_text}

**Selection Criteria:**
When evaluating each candidate schema, consider:

1. **Context Analysis**: What role does the phrase "{target_phrase}" play in the question?
   - Is it a filter condition (e.g., "in California", "greater than 50")?
   - Is it a query target (e.g., "school name", "student count")?
   - Is it a primary entity (e.g., "schools", "students")?

2. **Value Match (HIGH CONFIDENCE SIGNAL)**: 
   - ✅ EXACT VALUE MATCH indicates the column contains the exact value mentioned in the phrase
   - ⚠️ Value Match indicates partial or fuzzy match
   - This is a strong indicator that the schema is relevant

3. **Semantic Match**:
   - Does the schema's description/name semantically align with the phrase?
   - Consider synonyms, abbreviations, and domain-specific terminology

4. **Data Type Compatibility**:
   - Does the schema's data type make sense for the phrase?
   - (e.g., numeric types for numbers, text types for names/locations)

**Output Format:**
Return a JSON object with the following structure:
```json
{{
    "selected_schemas": ["table1.column1", "table2.column2", ...],
    "reasoning": "Brief explanation of why these schemas were selected, focusing on value matches and semantic alignment"
}}
```

**Important Notes:**
- Select exactly {top_k} schemas (or fewer if there are not enough good candidates)
- Prioritize schemas with EXACT VALUE MATCH (✅) as they have high confidence
- Consider the context of the full question when making selections
- If a schema has both good semantic match AND value match, it should be prioritized

**Your Selection:**"""


def get_value_extraction_prompt(filter_phrase: str) -> str:
    """
    Build prompt for value extraction from filter phrases
    
    Args:
        filter_phrase: The filter phrase to extract values from
        
    Returns:
        Formatted value extraction prompt
    """
    return f'''You are a value extraction expert. Given a filter phrase, extract any core values and provide common synonyms or abbreviations.

**Task**: Analyze the phrase and determine if it contains specific values that could be found in database columns.

**Rules**:
1. If the phrase contains a specific value (location, number, name, etc.), extract it
2. Generate common synonyms, abbreviations, or alternative representations
3. If it's purely a schema description (like "mailing address"), return null for value
4. Focus on values that would actually appear in database data

**Examples**:
- "in California" → {{"value": "California", "expansions": ["CA", "Calif"]}}
- "greater than 50" → {{"value": "50", "expansions": []}}
- "mailing state address" → {{"value": null, "expansions": []}}
- "in Orange County" → {{"value": "Orange County", "expansions": ["Orange", "OC"]}}

**Input**: "{filter_phrase}"

**Output** (JSON only):
```json
{{
    "value": "extracted_value_or_null",
    "expansions": ["synonym1", "synonym2"]
}}
```'''



