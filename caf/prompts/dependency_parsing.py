"""
Dependency Parsing Prompts

Prompts used for refining dependency parsing results.
"""


def get_dependency_refinement_prompt(question: str, stanza_summary: str) -> str:
    """
    Construct prompt for LLM to refine Stanza results
    
    Args:
        question: Original question
        stanza_summary: Formatted Stanza results
        
    Returns:
        Prompt string
    """
    return f"""You are given a question and its dependency parsing results from Stanza (a syntactic parser). Your task is to review, refine, and adjust these results based on semantic understanding.

Question: "{question}"

Stanza Parsing Results:
{stanza_summary}

Please review the Stanza results and:
1. Identify any missing constraints for entities
2. Correct any incorrect constraint assignments
3. Add constraints that should be associated with entities based on semantic understanding
4. Remove or adjust constraints that don't make semantic sense
5. Ensure each entity has ALL its complete constraint phrases
6. Identify cases where a constraint should modify multiple parallel entities
7. Correct cases where constraints are not properly attached to the right entities

CRITICAL REQUIREMENTS:
- Each constraint MUST be a COMPLETE modifying phrase (not fragments)
- Extract entities layer by layer (hierarchical extraction)
- Each constraint must be a continuous text fragment from the original question
- If an entity has no constraints, use an empty array []

Output your refined analysis in JSON format:
{{
  "entities": [
    {{
      "entity": "entity_name",
      "constraints": [
        "complete modifying phrase 1",
        "complete modifying phrase 2"
      ]
    }}
  ]
}}

Now review and refine the Stanza results:"""



