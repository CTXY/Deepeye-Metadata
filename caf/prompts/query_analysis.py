"""
Query Analysis Prompts

Prompts used for analyzing and decomposing natural language queries.
"""


def get_intent_prompt_template() -> str:
    """
    Build the LLM prompt template for global intent structuring
    
    Returns:
        Intent prompt template string with placeholder for {NL_QUESTION}
    """
    return (
        "You are an expert data analyst. Your task is to decompose a user's natural language "
        "question into a structured JSON format that represents the query's semantic intent. "
        "Do NOT assume any specific database schema. Focus only on the components present in the question.\n\n"
        "The JSON structure should identify the following components:\n"
        "- \"select_targets\": What is the user asking to see, return, or calculate? (e.g., a list of items, a count, an average).\n"
        "- \"primary_entity\": What are the main subjects or tables the user is asking about? (e.g., [\"schools\"], [\"products\", \"customers\"]).\n"
        "- \"conditions\": A list of filters or criteria applied to the data. Each condition should have:\n"
        "  - \"field\": The attribute being filtered (e.g., \"average score in Math\", \"funding type\").\n"
        "  - \"operator\": The comparison being made (e.g., \"over\", \"is\", \"not more than\").\n"
        "  - \"value\": The value used for comparison (e.g., \"560\", \"directly charter-funded\").\n"
        "- \"aggregations\": Any aggregation functions mentioned (e.g., \"how many\" -> COUNT, \"average\" -> AVG).\n"
        "- \"grouping\": Any attributes the results should be grouped by.\n"
        "- \"ordering\": How the results should be sorted.\n\n"
        "Now here is your task:\n"
        "User's question: \"{NL_QUESTION}\"\n\n"
        "Please provide the structured JSON output."
    )



