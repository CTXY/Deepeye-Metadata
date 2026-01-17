"""
Verification Prompts

Prompts used for verifying join paths and representation ambiguities.
"""


def get_join_path_verification_prompt(
    candidate_descriptions: list,
    similarity_scores: list,
    statistical_features: list
) -> str:
    """
    Build LLM prompt for batch join path verification
    
    Args:
        candidate_descriptions: List of candidate descriptions (source and target)
        similarity_scores: List of similarity scores
        statistical_features: List of statistical features dicts
        
    Returns:
        Formatted verification prompt
    """
    prompt_parts = [
        "I have identified several candidate join paths between database tables based on value overlap.",
        "For each candidate, I need you to determine if it represents a valid join relationship.",
        "",
        "A valid join path should:",
        "- Connect related entities (e.g., customer_id in orders table joins to id in customers table)",
        "- Have semantic meaning (not just coincidental value overlap)",
        "- Be useful for querying related data",
        "",
        "For each candidate, respond with:",
        "- 'A' if it represents a valid join path",
        "- 'B' if it does not represent a valid join path",
        "",
        "Here are the candidates to evaluate:",
        "",
    ]
    
    for idx, (candidate_desc, score, features) in enumerate(
        zip(candidate_descriptions, similarity_scores, statistical_features), 1
    ):
        prompt_parts.append(f"Candidate {idx}:")
        prompt_parts.append(f"  {candidate_desc}")
        prompt_parts.append(f"  Value Similarity: {score:.3f}")
        if features:
            features_str = ", ".join(f"{k}={v}" for k, v in features.items())
            prompt_parts.append(f"  Shared Features: {features_str}")
        prompt_parts.append("")
    
    prompt_parts.append(
        "Please respond with a JSON array where each element is either 'A' or 'B', "
        "corresponding to each candidate in order. Example: [\"A\", \"B\", \"A\"]"
    )
    
    return "\n".join(prompt_parts)


def get_representation_ambiguity_verification_prompt(
    pair_descriptions: list,
    statistics: list
) -> str:
    """
    Build LLM prompt for batch representation ambiguity verification
    
    Args:
        pair_descriptions: List of column pair descriptions
        statistics: List of statistics dicts with n_a, n_b, n_ab, ratio_a, ratio_b
        
    Returns:
        Formatted verification prompt
    """
    prompt_parts = [
        "I have identified several pairs of database columns that have a 1-to-1 data mapping.",
        "For each pair, I need you to determine if they represent the same real-world entity",
        "in different formats (e.g., ID vs Name vs Code), or if they are distinct attributes.",
        "",
        "IMPORTANT: Representation Ambiguity means the SAME entity in different formats.",
        "Examples of Representation Ambiguity:",
        "- school_id (integer) vs school_name (string) - same school, different format",
        "- product_code (string) vs product_id (integer) - same product, different format",
        "",
        "Examples that are NOT Representation Ambiguity:",
        "- county_name vs city_name - different entities (county contains multiple cities)",
        "- start_time vs end_time - different attributes of the same event",
        "- latitude vs longitude - different attributes of the same location",
        "",
        "For each pair, respond with:",
        "- 'A' if they represent the same entity in different formats (Representation Ambiguity)",
        "- 'B' if they are distinct attributes or different entities",
        "",
        "Here are the pairs to evaluate:",
        "",
    ]

    for idx, (pair_desc, stats) in enumerate(zip(pair_descriptions, statistics), 1):
        prompt_parts.append(f"Pair {idx}:")
        prompt_parts.append(f"  {pair_desc}")
        prompt_parts.append(
            f"  Statistics: N_A={stats['n_a']}, N_B={stats['n_b']}, N_AB={stats['n_ab']} "
            f"(ratio_A={stats['ratio_a']:.3f}, ratio_B={stats['ratio_b']:.3f})"
        )
        prompt_parts.append("")

    prompt_parts.append(
        "Please respond with a JSON array where each element is either 'A' or 'B', "
        "corresponding to each pair in order. Example: [\"A\", \"B\", \"A\"]"
    )

    return "\n".join(prompt_parts)



