"""
Centralized Prompt Management for CAF

This module provides centralized access to all prompts used throughout the CAF system.
Prompts are organized by functional area for better maintainability.
"""

from .sql_generation import (
    get_comment_prompt,
    get_cot_prompt,
)
from .sql_evaluation import (
    get_sql_evaluator_prompt,
)
from .query_analysis import (
    get_intent_prompt_template,
)
from .schema_matching import (
    get_schema_matching_prompt,
    get_value_extraction_prompt,
)
from .metadata_generation import (
    get_database_analysis_prompt,
    get_table_analysis_prompt,
    get_column_analysis_prompt,
    get_query_generation_prompt,
)
from .verification import (
    get_join_path_verification_prompt,
    get_representation_ambiguity_verification_prompt,
)
from .dependency_parsing import (
    get_dependency_refinement_prompt,
)

__all__ = [
    # SQL Generation
    'get_comment_prompt',
    'get_cot_prompt',
    # SQL Evaluation
    'get_sql_evaluator_prompt',
    # Query Analysis
    'get_intent_prompt_template',
    # Schema Matching
    'get_schema_matching_prompt',
    'get_value_extraction_prompt',
    # Metadata Generation
    'get_database_analysis_prompt',
    'get_table_analysis_prompt',
    'get_column_analysis_prompt',
    'get_query_generation_prompt',
    # Verification
    'get_join_path_verification_prompt',
    'get_representation_ambiguity_verification_prompt',
    # Dependency Parsing
    'get_dependency_refinement_prompt',
]



