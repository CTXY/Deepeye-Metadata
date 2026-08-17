from .prompt_template import *
from typing import List, Dict, Any, Optional
from app.db_utils import get_database_schema_profile
from app.config import config


class PromptFactory:
    
    @staticmethod
    def get_memory_aware_schema_profile(
        data_item,
        use_caf_mapping: bool = False,
        use_memory: bool = True,
    ) -> str:
        """
        Build database schema profile, taking memory overrides into account when available.
        Priority:
        1) memory_schema_overrides (from memory augmentation)
        2) database_schema_after_mapping (when use_caf_mapping is True)
        3) database_schema_after_schema_linking
        """
        schema_metadata = getattr(data_item, 'schema_metadata', None)
        join_relationships = getattr(data_item, 'join_relationships', None)

        if use_memory and getattr(data_item, "memory_schema_overrides", None) is not None:
            schema = data_item.memory_schema_overrides
        else:
            if use_caf_mapping and getattr(data_item, "database_schema_after_mapping", None) is not None:
                schema = data_item.database_schema_after_mapping
            else:
                schema = data_item.database_schema_after_schema_linking

        return get_database_schema_profile(
            schema,
            schema_metadata=schema_metadata,
            join_relationships=join_relationships
        )
    
    @staticmethod
    def get_enhanced_database_schema_profile(
        data_item,
        use_caf_mapping: bool = False,
        use_memory: bool = True,
    ) -> str:
        """Backward-compatible wrapper for callers; now delegates to memory-aware profile."""
        return PromptFactory.get_memory_aware_schema_profile(
            data_item,
            use_caf_mapping=use_caf_mapping,
            use_memory=use_memory,
        )

    @staticmethod
    def should_use_memory(stage: str) -> bool:
        memory_cfg = config.memory_augmentation_config
        if not memory_cfg.enabled:
            return False
        if stage == "generation":
            return memory_cfg.use_in_generation
        if stage == "revision":
            return memory_cfg.use_in_revision
        if stage == "selection":
            return memory_cfg.use_in_selection
        raise ValueError(f"Unknown stage: {stage}")

    @staticmethod
    def should_use_context_graph(stage: str) -> bool:
        if not PromptFactory.should_use_memory(stage):
            return False
        enabled_strategies = set(config.memory_augmentation_config.strategies)
        return "context_graph" in enabled_strategies or "mapping_based" in enabled_strategies
    
    @staticmethod
    def get_sql_generation_hint(
        data_item,
        use_caf_mapping: bool = False,
        use_memory: bool = True,
    ) -> str:
        """
        Get hint string for SQL generation/revision/selection prompts.
        When use_caf_mapping is True and data_item has mapping_hint, return evidence + mapping_hint.
        Priority: mapping_hint (user-verified) > guidance_hint (historical patterns)
        """
        from app.pipeline.memory_augmentation.context_graph_offline_loader import (
            format_mapping_historical_qa,
        )

        parts: List[str] = []
        if getattr(data_item, "evidence", None):
            parts.append(data_item.evidence)

        if use_caf_mapping:
            mapping_hint = getattr(data_item, "mapping_hint", None)
            if mapping_hint:
                parts.append(mapping_hint)
            mapping_historical_qa = getattr(data_item, "mapping_historical_qa", None)
            if mapping_historical_qa:
                parts.append(format_mapping_historical_qa(mapping_historical_qa))

        if use_memory:
            # Add guidance_hint if mapping_hint is not available
            # Priority: mapping_hint (user-verified) > guidance_hint (historical patterns)
            guidance_hint = getattr(data_item, "guidance_hint", None)
            if guidance_hint and not getattr(data_item, "mapping_hint", None):
                parts.append(guidance_hint)
            if getattr(data_item, "memory_summary", None):
                parts.append(data_item.memory_summary)

        return "\n\n".join([part for part in parts if part]).strip()
    
    @staticmethod
    def format_keywords_extraction_prompt(question: str, hint: str) -> str:
        return KEYWORDS_EXTRACTION_PROMPT.format(QUESTION=question, HINT=hint)
    
    @staticmethod
    def format_direct_linking_prompt(database_schema: str, question: str, hint: str) -> str:
        return DIRECT_LINKING_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint)
    
    @staticmethod
    def format_skeleton_sql_generation_prompt(database_schema: str, question: str, hint: str) -> str:
        return SKELETON_SQL_GENERATION_PROMPT.format(
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
        )
    
    @staticmethod
    def format_dc_sql_generation_prompt(database_schema: str, question: str, hint: str) -> str:
        return DC_SQL_GENERATION_PROMPT.format(
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
        )
    
    @staticmethod
    def format_icl_sql_generation_prompt(few_shot_examples: List[Dict[str, Any]], database_schema: str, question: str, hint: str) -> str:
        few_shot_examples = "\n".join(
            [f"- Example {i+1}:\nQuestion: {example['question']}\nSQL: {example['sql']}" for i, example in enumerate(few_shot_examples)]
        )
        return ICL_SQL_GENERATION_PROMPT.format(
            FEW_SHOT_EXAMPLES=few_shot_examples, 
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
        )

    @staticmethod
    def format_execution_checker_prompt(database_schema: str, question: str, hint: str, sql: str, execution_result: str) -> str:
        return EXECUTION_CHECKER_PROMPT.format(
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
            QUERY=sql, 
            RESULT=execution_result
        )
    
    @staticmethod
    def format_common_checker_prompt(database_schema: str, question: str, hint: str, sql: str, suggestions: str) -> str:
        return COMMON_CHECKER_PROMPT.format(
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
            QUERY=sql, 
            SUGGESTIONS=suggestions
        )
    
    @staticmethod
    def format_br_pair_selection_prompt(database_schema: str, question: str, hint: str, query_a: str, result_a: str, query_b: str, result_b: str) -> str:
        return BR_PAIR_SELECTION_PROMPT.format(
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
            QUERY_A=query_a, 
            RESULT_A=result_a, 
            QUERY_B=query_b, 
            RESULT_B=result_b
        )
    