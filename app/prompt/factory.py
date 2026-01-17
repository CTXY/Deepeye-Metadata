from .prompt_template import *
from typing import List, Dict, Any, Tuple, Optional
from app.db_utils import get_database_schema_profile


class PromptFactory:
    
    @staticmethod
    def get_enhanced_database_schema_profile(data_item) -> str:
        """
        Get database schema profile with optional schema_metadata and join_relationships.
        
        Args:
            data_item: DataItem with database_schema_after_schema_linking, 
                      optional schema_metadata, and optional join_relationships
            
        Returns:
            Database schema profile string with enhanced metadata and join paths if available
        """
        schema_metadata = getattr(data_item, 'schema_metadata', None)
        join_relationships = getattr(data_item, 'join_relationships', None)
        return get_database_schema_profile(
            data_item.database_schema_after_schema_linking,
            schema_metadata=schema_metadata,
            join_relationships=join_relationships
        )
    
    @staticmethod
    def format_sql_guidance(sql_guidance_items: Optional[List[Dict[str, Any]]]) -> str:
        """
        Format SQL guidance items from historical cases for prompt inclusion.
        
        Args:
            sql_guidance_items: List of guidance items with patterns and insights
            
        Returns:
            Formatted string for prompt
        """
        if not sql_guidance_items:
            return ""
        
        lines = ["## Historical SQL Patterns and Insights:", ""]
        lines.append("The following patterns are derived from analyzing historical NL2SQL cases across different databases.")
        lines.append("[CRITICAL] These patterns represent USER PREFERENCES and ALTERNATIVE APPROACHES, not absolute correctness.")
        lines.append("The 'correct' and 'incorrect' labels indicate user preference patterns, not absolute SQL rules. Note that these examples use DIFFERENT database schemas ")
        lines.append("")
        
        for idx, guidance in enumerate(sql_guidance_items, 1):
            intent = guidance.get('intent', 'Unknown intent')
            lines.append(f"### Case {idx}: {intent}")
            lines.append("")
            
            # Incorrect pattern
            incorrect_sql = guidance.get('qualified_incorrect_sql', '')
            incorrect_strategy = guidance.get('strategy_incorrect', {})
            if incorrect_sql and incorrect_strategy:
                lines.append("** Less Preferred Pattern:**")
                lines.append(f"```example sql")
                lines.append(incorrect_sql)
                lines.append("```")
                if incorrect_strategy.get('pattern'):
                    lines.append(f"  - Pattern: `{incorrect_strategy['pattern']}`")
                if incorrect_strategy.get('implication'):
                    lines.append(f"  - Issue: {incorrect_strategy['implication']}")
                lines.append("")
            
            # Correct pattern
            correct_sql = guidance.get('qualified_correct_sql', '')
            correct_strategy = guidance.get('strategy_correct', {})
            if correct_sql and correct_strategy:
                lines.append("** Preferred Pattern:**")
                lines.append(f"```example sql")
                lines.append(correct_sql)
                lines.append("```")
                if correct_strategy.get('pattern'):
                    lines.append(f"  - Pattern: `{correct_strategy['pattern']}`")
                if correct_strategy.get('implication'):
                    lines.append(f"  - Benefit: {correct_strategy['implication']}")
                lines.append("")
            
            # Actionable advice
            advice = guidance.get('actionable_advice', '')
            if advice:
                lines.append(f"**Actionable Advice:** {advice}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        lines.append("[REMINDER] These are cross-database examples showing PATTERNS, not exact SQL to copy.")
        lines.append("Extract the logical approach and adapt it to YOUR current database schema and question.")
        lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def get_sql_guidance(data_item) -> str:
        """
        Get SQL guidance from historical patterns as REFERENCE material.
        This is LOWER CONFIDENCE information - patterns from other databases for reference only.
        
        Args:
            data_item: DataItem with optional sql_guidance_items
            
        Returns:
            SQL guidance string (empty if not available)
        """
        if hasattr(data_item, 'sql_guidance_items') and data_item.sql_guidance_items:
            return PromptFactory.format_sql_guidance(data_item.sql_guidance_items)
        return ""
    
    @staticmethod
    def format_keywords_extraction_prompt(question: str, hint: str) -> str:
        return KEYWORDS_EXTRACTION_PROMPT.format(QUESTION=question, HINT=hint)
    
    @staticmethod
    def format_direct_linking_prompt(database_schema: str, question: str, hint: str) -> str:
        return DIRECT_LINKING_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint)
    
    @staticmethod
    def format_skeleton_sql_generation_prompt(database_schema: str, question: str, hint: str, sql_guidance: str = "") -> str:
        # sql_guidance already contains full formatting with headers
        guidance_section = f"\n{sql_guidance}" if sql_guidance else ""
        return SKELETON_SQL_GENERATION_PROMPT.format(
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
            SQL_GUIDANCE=guidance_section
        )
    
    @staticmethod
    def format_dc_sql_generation_prompt(database_schema: str, question: str, hint: str, sql_guidance: str = "") -> str:
        # sql_guidance already contains full formatting with headers
        guidance_section = f"\n{sql_guidance}" if sql_guidance else ""
        return DC_SQL_GENERATION_PROMPT.format(
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
            SQL_GUIDANCE=guidance_section
        )
    
    @staticmethod
    def format_icl_sql_generation_prompt(few_shot_examples: List[Dict[str, Any]], database_schema: str, question: str, hint: str, sql_guidance: str = "") -> str:
        few_shot_examples = "\n".join(
            [f"- Example {i+1}:\nQuestion: {example['question']}\nSQL: {example['sql']}" for i, example in enumerate(few_shot_examples)]
        )
        # sql_guidance already contains full formatting with headers
        guidance_section = f"\n{sql_guidance}" if sql_guidance else ""
        return ICL_SQL_GENERATION_PROMPT.format(
            FEW_SHOT_EXAMPLES=few_shot_examples, 
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
            SQL_GUIDANCE=guidance_section
        )

    @staticmethod
    def format_execution_checker_prompt(database_schema: str, question: str, hint: str, sql: str, execution_result: str, sql_guidance: str = "") -> str:
        # sql_guidance already contains full formatting with headers
        # guidance_section = f"\n{sql_guidance}" if sql_guidance else ""
        guidance_section = ""
        return EXECUTION_CHECKER_PROMPT.format(
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
            SQL_GUIDANCE=guidance_section,
            QUERY=sql, 
            RESULT=execution_result
        )
    
    @staticmethod
    def format_common_checker_prompt(database_schema: str, question: str, hint: str, sql: str, suggestions: str, sql_guidance: str = "") -> str:
        # sql_guidance already contains full formatting with headers
        guidance_section = f"\n{sql_guidance}" if sql_guidance else ""
        return COMMON_CHECKER_PROMPT.format(
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
            SQL_GUIDANCE=guidance_section,
            QUERY=sql, 
            SUGGESTIONS=suggestions
        )
    
    @staticmethod
    def format_br_pair_selection_prompt(database_schema: str, question: str, hint: str, sql_guidance: str, query_a: str, result_a: str, query_b: str, result_b: str) -> str:
        # sql_guidance already contains full formatting with headers
        guidance_section = f"\n{sql_guidance}" if sql_guidance else ""
        
        return BR_PAIR_SELECTION_PROMPT.format(
            DATABASE_SCHEMA=database_schema, 
            QUESTION=question, 
            HINT=hint,
            SQL_GUIDANCE=guidance_section,
            QUERY_A=query_a, 
            RESULT_A=result_a, 
            QUERY_B=query_b, 
            RESULT_B=result_b
        )
    