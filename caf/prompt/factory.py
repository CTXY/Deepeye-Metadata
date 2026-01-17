"""
Prompt Factory for CAF Metadata Generation

This module provides a PromptFactory class similar to app/prompt/factory.py,
which formats prompt templates with actual data for LLM calls.
"""

from typing import Dict, Any, List, Optional
import json
import pandas as pd

from .metadata_generation.prompt_template import (
    DATABASE_ANALYSIS_PROMPT,
    TABLE_ANALYSIS_PROMPT,
    TABLE_SEMANTICS_PROMPT,
    COLUMN_ANALYSIS_PROMPT,
    COLUMN_ANALYSIS_WITH_TOP_K_PROMPT,
    QUERY_GENERATION_PROMPT,
    NATURAL_DESCRIPTION_PROMPT,
    RELATIONSHIP_SEMANTICS_PROMPT
)


class PromptFactory:
    """
    Factory class for formatting prompt templates with actual data.
    
    Similar to app.prompt.factory.PromptFactory, this class provides
    static methods to format prompts used in CAF metadata generation.
    """
    
    @staticmethod
    def format_natural_description_prompt(
        table_name: str,
        column_sections: List[str]
    ) -> str:
        """
        Format natural language description generation prompt.
        
        Args:
            table_name: Name of the table
            column_sections: List of formatted column section strings
            
        Returns:
            Formatted prompt string
        """
        return NATURAL_DESCRIPTION_PROMPT.format(
            TABLE_NAME=table_name,
            COLUMN_SECTIONS="\n".join(column_sections)
        )

    @staticmethod
    def format_database_properties_prompt(
        database_id: str,
        table_info: List[str]
    ) -> str:
        """
        Format database properties analysis prompt.
        
        Args:
            database_id: Database identifier
            table_info: List of formatted table information strings
            
        Returns:
            Formatted prompt string for database analysis
        """
        return DATABASE_ANALYSIS_PROMPT.format(
            DATABASE_ID=database_id,
            TABLE_INFO="\n".join(table_info)
        )

    @staticmethod
    def format_column_analysis_prompt(
        table_md: str,
        instructions: List[str],
        output_schema: Dict[str, Any]
    ) -> str:
        """
        Format column analysis prompt.
        
        DEPRECATED: Use format_column_analysis_with_data_prompt instead.
        This method is kept for backward compatibility.
        
        Args:
            table_md: Markdown table with sample data
            instructions: List of analysis instructions
            output_schema: Expected output schema as dict
            
        Returns:
            Formatted prompt string for column analysis
        """
        return PromptFactory.format_column_analysis_with_data_prompt(
            table_markdown=table_md,
            instructions=instructions,
            output_schema=output_schema
        )
    
    @staticmethod
    def format_column_analysis_with_data_prompt(
        table_markdown: str,
        instructions: List[str],
        output_schema: Dict[str, Any],
        table_name: str = "",
        table_description: str = "",
        columns_to_analyze: Optional[List[str]] = None,
        column_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> str:
        """
        Format column analysis prompt with sample data.
        
        Args:
            table_markdown: Markdown table with sample data
            instructions: List of analysis instructions
            output_schema: Expected output schema as dict
            table_name: Name of the table (optional)
            table_description: Description of the table to provide context (optional)
            columns_to_analyze: List of column names to analyze (optional, for explicit instruction)
            column_metadata: Dictionary mapping column names to their metadata (optional, for whole_column_name)
            
        Returns:
            Formatted prompt string for column analysis
        """
        table_desc = table_description if table_description else "No description available"
        
        # Format columns to analyze list
        if columns_to_analyze:
            columns_list = ", ".join(columns_to_analyze)
            columns_text = f"- {columns_list}"
        else:
            columns_text = "- (All columns in the data table)"
        
        # Format column metadata (whole_column_name information)
        metadata_text = ""
        if column_metadata and columns_to_analyze:
            metadata_lines = []
            for col_name in columns_to_analyze:
                col_meta = column_metadata.get(col_name, {})
                whole_name = col_meta.get('whole_column_name', '')
                if whole_name and str(whole_name).strip() and whole_name != col_name:
                    metadata_lines.append(f"- {col_name}: Whole Column Name = {whole_name}")
            if metadata_lines:
                metadata_text = "\n**Column Metadata:**\n" + "\n".join(metadata_lines) + "\n"
        
        return COLUMN_ANALYSIS_PROMPT.format(
            TABLE_NAME=table_name,
            TABLE_DESCRIPTION=table_desc,
            TABLE_MD=table_markdown,
            COLUMNS_TO_ANALYZE=columns_text,
            COLUMN_METADATA=metadata_text,
            INSTRUCTIONS="\n".join(instructions),
            OUTPUT_SCHEMA=json.dumps(output_schema, indent=2)
        )
    
    @staticmethod
    def format_column_analysis_with_top_k_prompt(
        table_name: str,
        column_name: str,
        table_description: str,
        top_values_text: str,
        instructions: List[str],
        output_schema: Dict[str, Any],
        whole_column_name: Optional[str] = None
    ) -> str:
        """
        Format column analysis prompt using top_k_values.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            table_description: Table description
            top_values_text: Formatted text of top values
            instructions: List of analysis instructions
            output_schema: Expected output schema as dict
            whole_column_name: Whole column name (optional, for metadata context)
            
        Returns:
            Formatted prompt string for column analysis
        """
        whole_name = whole_column_name if whole_column_name and whole_column_name != column_name else column_name
        
        return COLUMN_ANALYSIS_WITH_TOP_K_PROMPT.format(
            TABLE_NAME=table_name,
            COLUMN_NAME=column_name,
            WHOLE_COLUMN_NAME=whole_name,
            TABLE_DESCRIPTION=table_description if table_description else 'N/A',
            TOP_VALUES_TEXT=top_values_text,
            INSTRUCTIONS="\n".join(instructions),
            OUTPUT_SCHEMA=json.dumps(output_schema, indent=2)
        )

    @staticmethod
    def format_query_generation_prompt(
        table_name: str,
        column_name: str,
        description: str
    ) -> str:
        """
        Format query generation prompt.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            description: Column description
            
        Returns:
            Formatted prompt string for query generation
        """
        desc = description or "No description available"
        return QUERY_GENERATION_PROMPT.format(
            COLUMN_NAME=column_name,
            DESCRIPTION=desc
        )

    @staticmethod
    def format_table_semantics_prompt(
        table_name: str,
        db_domain: str,
        db_description: str,
        topology_summary: str,
        table_detail: str,
        unique_columns: List[str]
    ) -> str:
        """
        Format table semantics analysis prompt.
        
        Args:
            table_name: Name of the table
            db_domain: Database domain
            db_description: Database description
            topology_summary: Topology summary string
            table_detail: Detailed table information
            unique_columns: List of unique column names
            
        Returns:
            Formatted prompt string
        """
        unique_cols_note = ""
        if unique_columns:
            unique_cols_note = (
                "\n\n**IMPORTANT: Pay special attention to the Unique Columns "
                "listed above. These columns define this table's unique value and purpose.**"
            )
        
        return TABLE_SEMANTICS_PROMPT.format(
            TABLE_NAME=table_name,
            DB_DOMAIN=db_domain,
            DB_DESCRIPTION=db_description,
            TOPOLOGY_SUMMARY=topology_summary,
            TABLE_DETAIL=table_detail,
            UNIQUE_COLUMNS_NOTE=unique_cols_note
        )

    @staticmethod
    def _format_column_description(metadata: Dict[str, Any]) -> str:
        """Format column description for LLM prompt"""
        parts = []
        
        table_name = metadata.get('table_name', '')
        column_name = metadata.get('column_name', '')
        whole_name = metadata.get('whole_column_name', '')
        name_to_use = whole_name if whole_name else column_name
        parts.append(f"{table_name}.{name_to_use}")
        
        # Add short_description first (contains global context)
        short_desc = metadata.get('short_description', '')
        if short_desc and str(short_desc).lower() not in ["nan", "none", "null", ""]:
            parts.append(f"Short Description: {short_desc}")
        
        
        data_type = metadata.get('data_type', '')
        if data_type and str(data_type).lower() not in ["nan", "none", "null", ""]:
            parts.append(f"Type: {data_type}")
        
        # Add PRIMARY KEY indicator
        is_pk = metadata.get('is_primary_key', False)
        if is_pk:
            parts.append("PRIMARY KEY")
        
        return " | ".join(parts)

    @staticmethod
    def format_join_path_verification_prompt(
        candidates: List[Dict[str, Any]],
        col_metadata: Dict[str, Dict[str, Any]],
        signatures: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format join path verification prompt for LLM evaluation.
        
        Args:
            candidates: List of candidate join paths (as dicts with keys:
                       source_table, source_column, target_table, target_column,
                       similarity_score, statistical_features)
            col_metadata: Dictionary mapping column IDs (table.column) to metadata dicts
            signatures: Optional dictionary mapping column IDs to ColumnSignature objects
        
        Returns:
            Formatted prompt string for LLM verification
        """
        prompt_parts = [
            "I have identified candidate join paths based on value overlap.",
            "Please evaluate whether each represents a valid FOREIGN KEY relationship.",
            "",
            "=== CORE PRINCIPLE ===",
            "",
            "**A column can be a join key ONLY if: (1) it serves as an IDENTIFIER, AND (2) it references the SAME business entity.**",
            "",
            "Two columns with overlapping values are NOT a valid join if they represent DIFFERENT business concepts:",
            "- ✗ tcgplayerGroupId (external platform ID) ←→ id (internal database ID) → INVALID",
            "- ✗ mcmId (Magic Card Market ID) ←→ id (internal ID) → INVALID",
            "- ✗ Charter School Number (charter attribute) ←→ School ID (school identifier) → INVALID",
            "",
            "✓ Valid joins: customer_id ←→ id (both reference customers), set_code ←→ code (both reference sets)",
            "",
            "**CRITICAL: Check column names and descriptions!**",
            "If descriptions mention different systems/platforms/contexts, they are NOT the same identifier.",
            "",
            "=== REJECT These Patterns ===",
            "",
            "1. **Two columns representing the SAME ATTRIBUTE** (both are descriptive properties, NOT identifiers)",
            "   → CRITICAL: If both columns describe the same property (e.g., both are 'county names', both are 'district names'),",
            "     they are ATTRIBUTES, not join keys. Even if values overlap, they should NOT be joined.",
            "   → Example: County (county name) ←→ cname (county name) → INVALID (both are attribute columns)",
            "   → Example: District (district name) ←→ District Name (district name) → INVALID (both are attribute columns)",
            "   → Example: County Name (county name) ←→ cname (county name) → INVALID (both are attribute columns)",
            "   → Key insight: Check short_description and description carefully!",
            "     If both columns describe the same semantic concept (location, name, type, etc.), they are attributes, not join keys.",
            "",
            "2. **Different business contexts** (external IDs vs internal IDs, different platforms)",
            "   → Example: tcgplayerGroupId (TCGplayer platform) ←→ id (internal DB) → INVALID",
            "   → Example: mcmId (Magic Card Market) ←→ id (internal) → INVALID",
            "   → Even if values overlap, they reference different systems!",
            "",
            "3. **Low-cardinality or categorical attributes** (low cardinality or contains 'type'/'status'/'class')",
            "   → Properties for filtering/grouping, NOT identifiers",
            "   → Example: 'District Type', 'School Type', 'status', 'category' are INVALID",
            "",
            "4. **Measurements or metrics** (size, count, amount, price, total, rate, etc.)",
            "   → Numeric values, NOT identifiers",
            "",
            "5. **Boolean/binary flags** (isActive, hasFoil, etc.)",
            "   → Create cartesian products with no business meaning",
            "",
            "6. **Descriptive text attributes** (description, note, comment, text, name)",
            "   → Narrative content, NOT references",
            "",
            "=== ACCEPT These Patterns ===",
            "",
            "✓ ID/Code fields: customer_id ←→ id, set_code ←→ code",
            "✓ Clear FK patterns: orders.customer_id ←→ customers.id",
            "✓ High-cardinality identifiers: UUIDs, serial numbers, unique codes",
            "",
            "=== Evaluation Checklist ===",
            "",
            "For each candidate, ask:",
            "1. **Are both columns attributes describing the SAME property?**",
            "   → Check short_description and description FIRST!",
            "   → If both columns describe the same semantic concept (e.g., 'county name', 'district name', 'charter school number'),",
            "     they are ATTRIBUTES, not join keys → REJECT even if values overlap.",
            "   → Example: County (county name) and cname (county name) are both attributes → REJECT",
            "   → Only proceed if one is clearly an IDENTIFIER (ID/Code) and the other is a FOREIGN KEY reference.",
            "",
            "2. **Business semantics**: Do BOTH columns reference the SAME entity/concept?",
            "   → Check column names, short_description, and descriptions carefully!",
            "   → External platform IDs (tcgplayer*, mcm*, etc.) ≠ Internal database IDs",
            "",
            "3. **Identifier nature**: Can BOTH columns serve as identifiers (not properties)?",
            "   → IDs/codes ✓ vs attributes/metrics ✗",
            "   → If both are attributes (descriptive properties), they should NOT be joined",
            "",
            "=== Candidates to Evaluate ===",
            "",
        ]
        
        for idx, candidate in enumerate(candidates, 1):
            # Handle both dict and object formats
            if isinstance(candidate, dict):
                source_table = candidate.get('source_table', '')
                source_column = candidate.get('source_column', '')
                target_table = candidate.get('target_table', '')
                target_column = candidate.get('target_column', '')
                similarity_score = candidate.get('similarity_score', 0.0)
                statistical_features = candidate.get('statistical_features')
            else:
                source_table = getattr(candidate, 'source_table', '')
                source_column = getattr(candidate, 'source_column', '')
                target_table = getattr(candidate, 'target_table', '')
                target_column = getattr(candidate, 'target_column', '')
                similarity_score = getattr(candidate, 'similarity_score', 0.0)
                statistical_features = getattr(candidate, 'statistical_features', None)
            
            col_a_id = f"{source_table}.{source_column}"
            col_b_id = f"{target_table}.{target_column}"
            
            meta_a = col_metadata.get(col_a_id, {})
            meta_b = col_metadata.get(col_b_id, {})
            
            col_a_desc = PromptFactory._format_column_description(meta_a)
            col_b_desc = PromptFactory._format_column_description(meta_b)
            
            # Get sample values and signatures (CRITICAL for LLM to understand the data!)
            sample_a = []
            sample_b = []
            sig_a = None
            sig_b = None
            if signatures:
                sig_a = signatures.get(col_a_id)
                sig_b = signatures.get(col_b_id)
                # Handle ColumnSignature objects (with attributes) or dicts
                if sig_a:
                    if hasattr(sig_a, 'sample_values'):
                        sample_a = sig_a.sample_values or []
                    elif isinstance(sig_a, dict):
                        sample_a = sig_a.get('sample_values', [])
                if sig_b:
                    if hasattr(sig_b, 'sample_values'):
                        sample_b = sig_b.sample_values or []
                    elif isinstance(sig_b, dict):
                        sample_b = sig_b.get('sample_values', [])
            
            prompt_parts.append(f"Candidate {idx}:")
            prompt_parts.append(f"  Source: {col_a_id}")
            if sig_a:
                # Handle ColumnSignature objects or dicts
                if hasattr(sig_a, 'data_type'):
                    data_type_a = sig_a.data_type or 'UNKNOWN'
                    distinct_count_a = getattr(sig_a, 'distinct_count', 'UNKNOWN')
                elif isinstance(sig_a, dict):
                    data_type_a = sig_a.get('data_type', 'UNKNOWN')
                    distinct_count_a = sig_a.get('distinct_count', 'UNKNOWN')
                else:
                    data_type_a = 'UNKNOWN'
                    distinct_count_a = 'UNKNOWN'
                prompt_parts.append(f"    - Data Type: {data_type_a}")
                prompt_parts.append(f"    - Cardinality: {distinct_count_a} distinct values")
            if col_a_desc:
                prompt_parts.append(f"    - Info: {col_a_desc}")
            if sample_a:
                sample_str = ", ".join(repr(v) for v in sample_a)
                prompt_parts.append(f"    - Sample Values: [{sample_str}]")
            
            prompt_parts.append(f"  Target: {col_b_id}")
            if sig_b:
                # Handle ColumnSignature objects or dicts
                if hasattr(sig_b, 'data_type'):
                    data_type_b = sig_b.data_type or 'UNKNOWN'
                    distinct_count_b = getattr(sig_b, 'distinct_count', 'UNKNOWN')
                elif isinstance(sig_b, dict):
                    data_type_b = sig_b.get('data_type', 'UNKNOWN')
                    distinct_count_b = sig_b.get('distinct_count', 'UNKNOWN')
                else:
                    data_type_b = 'UNKNOWN'
                    distinct_count_b = 'UNKNOWN'
                prompt_parts.append(f"    - Data Type: {data_type_b}")
                prompt_parts.append(f"    - Cardinality: {distinct_count_b} distinct values")
            if col_b_desc:
                prompt_parts.append(f"    - Info: {col_b_desc}")
            if sample_b:
                sample_str = ", ".join(repr(v) for v in sample_b)
                prompt_parts.append(f"    - Sample Values: [{sample_str}]")
            
            prompt_parts.append(f"  Containment Score: {similarity_score:.3f}")
            if statistical_features:
                features_str = ", ".join(f"{k}={v}" for k, v in statistical_features.items())
                prompt_parts.append(f"  Shared Features: {features_str}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "",
            "=== Response Format ===",
            "",
            "Respond with JSON array (one entry per candidate):",
            '{"candidate_id": N, "reasoning": "brief explanation", "decision": "Yes"|"No"}',
            "",
            "Example:",
            "[",
            '  {"candidate_id": 1, "reasoning": "sets.id is PK, set_translations.id is not PK but references sets - FK relationship with score 1.0", "decision": "Yes"},',
            '  {"candidate_id": 2, "reasoning": "tcgplayerGroupId is external platform ID, different system from internal DB id", "decision": "No"},',
            '  {"candidate_id": 3, "reasoning": "District Type is categorical attribute, not identifier", "decision": "No"},',
            '  {"candidate_id": 4, "reasoning": "County and cname are both county name attributes, not identifiers. Both describe the same property, should not be joined", "decision": "No"},',
            "]"
        ])
        
        return "\n".join(prompt_parts)

    @staticmethod
    def format_relationship_semantics_prompt(
        source_table: str,
        target_table: str,
        source_columns: List[str],
        target_columns: List[str],
        cardinality: Optional[str],
        source_description: str,
        target_description: str,
        source_role: Optional[str],
        target_role: Optional[str],
        source_sample_table_md: str,
        target_sample_table_md: str
    ) -> str:
        """
        Format relationship semantics analysis prompt.
        
        Args:
            source_table: Source table name
            target_table: Target table name
            source_columns: Source column names
            target_columns: Target column names
            cardinality: Cardinality string
            source_description: Source table description
            target_description: Target table description
            source_role: Source table role
            target_role: Target table role
            source_sample_table_md: Markdown sample table for source join keys + context columns
            target_sample_table_md: Markdown sample table for target join keys + context columns
            
        Returns:
            Formatted prompt string
        """
        source_cols_str = ', '.join(source_columns)
        target_cols_str = ', '.join(target_columns)
        
        # Safely handle cardinality (None, NaN, or empty string should show as 'Unknown')
        cardinality_str = 'Unknown'
        if cardinality:
            # Handle pandas NaN and None
            if pd.notna(cardinality) and str(cardinality).strip():
                cardinality_str = str(cardinality)
        
        return RELATIONSHIP_SEMANTICS_PROMPT.format(
            SOURCE_TABLE=source_table,
            SOURCE_COLUMNS=source_cols_str,
            SOURCE_DESCRIPTION=source_description if source_description else 'N/A',
            SOURCE_ROLE=source_role if source_role else 'N/A',
            SOURCE_SAMPLE_TABLE_MD=source_sample_table_md if source_sample_table_md else "No sample rows available",
            TARGET_TABLE=target_table,
            TARGET_COLUMNS=target_cols_str,
            TARGET_DESCRIPTION=target_description if target_description else 'N/A',
            TARGET_ROLE=target_role if target_role else 'N/A',
            TARGET_SAMPLE_TABLE_MD=target_sample_table_md if target_sample_table_md else "No sample rows available",
            CARDINALITY=cardinality_str
        )


