"""
Semantic Intent Analyzer - 语义意图维度分析器

实现 Ambiguity Discriminator Matrix 中的语义意图维度分析：
1. 实体/属性归属 (Entity Alignment)
2. 判别性场景生成 (Discriminative Scenario Generation)
"""

import logging
import json
from typing import Dict, List, Optional, Any

from ..types import SemanticIntentProfile, AmbiguousPair, DBElementRef
from ..stores.semantic import SemanticMemoryStore
from ...llm.client import BaseLLMClient

logger = logging.getLogger(__name__)


class SemanticIntentAnalyzer:
    """
    语义意图维度分析器.
    
    利用 LLM 的常识推理，挖掘设计者的意图和业务边界。
    生成判别性场景和触发词，帮助 Agent 正确选择字段。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        semantic_store: Optional[SemanticMemoryStore] = None,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            llm_client: LLM client for semantic analysis
            semantic_store: Optional semantic store for column metadata
            config: Configuration dict with optional parameters
        """
        self.llm_client = llm_client
        self.semantic_store = semantic_store
        self.config = config or {}
        
        # Configuration
        self.temperature = self.config.get("temperature", 0.3)
        self.max_retries = self.config.get("max_retries", 2)

    def analyze_pair(
        self,
        pair: AmbiguousPair,
        database_id: Optional[str] = None,
    ) -> Optional[SemanticIntentProfile]:
        """
        分析一对模糊字段的语义意图维度.
        
        Args:
            pair: 模糊字段对
            database_id: 数据库 ID (用于获取元数据)
        
        Returns:
            SemanticIntentProfile or None if analysis failed
        """
        try:
            # Get column metadata
            metadata_a = self._get_column_metadata(
                pair.database_id or database_id,
                pair.column_a.table_name,
                pair.column_a.column_name,
            )
            metadata_b = self._get_column_metadata(
                pair.database_id or database_id,
                pair.column_b.table_name,
                pair.column_b.column_name,
            )

            logger.info(
                "Analyzing semantic intent for pair: %s.%s <-> %s.%s",
                pair.column_a.table_name,
                pair.column_a.column_name,
                pair.column_b.table_name,
                pair.column_b.column_name,
            )

            # Build prompt
            prompt = self._build_discriminative_analysis_prompt(
                pair, metadata_a, metadata_b
            )

            # Call LLM
            response = self._call_llm_with_retry(prompt)
            if not response:
                return None

            # Parse response
            profile = self._parse_llm_response(response)

            if profile:
                logger.info(
                    "Semantic intent analysis completed: nuance=%s",
                    profile.semantic_nuance[:100] if profile.semantic_nuance else "N/A",
                )

            return profile

        except Exception as e:
            logger.error(
                "Failed to analyze semantic intent for pair %s: %s",
                pair.pair_id,
                e,
            )
            return None

    def _get_column_metadata(
        self,
        database_id: Optional[str],
        table_name: str,
        column_name: str,
    ) -> Dict[str, Any]:
        """
        从 semantic store 获取列元数据.
        """
        metadata = {
            "table_name": table_name,
            "column_name": column_name,
            "description": None,
            "data_type": None,
            "sample_values": [],
        }

        if not self.semantic_store or not database_id:
            return metadata

        try:
            # Bind to database (only if not already bound to avoid repeated loading)
            if self.semantic_store.current_database_id != database_id:
                self.semantic_store.bind_database(database_id)

            # Get column dataframe
            column_df = self.semantic_store.dataframes.get("column")
            if column_df is None or column_df.empty:
                return metadata

            # Find matching column
            col_rows = column_df[
                (column_df["table_name"] == table_name) &
                (column_df["column_name"] == column_name)
            ]

            if col_rows.empty:
                return metadata

            row = col_rows.iloc[0]

            # Extract metadata
            metadata["description"] = row.get("description") or row.get("short_description")
            metadata["data_type"] = row.get("data_type")
            
            # Get sample values from top_k_values
            top_k_values = row.get("top_k_values")
            if isinstance(top_k_values, dict):
                # Take top 5 most frequent values
                sample_values = list(top_k_values.keys())[:5]
                metadata["sample_values"] = [str(v) for v in sample_values]

        except Exception as e:
            logger.debug("Failed to get column metadata: %s", e)

        return metadata

    def _build_discriminative_analysis_prompt(
        self,
        pair: AmbiguousPair,
        metadata_a: Dict[str, Any],
        metadata_b: Dict[str, Any],
    ) -> str:
        """
        构建判别性分析的 LLM prompt.
        
        基于用户提供的模板，生成判别性场景和触发词。
        """
        # Format column information
        col_a_name = metadata_a["column_name"]
        col_a_desc = metadata_a.get("description") or "No description"
        col_a_samples = metadata_a.get("sample_values", [])
        col_a_samples_str = ", ".join(col_a_samples[:5]) if col_a_samples else "N/A"

        col_b_name = metadata_b["column_name"]
        col_b_desc = metadata_b.get("description") or "No description"
        col_b_samples = metadata_b.get("sample_values", [])
        col_b_samples_str = ", ".join(col_b_samples[:5]) if col_b_samples else "N/A"

        table_a = metadata_a["table_name"]
        table_b = metadata_b["table_name"]

        prompt = f"""# Role
You are a Database Expert. You are analyzing two semantically similar columns from a database to help a Text-to-SQL system distinguish them correctly.

# Input Context

**Table A**: {table_a}
**Column A**: {col_a_name}
- Description: {col_a_desc}
- Sample values: {col_a_samples_str}

**Table B**: {table_b}
**Column B**: {col_b_name}
- Description: {col_b_desc}
- Sample values: {col_b_samples_str}

# Background
These two columns have been identified as potentially ambiguous (high semantic similarity or value overlap). Your task is to help disambiguate them by analyzing their semantic intent and usage patterns.

# Task

Analyze the likely semantic difference between Column A and Column B based on:
1. Their names and descriptions
2. Their table context
3. Sample values
4. Standard data modeling practices

Then provide:

## 1. Semantic Nuance
A concise explanation (2-3 sentences) of the core semantic difference between these two columns.

## 2. Entity Alignment
Identify which entity or aspect each column describes:
- Column A Entity: [e.g., "Order entity - when the order was created"]
- Column B Entity: [e.g., "Shipment entity - when the package was shipped"]

## 3. Discriminative Scenarios
Generate TWO distinct user questions (Natural Language Queries):
- **Scenario A**: A question that MUST use Column A (not Column B)
- **Scenario B**: A question that MUST use Column B (not Column A)

Make these questions realistic and specific to the business domain.

## 4. Discriminative Logic
Explain WHY Column A is used for Scenario A and Column B is used for Scenario B. What is the key distinction?

## 5. Trigger Keywords
List trigger keywords that should map to each column:
- **Trigger Keywords A**: [comma-separated list of 3-5 keywords that suggest using Column A]
- **Trigger Keywords B**: [comma-separated list of 3-5 keywords that suggest using Column B]

# Output Format

Respond with a valid JSON object (no markdown code blocks):

{{
  "semantic_nuance": "...",
  "column_a_entity": "...",
  "column_b_entity": "...",
  "scenario_a": "...",
  "scenario_b": "...",
  "discriminative_logic": "...",
  "trigger_keywords_a": ["keyword1", "keyword2", ...],
  "trigger_keywords_b": ["keyword1", "keyword2", ...]
}}

# Important Guidelines

1. Be specific and concrete - avoid generic statements
2. Focus on the DIFFERENCE, not similarities
3. Keywords should be natural language terms users would type
4. Scenarios should be realistic business questions
5. If columns are truly synonyms, state that clearly in semantic_nuance
6. Consider temporal, hierarchical, or domain-specific distinctions

Generate the analysis now:"""

        return prompt

    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """
        调用 LLM，支持重试.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.call_with_messages(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                
                if response and response.strip():
                    return response.strip()
                
            except Exception as e:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                
                if attempt < self.max_retries - 1:
                    continue
                else:
                    return None
        
        return None

    def _parse_llm_response(self, response: str) -> Optional[SemanticIntentProfile]:
        """
        解析 LLM 响应，提取结构化信息.
        """
        try:
            # Clean response (remove markdown code blocks if present)
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                # Remove first and last lines (markdown markers)
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                # Also handle ```json
                if response.startswith("json"):
                    response = response[4:].strip()

            # Parse JSON
            data = json.loads(response)

            # Extract fields
            profile = SemanticIntentProfile(
                semantic_nuance=data.get("semantic_nuance"),
                column_a_entity=data.get("column_a_entity"),
                column_b_entity=data.get("column_b_entity"),
                entity_alignment=None,  # Can be computed from column entities
                scenario_a=data.get("scenario_a"),
                scenario_b=data.get("scenario_b"),
                discriminative_logic=data.get("discriminative_logic"),
                trigger_keywords_a=data.get("trigger_keywords_a", []),
                trigger_keywords_b=data.get("trigger_keywords_b", []),
            )

            # Build entity_alignment summary
            if profile.column_a_entity and profile.column_b_entity:
                profile.entity_alignment = (
                    f"Column A: {profile.column_a_entity} | "
                    f"Column B: {profile.column_b_entity}"
                )

            return profile

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON: %s", e)
            logger.debug("Response was: %s", response[:500])
            return None
        except Exception as e:
            logger.warning("Failed to parse LLM response: %s", e)
            return None












