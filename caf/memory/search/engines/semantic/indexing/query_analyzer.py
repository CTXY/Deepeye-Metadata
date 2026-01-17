# LLM-driven Query Analyzer - Intelligent query decomposition for semantic search

import logging
import json
from typing import Dict, Any, List
from dataclasses import dataclass

from caf.llm.client import BaseLLMClient

logger = logging.getLogger(__name__)

@dataclass
class QueryAnalysis:
    """Structured output from query analysis"""
    extracted_keywords: List[str]
    expanded_keywords: List[str]


@dataclass
class Condition:
    """Single condition component within an intent query plan"""
    field: str
    operator: str
    value: str


@dataclass
class Aggregation:
    """Aggregation operation possibly applied to a field"""
    operation: str  # e.g., COUNT, AVG
    field: str      # may be empty when not specified


@dataclass
class IntentAnalysis:
    """Structured global intent (pseudo-query plan) extracted from NL question"""
    select_targets: List[str]
    primary_entity: List[str]
    conditions: List[Condition]
    aggregations: List[Aggregation]
    grouping: List[str]
    ordering: List[str]

class LLMQueryAnalyzer:
    """
    LLM-driven Query Analyzer
    
    Replaces regex-based value extraction with intelligent LLM analysis.
    Provides structured keywords and literal values for downstream retrieval.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self._intent_prompt_template = self._build_intent_prompt_template()
        
        logger.debug("LLMQueryAnalyzer initialized")
    
    def _build_intent_prompt_template(self) -> str:
        """Build the LLM prompt template for global intent structuring"""
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
    
    def analyze_query(self, user_question: str) -> IntentAnalysis:
        """Stage 1: Structuring the Global Intent (Pseudo-Query Plan)"""
        prompt = self._intent_prompt_template.format(NL_QUESTION=user_question)

        logger.debug(f"Analyzing global intent: {user_question[:100]}...")
        response_text = self.llm_client.call_with_messages(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        print('*************************************')
        print(response_text)
        print('*************************************')
        intent = self._parse_intent_response(response_text)
        logger.debug(
            "Global intent analysis completed: %d select_targets, %d conditions",
            len(intent.select_targets),
            len(intent.conditions),
        )
        return intent
            
    def _parse_intent_response(self, llm_response: str) -> IntentAnalysis:
        """Parse LLM response for global intent into IntentAnalysis"""
        response_text = llm_response.strip()
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise ValueError("No valid JSON found in LLM response for intent")

        json_text = response_text[start_idx:end_idx + 1]
        parsed: Dict[str, Any] = json.loads(json_text)

        def _to_string_list(value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return [str(v).strip() for v in value if v is not None and str(v).strip()]
            if isinstance(value, str):
                text = value.strip()
                return [text] if text else []
            # Fallback: single non-list scalar
            return [str(value).strip()] if str(value).strip() else []

        select_targets = _to_string_list(parsed.get("select_targets", []))
        primary_entity = _to_string_list(parsed.get("primary_entity", []))

        # Conditions normalization
        conditions: List[Condition] = []
        conditions_raw = parsed.get("conditions", [])
        if isinstance(conditions_raw, dict):
            conditions_raw = [conditions_raw]
        if isinstance(conditions_raw, list):
            for cond in conditions_raw:
                if not isinstance(cond, dict):
                    continue
                field_val = cond.get("field", "")
                operator_val = cond.get("operator", "")
                value_val = cond.get("value", "")
                field = str(field_val).strip()
                operator = str(operator_val).strip()
                # Preserve numbers as strings for uniform downstream handling
                value = str(value_val).strip() if value_val is not None else ""
                if field or operator or value:
                    conditions.append(Condition(field=field, operator=operator, value=value))

        # Aggregations normalization: accept list of strings or list of objects {function, field}
        aggregations_raw = parsed.get("aggregations", [])
        aggregations: List[Aggregation] = []
        if isinstance(aggregations_raw, list):
            for item in aggregations_raw:
                if isinstance(item, dict):
                    func = str(item.get("function", "")).strip()
                    field_val = item.get("field", "")
                    field = str(field_val).strip() if field_val is not None else ""
                    if func or field:
                        aggregations.append(Aggregation(operation=func.upper() if func else "", field=field))
                elif isinstance(item, str):
                    text = item.strip()
                    if text:
                        aggregations.append(Aggregation(operation=text.upper(), field=""))
        elif isinstance(aggregations_raw, dict):
            func = str(aggregations_raw.get("function", "")).strip()
            field_val = aggregations_raw.get("field", "")
            field = str(field_val).strip() if field_val is not None else ""
            if func or field:
                aggregations.append(Aggregation(operation=func.upper() if func else "", field=field))
        elif isinstance(aggregations_raw, str):
            text = aggregations_raw.strip()
            if text:
                aggregations.append(Aggregation(operation=text.upper(), field=""))
        # Deduplicate while preserving order using (operation, field.lower())
        seen_pairs: set = set()
        deduped_aggs: List[Aggregation] = []
        for agg in aggregations:
            key = (agg.operation, agg.field.lower())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            deduped_aggs.append(agg)
        aggregations = deduped_aggs

        # Grouping normalization: allow string or list
        grouping_raw = parsed.get("grouping", [])
        grouping = _to_string_list(grouping_raw)

        # Ordering normalization: allow object {field, direction}, list of such objects, or strings
        ordering_raw = parsed.get("ordering", [])
        ordering: List[str] = []
        if isinstance(ordering_raw, dict):
            field = str(ordering_raw.get("field", "")).strip()
            # direction = str(ordering_raw.get("direction", "")).strip()
            if field:
                ordering.append(f"{field}" )
            elif field:
                ordering.append(field)
        elif isinstance(ordering_raw, list):
            for item in ordering_raw:
                if isinstance(item, dict):
                    field = str(item.get("field", "")).strip()
                    # direction = str(item.get("direction", "")).strip()
                    if field:
                        ordering.append(f"{field}")
                    elif field:
                        ordering.append(field)
                elif isinstance(item, str):
                    text = item.strip()
                    if text:
                        ordering.append(text)
        elif isinstance(ordering_raw, str):
            text = ordering_raw.strip()
            if text:
                ordering.append(text)

        return IntentAnalysis(
            select_targets=select_targets,
            primary_entity=primary_entity,
            conditions=conditions,
            aggregations=aggregations,
            grouping=grouping,
            ordering=ordering,
        )


                
