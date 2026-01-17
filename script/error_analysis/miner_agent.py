"""
Miner Agent - Extracts error patterns and generates guidance

This is the core component that analyzes the difference between incorrect and correct SQL,
identifies the error pattern, and generates actionable guidance.
"""

import logging
import json
import time
from typing import Optional, Dict, Any
from openai import OpenAI

from .config import Config
from .models import MinerOutput, RetrievalKey, GuidanceStructure, ComparativeStrategy

logger = logging.getLogger(__name__)


class MinerAgent:
    """
    Miner Agent that analyzes SQL errors and generates insights
    
    Core responsibilities:
    1. Difference Attribution: Identify WHY incorrect SQL fails
    2. Pattern Extraction: Extract retrieval keys (NL triggers + SQL risk atoms)
    3. Guidance Generation: Create comparative guidance
    """
    
    SYSTEM_PROMPT = """You are an expert SQL Analyst and Educator.
You are provided with a triplet:
1. A Natural Language Question (NLQ).
2. An Incorrect SQL query (with Schema Masked, e.g., T1.C1, values as V1, V2).
3. A Correct SQL query (with Schema Masked).

**Your Goal:**
Analyze strictly why the Incorrect SQL fails to meet the specific intent of the NLQ compared to the Correct SQL. Focus on logical patterns, side effects, and operational nuances (e.g., duplicates, precision loss, NULL handling, ranking logic).

**Task 1: Extract Retrieval Signature (The "Trigger")**
Identify the atomic features that would allow us to retrieve this case when a similar risk appears in the future.
- **NL Triggers:** Extract 2-4 **OPERATION/COMPUTATION-RELATED** keywords from the NLQ (NOT domain-specific nouns). 
  * Include: aggregation words (count, sum, average, total), comparison words (highest, lowest, most, least, greater, less), operation words (ratio, difference, rate, percentage, division, multiplication), quantifiers (each, all, any, top N), ranking/ordering words (rank, order, first, last).
  * Exclude: domain nouns (schools, students, websites, employees, products), specific column semantics (name, id, age, salary).
  * Examples: "highest" ✓, "ratio" ✓, "average difference" ✓, "top 5" ✓, "schools" ✗, "enrollment" ✗, "website" ✗
- **SQL Risk Atoms:** Extract the specific SQL keywords/operators in the *Incorrect SQL* that constitute the risky pattern (e.g., ["WHERE", "=", "MAX"], ["/", "No CAST"]). Do NOT include T1, C1, V1, etc.

**Task 2: Generate Comparative Guidance (The "Insight")**
Create a decision matrix comparing the two approaches.
- **Intent:** A SHORT, GENERIC, ABSTRACT description of the operation/computation goal (3-8 words). Focus on the OPERATION, not the domain.
  * Good: "Rank entities by aggregated metric", "Calculate ratio with precision", "Select single top value"
  * Bad: "Rank schools by Writing scores", "Calculate student enrollment rate", "Find highest salary employee"
  * Avoid: domain nouns (schools, students, salary), specific column semantics (Writing, enrollment)
  * Include: operation words (rank, calculate, select, filter, aggregate)
- **Incorrect Strategy:** Describe the logic of the incorrect SQL and its *negative side effect* (e.g., "Returns all tied rows", "Integer truncation", "Missing NULL filter").
- **Correct Strategy:** Describe the logic of the correct SQL and why it is better.
- **Actionable Advice:** A concise rule for future generation.

**Task 3: Evaluate Generalizability (CRITICAL)**
Determine if this error pattern is generalizable and reusable across different databases and domains.

✅ **Generalizable Errors** (Keep these):
- **Operational mistakes**: Wrong aggregation, missing CAST, incorrect JOIN type, NULL handling issues
- **SQL pattern errors**: MAX with WHERE = instead of ORDER BY LIMIT, integer division, missing window function
- **Logical errors**: Wrong ordering, missing GROUP BY, incorrect filtering logic
- **Examples**: "Missing CAST in division", "Using WHERE = MAX instead of LIMIT 1", "Missing NULL filter in ranking"

❌ **Non-Generalizable Errors** (Skip these):
- **Database-specific**: Relies on specific column values, table relationships, or data distributions
- **Question-specific**: Only makes sense in this particular question context
- **Trivial differences**: Only differ in ORDER BY direction, LIMIT number without conceptual difference
- **Schema-dependent**: Error only occurs with specific column types or table structures in this database
- **Examples**: "Wrong column in ORDER BY", "Missing specific table in FROM", "Wrong constant value"

**Generalizability Check:**
Ask yourself:
1. Could this error occur in a completely different database (e.g., movies, employees, products)? ✓
2. Does the error represent a general SQL operational/logical pattern? ✓
3. Would the advice help prevent similar errors in other domains? ✓
4. Is the error about HOW to use SQL, not WHAT schema to use? ✓

If all 4 checks pass → **is_generalizable: true**
If any check fails → **is_generalizable: false**

**Intent Guidelines (CRITICAL):**

✅ GOOD Intents (Abstract, Generic, Operation-Focused):
- "Select single top-value entity"
- "Calculate ratio with precision"
- "Rank entities by metric"
- "Filter by aggregated threshold"
- "Count with NULL handling"
- "Order subset by multiple criteria"

❌ BAD Intents (Question-Specific, Domain-Heavy):
- "Rank schools by Writing scores" → Too specific, mentions "schools" and "Writing"
- "Calculate student enrollment rate" → Domain-specific: "student", "enrollment"
- "Find highest salary employee" → Domain nouns: "salary", "employee"
- "List schools with meal count > 800" → Too specific to question

**Output Format:**
Return a valid JSON object with this structure:
{
  "retrieval_key": {
    "nl_triggers": ["keyword1", "keyword2"],
    "sql_risk_atoms": ["KEYWORD1", "KEYWORD2"]
  },
  "guidance": {
    "intent": "Short generic operation (3-8 words, NO domain nouns)",
    "comparison": {
      "strategy_incorrect": {
        "pattern": "Abstract SQL pattern",
        "implication": "Negative consequence"
      },
      "strategy_correct": {
        "pattern": "Abstract SQL pattern",
        "implication": "Why it's better"
      }
    },
    "actionable_advice": "Concrete rule"
  },
  "is_generalizable": true,  // or false
  "generalizability_reason": "Brief reason (1 sentence)"
}

**IMPORTANT**: If is_generalizable is false, the insight will be discarded. Only return false if the error is truly database-specific or question-specific."""
    
    FEW_SHOT_EXAMPLES = """
**Example 1:**
NLQ: "Which player has the highest overall rating?"
Incorrect (Masked): SELECT T1.C1 FROM T1 WHERE T1.C2 = (SELECT MAX(T1.C2) FROM T1)
Correct (Masked): SELECT T1.C1 FROM T1 ORDER BY T1.C2 DESC LIMIT 1

Output:
{
  "retrieval_key": {
    "nl_triggers": ["highest", "single result"],
    "sql_risk_atoms": ["WHERE", "=", "SELECT", "MAX"]
  },
  "guidance": {
    "intent": "Select single top-value entity",
    "comparison": {
      "strategy_incorrect": {
        "pattern": "WHERE col = (SELECT MAX(col)...)",
        "implication": "Returns ALL records that tie for the maximum value (non-deterministic cardinality)."
      },
      "strategy_correct": {
        "pattern": "ORDER BY col DESC LIMIT 1",
        "implication": "Enforces returning strictly ONE record, ignoring ties."
      }
    },
    "actionable_advice": "When the question implies a singular result ('Which player', 'Who has'), prefer 'ORDER BY ... LIMIT 1'. Use subquery MAX only if 'all tied entities' are explicitly requested."
  },
  "is_generalizable": true,
  "generalizability_reason": "This is a common SQL pattern error that applies to any query selecting a single top-value entity across all domains."
}

**Example 2:**
NLQ: "What is the eligible free rate (count/enrollment)?"
Incorrect (Masked): SELECT T1.C1 / T1.C2 FROM T1
Correct (Masked): SELECT CAST(T1.C1 AS REAL) / T1.C2 FROM T1

Output:
{
  "retrieval_key": {
    "nl_triggers": ["rate", "division"],
    "sql_risk_atoms": ["/"]
  },
  "guidance": {
    "intent": "Calculate ratio with precision",
    "comparison": {
      "strategy_incorrect": {
        "pattern": "col_a / col_b",
        "implication": "Performs integer division in many dialects, truncating decimal results (e.g., 5/2 = 2)."
      },
      "strategy_correct": {
        "pattern": "CAST(col_a AS REAL) / col_b",
        "implication": "Forces floating-point division to preserve precision."
      }
    },
    "actionable_advice": "For division operations involving potential integers, always cast the numerator to FLOAT/REAL to ensure precise decimal results."
  },
  "is_generalizable": true,
  "generalizability_reason": "Integer division precision loss is a universal SQL issue across all databases and domains involving ratio calculations."
}
"""
    
    def __init__(self, 
                 api_key: str = Config.OPENAI_API_KEY,
                 model: str = Config.OPENAI_MODEL,
                 temperature: float = Config.OPENAI_TEMPERATURE,
                 max_tokens: int = Config.OPENAI_MAX_TOKENS):
        """
        Initialize Miner Agent
        
        Args:
            api_key: OpenAI API key
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens for response
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def analyze(self,
                nlq: str,
                masked_incorrect_sql: str,
                masked_correct_sql: str,
                evidence: str = "") -> Optional[MinerOutput]:
        """
        Analyze error pattern and generate guidance
        
        Args:
            nlq: Natural language question
            masked_incorrect_sql: Masked incorrect SQL
            masked_correct_sql: Masked correct SQL
            evidence: Optional evidence/hints
        
        Returns:
            MinerOutput object or None if failed
        """
        try:
            # Build user prompt
            user_prompt = self._build_user_prompt(nlq, masked_incorrect_sql, masked_correct_sql, evidence)
            
            # Call OpenAI API
            logger.debug(f"Calling OpenAI API with model {self.model}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.FEW_SHOT_EXAMPLES},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            raw_output = response.choices[0].message.content
            logger.debug(f"Raw LLM output: {raw_output}")
            
            # Parse JSON
            output_dict = json.loads(raw_output)
            
            # Convert to MinerOutput
            miner_output = self._parse_output(output_dict)
            
            logger.info("Successfully generated miner output")
            return miner_output
            
        except Exception as e:
            logger.error(f"Miner analysis failed: {e}")
            return None
    
    def _build_user_prompt(self, nlq: str, masked_incorrect: str, masked_correct: str, evidence: str) -> str:
        """Build user prompt for current task"""
        prompt = f"""
**Current Task:**
NLQ: {nlq}
"""
        if evidence:
            prompt += f"Evidence/Hint: {evidence}\n"
        
        prompt += f"""Incorrect (Masked): {masked_incorrect}
Correct (Masked): {masked_correct}

Please analyze and return the JSON output.
"""
        return prompt
    
    def _parse_output(self, output_dict: Dict[str, Any]) -> MinerOutput:
        """
        Parse raw output dict into MinerOutput object
        
        Args:
            output_dict: Raw dict from LLM JSON output
        
        Returns:
            MinerOutput object
        """
        # Extract retrieval key
        retrieval_key_data = output_dict.get('retrieval_key', {})
        retrieval_key = RetrievalKey(
            nl_triggers=retrieval_key_data.get('nl_triggers', []),
            sql_risk_atoms=retrieval_key_data.get('sql_risk_atoms', [])
        )
        
        # Extract guidance
        guidance_data = output_dict.get('guidance', {})
        comparison_data = guidance_data.get('comparison', {})
        
        strategy_incorrect = ComparativeStrategy(
            pattern=comparison_data.get('strategy_incorrect', {}).get('pattern', ''),
            implication=comparison_data.get('strategy_incorrect', {}).get('implication', '')
        )
        
        strategy_correct = ComparativeStrategy(
            pattern=comparison_data.get('strategy_correct', {}).get('pattern', ''),
            implication=comparison_data.get('strategy_correct', {}).get('implication', '')
        )
        
        guidance = GuidanceStructure(
            intent=guidance_data.get('intent', ''),
            strategy_incorrect=strategy_incorrect,
            strategy_correct=strategy_correct,
            actionable_advice=guidance_data.get('actionable_advice', '')
        )
        
        # Extract generalizability
        is_generalizable = output_dict.get('is_generalizable', True)
        generalizability_reason = output_dict.get('generalizability_reason', '')
        
        return MinerOutput(
            retrieval_key=retrieval_key, 
            guidance=guidance,
            is_generalizable=is_generalizable,
            generalizability_reason=generalizability_reason
        )

