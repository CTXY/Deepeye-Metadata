#!/usr/bin/env python3
"""
NL2SQL Error Analysis Script

This script analyzes NL2SQL-Bugs-Benchmark data using LLM-based classification to:
1. First classify errors as CALCULATION-RELATED vs SEMANTIC-RELATED  
2. Extract generalizable SQL templates and insights only from calculation-related errors
3. Skip semantic-related errors (database-specific field/table understanding issues)

Usage Example:
export OPENAI_API_KEY="sk-RuCZ5VJcTEaGuy882795Bd35168a46B195B14404EaAb10F4"
export OPENAI_BASE_URL="https://vip.yi-zhan.top/v1"
python /home/yangchenyu/DeepEye-SQL-Metadata/script/caf/analyze_errors.py \
  --input /home/yangchenyu/DeepEye-SQL-Metadata/NL2SQL-Bugs-Benchmark/NL2SQL-Bugs-with-evidence.json \
  --output /home/yangchenyu/DeepEye-SQL-Metadata/output/insights.jsonl \
  --batch-size 5 \
  --delay 1.0

python /home/yangchenyu/DeepEye-SQL-Metadata/script/caf/analyze_errors.py \
  --input /home/yangchenyu/DeepEye-SQL-Metadata/results/bird-dev/qwen3-coder-30b-a3b_incorrect.json \
  --input-format bird_results \
  --question-file /home/yangchenyu/DeepEye-SQL-Metadata/data/bird/dev/dev.json \
  --output /home/yangchenyu/DeepEye-SQL-Metadata/output/bird_insights_qwen3-coder-30b-a3b_incorrect.jsonl \
  --batch-size 5 --delay 1.0

python /home/yangchenyu/DeepEye-SQL-Metadata/script/caf/analyze_errors.py \
  --input /home/yangchenyu/DeepEye-SQL-Metadata/NL2SQL-Bugs/gpt_4o_mini_damo_wo_memory_results_on_train_set.json \
  --input-format gpt4o_results \
  --output /home/yangchenyu/DeepEye-SQL-Metadata/output/gpt4o_mini_insights.jsonl \
  --batch-size 5 --delay 1.0
"""

import json
import os
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import openai
from openai import OpenAI
import logging
import argparse
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SQLPair:
    """Represents a pair of correct and incorrect SQL for the same question"""
    question: str
    db_id: str
    correct_sql: str
    incorrect_sql: str
    error_types: List[Dict]
    evidence: Optional[str] = None
    question_id: Optional[int] = None

@dataclass 
class ExtractedInsight:
    """Represents an extracted insight from SQL error analysis"""
    question: str
    db_id: str
    error_template: str
    comparative_insight: Dict
    original_correct_sql: str
    original_incorrect_sql: str
    question_id: Optional[int] = None

class NL2SQLErrorAnalyzer:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize the analyzer with OpenAI API settings.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # LLM will classify errors dynamically, no pre-defined types needed
        
        # System prompt for the LLM - Two-stage analysis
        self.system_prompt = """You are a Senior SQL Analyst designed to classify SQL errors and extract generalizable patterns.

Your task is to analyze a triplet of (NLQ, Incorrect SQL, Correct SQL) in two stages:

## Stage 1: Error Classification
First, determine if the error is:
- **CALCULATION-RELATED**: Logic errors, aggregation scope issues, function usage errors, mathematical operations, join logic problems, etc.
- **SEMANTIC-RELATED**: Wrong table/column selection, incorrect field understanding, database-specific attribute misunderstandings, etc.

## Stage 2: Pattern Extraction (Only for CALCULATION-RELATED errors)
If classified as CALCULATION-RELATED, generate `error_template` and `comparative_insight`:

### Rules for `error_template` (Anonymization):
1.  **Tables:** Replace table names with `T1`, `T2`, `T3`...
2.  **Columns:** Replace column names with `C1`, `C2`, `C3`...
    * *Crucial:* You must maintain **consistency**. If `district_id` is mapped to `C1`, it must be `C1` everywhere (in SELECT, JOIN, and WHERE).
3.  **Values:** Replace specific literals (numbers, strings) with `[VALUE]`.
4.  **Irrelevant Details:** If a WHERE clause is identical in both SQLs and unrelated to the error logic, replace it with `...` to keep the template clean.
5.  **Structure:** Keep all SQL keywords (`SELECT`, `JOIN`, `GROUP BY`, `AVG`, `IN`, `EXISTS`) exactly as they are.

### Rules for `comparative_insight` (Logic Contrast):
1.  Do NOT just say "Pred is wrong".
2.  **Format:** You must explain the logic of BOTH:
    * "**Pred Logic:** [What does this structure mathematically/logically calculate?]"
    * "**Gold Logic:** [What does this structure mathematically/logically calculate?]"
3.  **Abstraction:** Use the generic `T1`, `C1` terms in your explanation. Do not use specific business terms (like "Crime", "Bank", "Student") to ensure the insight is applicable to any database.

## Output Format:
For CALCULATION-RELATED errors:
{
  "error_classification": "CALCULATION-RELATED",
  "error_template": "...",
  "comparative_insight": {
    "pred_logic": "...",
    "gold_logic": "...",
    "key_difference": "..."
  }
}

For SEMANTIC-RELATED errors:
{
  "error_classification": "SEMANTIC-RELATED",
  "reason": "Brief explanation why this is semantic-related"
}"""

    def load_data(self, json_path: str, input_format: str = "nl2sql", question_path: Optional[str] = None) -> List[Dict]:
        """
        Load analysis data supporting multiple input formats.
        
        Args:
            json_path: path to primary input file.
            input_format: structure of the input file (nl2sql, bird_results, or gpt4o_results).
            question_path: optional path to question metadata (required for bird_results).
        """
        logger.info(f"Loading data from {json_path} (format={input_format})")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if input_format == "nl2sql":
            logger.info(f"Loaded {len(raw_data)} NL2SQL records")
            return raw_data
        
        if input_format == "bird_results":
            if not question_path:
                raise ValueError("--question-file is required when using input-format=bird_results")
            
            with open(question_path, 'r', encoding='utf-8') as qf:
                question_data = json.load(qf)
            
            question_index = {item["question_id"]: item for item in question_data}
            incorrect_entries = raw_data.get("incorrect_sqls", [])
            logger.info(f"Loaded {len(incorrect_entries)} incorrect SQL entries from model outputs")
            
            records = []
            for entry in incorrect_entries:
                question_id = entry.get("question_id")
                question_info = question_index.get(question_id)
                
                if not question_info:
                    logger.warning(f"Question metadata missing for question_id={question_id}, skipping")
                    continue
                
                gold_sql = entry.get("gold_sql")
                pred_sql = entry.get("pred_sql")
                
                if not gold_sql or not pred_sql:
                    logger.warning(f"Missing SQL for question_id={question_id}, skipping")
                    continue
                
                # Add gold SQL (treated as correct)
                records.append({
                    "question": question_info["question"],
                    "db_id": question_info["db_id"],
                    "sql": gold_sql,
                    "label": True,
                    "error_types": [],
                    "evidence": question_info.get("evidence"),
                    "question_id": question_id
                })
                
                # Add predicted SQL (treated as incorrect)
                records.append({
                    "question": question_info["question"],
                    "db_id": question_info["db_id"],
                    "sql": pred_sql,
                    "label": False,
                    "error_types": [],
                    "evidence": question_info.get("evidence"),
                    "question_id": question_id
                })
            
            logger.info(f"Constructed {len(records)} normalized records from bird_results input")
            return records
        
        if input_format == "gpt4o_results":
            results = raw_data.get("results", [])
            logger.info(f"Loaded {len(results)} total records from gpt4o_results")
            
            # Filter only records where ex_correct is False
            incorrect_records = [r for r in results if r.get("ex_correct") == False]
            logger.info(f"Filtered to {len(incorrect_records)} records with ex_correct=False")
            
            records = []
            for entry in incorrect_records:
                question_id = entry.get("question_id")
                ground_truth_sql = entry.get("ground_truth_sql")
                generated_sql = entry.get("generated_sql")
                
                if not ground_truth_sql or not generated_sql:
                    logger.warning(f"Missing SQL for question_id={question_id}, skipping")
                    continue
                
                # Add ground truth SQL (treated as correct)
                records.append({
                    "question": entry.get("question"),
                    "db_id": entry.get("db_id"),
                    "sql": ground_truth_sql,
                    "label": True,
                    "error_types": [],
                    "evidence": entry.get("evidence"),
                    "question_id": question_id
                })
                
                # Add generated SQL (treated as incorrect)
                records.append({
                    "question": entry.get("question"),
                    "db_id": entry.get("db_id"),
                    "sql": generated_sql,
                    "label": False,
                    "error_types": [],
                    "evidence": entry.get("evidence"),
                    "question_id": question_id
                })
            
            logger.info(f"Constructed {len(records)} normalized records from gpt4o_results input")
            return records
        
        raise ValueError(f"Unsupported input format: {input_format}")

    def group_by_question(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """Group records by question and db_id combination."""
        groups = defaultdict(list)
        for record in data:
            key = f"{record['question']}|||{record['db_id']}"
            groups[key].append(record)
        return groups

    def extract_sql_pairs(self, grouped_data: Dict[str, List[Dict]]) -> List[SQLPair]:
        """Extract SQL pairs (correct, incorrect) from grouped data."""
        sql_pairs = []
        
        for key, records in grouped_data.items():
            question, db_id = key.split('|||')
            
            correct_records = [r for r in records if r['label'] == True]
            incorrect_records = [r for r in records if r['label'] == False]
            
            # For each correct SQL, pair with incorrect SQLs
            for correct in correct_records:
                for incorrect in incorrect_records:
                    # Remove pre-filtering - let LLM classify all error types
                    pair = SQLPair(
                        question=question,
                        db_id=db_id,
                        correct_sql=correct['sql'],
                        incorrect_sql=incorrect['sql'],
                        error_types=incorrect['error_types'],
                        evidence=incorrect.get('evidence'),
                        question_id=incorrect.get('question_id') or correct.get('question_id')
                    )
                    sql_pairs.append(pair)
        
        logger.info(f"Extracted {len(sql_pairs)} SQL pairs for analysis (all error types)")
        return sql_pairs


    def analyze_sql_pair(self, pair: SQLPair) -> Optional[ExtractedInsight]:
        """Analyze a single SQL pair using OpenAI API with two-stage classification."""
        
        user_prompt = f"""Here is the data to analyze:

**Question:** {pair.question}

**Incorrect SQL (Pred):**
{pair.incorrect_sql}

**Correct SQL (Gold):**
{pair.correct_sql}

**Please analyze and generate the JSON response:**"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse the response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Find JSON content between ```json and ``` or just parse directly
                if '```json' in content:
                    json_start = content.find('```json') + 7
                    json_end = content.find('```', json_start)
                    json_content = content[json_start:json_end].strip()
                elif content.startswith('{') and content.endswith('}'):
                    json_content = content
                else:
                    # Try to find JSON pattern
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx != 0:
                        json_content = content[start_idx:end_idx]
                    else:
                        logger.warning(f"Could not find JSON in response: {content}")
                        return None
                
                result = json.loads(json_content)
                
                # Check error classification
                classification = result.get('error_classification', '').upper()
                
                if classification == 'SEMANTIC-RELATED':
                    logger.info(f"    🔍 Classified as SEMANTIC-RELATED: {result.get('reason', 'No reason provided')}")
                    return None  # Skip semantic-related errors
                
                elif classification == 'CALCULATION-RELATED':
                    # Extract template and insight for calculation-related errors
                    if 'error_template' not in result or 'comparative_insight' not in result:
                        logger.warning(f"    ⚠ Missing template/insight in CALCULATION-RELATED response")
                        return None
                    
                    insight = ExtractedInsight(
                        question=pair.question,
                        db_id=pair.db_id,
                        error_template=result['error_template'] if isinstance(result['error_template'], str) else result['error_template']['pred'],
                        comparative_insight=result['comparative_insight'],
                        original_correct_sql=pair.correct_sql,
                        original_incorrect_sql=pair.incorrect_sql,
                        question_id=pair.question_id
                    )
                    
                    logger.info(f"    🧮 Classified as CALCULATION-RELATED - template extracted")
                    return insight
                
                else:
                    logger.warning(f"    ❓ Unknown classification: {classification}")
                    return None
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {content}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None

    def process_batch(self, sql_pairs: List[SQLPair], 
                     batch_size: int = 10, 
                     delay: float = 1.0,
                     insight_callback: Optional[Callable[[ExtractedInsight], None]] = None) -> List[ExtractedInsight]:
        """Process SQL pairs in batches with rate limiting.

        Args:
            sql_pairs: list of SQLPair objects to process.
            batch_size: number of items per batch.
            delay: delay between API calls.
            insight_callback: optional function called immediately when an insight is extracted.
        """
        insights = []
        stats = {
            'calculation_related': 0,
            'semantic_related': 0, 
            'failed': 0
        }
        
        total_pairs = len(sql_pairs)
        logger.info(f"Processing {total_pairs} SQL pairs in batches of {batch_size}")
        
        for i in range(0, total_pairs, batch_size):
            batch = sql_pairs[i:i + batch_size]
            batch_insights = []
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_pairs + batch_size - 1)//batch_size}")
            
            for j, pair in enumerate(batch):
                logger.info(f"  Analyzing pair {i + j + 1}/{total_pairs}: {pair.question[:50]}...")
                
                insight = self.analyze_sql_pair(pair)
                if insight:
                    batch_insights.append(insight)
                    stats['calculation_related'] += 1
                    logger.info(f"    ✓ Successfully extracted calculation-related insight")
                    if insight_callback:
                        try:
                            insight_callback(insight)
                        except Exception as callback_error:
                            logger.error(f"    ⚠ Insight callback failed: {callback_error}")
                else:
                    # Check if it's semantic-related (skipped) or failed
                    # We can't easily distinguish here without re-parsing, so count as failed for now
                    stats['failed'] += 1
                    logger.warning(f"    ⏭ Skipped (semantic-related) or failed to analyze")
                
                # Rate limiting
                if j < len(batch) - 1:  # Don't sleep after the last item in batch
                    time.sleep(delay)
            
            insights.extend(batch_insights)
            logger.info(f"Batch completed: {len(batch_insights)}/{len(batch)} calculation-related insights extracted")
            
            # Longer delay between batches
            if i + batch_size < total_pairs:
                time.sleep(delay * 2)
        
        logger.info(f"Processing completed!")
        logger.info(f"  Total pairs analyzed: {total_pairs}")
        logger.info(f"  Calculation-related insights: {stats['calculation_related']}")
        logger.info(f"  Semantic-related/Failed: {stats['failed']}")
        logger.info(f"  Success rate: {stats['calculation_related']/total_pairs*100:.1f}%")
        
        return insights

    def save_results(self, insights: List[ExtractedInsight], output_path: str):
        """Save extracted insights to JSON file."""
        
        results = []
        for insight in insights:
            result = {
                "question": insight.question,
                "db_id": insight.db_id,
                "question_id": insight.question_id,
                "error_template": insight.error_template if isinstance(insight.error_template, str) else insight.error_template['pred'],
                "comparative_insight": insight.comparative_insight,
                "original_sqls": {
                    "question_id": insight.question_id,
                    "question": insight.question,
                    "correct": insight.original_correct_sql,
                    "incorrect": insight.original_incorrect_sql
                }
            }
            results.append(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")

def main():
    """Main function to run the error analysis."""
    parser = argparse.ArgumentParser(description="Analyze NL2SQL errors and extract insights")
    parser.add_argument("--input", "-i", required=True, help="Path to NL2SQL-Bugs JSON file")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing (default: 10)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls in seconds (default: 1.0)")
    parser.add_argument("--limit", type=int, help="Limit number of pairs to process (for testing)")
    parser.add_argument("--input-format", choices=["nl2sql", "bird_results", "gpt4o_results"], default="nl2sql",
                        help="Input format: 'nl2sql' for NL2SQL-Bugs (default), 'bird_results' for BIRD model outputs, or 'gpt4o_results' for GPT-4o results")
    parser.add_argument("--question-file", help="Path to question metadata file (required for bird_results)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("Please provide OpenAI API key via --api-key or OPENAI_API_KEY environment variable")
        return
    
    # Initialize analyzer
    analyzer = NL2SQLErrorAnalyzer(api_key=api_key, model=args.model)
    
    # Load and process data
    data = analyzer.load_data(args.input, input_format=args.input_format, question_path=args.question_file)
    grouped_data = analyzer.group_by_question(data)
    sql_pairs = analyzer.extract_sql_pairs(grouped_data)
    
    # Apply limit if specified (for testing)
    if args.limit:
        sql_pairs = sql_pairs[:args.limit]
        logger.info(f"Limited to {args.limit} pairs for testing")
    
    # Prepare streaming writer
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def write_insight(insight: ExtractedInsight, file_handle):
        """Write a single insight to output file immediately (JSONL)."""
        record = {
            "question": insight.question,
            "db_id": insight.db_id,
            "question_id": insight.question_id,
            "error_template": insight.error_template,
            "comparative_insight": insight.comparative_insight,
            "original_sqls": {
                "question_id": insight.question_id,
                "question": insight.question,
                "correct": insight.original_correct_sql,
                "incorrect": insight.original_incorrect_sql
            }
        }
        file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        file_handle.flush()

    # Process and stream results
    with open(args.output, 'w', encoding='utf-8') as output_file:
        insights = analyzer.process_batch(
            sql_pairs,
            args.batch_size,
            args.delay,
            insight_callback=lambda insight: write_insight(insight, output_file)
        )
    
    # Final summary is already logged in process_batch
    logger.info(f"Results saved to {args.output}")
    logger.info(f"Analysis completed successfully!")

if __name__ == "__main__":
    main()
