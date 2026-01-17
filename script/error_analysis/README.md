# Error Analysis System for SQL Generation

A comprehensive pipeline to analyze SQL generation errors and extract reusable insights for improving future SQL generation.

## Overview

This system processes triplets of (NLQ, Incorrect SQL, Correct SQL) and:

1. **Filters pure schema errors** - Identifies cases where only table/column selection is wrong
2. **Creates masked SQL patterns** - Abstracts SQLs by replacing schema with placeholders
3. **Mines error patterns** - Uses LLM to identify error signatures and generate guidance
4. **Verifies insights** - Validates guidance by executing SQLs on actual databases
5. **Outputs reusable insights** - Generates structured insights for future retrieval

## Architecture

```
script/error_analysis/
├── __init__.py              # Package init
├── config.py                # Configuration
├── models.py                # Pydantic data models
├── data_loader.py           # Load and merge data sources
├── schema_checker.py        # Step 1: Filter schema errors
├── sql_masker.py            # Step 2: Mask SQL to abstract patterns
├── miner_agent.py           # Step 3: Extract patterns with LLM
├── verifier_agent.py        # Step 4: Verify insights
├── utils.py                 # Utility functions
├── main.py                  # Main pipeline
└── README.md                # This file
```

## Pipeline Stages

### Stage 1: Data Loading
- Loads NLQ from `data/bird/dev/dev.json`
- Loads incorrect predictions from `results/bird-dev/qwen3-coder-30b-a3b_incorrect.json`
- Merges by `question_id`

### Stage 2: Schema Error Filtering
- Qualifies both SQLs (resolves aliases, adds table names to columns)
- Extracts schema (tables, columns) from both SQLs
- Calculates schema overlap score (Jaccard similarity)
- Filters out cases with overlap < threshold (default: 0.3)

**Example:**
```
Incorrect: SELECT schools.name FROM schools WHERE schools.id = 5
Correct:   SELECT frpm.school_name FROM frpm WHERE frpm.county = 'Alameda'
→ Filtered (completely different schemas)
```

### Stage 2.5: Value Error Filtering (NEW)
After masking SQLs, checks if the two SQLs differ **only in literal values** (V1, V2, ...).
If they have identical structure but different values, it's a "value selection error" (e.g., County = 'Alameda' vs 'Lake').

**Detection Method:**
```python
# Replace all V1, V2, ... with V_PLACEHOLDER
normalized_incorrect = re.sub(r'V\d+', 'V_PLACEHOLDER', masked_incorrect)
normalized_correct = re.sub(r'V\d+', 'V_PLACEHOLDER', masked_correct)

# If identical after normalization → value-only error → Filter out
if normalized_incorrect == normalized_correct:
    filter_out()
```

**Example:**
```sql
Incorrect: SELECT COUNT(*) FROM T1 WHERE T1.C1 = V1 AND T1.C2 < V2
Correct:   SELECT COUNT(*) FROM T1 WHERE T1.C1 = V3 AND T1.C2 < V2

Normalized both: SELECT COUNT(*) FROM T1 WHERE T1.C1 = V_PLACEHOLDER AND T1.C2 < V_PLACEHOLDER
→ Identical structure → Filtered (value-only error)
```

**Why Filter These?**
Value selection errors (wrong WHERE condition values) don't reveal useful **logical/operational** patterns. They're data-specific mistakes, not generalizable insights.

### Stage 3: SQL Masking
- Replaces table names: `schools` → `T1`, `frpm` → `T2`
- Replaces column names: `name` → `C1`, `id` → `C2`
- Replaces values: `5` → `V1`, `'Alameda'` → `V2`
- Ensures consistent mapping across both SQLs
- Removes aliases

**Example:**
```
Original:  SELECT schools.name FROM schools WHERE schools.rating = 5
Masked:    SELECT T1.C1 FROM T1 WHERE T1.C2 = V1
```

### Stage 4: Miner Agent
- Uses LLM (gpt-4o-mini) to analyze error patterns
- Extracts retrieval keys:
  - **NL Triggers**: **Operation/computation-related keywords** from NLQ (e.g., "highest", "ratio", "average difference")
    - ✅ Include: aggregation (count, sum, avg), comparison (highest, lowest), operations (ratio, division)
    - ❌ Exclude: domain nouns (schools, students, websites)
  - **SQL Risk Atoms**: SQL operators in incorrect SQL (e.g., ["WHERE", "=", "MAX"])
- Generates comparative guidance:
  - Intent
  - Incorrect strategy + implications
  - Correct strategy + benefits
  - Actionable advice

**Example Output:**
```json
{
  "retrieval_key": {
    "nl_triggers": ["highest", "single entity"],
    "sql_risk_atoms": ["WHERE", "=", "SELECT", "MAX"]
  },
  "guidance": {
    "intent": "Select a single entity with the top value",
    "comparison": {
      "strategy_incorrect": {
        "pattern": "WHERE col = (SELECT MAX(col)...)",
        "implication": "Returns ALL records that tie for max value"
      },
      "strategy_correct": {
        "pattern": "ORDER BY col DESC LIMIT 1",
        "implication": "Returns strictly ONE record"
      }
    },
    "actionable_advice": "Use ORDER BY...LIMIT 1 for singular results"
  }
}
```

### Stage 5: Verification
- Executes both incorrect and correct SQLs on actual database
- Compares results to confirm error exists
- Marks insight as verified if results differ

### Stage 6: Output
- Saves insights to `output/error_analysis/insights.jsonl`
- Saves all processed samples to `output/error_analysis/all_processed_samples.jsonl`
- Saves intermediate results (optional) to `output/error_analysis/intermediate/`

## Usage

### Prerequisites

1. **Python dependencies:**
   ```bash
   pip install openai pydantic sqlglot
   ```

2. **OpenAI API key:**
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

3. **Data files:**
   - `data/bird/dev/dev.json` (NLQ data)
   - `results/bird-dev/qwen3-coder-30b-a3b_incorrect.json` (incorrect predictions)
   - Database files referenced in the JSON

### Basic Usage

```bash
cd /home/yangchenyu/DeepEye-SQL-Metadata

# Run full pipeline
python script/error_analysis/main.py

# Test on first 10 samples
python script/error_analysis/main.py --limit 10

# Custom configuration
python script/error_analysis/main.py \
  --limit 50 \
  --delay 1.0 \
  --schema-overlap-threshold 0.3 \
  --openai-model gpt-4o-mini \
  --log-level DEBUG
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--bird-dev-json` | Path to bird dev.json | `data/bird/dev/dev.json` |
| `--incorrect-results-json` | Path to incorrect results | `results/bird-dev/qwen3-coder-30b-a3b_incorrect.json` |
| `--output-dir` | Output directory | `output/error_analysis` |
| `--limit` | Limit number of samples | None (all) |
| `--delay` | Delay between API calls (seconds) | 0.5 |
| `--schema-overlap-threshold` | Schema overlap threshold | 0.3 |
| `--openai-api-key` | OpenAI API key | From env |
| `--openai-model` | OpenAI model | `gpt-4o-mini` |
| `--openai-temperature` | LLM temperature | 0.0 |
| `--log-level` | Logging level | INFO |
| `--log-file` | Log file path | None (stdout) |

## Output Format

### Insights File (`insights.jsonl`)

Each line is a JSON object:

```json
{
  "insight_id": "insight_16",
  "retrieval_key": {
    "nl_triggers": ["highest", "top"],
    "sql_risk_atoms": ["WHERE", "=", "MAX"]
  },
  "guidance": {
    "intent": "...",
    "strategy_incorrect": {"pattern": "...", "implication": "..."},
    "strategy_correct": {"pattern": "...", "implication": "..."},
    "actionable_advice": "..."
  },
  "qualified_incorrect_sql": "SELECT schools.rating FROM schools WHERE schools.rating = (SELECT MAX(schools.rating) FROM schools)",
  "qualified_correct_sql": "SELECT schools.rating FROM schools ORDER BY schools.rating DESC LIMIT 1",
  "source_question_ids": [16],
  "verification_success_count": 1,
  "verification_total_count": 1,
  "verification_success_rate": 1.0,
  "created_at": "2024-01-01T00:00:00Z"
}
```

**New in v1.2.0**: Each insight now includes `qualified_incorrect_sql` and `qualified_correct_sql` as concrete examples to help future models understand the error pattern in context. These are the qualified versions of the SQLs (with aliases resolved and table names added to all columns).

### Processed Samples File (`all_processed_samples.jsonl`)

Contains full processing details for each sample, including:
- Original SQLs
- Qualified SQLs
- Masked SQLs
- Schema information
- Miner output
- Verification details

## Statistics and Monitoring

The pipeline prints statistics at the end:

```
PIPELINE STATISTICS
================================================================================
Total samples:              409
Schema errors filtered:     150
Value errors filtered:      30
Processing errors:          5
Miner success:              210
Miner failed:               14
Verification passed:        180
Verification failed:        30
================================================================================
```

## Error Handling

- **Schema extraction fails**: Falls back to original SQL, continues processing
- **SQL masking fails**: Returns original SQL with empty mapping
- **Miner Agent fails**: Marks sample as `miner_failed`, continues
- **Verification fails**: Still saves the insight, marks as unverified
- **All errors are logged**: Check logs for details

## Future Enhancements

1. **Insight Clustering**: Group similar patterns and merge insights
2. **Incremental Processing**: Resume from checkpoints
3. **Parallel Processing**: Process multiple samples concurrently
4. **Insight Retrieval**: Build index for retrieving relevant insights
5. **Auto-correction**: Apply insights to auto-fix new errors

## Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Set environment variable: `export OPENAI_API_KEY='your-key'`
   - Or use `--openai-api-key` flag

2. **"Database file not found"**
   - Check `db_path` in incorrect_results.json
   - Ensure database files exist at specified paths

3. **"Schema extraction failed"**
   - Check if SQLs are valid SQLite syntax
   - Review log files for detailed error messages

4. **Rate limiting errors**
   - Increase `--delay` parameter
   - Reduce batch size with `--limit`

## Contact

For issues or questions, contact the development team.

