# Quick Start Guide

## 5-Minute Setup and Run

### Step 1: Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Step 2: Test Components (Optional)

```bash
cd /home/yangchenyu/DeepEye-SQL-Metadata
python script/error_analysis/test_components.py
```

Expected output:
```
✓ Preprocessor tests completed
✓ SQL masking works correctly
✓ Schema checker initialized successfully
All component tests completed!
```

### Step 3: Run on 5 Samples (Recommended First Run)

```bash
python script/error_analysis/main.py --limit 5
```

This will:
1. Load data from BIRD dev set
2. Process 5 samples through the full pipeline
3. Generate insights in `output/error_analysis/insights.jsonl`

### Step 4: Check Results

```bash
# View insights
cat output/error_analysis/insights.jsonl | jq .

# View processing statistics
tail -20 output/error_analysis/all_processed_samples.jsonl
```

## What You Should See

### Console Output (Abbreviated)

```
================================================================================
Starting Error Analysis Pipeline
================================================================================

[Stage 1] Loading data...
Loaded 409 samples
Limited to first 5 samples

================================================================================
Processing sample 1/5 - Question ID: 16
================================================================================

[Stage 2] Checking schema...
✓ Passed schema check (overlap=0.857)

[Stage 3] Masking SQLs...
✓ Masked incorrect: SELECT T1.C1 FROM T1 INNER JOIN T2...
✓ Masked correct: SELECT T1.C1 FROM T1 INNER JOIN T2...

[Stage 4] Running Miner Agent...
✓ Miner Agent succeeded
  Intent: Count records matching specific criteria
  NL Triggers: ['county', 'count']
  SQL Risk Atoms: ['WHERE', '=']

[Stage 5] Running Verifier Agent...
✓ Verification PASSED

[Stage 6] Generating insights...
✓ Saved 4 insights to output/error_analysis/insights.jsonl

================================================================================
PIPELINE STATISTICS
================================================================================
Total samples:              5
Schema errors filtered:     1
Miner success:              4
Verification passed:        3
================================================================================
```

### Output Files

1. **Insights** (`output/error_analysis/insights.jsonl`)
   ```json
   {"insight_id": "insight_16", "retrieval_key": {...}, "guidance": {...}}
   {"insight_id": "insight_17", "retrieval_key": {...}, "guidance": {...}}
   ...
   ```

2. **All Samples** (`output/error_analysis/all_processed_samples.jsonl`)
   - Complete processing details for each sample

3. **Intermediate** (`output/error_analysis/intermediate/sample_*.json`)
   - Per-sample detailed output

## Next Steps

### 1. Run on More Samples

```bash
# 20 samples
python script/error_analysis/main.py --limit 20

# 50 samples
python script/error_analysis/main.py --limit 50
```

### 2. Run Full Analysis

```bash
# All 409 samples (~3-5 minutes)
python script/error_analysis/main.py --delay 1.0
```

### 3. Customize Configuration

```bash
python script/error_analysis/main.py \
  --limit 30 \
  --delay 1.0 \
  --schema-overlap-threshold 0.25 \
  --log-level DEBUG \
  --log-file logs/my_analysis.log
```

### 4. Analyze Results

```bash
# Count insights
wc -l output/error_analysis/insights.jsonl

# View specific insight
cat output/error_analysis/insights.jsonl | jq 'select(.insight_id == "insight_17")'

# Check verification rates
cat output/error_analysis/insights.jsonl | jq -r '.verification_success_rate' | \
  awk '{sum+=$1; count+=1} END {print "Avg:", sum/count}'

# Find insights about MAX pattern
cat output/error_analysis/insights.jsonl | jq 'select(.retrieval_key.sql_risk_atoms | contains(["MAX"]))'
```

## Common Issues

### Issue 1: "OpenAI API key not found"

**Solution**:
```bash
export OPENAI_API_KEY='sk-...'
# Or pass as argument
python script/error_analysis/main.py --openai-api-key 'sk-...'
```

### Issue 2: Rate limiting errors

**Solution**: Increase delay
```bash
python script/error_analysis/main.py --limit 10 --delay 2.0
```

### Issue 3: Database not found

**Check**: Verify database paths
```bash
cat results/bird-dev/qwen3-coder-30b-a3b_incorrect.json | \
  jq -r '.incorrect_sqls[0].db_path'
```

## Understanding the Output

### Insight Structure

```json
{
  "insight_id": "insight_17",
  "retrieval_key": {
    "nl_triggers": ["rank", "order"],        // Keywords from question
    "sql_risk_atoms": ["ORDER BY", "DESC"]   // SQL patterns that signal risk
  },
  "guidance": {
    "intent": "Rank entities by score",
    "strategy_incorrect": {
      "pattern": "ORDER BY col DESC",
      "implication": "Missing explicit rank numbers"
    },
    "strategy_correct": {
      "pattern": "RANK() OVER (ORDER BY col DESC)",
      "implication": "Explicitly computes rank values"
    },
    "actionable_advice": "Use window functions for ranking"
  },
  "verification_success_rate": 1.0  // 1.0 = verified
}
```

### How to Use Insights

1. **Retrieval**: Match `nl_triggers` and `sql_risk_atoms` to new cases
2. **Guidance**: Apply `actionable_advice` when generating SQL
3. **Validation**: Higher `verification_success_rate` = more trustworthy

## Performance Expectations

| Samples | Time (0.5s delay) | API Cost | Expected Insights |
|---------|-------------------|----------|-------------------|
| 5       | ~30 seconds       | ~$0.01   | 3-4               |
| 20      | ~2 minutes        | ~$0.05   | 12-15             |
| 50      | ~5 minutes        | ~$0.10   | 30-35             |
| 409     | ~30 minutes       | ~$1.00   | 200-250           |

## Tips for Best Results

1. **Start small**: Always test with `--limit 5` first
2. **Monitor logs**: Use `--log-level INFO` to see progress
3. **Save logs**: Use `--log-file` for debugging
4. **Adjust delay**: Increase if hitting rate limits
5. **Review failed cases**: Check logs to improve system

## Shell Script Alternative

```bash
# Set parameters
export LIMIT=10
export DELAY=1.0
export LOG_LEVEL=INFO

# Run
./script/error_analysis/run_analysis.sh
```

## Getting Help

1. **README.md** - Detailed documentation
2. **DESIGN.md** - Technical design
3. **EXAMPLE.md** - More examples
4. **PROJECT_SUMMARY.md** - Project overview

## Success Indicators

You know it's working when:
- ✅ Console shows "✓" marks for each stage
- ✅ `insights.jsonl` file is created
- ✅ Verification pass rate > 70%
- ✅ Insights contain actionable advice
- ✅ No Python tracebacks in output

## Ready to Start?

```bash
export OPENAI_API_KEY='your-key'
cd /home/yangchenyu/DeepEye-SQL-Metadata
python script/error_analysis/main.py --limit 5
```

**Happy analyzing! 🚀**

