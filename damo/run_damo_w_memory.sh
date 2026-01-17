#!/usr/bin/env bash

# DAMO NL2SQL Reasoning with CAF Memory (Semantic Only)
# This script processes BIRD dev dataset for multiple databases
# Runs NL2SQL tasks with CAF and semantic memory enabled on all test data

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# MODE can be: query | batch
MODE="query"

# Common options
CONFIG_PATH="$SCRIPT_DIR/config/config.yaml"  # Path to configuration file
USE_CAF="true"           # true | false
ENABLE_EPISODIC="false"  # true | false
ENABLE_SEMANTIC="true"   # true | false

# Query mode presets
QUESTION="In which city can you find the school in the state of California with the lowest latitude coordinates and what is its lowest grade? Indicate the school name."
DB_ID="california_schools"
EVIDENCE='''State of California refers to state = 'CA''''          # optional free text
GROUND_TRUTH_SQL=""      # optional
SESSION_ID=""            # optional (auto if empty)
OUTPUT_FILE=""           # optional, e.g. ./outputs/query_result.json
SHOW_RESULTS="true"      # true | false
SHOW_STEPS="false"       # true | false

# Batch mode presets
SPLIT="dev"              # train | dev
LIMIT=""                 # integer or empty
BATCH_OUTPUT_FILE="$SCRIPT_DIR/results/gpt-4o_damo_semantic_episodic_results_california_financial.json"     # output file for batch mode

# Target databases for batch processing
TARGET_DATABASES=("california_schools" "financial")
PERCENTAGE=100  # Use 100% of test data from each database


echo "🚀 DAMO NL2SQL Reasoning with CAF Memory"
echo "=============================================="
echo "Target databases: ${TARGET_DATABASES[*]}"
echo "Data percentage: ${PERCENTAGE}% (all test data)"
echo "CAF enabled: $([ "$USE_CAF" == "true" ] && echo "Yes" || echo "No")"
echo "Episodic memory: $([ "$ENABLE_EPISODIC" == "true" ] && echo "Enabled" || echo "Disabled")"
echo "Semantic memory: $([ "$ENABLE_SEMANTIC" == "true" ] && echo "Enabled" || echo "Disabled")"
echo ""

# Create necessary directories
mkdir -p "$SCRIPT_DIR/results"

# Change to damo directory for proper imports (since imports are now relative)
cd "$SCRIPT_DIR"

if [ "$MODE" == "query" ]; then
    echo "🤖 Processing single query with NL2SQL reasoning..."
    echo "Question: $QUESTION"
    echo "Database: $DB_ID"
    echo ""
    
    python3 main.py \
    query \
    "$QUESTION" \
    "$DB_ID" \
    ${CONFIG_PATH:+--config "$CONFIG_PATH"} \
    --evidence "$EVIDENCE" \
    --ground-truth-sql "$GROUND_TRUTH_SQL" \
    --session-id "$SESSION_ID" \
    --output "$OUTPUT_FILE" \
    ${USE_CAF:+--use-caf} \
    $([ "$ENABLE_EPISODIC" == "true" ] && echo "--enable-episodic" || echo "--disable-episodic") \
    $([ "$ENABLE_SEMANTIC" == "true" ] && echo "--enable-semantic" || echo "--disable-semantic") \
    ${SHOW_RESULTS:+--show-results} \
    ${SHOW_STEPS:+--show-steps}
    
elif [ "$MODE" == "batch" ]; then
    echo "🤖 Processing BIRD dev dataset with NL2SQL reasoning..."
    echo "Output file: $BATCH_OUTPUT_FILE"
    echo ""
    
    # Run main.py batch mode with database filtering and percentage selection
    # Note: Running from damo directory, so main.py is in current directory
    python3 main.py batch "$SPLIT" \
        --databases "${TARGET_DATABASES[@]}" \
        --percentage "$PERCENTAGE" \
        --config "$CONFIG_PATH" \
        --output "$BATCH_OUTPUT_FILE" \
        ${USE_CAF:+--use-caf} \
        ${LIMIT:+--limit "$LIMIT"} \
        $([ "$ENABLE_EPISODIC" == "true" ] && echo "--enable-episodic" || echo "--disable-episodic") \
        $([ "$ENABLE_SEMANTIC" == "true" ] && echo "--enable-semantic" || echo "--disable-semantic")
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: NL2SQL processing failed"
        exit 1
    fi
fi

echo ""
echo "✅ NL2SQL processing completed successfully"
echo ""
echo "🎉 Experiment completed successfully!"

if [ "$MODE" == "batch" ]; then
    echo "📁 Results saved to: $BATCH_OUTPUT_FILE"
elif [ "$MODE" == "query" ] && [ -n "$OUTPUT_FILE" ]; then
    echo "📁 Results saved to: $OUTPUT_FILE"
fi

echo ""
echo "🏁 All tasks completed successfully!"
echo "====================================="


