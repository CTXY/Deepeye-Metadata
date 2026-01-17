#!/bin/bash

# DAMO Error Analysis Pipeline Runner

set -e

# Change to project root
cd "$(dirname "$0")/../.."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}DAMO Error Analysis Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY environment variable is not set${NC}"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Default values
# LIMIT=${LIMIT:-10}  # Process 10 samples by default (for testing)
LIMIT=5227
DELAY=${DELAY:-1.0}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Default DAMO results path
DAMO_RESULTS=${DAMO_RESULTS:-"/home/yangchenyu/Text2SQL/reasoning_modules/damo/results/gpt_4o_mini_damo_wo_memory_results_on_train_set.json"}

echo -e "${YELLOW}Configuration:${NC}"
echo "  Results file: $DAMO_RESULTS"
echo "  Limit: $LIMIT samples"
echo "  Delay: $DELAY seconds"
echo "  Log level: $LOG_LEVEL"
echo ""

# Check if results file exists
if [ ! -f "$DAMO_RESULTS" ]; then
    echo -e "${RED}Error: DAMO results file not found: $DAMO_RESULTS${NC}"
    echo "Please check the path or set DAMO_RESULTS environment variable"
    exit 1
fi

# Create log directory
mkdir -p logs

# Run the pipeline
echo -e "${GREEN}Starting pipeline...${NC}"
python script/error_analysis/run_damo_analysis.py \
  --damo-results-path "$DAMO_RESULTS" \
  --limit $LIMIT \
  --delay $DELAY \
  --log-level $LOG_LEVEL \
  --log-file logs/damo_analysis_$(date +%Y%m%d_%H%M%S).log \
  --save-intermediate \
  "$@"

# Check if successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Pipeline completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Output files:"
    echo "  - Insights: output/error_analysis/damo/insights.jsonl"
    echo "  - All samples: output/error_analysis/damo/all_processed_samples.jsonl"
    echo "  - Intermediate: output/error_analysis/intermediate/damo_sample_*.json"
    echo ""
else
    echo -e "${RED}Pipeline failed. Check logs for details.${NC}"
    exit 1
fi


