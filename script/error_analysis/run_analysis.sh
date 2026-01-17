#!/bin/bash

# Error Analysis Pipeline Runner
# This script runs the error analysis pipeline with recommended settings

set -e

# Change to project root
cd "$(dirname "$0")/../.."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Error Analysis Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY environment variable is not set${NC}"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Default values
LIMIT=${LIMIT:-10}  # Process 10 samples by default (for testing)
DELAY=${DELAY:-1.0}
LOG_LEVEL=${LOG_LEVEL:-INFO}
OUTPUT_DIR=${OUTPUT_DIR:-"output/error_analysis"}

echo -e "${YELLOW}Configuration:${NC}"
echo "  Limit: $LIMIT samples"
echo "  Delay: $DELAY seconds"
echo "  Log level: $LOG_LEVEL"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create log directory
mkdir -p logs

# Run the pipeline
echo -e "${GREEN}Starting pipeline...${NC}"
python script/error_analysis/main.py \
  --limit $LIMIT \
  --delay $DELAY \
  --log-level $LOG_LEVEL \
  --log-file logs/error_analysis_$(date +%Y%m%d_%H%M%S).log \
  --save-intermediate \
  "$@"

# Check if successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Pipeline completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Output files:"
    echo "  - Insights: $OUTPUT_DIR/insights.jsonl"
    echo "  - All samples: $OUTPUT_DIR/all_processed_samples.jsonl"
    echo "  - Intermediate: $OUTPUT_DIR/intermediate/"
    echo ""
else
    echo -e "${RED}Pipeline failed. Check logs for details.${NC}"
    exit 1
fi

