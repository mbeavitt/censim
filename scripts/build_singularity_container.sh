#!/bin/bash
# Script to build the censim Singularity container
# Usage: sudo ./scripts/build_singularity_container.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SINGULARITY_BIN="/home/mbeavitt/micromamba/envs/singularity/bin/singularity"
DEF_FILE="assets/censim.def"
OUTPUT_FILE="censim.sif"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: This script must be run with sudo${NC}"
    echo "Usage: sudo ./scripts/build_singularity_container.sh"
    exit 1
fi

# Check if singularity binary exists
if [ ! -f "$SINGULARITY_BIN" ]; then
    echo -e "${RED}Error: Singularity binary not found at $SINGULARITY_BIN${NC}"
    exit 1
fi

# Check if definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo -e "${RED}Error: Definition file not found at $DEF_FILE${NC}"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project directory
cd "$PROJECT_DIR"

echo -e "${GREEN}Building Singularity container...${NC}"
echo "Definition file: $DEF_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Project directory: $PROJECT_DIR"
echo ""

# Remove old container if it exists
if [ -f "$OUTPUT_FILE" ]; then
    echo -e "${YELLOW}Removing existing container: $OUTPUT_FILE${NC}"
    rm -f "$OUTPUT_FILE"
fi

# Build the container
echo -e "${GREEN}Running singularity build...${NC}"
"$SINGULARITY_BIN" build "$OUTPUT_FILE" "$DEF_FILE"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Container built successfully: $OUTPUT_FILE${NC}"
    echo ""
    echo "To use the container:"
    echo "  singularity exec $OUTPUT_FILE python script.py [args]"
    echo "  singularity shell $OUTPUT_FILE"
else
    echo -e "${RED}Container build failed${NC}"
    exit 1
fi
