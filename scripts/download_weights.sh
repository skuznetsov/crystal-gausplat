#!/bin/bash
# Download model weights for Crystal 3D Scanner
# MASt3R: https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Crystal 3D Scanner - Model Downloader"
echo "=============================================="

# Create models directory
mkdir -p "$MODELS_DIR/mastr"

# MASt3R weights (2.75 GB)
MASTR_URL="https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/main/model.safetensors"
MASTR_FILE="$MODELS_DIR/mastr/model.safetensors"
MASTR_SIZE="2.75 GB"

download_file() {
    local url=$1
    local output=$2
    local name=$3
    local size=$4

    if [ -f "$output" ]; then
        echo -e "${GREEN}[OK]${NC} $name already exists"
        return 0
    fi

    echo -e "${YELLOW}[DOWNLOADING]${NC} $name ($size)..."
    echo "    Source: $url"
    echo ""

    # Use curl with progress bar
    if command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$output.tmp" "$url"
    elif command -v wget &> /dev/null; then
        wget --show-progress -O "$output.tmp" "$url"
    else
        echo -e "${RED}[ERROR]${NC} Neither curl nor wget found. Please install one."
        exit 1
    fi

    # Move temp file to final location
    mv "$output.tmp" "$output"
    echo -e "${GREEN}[OK]${NC} $name downloaded successfully"
}

# Download MASt3R
echo ""
echo "=== MASt3R (Dense Stereo Matching) ==="
echo "License: CC BY-NC-SA 4.0"
echo "Paper: https://arxiv.org/abs/2406.09756"
echo ""
download_file "$MASTR_URL" "$MASTR_FILE" "MASt3R ViT-Large" "$MASTR_SIZE"

echo ""
echo "=============================================="
echo -e "${GREEN}All models downloaded successfully!${NC}"
echo "=============================================="
echo ""
echo "Models location: $MODELS_DIR"
ls -lh "$MODELS_DIR/mastr/"
