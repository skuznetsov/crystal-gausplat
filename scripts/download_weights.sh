#!/bin/bash
# Download model weights for Crystal 3D Scanner
# MASt3R: https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
# YOLOv8: https://github.com/ultralytics/ultralytics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Crystal 3D Scanner - Model Downloader"
echo "=============================================="

# Create models directories
mkdir -p "$MODELS_DIR/mastr"
mkdir -p "$MODELS_DIR/yolo"

# MASt3R weights (2.75 GB)
MASTR_URL="https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/main/model.safetensors"
MASTR_FILE="$MODELS_DIR/mastr/model.safetensors"
MASTR_SIZE="2.75 GB"

# YOLOv8 CoreML models from HuggingFace
# Source: https://huggingface.co/TheCluster/YOLOv8-CoreML
YOLO_HF_REPO="TheCluster/YOLOv8-CoreML"
YOLO_MODEL="${YOLO_MODEL:-yolov8n}"  # Default: nano (fastest)

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

# Download YOLOv8
echo ""
echo "=== YOLOv8 (Object Detection for CoreML) ==="
echo "License: AGPL-3.0"
echo "Repo: https://github.com/ultralytics/ultralytics"
echo ""
echo -e "${CYAN}Available models:${NC} yolov8n (6MB), yolov8s (22MB), yolov8m (50MB)"
echo -e "${CYAN}Selected:${NC} $YOLO_MODEL (set YOLO_MODEL=yolov8s to change)"
echo ""

# Model sizes
case "$YOLO_MODEL" in
    yolov8n) YOLO_SIZE="~13 MB" ;;
    yolov8s) YOLO_SIZE="~45 MB" ;;
    yolov8m) YOLO_SIZE="~104 MB" ;;
    yolov8l) YOLO_SIZE="~175 MB" ;;
    yolov8x) YOLO_SIZE="~273 MB" ;;
    *) YOLO_SIZE="unknown" ;;
esac

YOLO_DIR="$MODELS_DIR/yolo/${YOLO_MODEL}.mlpackage"

if [ -d "$YOLO_DIR" ]; then
    echo -e "${GREEN}[OK]${NC} YOLOv8 ($YOLO_MODEL) already exists"
else
    echo -e "${YELLOW}[DOWNLOADING]${NC} YOLOv8 ($YOLO_MODEL) ($YOLO_SIZE)..."

    # Check for git-lfs
    if ! command -v git-lfs &> /dev/null; then
        echo -e "${RED}[ERROR]${NC} git-lfs is required for YOLO download"
        echo "Install with: brew install git-lfs && git lfs install"
        exit 1
    fi

    # Clone only the specific model directory using sparse checkout
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    git clone --depth 1 --filter=blob:none --sparse \
        "https://huggingface.co/${YOLO_HF_REPO}" yolo-coreml 2>&1 | grep -v "^remote:"

    cd yolo-coreml
    git sparse-checkout set "${YOLO_MODEL}.mlpackage"
    git lfs pull --include="${YOLO_MODEL}.mlpackage/*"

    # Move to models directory
    mv "${YOLO_MODEL}.mlpackage" "$YOLO_DIR"

    # Cleanup
    cd "$PROJECT_DIR"
    rm -rf "$TEMP_DIR"

    echo -e "${GREEN}[OK]${NC} YOLOv8 ($YOLO_MODEL) downloaded"
fi

echo ""
echo "=============================================="
echo -e "${GREEN}All models downloaded successfully!${NC}"
echo "=============================================="
echo ""
echo "Models location: $MODELS_DIR"
echo ""
echo "MASt3R:"
ls -lh "$MODELS_DIR/mastr/" 2>/dev/null || echo "  (not downloaded)"
echo ""
echo "YOLOv8:"
ls -lhd "$MODELS_DIR/yolo/"*.mlpackage 2>/dev/null || echo "  (not downloaded)"
