#!/bin/bash

# Usage information
usage() {
    echo "Usage: $(basename "$0") [-h] [-d <wsi_dir>] [-c <cache_dir>] [-o <output_dir>] [-m <model_file>] [-g <gpu_ids>]"
    echo ""
    echo "Options:"
    echo "  -h               Show this help message and exit"
    echo "  -d <wsi_dir>     Path to the directory containing WSI files (default: wsi_samples/)"
    echo "  -c <cache_dir>   Path to the directory to store the cached image patches (default: workspace/output/)"
    echo "  -o <output_dir>  Path to the directory to store the output normalized WSIs (default: output/)"
    echo "  -m <model_file>  Path to the model file (default: mlcontext/best_ckpt.pth)"
    echo "  -g <gpu_ids>     Comma-separated list of GPU IDs to use (default: 1)"
    echo ""
}

# Default values
wsi_dir="wsi_samples/"
cache_dir="workspace/output/"
output_dir="output/"
model_file="mlcontext/best_ckpt.pth"
gpu_ids="1"

# Process command-line arguments
while getopts "hd:c:o:m:g:" opt; do
    case "$opt" in
        h ) usage; exit 0 ;;
        d ) wsi_dir="$OPTARG" ;;
        c ) cache_dir="$OPTARG" ;;
        o ) output_dir="$OPTARG" ;;
        m ) model_file="$OPTARG" ;;
        g ) gpu_ids="$OPTARG" ;;
        ? ) usage; exit 1 ;;
    esac
done

# Set CUDA_VISIBLE_DEVICES to specified GPU IDs
echo "Using CUDA devices $gpu_ids"
export CUDA_VISIBLE_DEVICES="$gpu_ids"

# Get the absolute path of the current directory
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Run the WSI normalization script
python wsi-norm.py \
    --wsi-dir "$dir/$wsi_dir" \
    --cache-dir "$dir/$cache_dir" \
    -o "$dir/$output_dir" \
    -m "$dir/$model_file"
