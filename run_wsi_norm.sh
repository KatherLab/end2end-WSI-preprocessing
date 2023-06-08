#!/bin/bash


##### ONLY THINGS TO FILL IN
# Default values, use absolute paths only!
wsi_dir="/scratch/ws/1/s1787956-tim-cpath/NUCIFORO-VHIO-BRCA-IMGS"
cache_dir="/scratch/ws/1/s1787956-tim-cpath/NUCIFORO-VHIO-BRCA-IMGS/cache"
output_dir="/scratch/ws/1/s1787956-tim-cpath/NUCIFORO-VHIO-BRCA-IMGS/features"
gpu_ids="0"
extract="ctranspath"
model_file="mlcontext/ctranspath.pth"
#####


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

# Process command-line arguments
while getopts "hd:c:o:m:g:e:" opt; do
    case "$opt" in
        h ) usage; exit 0 ;;
        d ) wsi_dir="$OPTARG" ;;
        c ) cache_dir="$OPTARG" ;;
        o ) output_dir="$OPTARG" ;;
        m ) model_file="$OPTARG" ;;
        g ) gpu_ids="$OPTARG" ;;
        e ) extract="$OPTARG" ;;
        ? ) usage; exit 1 ;;
    esac
done
if [ "$extract" = "ctranspath" ]; then
    model_file="mlcontext/ctranspath.pth"
elif [ "$extract" = "retccl" ]; then
    model_file="mlcontext/best_ckpt.pth"
fi

# Set CUDA_VISIBLE_DEVICES to specified GPU IDs
echo "Using CUDA devices $gpu_ids"
export CUDA_VISIBLE_DEVICES="$gpu_ids"

# Run the WSI normalization script
python wsi-norm.py \
    --wsi-dir "$wsi_dir" \
    --cache-dir "$cache_dir" \
    -o "$output_dir" \
    -m "$model_file" \
    -e $extract --only-fex 
