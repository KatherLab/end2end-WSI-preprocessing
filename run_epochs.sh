#!/bin/bash

epochs=("01" "05" "10" "15" "20" "25" "30" "35" "40" "45" "50" "55") 
for e in ${epochs[@]}; do
    echo "Extracting features for epoch ${e}"
    wsi_dir="/mnt/bulk/timlenz/tumpe/CACHE-DACHS_CRC"
    cache_dir="/mnt/bulk/timlenz/tumpe/CACHE-DACHS_CRC"
    output_dir="/mnt/bulk/timlenz/tumpe/TCGA-results/feats-dachs/swin-zoom-canny-e${e}"
    gpu_ids="0"
    extract="moco-swin-epoch${e}"
    model_file="/mnt/bulk/timlenz/tumpe/models/swin-moco-tcga-zoom-unorm_00${e}.pth"

    #sh run_wsi_norm.sh -d "$wsi_dir" -c "$cache_dir" -o "$output_dir" -m "$model_file" -g "$gpu_ids"
    echo "Using CUDA devices $gpu_ids"
    export CUDA_VISIBLE_DEVICES="$gpu_ids"

    # Run the WSI normalization script
    python wsi-norm.py \
        --wsi-dir "$wsi_dir" \
        --cache-dir "$cache_dir" \
        -o "$output_dir" \
        -m "$model_file" \
        -e $extract \
        --only-fex -z 
    echo "Done extracting features for epoch ${e}!"
done


