Note: Requires Python 3.8+
# End-to-end WSI processing pipeline
This repository contains a pipeline for the pre-processing of Whole Slide Images (WSIs) as an initial step for histopathological deep learning.
In this pipeline, we are using the Macenko normalization adapted method from https://github.com/wanghao14/Stain_Normalization.git

F
0. Clone and enter this repository on your device
1. Install the Singularity dependencies and build container, requires (fake) root access
```
  cd mlcontext
  sh setup.sh
  cd ..
```
2. Edit [run_wsi_norm.sh](run_wsi_norm.sh) and specify your paths. Observe the following arguments:

Input Variable name | Description
--- | --- 
-o | Path to the output folder where features are saved | 
--wsi-dir | Path to the WSI folder
--cache-dir | Path to the output folder where intermediate slide JPGs are saved
-m | Path to the SSL model used for feature extraction
-e | Feature extractor, 'retccl' or 'ctranspath'
-c | Number of CPU cores, optional
--del_slide | Delete original slide from your drive, optional
--no-norm | Do not apply Macenko normalization, optional

Example usage: 
```python
python wsi-norm.py \
    -o FEATURE_OUTPUT_PATH \
    --wsi-dir INPUT_PATH \ 
    --cache-dir IMAGES_OUTPUT_PATH \
    -m MODEL_PATH \
    -e FEATURE_EXTRACTOR \
    -c NUM_OF_CPU_CORES \
    --del-slide \
    --no-norm 
```
3. Run the script inside container env with [run_wsi_norm.sh](run_wsi_norm.sh):
`singularity run --nv -B /:/ e2e_container.sif run_wsi_norm.sh`

