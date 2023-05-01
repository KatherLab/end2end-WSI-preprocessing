Note: Requires Python 3.8+
# End-to-end WSI processing pipeline
This repository contains two main scripts for the preProcessing of the Whole Slide Images (WSIs) as an initial step for histopathological deep learning.


0. Clone and enter this repository on your device
1. _For lab members_: download ```e2e_container.sif``` from the ZIH at ```/glw/ekfz_ai/ekfz_proj/Omar```
2. _For externals_: Install the Singularity dependencies and container with
```
  cd mlcontext
  sh setup.sh
  cd ..
```
3. Edit [run_wsi_norm.sh](run_wsi_norm.sh) and specify your paths. Observe the following arguments:

Input Variable name | Description
--- | --- 
-o | Path to the output folder where normalised .JPGs and normalised features are saved | 
--wsi-dir | Path to the WSI folder
--cache-dir | Path to the output folder where tiles are saved
-m | Path to the SSL model used for feature extraction
-e | Feature extractor, 'retccl' or 'ctranspath'

usage: 
```python
python wsi-norm.py \
    -o OUTPUTPATH \
    --wsi-dir INPUTPATH \ 
    --cache-dir OUTPUTPATH \
    -m MODELPATH \
    -e FEATUREEXTRACTOR
```
4. Run the script inside container env with [run_wsi_norm.sh](run_wsi_norm.sh):
`singularity run --nv e2e_container.sif run_wsi_norm.sh`

In this script, we are using the Macenko normalization adapted method from https://github.com/wanghao14/Stain_Normalization.git
