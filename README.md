Note: Requires Python 3.8+
# End-to-end WSI processing pipeline
This repository contains two main scripts for the preProcessing of the Whole Slide Images (WSIs) as an initial step for histopathological deep learning.

0. Install the conda packages with ```conda create --name cenv --file cenv_conda.txt python=3.8```
1. Activate the conda environment with ```conda activate cenv```
2. Install PyTorch at pytorch.org/get-started/locally, select pip and the required CUDA platform, and run the pip command
3. Install the pip packages with ```pip install openslide-python numba scikit-learn```
4. Run the pipeline with the following arguments:

Input Variable name | Description
--- | --- 
-o | Path to the output folder where normalised .JPGs and normalised features are saved | 
--wsi-dir | Path to the WSI folder
--cache-dir | Path to the output folder where tiles are saved
-m | Path to the Xiyue Wang RetCCL model used for feature extraction

usage: python wsi-norm.py -o OUTPUTPATH --wsi-dir INPUTPATH --cache-dir OUTPUTPATH -m MODELPATH

See ```run_wsi_norm.sh``` for an example inside a bash script.

In this script, we are using the Macenko normalization method from https://github.com/wanghao14/Stain_Normalization.git repository.
