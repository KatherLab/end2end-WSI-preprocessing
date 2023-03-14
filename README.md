Note: Requires Python 3.8+
# End-to-end WSI processing pipeline
This repository contains two main scripts for the preProcessing of the Whole Slide Images (WSIs) as an initial step for histopathological deep learning.


0. Install the dependencies and container env with 

```cd mlcontext```

```sh setup.sh```

```cd ..```
1. Edit [run_wsi_norm.sh](run_wsi_norm.sh) and specify your paths. Observe the following arguments:

Input Variable name | Description
--- | --- 
-o | Path to the output folder where normalised .JPGs and normalised features are saved | 
--wsi-dir | Path to the WSI folder
--cache-dir | Path to the output folder where tiles are saved
-m | Path to the Xiyue Wang RetCCL model used for feature extraction

usage: python wsi-norm.py -o OUTPUTPATH --wsi-dir INPUTPATH --cache-dir OUTPUTPATH -m MODELPATH

2. Run the script inside container env with

```sh run.sh```

In this script, we are using the Macenko normalization method from https://github.com/wanghao14/Stain_Normalization.git repository.
