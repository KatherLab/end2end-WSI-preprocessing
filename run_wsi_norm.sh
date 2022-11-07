dir=/scratch/ws/0/omel987e-wsi-pipeline 
python wsi-norm.py \
	-o ${dir}/CACHE-E2 \
	--wsi-dir ${dir}/TCGA-BRCA-DX-E2 \
	--cache-dir ${dir}/CACHE-E2 \
	-m ${dir}/best_ckpt.pth
