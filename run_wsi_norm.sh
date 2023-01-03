echo "Using CUDA device 1"
CUDA_VISIBLE_DEVICES=1, python wsi-norm.py \
	-o /home/omarelnahhas/CACHE_MSK \
	--wsi-dir /mnt/KATHER-T03/MSKCC-DX-WSI/msk_wsi1 \
	--cache-dir /home/omarelnahhas/CACHE_MSK \
	-m /home/omarelnahhas/best_ckpt.pth
