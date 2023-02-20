echo "Using CUDA device 1"
CUDA_VISIBLE_DEVICES=1, python wsi-norm.py \
	-o /output \
	--wsi-dir /home/jeff/PycharmProjects/end2end-WSI-preprocessing/input \
	--cache-dir /home/jeff/PycharmProjects/end2end-WSI-preprocessing/jeff \
	-m mlcontext/e2e_container/best_ckpt.pth
