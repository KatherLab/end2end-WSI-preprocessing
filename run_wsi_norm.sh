echo "Using CUDA device 1"
# get the absolute path of the current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CUDA_VISIBLE_DEVICES=1, python wsi-norm.py \
	-o /output \
	--wsi-dir $DIR/wsi_samples/ \
	--cache-dir $DIR/workspace/output \
	-m mlcontext/best_ckpt.pth