import argparse
import io
from pathlib import Path
import sys
import shutil
from typing import Dict, Tuple
from concurrent import futures
# from urllib.parse import urlparse
import warnings
import glob
import logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Normalise WSI directly.')

    parser.add_argument('-o', '--output-path', type=Path, required=True,
                        help='Path to save results to.')
    parser.add_argument('--wsi-dir', metavar='DIR', type=Path, required=True,
                        help='Path of where the whole-slide images are.')
    parser.add_argument('-m', '--model', metavar='DIR', type=Path, required=True,
                        help='Path of where the whole-slide images are.')
    parser.add_argument('--cache-dir', type=Path, default=None,
        help='Directory to cache extracted features etc. in.')
    # parser.add_argument('--cache-dir', type=Path, default=None,
    #                     help='Directory to cache extracted features etc. in.')

    args = parser.parse_args()




# if (p := './RetCCL') not in sys.path:
#     sys.path = [p] + sys.path
# import ResNet
# import torch.nn as nn
# import torch
# from torchvision import transforms
import os
from matplotlib import pyplot as plt
import openslide
from tqdm import tqdm
import numpy as np
# from fastai.vision.all import load_learner
# from pyzstd import ZstdFile
import PIL
import stainNorm_Macenko
import cv2
from numba import jit

# supress DecompressionBombWarning: yes, our files are really that big (‘-’*)
PIL.Image.MAX_IMAGE_PIXELS = None


def _load_tile(
    slide: openslide.OpenSlide, pos: Tuple[int, int], stride: Tuple[int, int], target_size: Tuple[int, int]
) -> np.ndarray:
    # Loads part of a WSI. Used for parallelization with ThreadPoolExecutor
    tile = slide.read_region(pos, 0, stride).convert('RGB').resize(target_size)
    return np.array(tile)


def load_slide(slide: openslide.OpenSlide, target_mpp: float = 256/224) -> np.ndarray:
    """Loads a slide into a numpy array."""
    # We load the slides in tiles to
    #  1. parallelize the loading process
    #  2. not use too much data when then scaling down the tiles from their
    #     initial size
    steps = 8
    stride = np.ceil(np.array(slide.dimensions)/steps).astype(int)
    slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    tile_target_size = np.round(stride*slide_mpp/target_mpp).astype(int)
    #changed max amount of threads used
    with futures.ThreadPoolExecutor(os.cpu_count()) as executor:
        # map from future to its (row, col) index
        future_coords: Dict[futures.Future, Tuple[int, int]] = {}
        for i in range(steps):  # row
            for j in range(steps):  # column
                future = executor.submit(
                    _load_tile, slide, (stride*(j, i)), stride, tile_target_size)
                future_coords[future] = (i, j)

        # write the loaded tiles into an image as soon as they are loaded
        im = np.zeros((*(tile_target_size*steps)[::-1], 3), dtype=np.uint8)
        for tile_future in tqdm(futures.as_completed(future_coords), total=steps*steps, desc='Reading WSI tiles', leave=False):
            i, j = future_coords[tile_future]
            tile = tile_future.result()
            x, y = tile_target_size * (j, i)
            im[y:y+tile.shape[0], x:x+tile.shape[1], :] = tile

    return im

import time
from datetime import timedelta

#xiyue wang fex
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
from marugoto.marugoto.extract.extract import extract_features_
from marugoto.marugoto.extract.xiyue_wang.RetCLL import ResNet
from concurrent_canny_rejection import reject_background
from PIL import Image

# %%

def extract_xiyuewang_features_(norm_wsi_img: PIL.Image, wsi_name: str, coords: list, checkpoint_path: str, outdir: Path, **kwargs):
    """Extracts features from slide tiles.
    Args:
        checkpoint_path:  Path to the model checkpoint file.  Can be downloaded
            from <https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL>.
    """
   # calculate checksum of model
    sha256 = hashlib.sha256()
    with open(checkpoint_path, 'rb') as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)

    assert sha256.hexdigest() == '931956f31d3f1a3f6047f3172b9e59ee3460d29f7c0c2bb219cbc8e9207795ff'

    model = ResNet.resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
    #put the model on the CPU for HPC
    pretext_model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)

    if torch.cuda.is_available():
        model=model.cuda()
    
    #TODO: replace slide_tile_paths with the actual tiles which are in memory
    return extract_features_(norm_wsi_img=norm_wsi_img, wsi_name=wsi_name, coords=coords, model=model, outdir=outdir, model_name='xiyuewang-retcll-931956f3', **kwargs) #removed model.cuda()


if __name__ == "__main__":
    logdir = args.cache_dir/'logfile'
    logging.basicConfig(filename=logdir, force=True)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f'Stored logfile in {logdir}')
    #init the Macenko normaliser
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    has_gpu=torch.cuda.is_available()
    print(f"GPU is available: {has_gpu}")
    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}")

    print("\nInitialising Macenko normaliser...")
    target = cv2.imread('normalization_template.jpg') #TODO: make scaleable with path
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    normalizer = stainNorm_Macenko.Normalizer()
    normalizer.fit(target)
    # norm = MacenkoNormalizer()
    # norm.fit(target)
    logging.info('Running WSI to normalised feature extraction...')
    total_start_time = time.time()
    svs_dir = glob.glob(f"{args.wsi_dir}/*.svs")
    for slide_url in (progress := tqdm(svs_dir, leave=False)):
        # breakpoint()
        slide_name = Path(slide_url).stem
        progress.set_description(slide_name)
        slide_cache_dir = args.cache_dir/slide_name
        slide_cache_dir.mkdir(parents=True, exist_ok=True)
        # print('\n')
        # print((f'{args.cache_dir}/{slide_name}.h5'))
        # print((os.path.exists((f'{args.cache_dir}/{slide_name}.h5'))))
        if not (os.path.exists((f'{args.cache_dir}/{slide_name}.h5'))):
            # Load WSI as one image
            if (slide_jpg := slide_cache_dir/'norm_slide.jpg').exists():
                img_norm_wsi_jpg = PIL.Image.open(slide_jpg)
            else:
                logging.info(f"\nLoading {slide_name}")
                try:
                    slide = openslide.OpenSlide(str(slide_url))
                except Exception as e:
                    logging.error(f"Failed loading {slide_name}, error: {e}")
                    continue
                    
                #measure time performance
                start_time = time.time()
                slide_array = load_slide(slide)
                #save raw .svs jpg
                (Image.fromarray(slide_array)).save(f'{slide_cache_dir}/slide.jpg')

                #remove .SVS from memory (couple GB)
                del slide
                
                print("\n--- Loaded slide: %s seconds ---" % (time.time() - start_time))
                #########################

                #########################
                #Do edge detection here and reject unnecessary tiles BEFORE normalisation
                bg_reject_array, rejected_tile_array, patch_shapes = reject_background(img = slide_array, patch_size=(224,224), step=224, outdir=args.cache_dir, save_tiles=False)

                logging.info(f"Normalising {slide_name}...")
                #measure time performance
                start_time = time.time()
                #pass raw slide_array for getting the initial concentrations, bg_reject_array for actual normalisation

                canny_img, img_norm_wsi_jpg, img_norm_wsi_list, coords_list = normalizer.transform(slide_array, bg_reject_array, rejected_tile_array, patch_shapes)
                
                print("Saving Canny background rejected image...")
                canny_img.save(f'{slide_cache_dir}/canny_slide.jpg')
                # norm_wsi_jpg = norm.transform(np.array(slide_array))
                
                #remove original slide jpg from memory
                del slide_array

                print(f"\n--- Normalised slide {slide_name}: {(time.time() - start_time)} seconds ---")
                #########################

                # img_norm_wsi_jpg = PIL.Image.fromarray(norm_wsi_jpg)
                img_norm_wsi_jpg.save(slide_jpg) #save WSI.svs -> WSI.jpg

            print(f"Extracting xiyue-wang macenko features from {slide_name}")
            #FEATURE EXTRACTION
            #measure time performance
            start_time = time.time()
            extract_xiyuewang_features_(norm_wsi_img=np.asarray(img_norm_wsi_list), wsi_name=slide_name, coords=coords_list, checkpoint_path=args.model, outdir=slide_cache_dir)
            print("\n--- Extracted features from slide: %s seconds ---" % (time.time() - start_time))
            #########################
        else:
            print(f"{slide_name}.h5 already exists. Skipping...")

    print(f"--- End-to-end processing time of {len(svs_dir)} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} ---")
