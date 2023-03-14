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
                        help='Path of where model for the feature extractor is.')
    parser.add_argument('--cache-dir', type=Path, default=None,
        help='Directory to cache extracted features etc. in.')
    parser.add_argument('-e', '--extractor', type=str, help='Feature extractor to use.')
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
from common import supported_extensions
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
    try:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        print(f"Read slide MPP of {slide_mpp} from meta-data")
    except:
        print(f"Error: couldn't load MPP from slide!")
        return None
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
from marugoto.marugoto.extract.ctranspath.swin_transformer import swin_tiny_patch4_window7_224, ConvStem
from concurrent_canny_rejection import reject_background
from PIL import Image

# %%

import hashlib
import torch
import torch.nn as nn
from marugoto.marugoto.extract.extract import extract_features_
from marugoto.marugoto.extract.xiyue_wang.RetCLL import ResNet
from marugoto.marugoto.extract.ctranspath.swin_transformer import swin_tiny_patch4_window7_224, ConvStem
from PIL import Image

class FeatureExtractor:
    def __init__(self, model_type):
        self.model_type = model_type

    def extract_features(self, norm_wsi_img: PIL.Image, wsi_name: str, coords: list, checkpoint_path: str, outdir: Path, **kwargs):
        """Extracts features from slide tiles.
        Args:
            checkpoint_path:  Path to the model checkpoint file.
        """
        sha256 = hashlib.sha256()
        with open(checkpoint_path, 'rb') as f:
            while True:
                data = f.read(1 << 16)
                if not data:
                    break
                sha256.update(data)

        if self.model_type == 'xiyue_wang':
            assert sha256.hexdigest() == '931956f31d3f1a3f6047f3172b9e59ee3460d29f7c0c2bb219cbc8e9207795ff'

            model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            pretext_model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.fc = nn.Identity()
            model.load_state_dict(pretext_model, strict=True)

            if torch.cuda.is_available():
                model = model.cuda()

            return extract_features_(norm_wsi_img=norm_wsi_img, wsi_name=wsi_name, coords=coords, model=model, outdir=outdir, model_name='xiyuewang-retcll-931956f3', **kwargs)

        elif self.model_type == 'ctranspath':
            assert sha256.hexdigest() == '7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539'

            model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
            model.head = nn.Identity()

            ctranspath = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            # print keys and values of ctranspath

            model.load_state_dict(ctranspath['model'], strict=True)

            return extract_features_(norm_wsi_img=norm_wsi_img, wsi_name=wsi_name, coords=coords, model=model, outdir=outdir,
                                     model_name='xiyuewang-ctranspath-7c998680', **kwargs)

        else:
            raise ValueError('Invalid model type')

def get_raw_tile_list(I_shape: tuple, bg_reject_array: np.array, rejected_tile_array: np.array, patch_shapes: np.array):
    canny_output_array=[]
    for i in range(len(bg_reject_array)):
        if not rejected_tile_array[i]:
            canny_output_array.append(np.array(bg_reject_array[i]))

    canny_img = Image.new("RGB", (I_shape[1], I_shape[0]))
    coords_list=[]
    i_range = range(I_shape[0]//patch_shapes[0][0])
    j_range = range(I_shape[1]//patch_shapes[0][1])

    for i in i_range:
        for j in j_range:
            idx = i*len(j_range) + j
            canny_img.paste(Image.fromarray(np.array(bg_reject_array[idx])), (j*patch_shapes[idx][1], 
            i*patch_shapes[idx][0],j*patch_shapes[idx][1]+patch_shapes[idx][1],i*patch_shapes[idx][0]+patch_shapes[idx][0]))
            
            if not rejected_tile_array[idx]:
                coords_list.append((j*patch_shapes[idx][1], i*patch_shapes[idx][0]))

    return canny_img, canny_output_array, coords_list

if __name__ == "__main__":
    # print current dir
    print(f"Current working directory: {os.getcwd()}")
    Path(args.cache_dir).mkdir(exist_ok=True, parents=True)
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
    norm=True
    if norm:
        print("\nInitialising Macenko normaliser...")
        target = cv2.imread('normalization_template.jpg') #TODO: make scaleable with path
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        normalizer = stainNorm_Macenko.Normalizer()
        normalizer.fit(target)
        logging.info('Running WSI to normalised feature extraction...')
    else:
        logging.info('Running WSI to raw feature extraction...')
    # norm = MacenkoNormalizer()
    # norm.fit(target)
    total_start_time = time.time()
    svs_dir = sum((list(args.wsi_dir.glob(f'**/*.{ext}'))
                  for ext in supported_extensions),
                 start=[])
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
                image_array = np.array(img_norm_wsi_jpg)
                canny_norm_patch_list = []
                coords_list=[]
                total=0
                patch_saved=0
                for i in range(0, image_array.shape[0]-224, 224):
                    for j in range(0, image_array.shape[1]-224, 224):
                        total+=1
                        patch = image_array[j:j+224, i:i+224, :]
                        if not np.all(patch):
                            canny_norm_patch_list.append(patch)
                            coords_list.append((j,i))
                            patch_saved+=1
                print(f"Loaded normalised canny image, {patch_saved}/{total} tiles remain")
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
                if slide_array is None:
                    print(f"Skipping slide and deleting {slide_url} due to missing MPP...")
                    os.remove(str(slide_url))
                    continue
                #save raw .svs jpg
                (Image.fromarray(slide_array)).save(f'{slide_cache_dir}/slide.jpg')

                #remove .SVS from memory (couple GB)
                #del slide
                
                print("\n--- Loaded slide: %s seconds ---" % (time.time() - start_time))
                #########################

                #########################
                #Do edge detection here and reject unnecessary tiles BEFORE normalisation
                bg_reject_array, rejected_tile_array, patch_shapes = reject_background(img = slide_array, patch_size=(224,224), step=224, outdir=args.cache_dir, save_tiles=False)

                #measure time performance
                start_time = time.time()
                #pass raw slide_array for getting the initial concentrations, bg_reject_array for actual normalisation
                if norm:
                    logging.info(f"Normalising {slide_name}...")
                    canny_img, img_norm_wsi_jpg, canny_norm_patch_list, coords_list = normalizer.transform(slide_array, bg_reject_array, rejected_tile_array, patch_shapes)
                    print(f"\n--- Normalised slide {slide_name}: {(time.time() - start_time)} seconds ---")
                    img_norm_wsi_jpg.save(slide_jpg) #save WSI.svs -> WSI.jpg

                else:
                    canny_img, canny_norm_patch_list, coords_list = get_raw_tile_list(slide_array.shape, bg_reject_array, rejected_tile_array, patch_shapes)

                print("Saving Canny background rejected image...")
                canny_img.save(f'{slide_cache_dir}/canny_slide.jpg')
                # norm_wsi_jpg = norm.transform(np.array(slide_array))
                
                #remove original slide jpg from memory
                del slide_array
                #print(f"Deleting slide {slide_name} from local folder...")
                #os.remove(str(slide_url))


                # img_norm_wsi_jpg = PIL.Image.fromarray(norm_wsi_jpg)
                img_norm_wsi_jpg.save(slide_jpg) #save WSI.svs -> WSI.jpg

            print(f"Extracting {args.extractor} features from {slide_name}")
            #FEATURE EXTRACTION
            #measure time performance
            start_time = time.time()
            extractor = FeatureExtractor(args.extractor)
            features = extractor.extract_features(norm_wsi_img=canny_norm_patch_list, wsi_name=slide_name, coords=coords_list, checkpoint_path=args.model, outdir=slide_cache_dir)
            print("\n--- Extracted features from slide: %s seconds ---" % (time.time() - start_time))
            #########################
            #print(f"Deleting slide {slide_name} from local folder...")
            #os.remove(str(slide_url))

        else:
            print(f"{slide_name}.h5 already exists. Skipping...")
            #print(f"Deleting slide {slide_name} from local folder...")
            #os.remove(str(slide_url))

    print(f"--- End-to-end processing time of {len(svs_dir)} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} ---")

# test get_raw_tile_list function
def test_get_raw_tile_list():
    img = np.random.randint(0, 255, size=(1000, 1000, 3), dtype=np.uint8)
    canny_img, canny_patch_list, coords_list = get_raw_tile_list(img.shape, img, img, (224,224))
    assert len(canny_patch_list) == 4
    assert len(coords_list) == 4
    assert canny_patch_list[0].shape == (224,224,3)
    assert coords_list[0] == (0,0)

# test reject_background function
def test_reject_background():
    img = np.random.randint(0, 255, size=(1000, 1000, 3), dtype=np.uint8)
    bg_reject_array, rejected_tile_array, patch_shapes = reject_background(img = img, patch_size=(224,224), step=224, outdir='.', save_tiles=False)
    assert bg_reject_array.shape == (1000, 1000, 3)
    assert rejected_tile_array.shape == (1000, 1000, 3)
    assert patch_shapes == (224,224)

    # test that the rejected tiles are all black
    assert np.all(rejected_tile_array == 0)

# test extract_xiyuewang_features_ function
def test_extract_xiyuewang_features_():
    img = np.random.randint(0, 255, size=(1000, 1000, 3), dtype=np.uint8)
    feature_extractor = FeatureExtractor('xiyuewang')
    feature_extractor.extract_features(norm_wsi_img=img, wsi_name='test', coords=[(0,0)], checkpoint_path='.', outdir='.')

    # test that the output file exists
    assert os.path.exists('test.h5')

    # test that the output file is not empty
    assert os.path.getsize('test.h5') > 0

    # remove the output file
    os.remove('test.h5')
