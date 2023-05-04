import numpy as np
import cv2
import PIL
from concurrent import futures
from pathlib import Path
import time
from typing import Dict, Tuple, List, Any
import os

from numpy import ndarray


def canny_fcn(patch: np.array) -> Tuple[np.array, bool]:
    patch_img = PIL.Image.fromarray(patch)
    tile_to_greyscale = patch_img.convert('L')
    # tile_to_greyscale is an PIL.Image.Image with image mode L
    # Note: If you have an L mode image, that means it is
    # a single channel image - normally interpreted as greyscale.
    # The L means that is just stores the Luminance.
    # It is very compact, but only stores a greyscale, not colour.

    tile2array = np.array(tile_to_greyscale)

    # hardcoded thresholds
    edge = cv2.Canny(tile2array, 40, 100)

    # avoid dividing by zero
    edge = (edge / np.max(edge) if np.max(edge) != 0 else 0)
    edge = (((np.sum(np.sum(edge)) / (tile2array.shape[0]*tile2array.shape[1])) * 100)
        if (tile2array.shape[0]*tile2array.shape[1]) != 0 else 0)

    # hardcoded limit. Less or equal to 2 edges will be rejected (i.e., not saved)
    if(edge < 2.):
        #return a black image + rejected=True
        return (np.zeros_like(patch), True)
    else:
        #return the patch + rejected=False
        return (patch, False)


def reject_background(img: np.array, patch_size: Tuple[int,int], step: int, save_tiles: bool = False, outdir: Path = None, cores: int = 8) -> \
Tuple[ndarray, ndarray, List[Any]]:
    img_shape = img.shape
    print(f"\nSize of WSI: {img_shape}")

    split=True
    x=(img_shape[0]//patch_size[0])*(img_shape[1]//patch_size[1])

    print(f"Splitting WSI into {x} tiles and Canny background rejection...")
    begin = time.time()
    patches_shapes_list=[]

    with futures.ThreadPoolExecutor(cores) as executor: #os.cpu_count()
        future_coords: Dict[futures.Future, int] = {}
        i_range = range(img_shape[0]//patch_size[0])
        j_range = range(img_shape[1]//patch_size[1])
        for i in i_range:
            for j in j_range:
                patch = img[(i*patch_size[0]):(i*patch_size[0]+step), (j*patch_size[1]):(j*patch_size[1]+step)]
                #(PIL.Image.fromarray(patch)).save(f'{outdir}/patch_{i*len(j_range) + j}.jpg')
                patches_shapes_list.append(patch.shape)
                future = executor.submit(canny_fcn, patch)
                # begin_time_list.append(time.time())
                future_coords[future] = i*len(j_range) + j # index 0 - 3. (0,0) = 0, (0,1) = 1, (1,0) = 2, (1,1) = 3

        del img
        
        begin = time.time()
        #num of patches x 224 x 224 x 3 for RGB patches
        ordered_patch_list = np.zeros((x, patch_size[0], patch_size[1], 3), dtype=np.uint8)
        rejected_tile_list = np.zeros(x, dtype=bool)
        for tile_future in futures.as_completed(future_coords):
            i = future_coords[tile_future]
            #print(f'Received normalised patch #{i} from thread in {time.time()-begin_time_list[i]} seconds')
            patch, is_rejected = tile_future.result()
            ordered_patch_list[i] = patch
            rejected_tile_list[i] = is_rejected

        end = time.time()

    print(f"\nFinished Canny background rejection, rejected {np.sum(rejected_tile_list)} tiles: {end-begin}")
    return ordered_patch_list, rejected_tile_list, patches_shapes_list


# test canny_fcn
def test_canny_fcn():
    img = cv2.imread('test.jpg')
    patch = img[0:224, 0:224]
    patch, is_rejected = canny_fcn(patch)
    assert patch.shape == (224, 224, 3)
    assert is_rejected == False

# test reject_background function
def test_reject_background():
    img = np.random.randint(0, 255, size=(1000, 1000, 3), dtype=np.uint8)
    bg_reject_array, rejected_tile_array, patch_shapes = reject_background(img = img, patch_size=(224,224), step=224, outdir='.', save_tiles=False)
    assert bg_reject_array.shape == (1000, 1000, 3)
    assert rejected_tile_array.shape == (1000, 1000, 3)
    assert patch_shapes == (224,224)

    # test that the rejected tiles are all black
    assert np.all(rejected_tile_array == 0)
