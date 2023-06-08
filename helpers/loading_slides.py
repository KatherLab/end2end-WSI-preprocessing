from typing import Dict, Tuple
from concurrent import futures
import logging
import openslide
from tqdm import tqdm
import numpy as np
import PIL
from scipy import ndimage
PIL.Image.MAX_IMAGE_PIXELS = None
import multiprocessing

def _load_tile(
    slide: openslide.OpenSlide, pos: Tuple[int, int], stride: Tuple[int, int], target_size: Tuple[int, int]
) -> np.ndarray:
    # Loads part of a WSI. Used for parallelization with ThreadPoolExecutor
    tile = slide.read_region(pos, 0, stride).convert('RGB').resize(target_size)
    return np.array(tile)


def load_slide(slide: openslide.OpenSlide, target_mpp: float = 256/224, cores: int = 8) -> np.ndarray:
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
    except KeyError:
        #if it fails, then try out missing mpp handler
        #TODO: create handlers for different image types
        try:
            slide_mpp = handle_missing_mpp(slide)
        except:
            print(f"Error: couldn't load MPP from slide!")
            return None
    tile_target_size = np.round(stride*slide_mpp/target_mpp).astype(int)
    #changed max amount of threads used
    with futures.ThreadPoolExecutor(cores) as executor:
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

def handle_missing_mpp(slide: openslide.OpenSlide) -> float:
    logging.exception("Missing mpp in metadata of this file format, reading mpp from metadata")
    import xml.dom.minidom as minidom
    xml_path = slide.properties['tiff.ImageDescription']
    doc = minidom.parseString(xml_path)
    collection = doc.documentElement
    images = collection.getElementsByTagName("Image")
    pixels = images[0].getElementsByTagName("Pixels")
    #tile_size_px = um_per_tile / float(pixels[0].getAttribute("PhysicalSizeX"))
    mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
    return mpp

def get_raw_tile_list(I_shape: tuple, bg_reject_array: np.array, rejected_tile_array: np.array, patch_shapes: np.array):
    canny_output_array=[]
    for i in range(len(bg_reject_array)):
        if not rejected_tile_array[i]:
            canny_output_array.append(np.array(bg_reject_array[i]))

    canny_img = PIL.Image.new("RGB", (I_shape[1], I_shape[0]))
    coords_list=[]
    zoom_list = []
    i_range = range(I_shape[0]//patch_shapes[0][0])
    j_range = range(I_shape[1]//patch_shapes[0][1])

    for i in i_range:
        for j in j_range:
            idx = i*len(j_range) + j
            canny_img.paste(PIL.Image.fromarray(np.array(bg_reject_array[idx])), (j*patch_shapes[idx][1], 
            i*patch_shapes[idx][0],j*patch_shapes[idx][1]+patch_shapes[idx][1],i*patch_shapes[idx][0]+patch_shapes[idx][0]))
            
            if not rejected_tile_array[idx]:
                coords_list.append((j*patch_shapes[idx][1], i*patch_shapes[idx][0]))
                zoom_list.append(1)

    return canny_img, canny_output_array, coords_list, zoom_list

def process_patch(patch, i, j, z):
    canny_norm_patch_list = []
    coords_list = []
    zoom_list = []

    if z > 1:
        if (np.count_nonzero(patch) / patch.size) >= min(1 / 4, 4 / z ** 2):
            patch = ndimage.zoom(patch, (224 / patch.shape[0], 224 / patch.shape[1], 1))
            canny_norm_patch_list.append(patch)
            coords_list.append((i, j))
            zoom_list.append(z)
    else:
        if np.sum(patch) > 0:
            canny_norm_patch_list.append(patch)
            coords_list.append((i, j))
            zoom_list.append(z)

    return canny_norm_patch_list, coords_list, zoom_list

def process_slide_jpg(slide_jpg: PIL.Image, zoom=False, cores=8):
    img_norm_wsi_jpg = PIL.Image.open(slide_jpg)
    image_array = np.array(img_norm_wsi_jpg)
    zoom_levels = [1, 2, 4, 8, 16]
    canny_norm_patch_list = []
    coords_list = []
    zoom_list = []
    total = 0
    patch_saved = 0

    if zoom:
        for z in zoom_levels:
            img_pxl = 224 * z
            patch_data = []
            for i in range(0, image_array.shape[0] - img_pxl, img_pxl):
                for j in range(0, image_array.shape[1] - img_pxl, img_pxl):
                    total += 1
                    patch = image_array[i:i + img_pxl, j:j + img_pxl, :]
                    patch_data.append((patch, i, j, z))

            with multiprocessing.Pool(cores) as pool:
                results = pool.starmap(process_patch, patch_data)
                for result in results:
                    canny_norm_patch_list.extend(result[0])
                    coords_list.extend(result[1])
                    zoom_list.extend(result[2])

            patch_saved += len(canny_norm_patch_list)

    else:
        patch_data = []
        for i in range(0, image_array.shape[0] - 224, 224):
            for j in range(0, image_array.shape[1] - 224, 224):
                total += 1
                patch = image_array[i:i + 224, j:j + 224, :]
                patch_data.append((patch, i, j, 1))

        with multiprocessing.Pool(cores) as pool:
            results = pool.starmap(process_patch, patch_data)
            for result in results:
                canny_norm_patch_list.extend(result[0])
                coords_list.extend(result[1])
                zoom_list.extend(result[2])

        patch_saved += len(canny_norm_patch_list)

    return canny_norm_patch_list, coords_list, zoom_list, patch_saved, total
    

# def process_slide_jpg(slide_jpg: PIL.Image, zoom=False, num_cores=8):
#     img_norm_wsi_jpg = PIL.Image.open(slide_jpg)
#     image_array = np.array(img_norm_wsi_jpg)
#     zoom_levels = [1,2,4,8,16] 
#     canny_norm_patch_list = []
#     coords_list=[]
#     zoom_list=[]
#     total=0
#     patch_saved=0
#     # if (np.count_nonzero(patch)/patch.size) >= 1/4:
#     if zoom:
#         for z in zoom_levels:
#             img_pxl = 224*z
#             for i in range(0, image_array.shape[0]-img_pxl, img_pxl):
#                 for j in range(0, image_array.shape[1]-img_pxl, img_pxl):
#                     total+=1
#                     patch = image_array[i:i+img_pxl, j:j+img_pxl, :]
#                     # if patch is not fully black (i.e. rejected previously)
#                     if z>1:
#                         if (np.count_nonzero(patch)/patch.size)>=min(1/4,4/z**2):
#                             patch = ndimage.zoom(patch, (224 / patch.shape[0], 224 / patch.shape[1], 1))
#                             canny_norm_patch_list.append(patch)
#                             coords_list.append((i,j))
#                             zoom_list.append(z)
#                             patch_saved+=1
#                     else:    
#                         if np.sum(patch) > 0:
#                             canny_norm_patch_list.append(patch)
#                             coords_list.append((i,j))
#                             zoom_list.append(z)
#                             patch_saved+=1
#     else:
#         for i in range(0, image_array.shape[0]-224, 224):
#             for j in range(0, image_array.shape[1]-224, 224):
#                 total+=1
#                 patch = image_array[i:i+224, j:j+224, :]
#                 # if patch is not fully black (i.e. rejected previously)
#                 if np.sum(patch) > 0:
#                     canny_norm_patch_list.append(patch)
#                     coords_list.append((i,j))
#                     zoom_list.append(1)
#                     patch_saved+=1
#     return canny_norm_patch_list, coords_list, zoom_list, patch_saved, total


# test get_raw_tile_list function
def test_get_raw_tile_list():
    img = np.random.randint(0, 255, size=(1000, 1000, 3), dtype=np.uint8)
    canny_img, canny_patch_list, coords_list = get_raw_tile_list(img.shape, img, img, (224,224))
    assert len(canny_patch_list) == 4
    assert len(coords_list) == 4
    assert canny_patch_list[0].shape == (224,224,3)
    assert coords_list[0] == (0,0)
