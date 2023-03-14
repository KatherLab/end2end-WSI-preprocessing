#!/usr/bin/env python3
import argparse
import io
from pathlib import Path
import sys
import shutil
from typing import Dict, Tuple
from concurrent import futures
from urllib.parse import urlparse
import warnings


# loading all the below packages takes quite a bit of time, so get cli parsing
# out of the way beforehand so it's more responsive in case of errors
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create heatmaps for MIL models.')
    parser.add_argument('slide_urls', metavar='SLIDE_URL', type=urlparse,
                        nargs='+', help='Slides to create heatmaps for.')
    parser.add_argument('-m', '--model-path', type=Path, required=True,
                        help='MIL model used to generate attention / score maps.')
    parser.add_argument('-o', '--output-path', type=Path, required=True,
                        help='Path to save results to.')
    parser.add_argument('-t', '--true-class', type=str, required=True,
                        help='Class to be rendered as "hot" in the heatmap.')
    parser.add_argument('--from-file', metavar='FILE', type=Path,
                        help='File containing a list of slides to create heatmaps for.')
    parser.add_argument('--blur-kernel-size', metavar='SIZE', type=int, default=15,
                        help='Size of gaussian pooling filter. 0 disables pooling.')
    parser.add_argument('--cache-dir', type=Path, default=None,
                        help='Directory to cache extracted features etc. in.')
    threshold_group = parser.add_argument_group(
        'thresholds', 'thresholds for scaling attention / score values')
    threshold_group.add_argument('--mask-threshold', metavar='THRESH', type=int, default=224,
                                 help='Brightness threshold for background removal.')
    threshold_group.add_argument('--att-upper-threshold', metavar='THRESH', type=float, default=1.,
                                 help='Quantile to squash attention from during attention scaling '
                                 ' (e.g. 0.99 will lead to the top 1%% of attention scores to become 1)')
    threshold_group.add_argument('--att-lower-threshold', metavar='THRESH', type=float, default=.01,
                                 help='Quantile to squash attention to during attention scaling '
                                 ' (e.g. 0.01 will lead to the bottom 1%% of attention scores to become 0)')
    threshold_group.add_argument('--score-threshold', metavar='THRESH', type=float, default=.95,
                                 help='Quantile to consider in score scaling '
                                 '(e.g. 0.95 will discard the top / bottom 5%% of score values as outliers)')
    colormap_group = parser.add_argument_group(
        'colors',
        'color maps to use for attention / score maps (see https://matplotlib.org/stable/tutorials/colors/colormaps.html)')
    colormap_group.add_argument('--att-cmap', metavar='CMAP', type=str, default='magma',
                                help='Color map to use for the attention heatmap.')
    colormap_group.add_argument('--score-cmap', metavar='CMAP', type=str, default='coolwarm',
                                help='Color map to use for the score heatmap.')
    colormap_group.add_argument('--att-alpha', metavar='ALPHA', type=float, default=.5,
                                help='Opaqueness of attention map.')
    colormap_group.add_argument('--score-alpha', metavar='ALPHA', type=float, default=1.,
                                help='Opaqueness of score map at highest-attention location.')
    args = parser.parse_args()
    if not args.cache_dir:
        warnings.warn(
            'no cache directory specified! If you are generating heat maps for multiple targets, '
            'it is HIGHLY recommended to manually set a cache directory. This directory should be '
            'the SAME for each run.')
        args.cache_dir = args.output_path/'cache'

    assert args.att_upper_threshold >= 0 and args.att_upper_threshold <= 1, \
        'threshold needs to be between 0 and 1.'
    assert args.att_lower_threshold >= 0 and args.att_lower_threshold <= 1, \
        'threshold needs to be between 0 and 1.'
    assert args.att_lower_threshold < args.att_upper_threshold, \
        'lower attention threshold needs to be lower than upper attention threshold.'


if (p := './RetCCL') not in sys.path:
    sys.path = [p] + sys.path
import ResNet

import torch.nn as nn
import torch
from torchvision import transforms
import os
from matplotlib import pyplot as plt
import openslide
from tqdm import tqdm
import numpy as np
from fastai.vision.all import load_learner
from pyzstd import ZstdFile
import PIL
from wsiProcessing.sftp import get_wsi

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
        for tile_future in tqdm(futures.as_completed(future_coords), total=steps*steps, desc='Loading WSI', leave=False):
            i, j = future_coords[tile_future]
            tile = tile_future.result()
            x, y = tile_target_size * (j, i)
            im[y:y+tile.shape[0], x:x+tile.shape[1], :] = tile

    return im


def batch1d_to_batch_2d(batch1d):
    batch2d = nn.BatchNorm2d(batch1d.num_features)
    batch2d.state_dict = batch1d.state_dict
    return batch2d


def dropout1d_to_dropout2d(dropout1d):
    return nn.Dropout2d(dropout1d.p)


def linear_to_conv2d(linear):
    """Converts a fully connected layer to a 1x1 Conv2d layer with the same weights."""
    conv = nn.Conv2d(in_channels=linear.in_features,
                     out_channels=linear.out_features, kernel_size=1)
    conv.load_state_dict({
        "weight": linear.weight.view(conv.weight.shape),
        "bias": linear.bias.view(conv.bias.shape),
    })
    return conv


if __name__ == '__main__':
    # use all the threads
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(os.cpu_count())

    # default imgnet transforms
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # load base fully convolutional model (w/o pooling / flattening or head)
    # In this case we're loading the xiyue wang RetCLL model, change this bit for other networks
    import ResNet
    base_model = ResNet.resnet50(num_classes=128, mlp=False,
                                    two_branch=False, normlinear=True)
    pretext_model = torch.load('./xiyue-wang.pth')
    base_model.avgpool = nn.Identity()
    base_model.flatten = nn.Identity()
    base_model.fc = nn.Identity()
    base_model.load_state_dict(pretext_model, strict=True)
    base_model = base_model.eval().cuda()

    # transform MIL model into fully convolutional equivalent
    learn = load_learner(args.model_path)
    classes = learn.dls.train.dataset._datasets[-1].encode.categories_[0]
    assert args.true_class in classes, \
        f'{args.true_class} not a target of {args.model_path}! ' \
        f'(Did you mean any of {list(classes)}?)'
    true_class_idx = (classes == args.true_class).argmax()
    att = nn.Sequential(
        linear_to_conv2d(learn.encoder[0]),
        nn.ReLU(),
        linear_to_conv2d(learn.attention[0]),
        nn.Tanh(),
        linear_to_conv2d(learn.attention[2]),
    ).eval().cuda()

    score = nn.Sequential(
        linear_to_conv2d(learn.encoder[0]),
        nn.ReLU(),
        batch1d_to_batch_2d(learn.head[1]),
        dropout1d_to_dropout2d(learn.head[2]),
        linear_to_conv2d(learn.head[3]),
    ).eval().cuda()

    # we operate in two steps: we first collect all attention values / scores,
    # the entirety of which we then calculate our scaling parameters from.  Only
    # then we output the actual maps.
    attention_maps: Dict[Path, torch.Tensor] = {}
    score_maps: Dict[Path, torch.Tensor] = {}
    masks: Dict[Path, torch.Tensor] = {}

    print('Extracting features, attentions and scores...')
    for slide_url in (progress := tqdm(args.slide_urls, leave=False)):
        slide_name = Path(slide_url.path).stem
        progress.set_description(slide_name)
        slide_cache_dir = args.cache_dir/slide_name
        slide_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load WSI as one image
        if (slide_jpg := slide_cache_dir/'slide.jpg').exists():
            slide_array = np.array(PIL.Image.open(slide_jpg))
        else:
            slide_path = get_wsi(slide_url, cache_dir=args.cache_dir)
            slide = openslide.OpenSlide(str(slide_path))
            slide_array = load_slide(slide)
            PIL.Image.fromarray(slide_array).save(slide_jpg)

        # pass the WSI through the fully convolutional network'
        # since our RAM is still too small, we do this in two steps
        # (if you run out of RAM, try upping the number of slices)
        if (feats_pt := slide_cache_dir/'feats.pt.zst').exists():
            with ZstdFile(feats_pt, mode='rb') as fp:
                feat_t = torch.load(io.BytesIO(fp.read()))
            feat_t = feat_t.float()
        elif (slide_cache_dir/'feats.pt').exists():
            feat_t = torch.load(slide_cache_dir/'feats.pt').float()
        else:
            max_slice_size = 0xa800000  # experimentally determined
            # ceil(pixels/max_slice_size)
            no_slices = (np.prod(slide_array.shape)
                         + max_slice_size-1) // max_slice_size
            step = slide_array.shape[1]//no_slices
            slices = []
            for slice_i in range(no_slices):
                x = tfms(slide_array[:, slice_i*step:(slice_i+1)*step, :])
                with torch.inference_mode():
                    res = base_model(x.unsqueeze(0).cuda())
                    slices.append(res.detach().cpu())
            feat_t = torch.concat(slices, 3).squeeze()
            # save the features (with compression)
            with ZstdFile(feats_pt, mode='wb') as fp:
                torch.save(feat_t, fp)

        feat_t = feat_t.cuda()
        # pool features, but use gaussian blur instead of avg pooling to reduce artifacts
        if args.blur_kernel_size:
            feat_t = transforms.functional.gaussian_blur(
                feat_t, kernel_size=args.blur_kernel_size)

        # calculate attention / classification scores according to the MIL model
        with torch.inference_mode():
            att_map = att(feat_t).squeeze().cpu()
            score_map = score(feat_t.unsqueeze(0)).squeeze()
            score_map = torch.softmax(score_map, 0).cpu()

        # compute foreground mask
        mask = np.array(PIL.Image.fromarray(slide_array).resize(
            att_map.shape[::-1]).convert('L')) < args.mask_threshold

        attention_maps[slide_name] = att_map
        score_maps[slide_name] = score_map
        masks[slide_name] = mask

    # now we can use all of the features to calculate the scaling factors
    all_attentions = torch.cat(
        [attention_maps[s].view(-1)[masks[s].reshape(-1)] for s in score_maps.keys()])
    att_lower = all_attentions.quantile(args.att_lower_threshold)
    att_upper = all_attentions.quantile(args.att_upper_threshold)

    all_scores = torch.cat([
        # mask out background scores, then linearize them
        score_maps[s].view(2, -1) \
        .permute(1, 0)[masks[s].reshape(-1)] \
        .permute(1, 0)
        for s in score_maps.keys()],
        dim=1)
    centered_score = all_scores[true_class_idx] - (1/len(classes))
    scale_factor = torch.quantile(
        centered_score.abs(), args.score_threshold) * 2

    print('Writing heatmaps...')
    for slide_url in (progress := tqdm(args.slide_urls, leave=False)):
        slide_name = Path(slide_url.path).stem
        slide_cache_dir = args.cache_dir/slide_name
        slide_outdir = args.output_path/slide_name
        slide_outdir.mkdir(parents=True, exist_ok=True)

        progress.set_description(slide_name)
        slide_outdir = args.output_path/slide_name

        slide_im = PIL.Image.open(slide_cache_dir/'slide.jpg')
        if not (slide_outdir/'slide.jpg').exists():
            shutil.copyfile(slide_cache_dir/'slide.jpg',
                            slide_outdir/'slide.jpg')

        mask = masks[slide_name]

        # attention map
        att_map = (attention_maps[slide_name] - att_lower) \
            / (att_upper - att_lower)
        att_map = att_map.clamp(0, 1)

        # bare attention
        im = plt.get_cmap(args.att_cmap)(att_map)
        im[:, :, 3] = mask
        PIL.Image.fromarray(
            np.uint8(im*255.)).save(slide_outdir/'attention.png')
        # attention map (blended with slide)
        im[:, :, 3] *= args.att_alpha
        map_im = PIL.Image.fromarray(np.uint8(im*255.))
        map_im = map_im.resize(slide_im.size, PIL.Image.Resampling.NEAREST)
        x = slide_im.copy().convert('RGBA')
        x.paste(map_im, mask=map_im)
        x.convert('RGB').save(slide_outdir/'attention_overlayed.jpg')

        # score map
        scaled_score_map = (
            (score_maps[slide_name][true_class_idx] - 1/len(classes))
            / scale_factor
            + 1/len(classes))
        scaled_score_map = (scaled_score_map * mask).clamp(0, 1)

        # create image with RGB from scores, Alpha from attention
        im = plt.get_cmap(args.score_cmap)(scaled_score_map)
        im[:, :, 3] = att_map * mask * args.score_alpha
        map_im = PIL.Image.fromarray(np.uint8(im*255.))
        map_im.save(slide_outdir/'map.png')
        # overlayed onto slide
        map_im = map_im.resize(slide_im.size, PIL.Image.Resampling.NEAREST)
        x = slide_im.copy().convert('RGBA')
        x.paste(map_im, mask=map_im)
        x.convert('RGB').save(slide_outdir/'map_overlayed.jpg')
