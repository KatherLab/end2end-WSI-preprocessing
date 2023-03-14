#!/usr/bin/env python3
import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from marugoto.extract.extract import extract_features_
from .swin_transformer import swin_tiny_patch4_window7_224, ConvStem


def extract_ctranspath_features_(*slide_tile_paths: Path, checkpoint_path: str, **kwargs):
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

    assert sha256.hexdigest() == '7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539'

    model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
    model.head = nn.Identity()

    ctranspath = torch.load(checkpoint_path)
    model.load_state_dict(ctranspath['model'], strict=True)

    return extract_features_(slide_tile_paths=slide_tile_paths, model=model.cuda(), model_name='xiyuewang-ctranspath-7c998680', **kwargs)
