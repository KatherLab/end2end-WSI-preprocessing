# %%
import os
import re
import json
from typing import Optional, Sequence
import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import PIL
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import h5py
import os
from tqdm import tqdm
import numpy as np
import PIL
from pathlib import Path
from . import __version__


__all__ = ['extract_features_']

# supress DecompressionBombWarning: yes, our files are really that big (‘-’*)
PIL.Image.MAX_IMAGE_PIXELS = None



class SlideTileDataset(Dataset):
    def __init__(self, patches: np.array, transform=None, *, repetitions: int = 1) -> None:
        self.tiles = patches
        #assert self.tiles, f'no tiles found in {slide_dir}'
        self.tiles *= repetitions
        self.transform = transform

    # patchify returns a NumPy array with shape (n_rows, n_cols, 1, H, W, N), if image is N-channels.
    # H W N is Height Width N-channels of the extracted patch
    # n_rows is the number of patches for each column and n_cols is the number of patches for each row
    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        image = PIL.Image.fromarray(self.tiles[i])
        if self.transform:
            image = self.transform(image)

        return image


def _get_coords(filename) -> Optional[np.ndarray]:
    if matches := re.match(r'.*\((\d+),(\d+)\)\.jpg', str(filename)):
        coords = tuple(map(int, matches.groups()))
        assert len(coords) == 2, 'Error extracting coordinates'
        return np.array(coords)
    else:
        return None

def get_mask_from_thumb(thumb, threshold: int) -> np.ndarray:
    thumb = thumb.convert('L')
    return np.array(thumb) < threshold


def extract_features_(
        *,
        model, model_name, norm_wsi_img: np.ndarray, coords: list, wsi_name: str, outdir: Path,
        augmented_repetitions: int = 0, cores: int = 8, is_norm: bool = True
) -> None:
    """Extracts features from slide tiles.

    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """

    normal_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    augmenting_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.RandomVerticalFlip(p=.5),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=.5),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=.1, contrast=.2, saturation=.25, hue=.125)], p=.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    extractor_string = f'marugoto-extract-v{__version__}_{model_name}'
    with open(outdir.parent/'info.json', 'w') as f:
        json.dump({'extractor': extractor_string,
                  'augmented_repetitions': augmented_repetitions}, f)

    unaugmented_ds = SlideTileDataset(norm_wsi_img, normal_transform)
    augmented_ds = []

    #clean up memory
    del norm_wsi_img

    ds = ConcatDataset([unaugmented_ds, augmented_ds])
    dl = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False, num_workers=cores, drop_last=False)

    model = model.eval()

    feats = []
    for batch in tqdm(dl, leave=False):
        feats.append(
            model(batch.type_as(next(model.parameters()))).half().cpu().detach())

    with h5py.File(f'{outdir}.h5', 'w') as f:
        f['coords'] = coords
        f['feats'] = torch.concat(feats).cpu().numpy()
        f['augmented'] = np.repeat(
            [False, True], [len(unaugmented_ds), len(augmented_ds)])
        assert len(f['feats']) == len(f['augmented'])
        f.attrs['extractor'] = extractor_string


if __name__ == '__main__':
    import fire
    fire.Fire(extract_features_)

# %%
