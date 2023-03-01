#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

from pathlib import Path
import h5py
retccl_feature_dimension = 2048
true_tiles_number = {
    '5421.h5'           : 16,
    '1031228_001.h5'    : 77,
}


def test_h5_file():
    output_dir = Path('workspace/output')
    # get file end with .h5
    h5_files = list(output_dir.glob('**/*.h5'))
    for file in h5_files:
        print('\n checking: ', file.name)
        with h5py.File(file, "r") as f:
            assert true_tiles_number[file.name] == f['feats'].shape[0]

