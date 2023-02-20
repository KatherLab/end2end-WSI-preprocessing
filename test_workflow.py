#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

from pathlib import Path
import h5py

def test_h5_file():
    output_dir = Path('workspace/output')
    # get file end with .h5
    h5_files = list(output_dir.glob('**/*.h5'))
    file = h5_files[0]
    print(file)
    with h5py.File(file, "r") as f:
        print(f.keys())
        assert f['feats'].shape == (4059, 2048)
        assert f['coords'].shape == (4059, 2)

