#!/usr/bin/env python3
from tqdm import tqdm
import multiprocessing
from tqdm.contrib import tzip
import os
from pathlib import Path
import subprocess

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

start_path = '/mnt/SATELLITE_04/FOXTROT-CRC-DX-BLOCKS-NORM'

dir_list = set()
for dirpath, dirnames, filenames in os.walk(start_path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            dir_list.add(os.path.dirname(os.path.join(dirpath, filename)))

dir_list = list(dict.fromkeys(sorted(dir_list)))
output_list = [d.replace('FOXTROT-CRC-DX-BLOCKS-NORM', 'FOXTROT-CRC-DX-NORM-features') for d in dir_list]

dir_list1 = [str(Path(d).parent) for d in dir_list]
output_list1 = [str(Path(d).parent) for d in output_list]

dir_list2 = list(dict.fromkeys(sorted(dir_list1)))
output_list2 = list(dict.fromkeys(sorted(output_list1)))

#print(output_list2[:10])
#print(dir_list2[:10])

def normalize(args):
    input_dir, output_dir = args
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_dir += '/*'

    try:
        command = f"python -m marugoto.extract.xiyue_wang --checkpoint-path /home/jeff/Downloads/best_ckpt.pth --outdir {output_dir} {input_dir}"
        result = subprocess.run(command, shell=True, check=True)
        print(command)
    except Exception as e:
        print(f"An exception occurred: {e}")
        return

if __name__ == '__main__':
    with multiprocessing.Pool(processes=1) as pool:
        for _ in tqdm(pool.imap_unordered(normalize, zip(dir_list2, output_list2)), total=len(dir_list2)):
            pass

