#!/usr/bin/env python3
from tqdm.contrib import tzip
import multiprocessing
import argparse
import os
from pathlib import Path
import subprocess
import tqdm

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

def normalize(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Working on", input_dir)
    try:
        subprocess.run(['python', 'Normalize.py', '-ip', input_dir, '-op', output_dir], check=True)
    except Exception as e:
        print(f"An exception occurred: {e}")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_path", help="Path to search for .jpg files", required=True)
    args = parser.parse_args()

    start_path = args.start_path

    input_directories = set()
    for dirpath, dirnames, filenames in os.walk(start_path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                input_directories.add(os.path.dirname(os.path.join(dirpath, filename)))

    input_directories = list(dict.fromkeys(sorted(input_directories)))
    output_directories = [d.replace('FOXTROT-CRC-DX-BLOCKS', 'FOXTROT-CRC-DX-BLOCKS-NORM') for d in input_directories]

    input_parent_directories = [str(Path(d).parent) for d in input_directories]
    output_parent_directories = [str(Path(d).parent) for d in output_directories]

    input_parent_directories = list(dict.fromkeys(sorted(input_parent_directories)))
    output_parent_directories = list(dict.fromkeys(sorted(output_parent_directories)))

    #print(output_parent_directories[:10])
    #print(input_parent_directories[:10])

    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(normalize, tzip(input_parent_directories, output_parent_directories))
