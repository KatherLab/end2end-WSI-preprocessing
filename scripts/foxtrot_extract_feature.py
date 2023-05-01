#!/usr/bin/env python3
from tqdm import tqdm
import multiprocessing
from tqdm.contrib import tzip
import argparse
import os
from pathlib import Path
import subprocess
import csv

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

def extract_features(args, checkpoint_path):
    input_dir, output_dir = args
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        command = f"python -m marugoto.extract.ctranspath --checkpoint-path {checkpoint_path} --outdir {output_dir} {input_dir} --histaugan True"
        result = subprocess.run(command, shell=True, check=True)
        print(command)
    except Exception as e:
        print(f"An exception occurred: {e}")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_path", help="Path to search for .jpg files", required=True)
    parser.add_argument("--table_path", help="Path to the CSV table", required=True)
    parser.add_argument("--checkpoint_path", help="Path to the checkpoint file", required=True)
    args = parser.parse_args()

    start_path = args.start_path
    table_path = args.table_path
    checkpoint_path = args.checkpoint_path

    filenames_csv = []
    with open(table_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            filenames_csv.append(row[0])

    input_directories = set()
    for dirpath, dirnames, filenames in os.walk(start_path):
        last_part = os.path.basename(dirpath)
        if last_part not in filenames_csv:
            continue

        for filename in filenames:
            if filename.endswith('.jpg'):
                input_directories.add(os.path.join(dirpath, filename))

    input_directories = list(dict.fromkeys(sorted(input_directories)))
    output_directories = [d.replace('FOXTROT-CRC-DX-BLOCKS', 'FOXTROT-CRC-DX-features-histaugan') for d in input_directories]

    input_parent_directories = [str(Path(d).parent) for d in input_directories]
    output_parent_directories = [str(Path(d).parent) for d in output_directories]

    input_parent_directories = list(dict.fromkeys(sorted(input_parent_directories)))
    output_parent_directories = list(dict.fromkeys(sorted(output_parent_directories)))

    with multiprocessing.Pool(processes=1) as pool:
        for _ in tqdm(pool.imap_unordered(lambda args: extract_features(args, checkpoint_path), zip(input_parent_directories, output_parent_directories)), total=len(input_parent_directories)):
            pass
