#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import os
import glob
import subprocess
import argparse
from tqdm import tqdm
from multiprocessing import Pool

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Convert .vsi files to .ome.tiff files using bfconvert.')
parser.add_argument('path_to_bftools', type=str, help='the path to the bfconvert tool')
parser.add_argument('input_folder', type=str, help='the folder containing the .vsi files to convert')
parser.add_argument('output_folder', type=str, help='the folder to save the converted .ome.tiff files')
parser.add_argument('series', type=str, help='plane to extract')

# Parse the command-line arguments
args = parser.parse_args()

# Find all .vsi files in the input folder
vsi_files = glob.glob(os.path.join(args.input_folder, '*.vsi'))

# Check if the output folder exists, create it if it doesn't
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

# Remove files that have already been converted
converted = glob.glob(os.path.join(args.output_folder, '*.tiff'))
converted_filenames = [os.path.basename(filename).split('.')[0] for filename in converted]
vsi_files = [filename for filename in vsi_files if os.path.basename(filename).split('.')[0] not in converted_filenames]

# Define a function to run bfconvert command
def convert_file(vsi_file):
    output_file = os.path.join(args.output_folder, ('.').join((os.path.basename(vsi_file).split('.'))[:-1]) + '.ome.tiff')
    command = [os.path.join(args.path_to_bftools, 'bfconvert'), '-series', args.series, vsi_file, output_file]
    subprocess.run(command)

if __name__ == '__main__':
    # Use a pool of 16 processes to run the conversion in parallel
    with Pool(16) as p:
        list(tqdm(p.imap(convert_file, vsi_files), total=len(vsi_files)))

