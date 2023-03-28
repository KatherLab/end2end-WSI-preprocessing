#!/usr/bin/env python3
from tqdm.contrib import tzip
import multiprocessing

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import os
from pathlib import Path

# Define the start path for the search
start_path = '/mnt/SATELLITE_04/FOXTROT-CRC-DX-BLOCKS/TRANSFER_FOXTROT_1000_to_1099'

# Search for all .jpg files and extract their parent directories
dir_list = set()
for dirpath, dirnames, filenames in os.walk(start_path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            dir_list.add(os.path.dirname(os.path.join(dirpath, filename)))

# Sort and de-duplicate the list of directories
dir_list = list(dict.fromkeys(sorted(dir_list)))

# Replace 'FOXTROT-CRC-DX-BLOCKS' with 'FOXTROT-CRC-DX-BLOCKS-NORM'
output_list = [d.replace('FOXTROT-CRC-DX-BLOCKS', 'FOXTROT-CRC-DX-BLOCKS-NORM') for d in dir_list]



# replace the elements in the list with Path objects and extract the parent directories
output_list1 = [str(Path(d).parent) for d in output_list]
dir_list1 = [str(Path(d).parent) for d in dir_list]
dir_list2 = list(dict.fromkeys(sorted(dir_list1)))
output_list2 = list(dict.fromkeys(sorted(output_list1)))
# Print the resulting list
print(output_list2[:10])
print(dir_list2[:10])

import tqdm
# Create the output directories if they don't exist
#for d in output_list2:
    #if not os.path.exists(d):
        #os.makedirs(d)
import subprocess

# Loop over the input and output directories, and call the Normalize.py script on each pair
import multiprocessing

def normalize(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("working on ", input_dir)
    try:
        subprocess.run(['python', 'Normalize.py', '-ip', input_dir, '-op', output_dir], check=True)
    except Exception as e:
        print(f"An exception occurred: {e}")
        return

if __name__ == '__main__':
    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(normalize, tzip(dir_list2, output_list2))
        #for _ in tqdm.tqdm(pool.imap_unordered(normalize, tzip(dir_list2, output_list2)), total=len(dir_list2)):
            #pass


