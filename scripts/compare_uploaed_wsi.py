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


def get_local_filenames(path, suffix):
    """
    Get a list of filenames with the input suffix by recursively checking a local folder with the input path.
    """
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix):
                filenames.append(os.path.join(root, file))
    return filenames


if __name__ == '__main__':
    # Define the command-line arguments
    parser = argparse.ArgumentParser(
        description='Count the number of files with a specified suffix in a remote directory.')
    parser.add_argument('username', type=str, help='the username to use for the remote connection')
    parser.add_argument('remote_path', type=str, help='the remote path to the directory to check')
    parser.add_argument('suffix', type=str, help='the suffix to check for')
    parser.add_argument('local_path', type=str, help='the local path to the directory to check')
    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the list of filenames with the input suffix from the local directory
    local_filenames = get_local_filenames(args.local_path, args.suffix)

    # Run the rsync command and get the list of filenames with the specified suffix from the remote directory
    output = subprocess.check_output(["rsync", "-r", "--list-only", f"{args.username}@dgw.zih.tu-dresden.de:{args.remote_path}"])
    lines = output.decode().splitlines()
    remote_filenames = [line.split()[-1] for line in lines[1:]] # skip the first line, which is the total size of files

    # Compare the lists of filenames and print out the missing files and the number of the lists
    missing_files = set(local_filenames) - set(remote_filenames)
    print(f"Number of files with suffix '{args.suffix}' in local directory: {len(local_filenames)}")
    print(f"Number of files with suffix '{args.suffix}' in remote directory: {len(remote_filenames)}")
    print(f"Number of missing files: {len(missing_files)}")
    print("Missing files:")
    for file in missing_files:
        print(file)
