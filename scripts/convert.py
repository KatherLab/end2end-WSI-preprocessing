import os
import glob
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
from getfiles import filenames
path_to_bftools = '/mnt/Sirius_03_empty/bftools'
input_folder = '/mnt/Sirius_03_empty/IGR_HRD_OV/IGR-SLIDES-DX'
output_folder = '/mnt/SATELLITE_05_CHI/IGR_HRD_OV/IGR-SLIDES-DX-converted'

# Find all .vsi files in the input folder
vsi_files = glob.glob(os.path.join(input_folder, '*.vsi'))
converted= glob.glob(os.path.join('/mnt/SATELLITE_05_CHI/IGR_HRD_OV/IGR-SLIDES-DX-converted', '*.tiff'))
print(len(filenames))
print(len(vsi_files))
#print(len(converted))
'''
for filename in vsi_files:
    basename = os.path.basename(filename).split('.')[0]
    print(basename)
    if basename in filenames:
        #print(filename1)
        vsi_files.remove(filename)
print(len(vsi_files))
'''
vsi_files = [filename for filename in vsi_files if os.path.basename(filename).split('.')[0] not in filenames]
print(len(vsi_files))
'''
for filename in vsi_files:
    basename = os.path.basename(filename).split('.')[0]
    #print(basename)
    if any(basename in filename1.split('.')[0] for filename1 in converted):
        #print(filename1)
        vsi_files.remove(filename)

print(len(vsi_files))
'''
print(vsi_files)
# Define a function to run bfconvert command
def convert_file(vsi_file):
    #print(os.path.basename(vsi_file))
    output_file = os.path.join(output_folder, ('.').join((os.path.basename(vsi_file).split('.'))[:-1]) + '.ome.tiff')
    #print(output_file)
    command = [os.path.join(path_to_bftools, 'bfconvert'), '-series', '6', vsi_file, output_file]
    subprocess.run(command)

# Use a pool of 4 processes to run the conversion in parallel
#with Pool(36) as p:
    #list(tqdm(p.imap(convert_file, vsi_files), total=len(vsi_files)))

