import subprocess

# Replace "jizh551e@dgw.zih.tu-dresden.de:/path/to/directory" with the actual remote path to the directory
output = subprocess.check_output(["rsync", "-r", "--list-only", "jizh551e@dgw.zih.tu-dresden.de:/svm/cephnfs/ekfz_ai/data/other-wsi/IGR-OV-DX-IMGS/IGR_converted"])

# Split the output into lines and extract the filenames
lines = output.decode().splitlines()
filenames = [line.split()[-1].split('.')[0].split('/')[-1] for line in lines[1:]] # skip the first line, which is the total size of files

# Print the filenames
print(filenames)

