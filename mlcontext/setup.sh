#!/bin/sh

REQUIRED_PKG="singularity"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  echo "Installing dependencies..."
  sudo apt-get update && sudo apt-get install -y     build-essential     uuid-dev     libgpgme-dev     squashfs-tools     libseccomp-dev     wget     pkg-config     git     cryptsetup-bin
  export VERSION=1.12 OS=linux ARCH=amd64 &&     wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz &&     sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz &&     rm go$VERSION.$OS-$ARCH.tar.gz
  wget https://github.com/apptainer/singularity/releases/download/v3.8.7/singularity-container_3.8.7_amd64.deb
  sudo dpkg -i singularity-container_3.8.7_amd64.deb
fi


sudo singularity build --sandbox e2e_container container.def