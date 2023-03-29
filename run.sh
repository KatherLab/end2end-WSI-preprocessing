#!/bin/sh
set -eux
singularity run --nv mlcontext/e2e_container.sif run_wsi_norm.sh