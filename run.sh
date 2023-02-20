#!/bin/sh
set -eux
singularity run -W workspace mlcontext/e2e_container run_wsi_norm.sh