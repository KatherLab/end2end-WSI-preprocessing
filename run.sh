#!/bin/sh
set -eux

singularity run -W test mlcontext/e2e_container run_wsi_norm.sh