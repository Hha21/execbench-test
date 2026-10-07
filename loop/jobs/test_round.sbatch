#!/bin/bash --login
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 0-2
#SBATCH -J solx_round
#SBATCH -o /scratch/t95317ha/solx/logs/%x_%j.out
#SBATCH -e /scratch/t95317ha/solx/logs/%x_%j.err
# Submitted by `loop/round.py test` with ROUND exported. Usage by hand:
#   sbatch -A <acct> -p <part> --export=ALL,ROUND=<round>[,PATTERN=<glob>][,CONTAINER=0][,STATIC_ONLY=1] loop/jobs/test_round.sbatch
# Default: run inside NVIDIA's CUDA 13.1 container (pulled by pull_cuda13.sbatch), because CSF3's newest module,
# CUDA 12.8, has a ptxas that rejects Blackwell's 256-bit loads (PTX ISA 8.8). CONTAINER=0 uses the host module.
set -eo pipefail
SOLX=/scratch/t95317ha/solx
SIF=$SOLX/images/cuda-13.1.1-cudnn-devel-ubuntu24.04.sif
if [ "${CONTAINER:-1}" != 0 ] && [ -f $SIF ]; then
    apptainer exec --nv -B /scratch,/net/scratch --env CUDA_HOME=/usr/local/cuda $SIF bash $SOLX/loop/jobs/round_body.sh
else
    module purge; module load libs/cuda/12.8.1
    CUDA_HOME=/opt/apps/libs/nvidia-cuda/toolkit/12.8.1 bash $SOLX/loop/jobs/round_body.sh
fi
