#!/bin/bash --login
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 0-4
#SBATCH -J solx_tools
#SBATCH -o /scratch/t95317ha/solx/logs/%x_%j.out
#SBATCH -e /scratch/t95317ha/solx/logs/%x_%j.err
# GPU tool server for interactive design sessions (loop/designer.py), started by `round.py ... --interactive`.
# Needs QUEUE (directory under $SOLX/toolq) and REF (reference solution, relative to $SOLX) exported.
set -eo pipefail
SOLX=/scratch/t95317ha/solx
SIF=$SOLX/images/cuda-13.1.1-cudnn-devel-ubuntu24.04.sif
apptainer exec --nv -B /scratch,/net/scratch --env CUDA_HOME=/usr/local/cuda $SIF bash -c \
    "source $SOLX/env.sh && export PATH=/usr/local/cuda/bin:\$PATH && cd $SOLX && \
     python loop/toolserver.py --queue $QUEUE --ref $REF --problem \$PROBLEM038"
