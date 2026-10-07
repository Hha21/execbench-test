#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 0-2
#SBATCH -J solx_pull
#SBATCH -o /scratch/t95317ha/solx/logs/%x_%j.out
#SBATCH -e /scratch/t95317ha/solx/logs/%x_%j.err
# One-off: pull NVIDIA's CUDA 13.1 devel image as an Apptainer .sif for test_round.sbatch (CPU node, ~10 GB).
set -eo pipefail
export APPTAINER_CACHEDIR=/scratch/t95317ha/solx/cache/apptainer
export APPTAINER_TMPDIR=/scratch/t95317ha/solx/cache/apptainer_tmp
SIF=/scratch/t95317ha/solx/images/cuda-13.1.1-cudnn-devel-ubuntu24.04.sif
apptainer pull --force $SIF docker://nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04
ls -la $SIF
apptainer exec $SIF nvcc --version
echo PULL_DONE
