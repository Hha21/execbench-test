#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 0-2
#SBATCH -J solx_buildsim
#SBATCH -o /scratch/t95317ha/solx/logs/%x_%j.out
#SBATCH -e /scratch/t95317ha/solx/logs/%x_%j.err
set -eo pipefail
module purge
module load libs/cuda/12.8.1
export CUDA_INSTALL_PATH=$(dirname $(dirname $(which nvcc)))
cd /scratch/t95317ha/solx/flashgpusim
source setup_environment
make -j $SLURM_CPUS_PER_TASK
echo BUILD_OK
cd tutorials/vectorAdd && ./run.sh 2>&1 | tail -20
