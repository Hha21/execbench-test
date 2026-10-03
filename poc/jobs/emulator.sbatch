#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 0-3
#SBATCH -J solx_emulator
#SBATCH -o /scratch/t95317ha/solx/logs/%x_%j.out
#SBATCH -e /scratch/t95317ha/solx/logs/%x_%j.err
# Usage: sbatch jobs/emulator.sbatch [extra emulator.py args]
set -eo pipefail
source /scratch/t95317ha/solx/env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd $SOLX/poc
python -u emulator.py "$@"
