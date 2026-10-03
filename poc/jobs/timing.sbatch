#!/bin/bash --login
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 0-6
#SBATCH -J solx_timing
#SBATCH -o /scratch/t95317ha/solx/logs/%x_%j.out
#SBATCH -e /scratch/t95317ha/solx/logs/%x_%j.err
# Usage: sbatch -A <account> -p <gpuA|gpuL|gpuH_short> jobs/timing.sbatch
set -eo pipefail
source /scratch/t95317ha/solx/env.sh
cd $SOLX/poc
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,clocks.max.sm --format=csv
python run_timing.py --problem $PROBLEM038
