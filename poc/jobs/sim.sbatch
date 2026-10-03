#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-4
#SBATCH -J solx_sim
#SBATCH -o /scratch/t95317ha/solx/logs/%x_%A_%a.out
#SBATCH -e /scratch/t95317ha/solx/logs/%x_%A_%a.err
# Usage: sbatch -a 0-71%16 jobs/sim.sbatch [config] [arch]
set -eo pipefail
CONFIG=${1:-SM90_H100}
ARCH=${2:-sm90}
source /scratch/t95317ha/solx/env.sh
module purge
module load libs/cuda/12.8.1
export CUDA_HOME=/opt/apps/libs/nvidia-cuda/toolkit/12.8.1
cd $SOLX/poc
VID=v$(printf %03d $SLURM_ARRAY_TASK_ID)
/usr/bin/time -v python simulate.py --vid $VID --config $CONFIG --arch $ARCH 2>&1 | grep -E "^\{|Elapsed|Maximum resident"
