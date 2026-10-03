#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-0:30
#SBATCH -J solx_b200plan
#SBATCH -o /scratch/t95317ha/solx/logs/%x_%j.out
#SBATCH -e /scratch/t95317ha/solx/logs/%x_%j.err
set -eo pipefail
source /scratch/t95317ha/solx/env.sh
cd $SOLX/poc
python -u b200_plan.py "$@"
