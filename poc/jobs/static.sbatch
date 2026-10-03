#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-2
#SBATCH -J solx_static
#SBATCH -o /scratch/t95317ha/solx/logs/%x_%j.out
#SBATCH -e /scratch/t95317ha/solx/logs/%x_%j.err
set -eo pipefail
source /scratch/t95317ha/solx/env.sh
cd $SOLX/poc
python static_features.py
