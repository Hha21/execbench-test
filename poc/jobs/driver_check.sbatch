#!/bin/bash --login
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 0-0:10
#SBATCH -J solx_driver
#SBATCH -o /scratch/t95317ha/solx/logs/%x_%j.out
#SBATCH -e /scratch/t95317ha/solx/logs/%x_%j.err
hostname
nvidia-smi --query-gpu=name,driver_version,compute_cap,clocks.max.sm,clocks.max.mem,memory.total --format=csv
nvidia-smi | head -5
