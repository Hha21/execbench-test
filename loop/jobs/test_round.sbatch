#!/bin/bash --login
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 0-2
#SBATCH -J solx_round
#SBATCH -o /scratch/t95317ha/solx/logs/%x_%j.out
#SBATCH -e /scratch/t95317ha/solx/logs/%x_%j.err
# Submitted by `loop/round.py test` with ROUND exported: correctness + timing through NVIDIA's harness, then the
# sm_100a compile capture. Usage by hand: sbatch -A <acct> -p <part> --export=ALL,ROUND=<round> loop/jobs/test_round.sbatch
set -eo pipefail
source /scratch/t95317ha/solx/env.sh
module purge; module load libs/cuda/12.8.1            # nvcc for CUDA C++ candidates (sm_100a needs >= 12.8)
export CUDA_HOME=/opt/apps/libs/nvidia-cuda/toolkit/12.8.1
cd $SOLX
R=loop/rounds/$ROUND
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python poc/run_timing.py --problem $PROBLEM038 --variants $R/candidates --pattern "*.json" --out $R/results/timing --replicates 0
GPU=$(python -c "import torch; n = torch.cuda.get_device_name(0); print(next((t for t in ('B200', 'H200', 'H100', 'A100', 'L40S') if t in n), 'GPU'))")
python loop/static_any.py --candidates $R/candidates --problem $PROBLEM038 --out $R/results/static_$GPU.jsonl
echo ROUND_DONE
