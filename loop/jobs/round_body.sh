#!/bin/bash
# Body of test_round.sbatch: correctness + timing through NVIDIA's harness, then the sm_100a compile capture.
# Runs either inside the CUDA 13.1 container or on the host with the CUDA 12.8 module; CUDA_HOME is set by the caller.
set -eo pipefail
source /scratch/t95317ha/solx/env.sh
export PATH=$CUDA_HOME/bin:$PATH
cd $SOLX
R=loop/rounds/$ROUND
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "nvcc: $(nvcc --version | tail -1)"
[ -n "$STATIC_ONLY" ] || python poc/run_timing.py --problem $PROBLEM038 --variants $R/candidates --pattern "${PATTERN:-*.json}" --out $R/results/timing --replicates 0
GPU=$(python -c "import torch; n = torch.cuda.get_device_name(0); print(next((t for t in ('B200', 'H200', 'H100', 'A100', 'L40S') if t in n), 'GPU'))")
python loop/static_any.py --candidates $R/candidates --problem $PROBLEM038 --out $R/results/static_$GPU.jsonl
echo ROUND_DONE
