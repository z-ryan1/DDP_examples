#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Single-GPU nsys baseline — Tillicum (UW HPC)
#
# QoS: debug  (max 1 GPU, max 1 hr — sufficient for a profiling run)
# Submit:  sbatch slurm/single_gpu_nsys.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=single-gpu-nsys
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8          # Tillicum allocates 8 CPUs per GPU
#SBATCH --mem=200G                 # ~200 GB available per GPU on H200 nodes
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

# ── Environment ───────────────────────────────────────────────────────────────
module load gcc/13.4.0
module load cuda/12.9.1            # nsys ships with the CUDA toolkit
module load conda

conda activate ${CONDA_ENV:?Set CONDA_ENV to your environment name}

# ── Output paths on GPFS ──────────────────────────────────────────────────────
PROFILE_DIR="/gpfs/scrubbed/${USER}/ddp_profiles"
mkdir -p "${PROFILE_DIR}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "nsys:        $(nsys --version 2>&1 | head -1)"
echo "PyTorch:     $(python -c 'import torch; print(torch.__version__)')"
echo "Profile dir: ${PROFILE_DIR}"

# ── Profile ───────────────────────────────────────────────────────────────────
# --capture-range=cudaProfilerApi   only record between cudaProfilerStart/Stop
#                                   (warmup epochs are excluded automatically)
# --capture-range-end=stop          stop tracing at cudaProfilerStop
# -t cuda,nvtx,...                  trace CUDA kernels, NVTX ranges, OS runtime,
#                                   cuDNN, and cuBLAS
nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -o "${PROFILE_DIR}/single_gpu_${SLURM_JOB_ID}" \
    python "${SCRIPT_DIR}/single_gpu_nsys.py" \
        --total-epochs 5 \
        --warmup-epochs 2 \
        --batch-size 32 \
        --num-workers 4 \
        --d-model 1024 \
        --n-heads 16 \
        --n-layers 12 \
        --seq-len 256

echo "Profile written to: ${PROFILE_DIR}/single_gpu_${SLURM_JOB_ID}.nsys-rep"
echo "Open with: nsys-ui ${PROFILE_DIR}/single_gpu_${SLURM_JOB_ID}.nsys-rep"
