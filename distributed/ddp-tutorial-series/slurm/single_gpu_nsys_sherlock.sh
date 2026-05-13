#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Single-GPU nsys baseline — Sherlock (Stanford HPC)
#
# Partition: gpu  (1 GPU, max 1 hr — sufficient for a profiling run)
# Submit:  sbatch slurm/single_gpu_nsys_sherlock.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=single-gpu-nsys
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=GPU_GEN:HPR
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

# ── Environment ───────────────────────────────────────────────────────────────
module load py-pytorch/2.4.1_py312
module load cuda/12.6.1            # nsys ships with the CUDA toolkit

# ── Output paths ──────────────────────────────────────────────────────────────
PROFILE_DIR="${SCRATCH}/ddp_profiles"
mkdir -p "${PROFILE_DIR}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "nsys:        $(nsys --version 2>&1 | head -1)"
echo "PyTorch:     $(python3 -c 'import torch; print(torch.__version__)')"
echo "Profile dir: ${PROFILE_DIR}"

# ── Profile ───────────────────────────────────────────────────────────────────
nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -o "${PROFILE_DIR}/single_gpu_${SLURM_JOB_ID}" \
    python3 "${SCRIPT_DIR}/single_gpu_nsys.py" \
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
