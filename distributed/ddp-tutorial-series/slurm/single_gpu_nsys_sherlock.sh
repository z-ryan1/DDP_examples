#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Single-GPU nsys baseline — Sherlock (Stanford HPC), serc partition via Apptainer
#
# Submit:  sbatch slurm/single_gpu_nsys_sherlock.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=single-gpu-nsys
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
SIF="${SCRATCH}/pytorch-24.10.sif"
PROFILE_DIR="${SCRATCH}/ddp_profiles"
mkdir -p "${PROFILE_DIR}"

# ── Pull container once (idempotent) ──────────────────────────────────────────
if [ ! -f "${SIF}" ]; then
    echo "Pulling NGC PyTorch container to ${SIF}..."
    apptainer pull "${SIF}" docker://nvcr.io/nvidia/pytorch:24.10-py3
fi

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
apptainer exec --nv "${SIF}" python3 -c \
    'import torch; print(f"GPU:         {torch.cuda.get_device_name(0)}"); print(f"PyTorch:     {torch.__version__}")'
apptainer exec --nv "${SIF}" bash -c 'echo "nsys:        $(nsys --version 2>&1 | head -1)"'
echo "Profile dir: ${PROFILE_DIR}"

# ── Profile ───────────────────────────────────────────────────────────────────
apptainer exec --nv "${SIF}" \
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
