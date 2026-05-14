#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU DDP nsys profile — Sherlock (Stanford HPC), serc partition via Apptainer
#
# Do not submit this script directly — use the wrapper:
#   ./slurm/submit_ddp_sherlock.sh [NGPUS]    e.g.  ./slurm/submit_ddp_sherlock.sh 4
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ddp-nsys
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4                   # default — overridden by submit_ddp_sherlock.sh
#SBATCH --cpus-per-task=32         # 8 CPUs per GPU × 4 — overridden by submit_ddp_sherlock.sh
#SBATCH --mem=256G                 # 64 GB per GPU × 4 — overridden by submit_ddp_sherlock.sh
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

NGPUS=${NGPUS:-4}
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
echo "GPUs:        ${NGPUS}"
apptainer exec --nv "${SIF}" python3 -c \
    'import torch; print(f"GPU:         {torch.cuda.get_device_name(0)}"); print(f"PyTorch:     {torch.__version__}")'
apptainer exec --nv "${SIF}" bash -c 'echo "nsys:        $(nsys --version 2>&1 | head -1)"'
echo "Profile dir: ${PROFILE_DIR}"

# ── Profile ───────────────────────────────────────────────────────────────────
# --trace-fork-before-exec=true   follow worker processes torchrun spawns;
#                                  all ranks appear as separate rows in nsys-ui
apptainer exec --nv "${SIF}" \
    nsys profile \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite=true \
        --trace-fork-before-exec=true \
        -t cuda,nvtx,osrt,cublas,cudnn \
        -o "${PROFILE_DIR}/ddp_${NGPUS}gpu_${SLURM_JOB_ID}" \
        torchrun \
            --nproc_per_node="${NGPUS}" \
            --rdzv-backend=c10d \
            --rdzv-endpoint=127.0.0.1:29500 \
            "${SCRIPT_DIR}/multigpu_ddp_nsys.py" \
                --total-epochs 5 \
                --warmup-epochs 2 \
                --batch-size 32 \
                --num-workers 4 \
                --d-model 1024 \
                --n-heads 16 \
                --n-layers 12 \
                --seq-len 256

echo "Profile written to: ${PROFILE_DIR}/ddp_${NGPUS}gpu_${SLURM_JOB_ID}.nsys-rep"
echo "Open with: nsys-ui ${PROFILE_DIR}/ddp_${NGPUS}gpu_${SLURM_JOB_ID}.nsys-rep"
