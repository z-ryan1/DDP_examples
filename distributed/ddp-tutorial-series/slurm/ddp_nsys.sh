#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU DDP nsys profile — Tillicum (UW HPC)
#
# All GPUs are on a single node connected via NVLink 4.0 (900 GB/s).
# QoS: normal  (up to 16 GPUs, up to 24 hr)
#
# Do not submit this script directly — use the wrapper:
#   ./slurm/submit_ddp.sh [NGPUS]       e.g.  ./slurm/submit_ddp.sh 4
#
# The wrapper computes --cpus-per-task and --mem from NGPUS and passes them
# as sbatch command-line arguments, which override the defaults below.
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ddp-nsys
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # torchrun manages worker processes internally
#SBATCH --gres=gpu:4               # default — overridden by submit_ddp.sh
#SBATCH --cpus-per-task=32         # 8 CPUs per GPU × 4 — overridden by submit_ddp.sh
#SBATCH --mem=800G                 # 200 GB per GPU × 4 — overridden by submit_ddp.sh
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# ── Environment ───────────────────────────────────────────────────────────────
module load gcc/13.4.0
module load cuda/12.9.1
module load conda

conda activate ${CONDA_ENV:?Set CONDA_ENV to your environment name}

# NGPUS is exported by submit_ddp.sh; fall back to the #SBATCH default if
# someone runs sbatch directly
NGPUS=${NGPUS:-4}

# ── Output paths on GPFS ──────────────────────────────────────────────────────
PROFILE_DIR="/gpfs/scrubbed/${USER}/ddp_profiles"
mkdir -p "${PROFILE_DIR}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo "GPUs:        ${NGPUS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "nsys:        $(nsys --version 2>&1 | head -1)"
echo "PyTorch:     $(python -c 'import torch; print(torch.__version__)')"
echo "Profile dir: ${PROFILE_DIR}"

# ── Profile ───────────────────────────────────────────────────────────────────
# --trace-fork-before-exec=true   follow the worker processes torchrun spawns;
#                                  all ranks appear as separate rows in nsys-ui
# -t ...,nccl                     traces NCCL so AllReduce calls are visible
#                                  inside the 'backward' NVTX range
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

# ── Optional: re-run with large bucket to show suppressed overlap ─────────────
# Uncomment to produce a second profile where all gradients land in a single
# AllReduce bucket.  Compare with the default to see the overlap disappear.
#
# nsys profile \
#     --capture-range=cudaProfilerApi \
#     --capture-range-end=stop \
#     --force-overwrite=true \
#     --trace-fork-before-exec=true \
#     -t cuda,nvtx,osrt,cublas,cudnn \
#     -o "${PROFILE_DIR}/ddp_${NGPUS}gpu_nobucket_${SLURM_JOB_ID}" \
#     torchrun \
#         --nproc_per_node="${NGPUS}" \
#         --rdzv-backend=c10d \
#         --rdzv-endpoint=127.0.0.1:29500 \
#         "${SCRIPT_DIR}/multigpu_ddp_nsys.py" \
#             --total-epochs 5 \
#             --warmup-epochs 2 \
#             --batch-size 32 \
#             --num-workers 4 \
#             --bucket-cap-mb 2048
