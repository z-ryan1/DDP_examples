#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU DDP nsys profile — Sherlock (Stanford HPC)
#
# Do not submit this script directly — use the wrapper:
#   ./slurm/submit_ddp_sherlock.sh [NGPUS]    e.g.  ./slurm/submit_ddp_sherlock.sh 4
#
# The wrapper computes --cpus-per-task and --mem from NGPUS and passes them
# as sbatch command-line arguments, which override the defaults below.
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ddp-nsys
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4                   # default — overridden by submit_ddp_sherlock.sh
#SBATCH --constraint=GPU_GEN:HPR
#SBATCH --cpus-per-task=32         # 8 CPUs per GPU × 4 — overridden by submit_ddp_sherlock.sh
#SBATCH --mem=256G                 # 64 GB per GPU × 4 — overridden by submit_ddp_sherlock.sh
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# ── Environment ───────────────────────────────────────────────────────────────
module load py-pytorch/2.4.1_py312
module load cuda/12.6.1            # nsys ships with the CUDA toolkit

NGPUS=${NGPUS:-4}

# ── Output paths ──────────────────────────────────────────────────────────────
PROFILE_DIR="${SCRATCH}/ddp_profiles"
mkdir -p "${PROFILE_DIR}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo "GPUs:        ${NGPUS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "nsys:        $(nsys --version 2>&1 | head -1)"
echo "PyTorch:     $(python3 -c 'import torch; print(torch.__version__)')"
echo "Profile dir: ${PROFILE_DIR}"

# ── Profile ───────────────────────────────────────────────────────────────────
# --trace-fork-before-exec=true   follow the worker processes torchrun spawns;
#                                  all ranks appear as separate rows in nsys-ui
# NCCL is not a valid -t flag on nsys 2025+; AllReduce is still visible as
# ncclKernel_* CUDA kernels on the default CUDA trace
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
