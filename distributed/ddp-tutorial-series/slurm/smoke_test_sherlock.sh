#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Smoke test — Sherlock (Stanford HPC), serc partition via Apptainer
#
# Runs both training scripts with a tiny model and dataset to confirm the
# container + GPU access work before submitting a full profiling job.
#
# Submit:  sbatch slurm/smoke_test_sherlock.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ddp-smoke
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
SIF="${SCRATCH}/pytorch-24.10.sif"

# ── Pull container once (idempotent) ──────────────────────────────────────────
if [ ! -f "${SIF}" ]; then
    echo "Pulling NGC PyTorch container to ${SIF}..."
    apptainer pull "${SIF}" docker://nvcr.io/nvidia/pytorch:24.10-py3
fi

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "===== Environment ====="
echo "Node:     ${SLURMD_NODENAME}"
apptainer exec --nv "${SIF}" python3 -c \
    'import torch; print(f"PyTorch:  {torch.__version__}"); print(f"CUDA:     {torch.version.cuda}"); print(f"GPU:      {torch.cuda.get_device_name(0)}")'
echo ""

rm -f "${SCRIPT_DIR}/snapshot_single.pt" "${SCRIPT_DIR}/snapshot_ddp.pt"

TINY_ARGS=(
    --total-epochs 2
    --warmup-epochs 1
    --batch-size 4
    --dataset-size 64
    --d-model 64
    --n-heads 4
    --n-layers 2
    --seq-len 32
    --num-workers 0
    --save-every 999
)

# ── Test 1: single GPU ────────────────────────────────────────────────────────
echo "===== Test 1: single_gpu_nsys.py ====="
apptainer exec --nv "${SIF}" \
    python3 "${SCRIPT_DIR}/single_gpu_nsys.py" "${TINY_ARGS[@]}"
echo "PASSED"
echo ""

# ── Test 2: DDP via torchrun (1 GPU — tests the full DDP code path) ──────────
echo "===== Test 2: multigpu_ddp_nsys.py (1 GPU via torchrun) ====="
apptainer exec --nv "${SIF}" \
    torchrun \
        --nproc_per_node=1 \
        --rdzv-backend=c10d \
        --rdzv-endpoint=127.0.0.1:29500 \
        "${SCRIPT_DIR}/multigpu_ddp_nsys.py" "${TINY_ARGS[@]}"
echo "PASSED"
echo ""

echo "===== Smoke test complete — ready for profiling runs ====="
