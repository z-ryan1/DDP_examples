#!/bin/bash
set -euo pipefail
# ──────────────────────────────────────────────────────────────────────────────
# Smoke test — Sherlock (Stanford HPC)
#
# Runs both training scripts with a tiny model and dataset to confirm the
# environment is correct before submitting a full profiling job.
# No nsys — just checks that the Python code executes without error.
#
# Partition: gpu  (1 GPU, 30 min)
# Submit:  sbatch slurm/smoke_test_sherlock.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ddp-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

# ── Environment ───────────────────────────────────────────────────────────────
module load py-pytorch/2.4.1_py312

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "===== Environment ====="
echo "Node:     ${SLURMD_NODENAME}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:     $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Remove stale snapshots — prevents shape mismatch if a previous run used a different model config
rm -f "${SCRIPT_DIR}/snapshot_single.pt" "${SCRIPT_DIR}/snapshot_ddp.pt"

# Tiny config — fast to run, exercises the full code path
TINY_ARGS=(
    --total-epochs 2
    --warmup-epochs 1
    --batch-size 4
    --dataset-size 64
    --d-model 64
    --n-heads 4
    --n-layers 2
    --seq-len 32
    --num-workers 0      # avoid DataLoader worker issues in smoke test
    --save-every 999     # suppress checkpoint writes
)

# ── Test 1: single GPU ────────────────────────────────────────────────────────
echo "===== Test 1: single_gpu_nsys.py ====="
python "${SCRIPT_DIR}/single_gpu_nsys.py" "${TINY_ARGS[@]}"
echo "PASSED"
echo ""

# ── Test 2: DDP via torchrun (still 1 GPU — tests the full DDP code path) ────
echo "===== Test 2: multigpu_ddp_nsys.py (1 GPU via torchrun) ====="
torchrun \
    --nproc_per_node=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=127.0.0.1:29500 \
    "${SCRIPT_DIR}/multigpu_ddp_nsys.py" "${TINY_ARGS[@]}"
echo "PASSED"
echo ""

echo "===== Smoke test complete — ready for profiling runs ====="
