#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Smoke test — Brev L40S 2-GPU instance
#
# Runs both training scripts with a tiny model and dataset before profiling.
#
# Submit:  bash brev/smoke_test.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not found. Run: bash .brev/setup.sh"
    exit 1
fi

if ! python3 -c 'import torch' >/dev/null 2>&1; then
    echo "PyTorch not found. Run: bash .brev/setup.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Sanity check ──────────────────────────────────────────────────────────────
python3 -c 'import torch; assert torch.cuda.device_count() >= 1, "No CUDA GPU visible"'

echo "===== Environment ====="
echo "PyTorch:  $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA:     $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "GPU:      $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "GPUs:     $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "nsys:     $(nsys --version 2>&1 | head -1)"
echo ""

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
python3 "${SCRIPT_DIR}/single_gpu_nsys.py" "${TINY_ARGS[@]}"
echo "PASSED"
echo ""

# ── Test 2: DDP via torchrun (1 GPU) ──────────────────────────────────────────
echo "===== Test 2: multigpu_ddp_nsys.py (1 GPU via torchrun) ====="
torchrun \
    --standalone \
    --nproc_per_node=1 \
    "${SCRIPT_DIR}/multigpu_ddp_nsys.py" "${TINY_ARGS[@]}"
echo "PASSED"
echo ""

echo "===== Smoke test complete — ready for profiling runs ====="
