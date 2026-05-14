#!/bin/bash
# Quick sanity check — tiny model, no nsys.
# Run from the ddp-tutorial-series directory:
#   bash brev/smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "===== Environment ====="
echo "PyTorch:  $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA:     $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "GPU:      $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "GPUs:     $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "nsys:     $(nsys --version 2>&1 | head -1)"
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

echo "===== Test 1: single_gpu_nsys.py ====="
python3 "${SCRIPT_DIR}/single_gpu_nsys.py" "${TINY_ARGS[@]}"
echo "PASSED"
echo ""

echo "===== Test 2: multigpu_ddp_nsys.py (1 GPU via torchrun) ====="
torchrun \
    --nproc_per_node=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=127.0.0.1:29500 \
    "${SCRIPT_DIR}/multigpu_ddp_nsys.py" "${TINY_ARGS[@]}"
echo "PASSED"
echo ""

echo "===== Smoke test complete — ready for profiling runs ====="
