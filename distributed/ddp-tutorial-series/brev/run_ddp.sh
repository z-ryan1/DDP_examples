#!/bin/bash
# Multi-GPU DDP nsys profile — Brev (direct, no SLURM)
#
# Run from the ddp-tutorial-series directory:
#   bash brev/run_ddp.sh [NGPUS]   e.g.  bash brev/run_ddp.sh 4

set -euo pipefail

NGPUS=${1:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE_DIR="${HOME}/ddp_profiles"
mkdir -p "${PROFILE_DIR}"

echo "GPUs:        ${NGPUS}"
echo "GPU:         $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "nsys:        $(nsys --version 2>&1 | head -1)"
echo "PyTorch:     $(python3 -c 'import torch; print(torch.__version__)')"
echo "Profile dir: ${PROFILE_DIR}"

rm -f "${SCRIPT_DIR}/snapshot_ddp.pt"

OUTPUT="${PROFILE_DIR}/ddp_${NGPUS}gpu_$(date +%Y%m%d_%H%M%S)"

nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --trace-fork-before-exec=true \
    -t cuda,nvtx,osrt,cublas,cudnn \
    -o "${OUTPUT}" \
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

echo "Profile written to: ${OUTPUT}.nsys-rep"
echo "Open with: nsys-ui ${OUTPUT}.nsys-rep"
