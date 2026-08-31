#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU DDP nsys profile — Brev L40S 2-GPU instance
#
# Submit:  bash brev/run_ddp.sh [NGPUS]       e.g.  bash brev/run_ddp.sh 2
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

AVAILABLE_GPUS=$(python3 -c 'import torch; print(torch.cuda.device_count())')
NGPUS=${1:-${AVAILABLE_GPUS}}

if (( AVAILABLE_GPUS < 1 )); then
    echo "No CUDA GPUs visible."
    exit 1
fi

if (( NGPUS > AVAILABLE_GPUS )); then
    echo "Requested ${NGPUS} GPU(s), but only ${AVAILABLE_GPUS} visible."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE_DIR="${HOME}/ddp_profiles"
mkdir -p "${PROFILE_DIR}"

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "GPUs:        ${NGPUS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "nsys:        $(nsys --version 2>&1 | head -1)"
echo "PyTorch:     $(python3 -c 'import torch; print(torch.__version__)')"
echo "Profile dir: ${PROFILE_DIR}"

rm -f "${SCRIPT_DIR}/snapshot_ddp.pt"

OUTPUT="${PROFILE_DIR}/ddp_${NGPUS}gpu_$(date +%Y%m%d_%H%M%S)"

# ── Profile ──────────────────────────────────────────────────────────────────
nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --trace-fork-before-exec=true \
    -t cuda,nvtx,osrt,cublas,cudnn \
    -o "${OUTPUT}" \
    torchrun \
        --standalone \
        --nproc_per_node="${NGPUS}" \
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
