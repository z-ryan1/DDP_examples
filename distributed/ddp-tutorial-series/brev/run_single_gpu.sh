#!/bin/bash
# Single-GPU nsys baseline — Brev (direct, no SLURM)
#
# Run from the ddp-tutorial-series directory:
#   bash brev/run_single_gpu.sh

set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
    echo "Installing nsys..."
    apt-get update -qq && apt-get install -y --no-install-recommends cuda-nsight-systems-12-1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE_DIR="${HOME}/ddp_profiles"
mkdir -p "${PROFILE_DIR}"

echo "GPU:         $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "nsys:        $(nsys --version 2>&1 | head -1)"
echo "PyTorch:     $(python3 -c 'import torch; print(torch.__version__)')"
echo "Profile dir: ${PROFILE_DIR}"

rm -f "${SCRIPT_DIR}/snapshot_single.pt"

OUTPUT="${PROFILE_DIR}/single_gpu_$(date +%Y%m%d_%H%M%S)"

nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -o "${OUTPUT}" \
    python3 "${SCRIPT_DIR}/single_gpu_nsys.py" \
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
