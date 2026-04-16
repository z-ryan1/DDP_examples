#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Nsight Systems profiling workflow — single GPU vs. DDP comparison
#
# NOTE: For Tillicum (UW HPC) or any SLURM cluster, use the sbatch scripts:
#
#   sbatch slurm/single_gpu_nsys.sh          # baseline
#   NGPUS=4 sbatch slurm/ddp_nsys.sh         # DDP (set NGPUS as needed)
#
# This script is for reference / local workstations with nsys in PATH.
# Prerequisites:
#   - NVIDIA Nsight Systems installed (nsys in PATH)
#   - PyTorch >= 2.0 with CUDA
#
# Output: .nsys-rep files under profiles/
# Viewer:  nsys-ui profiles/<name>.nsys-rep
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-2}          # override: NUM_GPUS=4 ./run_nsys_profile.sh
EPOCHS=${EPOCHS:-5}
WARMUP=${WARMUP:-2}              # epochs before capture window opens
BATCH=${BATCH:-32}               # per-GPU batch size

mkdir -p profiles

# ── Common nsys flags ─────────────────────────────────────────────────────────
# --capture-range=cudaProfilerApi   only record between cudaProfilerStart/Stop
#                                   (the warmup epochs are excluded)
# --capture-range-end=stop          stop tracing when cudaProfilerStop is called
# -t cuda,nvtx,...                  trace CUDA activity, NVTX ranges, OS runtime,
#                                   cuDNN, cuBLAS (add nccl for DDP)
# ──────────────────────────────────────────────────────────────────────────────

NSYS_COMMON=(
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --force-overwrite=true
)

# ── 1. Single GPU baseline ────────────────────────────────────────────────────
echo ">>> Profiling single-GPU baseline..."

nsys profile \
    "${NSYS_COMMON[@]}" \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -o profiles/single_gpu \
    python single_gpu_nsys.py \
        --total-epochs "${EPOCHS}" \
        --warmup-epochs "${WARMUP}" \
        --batch-size "${BATCH}"

echo ">>> Single-GPU profile written to profiles/single_gpu.nsys-rep"

# ── 2. Multi-GPU DDP ──────────────────────────────────────────────────────────
echo ">>> Profiling DDP on ${NUM_GPUS} GPUs..."

# --trace-fork-before-exec=true tells nsys to follow the worker processes that
# torchrun spawns.  All ranks appear as separate process rows in the timeline.
# Add --nccl to -t to see the AllReduce calls that overlap with backward.
nsys profile \
    "${NSYS_COMMON[@]}" \
    -t cuda,nvtx,osrt,nccl,cublas,cudnn \
    --trace-fork-before-exec=true \
    -o "profiles/ddp_${NUM_GPUS}gpu" \
    torchrun --nproc_per_node="${NUM_GPUS}" multigpu_ddp_nsys.py \
        --total-epochs "${EPOCHS}" \
        --warmup-epochs "${WARMUP}" \
        --batch-size "${BATCH}"

echo ">>> DDP profile written to profiles/ddp_${NUM_GPUS}gpu.nsys-rep"

# ── 3. Optional: DDP with a large bucket to suppress overlap ─────────────────
# Uncomment to generate a third profile where all gradients are bucketed into
# one large AllReduce.  Compare with the default to see how overlap disappears.
#
# nsys profile \
#     "${NSYS_COMMON[@]}" \
#     -t cuda,nvtx,osrt,nccl,cublas,cudnn \
#     --trace-fork-before-exec=true \
#     -o "profiles/ddp_${NUM_GPUS}gpu_nobucket" \
#     torchrun --nproc_per_node="${NUM_GPUS}" multigpu_ddp_nsys.py \
#         --total-epochs "${EPOCHS}" \
#         --warmup-epochs "${WARMUP}" \
#         --batch-size "${BATCH}" \
#         --bucket-cap-mb 2048

# ── What to look for in nsys-ui ──────────────────────────────────────────────
cat <<'EOF'

── Reading the profiles ──────────────────────────────────────────────────────

Open each .nsys-rep with:   nsys-ui profiles/<name>.nsys-rep

Single-GPU timeline (single_gpu.nsys-rep):
  NVTX row:  forward | loss | backward | optimizer  (one row per batch)
  CUDA row:  cuBLAS GEMMs for QKV projections, LayerNorm, softmax kernels

DDP timeline (ddp_<N>gpu.nsys-rep):
  One process block per rank; within each block:
  NVTX row:  same forward | loss | backward | optimizer bands
  NCCL row:  AllReduce ops — watch them start partway through 'backward'
             and run concurrently with gradient kernels from earlier layers.
             This overlap is DDP's gradient-bucketing at work.

Key comparisons:
  1. Wall-clock 'backward' duration:    single-GPU vs. DDP rank-0
     (DDP backward is longer in absolute time, but AllReduce is hidden inside it)
  2. GPU utilization (SM Active %):     higher sustained utilization with DDP
  3. 'optimizer' step:                  should be similar — no communication here
  4. Re-run with --bucket-cap-mb 2048:  AllReduce moves to *after* backward;
     the overlap disappears and end-to-end step time increases.

EOF
