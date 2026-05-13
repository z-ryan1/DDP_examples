#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Submission wrapper for ddp_nsys_sherlock.sh — Sherlock (Stanford HPC)
#
# Computes --cpus-per-task and --mem from the requested GPU count and passes
# them as sbatch command-line arguments (which override #SBATCH defaults).
#
# Usage:
#   ./slurm/submit_ddp_sherlock.sh [NGPUS]
#
# Examples:
#   ./slurm/submit_ddp_sherlock.sh        # 4 GPUs (default)
#   ./slurm/submit_ddp_sherlock.sh 2
#   ./slurm/submit_ddp_sherlock.sh 8
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

NGPUS=${1:-4}

# Sherlock GPU nodes: 8 CPUs and ~64 GB RAM per GPU
CPUS=$(( NGPUS * 8 ))
MEM=$(( NGPUS * 64 ))G

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "Submitting DDP profile job: ${NGPUS} GPU(s), ${CPUS} CPUs, ${MEM} RAM"

sbatch \
    --gpus="${NGPUS}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --export=ALL,NGPUS="${NGPUS}" \
    "${SCRIPT_DIR}/ddp_nsys_sherlock.sh"
