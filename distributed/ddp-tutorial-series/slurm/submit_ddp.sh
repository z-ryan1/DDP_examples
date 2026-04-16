#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Submission wrapper for ddp_nsys.sh — Tillicum (UW HPC)
#
# Computes --cpus-per-task and --mem from the requested GPU count and passes
# them as sbatch command-line arguments (which override #SBATCH defaults).
#
# Usage:
#   ./slurm/submit_ddp.sh [NGPUS]
#
# Examples:
#   ./slurm/submit_ddp.sh        # 4 GPUs (default)
#   ./slurm/submit_ddp.sh 2
#   ./slurm/submit_ddp.sh 8      # full node
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

NGPUS=${1:-4}

# Tillicum allocates 8 CPUs and ~200 GB RAM per GPU
CPUS=$(( NGPUS * 8 ))
MEM=$(( NGPUS * 200 ))G

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "Submitting DDP profile job: ${NGPUS} GPU(s), ${CPUS} CPUs, ${MEM} RAM"

sbatch \
    --gres=gpu:"${NGPUS}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --export=ALL,NGPUS="${NGPUS}" \
    "${SCRIPT_DIR}/ddp_nsys.sh"
