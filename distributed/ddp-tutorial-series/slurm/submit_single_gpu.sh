#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Submission wrapper for single_gpu_nsys.sh — Tillicum (UW HPC)
#
# Creates the profile output directory before submitting so nsys has
# somewhere to write.
#
# Usage:
#   ./slurm/submit_single_gpu.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

mkdir -p "/gpfs/scrubbed/${USER}/ddp_profiles"

echo "Submitting single-GPU baseline profile job..."
sbatch "${SCRIPT_DIR}/single_gpu_nsys.sh"
