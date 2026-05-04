#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Submission wrapper for single_gpu_nsys_sherlock.sh — Sherlock (Stanford HPC)
#
# Usage:
#   ./slurm/submit_single_gpu_sherlock.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "Submitting single-GPU baseline profile job..."
sbatch "${SCRIPT_DIR}/single_gpu_nsys_sherlock.sh"
