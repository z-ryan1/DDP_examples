#!/bin/bash
# Brev launchable setup — runs once when the instance starts.
# Brev clones the repo automatically; this script just prepares the environment.
#
# Base image (set in Brev UI when creating the launchable):
#   nvcr.io/nvidia/pytorch:24.10-py3   (PyTorch 2.5, CUDA 12.6, nsys pre-installed)

set -euo pipefail

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

mkdir -p "${HOME}/ddp_profiles"

WORKDIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Setup complete."
echo "  Working directory: ${WORKDIR}"
echo "  Profiles will be written to: ${HOME}/ddp_profiles"
echo ""
echo "Quick start:"
echo "  cd ${WORKDIR}"
echo "  bash brev/smoke_test.sh"
