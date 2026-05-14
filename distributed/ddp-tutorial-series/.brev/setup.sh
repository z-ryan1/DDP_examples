#!/bin/bash
# Brev launchable setup — runs inside the container on first launch.
# Base image: Pytorch Devel (PyTorch 2.2, CUDA 12.1, cuDNN 8)

set -euo pipefail

apt-get update -qq
apt-get install -y --no-install-recommends cuda-nsight-systems-12-1

mkdir -p "${HOME}/ddp_profiles"

echo "Setup complete."
echo "  Profiles will be written to: ${HOME}/ddp_profiles"
echo ""
echo "Quick start:"
echo "  bash brev/smoke_test.sh"
