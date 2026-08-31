#!/bin/bash
# Brev launchable setup — runs inside the container on first launch.
# Base image: nvcr.io/nvidia/pytorch:24.10-py3

set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends cuda-nsight-systems-12-1
fi

mkdir -p "${HOME}/ddp_profiles"

echo "Setup complete."
echo "  Profiles will be written to: ${HOME}/ddp_profiles"
echo ""
echo "Quick start:"
echo "  bash brev/smoke_test.sh"
