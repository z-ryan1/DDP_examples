#!/bin/bash
# Brev launchable setup — runs inside the container on first launch.
# Base image: nvcr.io/nvidia/pytorch:24.10-py3

set -euo pipefail

install_apt_packages() {
    if ! command -v apt-get >/dev/null 2>&1; then
        return
    fi

    APT=(apt-get)
    if [[ "$(id -u)" != "0" ]] && command -v sudo >/dev/null 2>&1; then
        APT=(sudo apt-get)
    fi

    "${APT[@]}" update -qq
    "${APT[@]}" install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3-pip \
        python3-venv \
        cuda-nsight-systems-12-1
}

if ! command -v nsys >/dev/null 2>&1; then
    install_apt_packages
fi

if ! python3 -c 'import torch' >/dev/null 2>&1; then
    if ! python3 -m pip --version >/dev/null 2>&1; then
        python3 -m ensurepip --upgrade || true
    fi

    if ! python3 -m pip --version >/dev/null 2>&1; then
        install_apt_packages
    fi

    if ! python3 -m pip --version >/dev/null 2>&1; then
        curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        python3 /tmp/get-pip.py
    fi

    python3 -m pip install --upgrade pip
    python3 -m pip install numpy
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

mkdir -p "${HOME}/ddp_profiles"

echo "Setup complete."
echo "  Profiles will be written to: ${HOME}/ddp_profiles"
echo ""
echo "Quick start:"
echo "  bash brev/smoke_test.sh"
