#!/usr/bin/env bash

PYTHON_VERSIONS=("3.11" "3.12" "3.13")
CUDA_ARCHS="all"
AUDITWHEEL_PLAT="manylinux_2_28_x86_64"
IMAGE_NAME="jztree-cu12-manylinux228-wheel-builder"
OUTPUT_DIR="packaging/output/cu12"
UV_VERSION="0.7.2"
