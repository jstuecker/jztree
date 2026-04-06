#!/usr/bin/env bash

PYTHON_VERSIONS=("3.11" "3.12" "3.13" "3.14")
CUDA_ARCHS="all"
PACKAGE_DIR="packaging/jztree-cu12"
AUDITWHEEL_PLAT="manylinux_2_17_x86_64"
IMAGE_NAME="jztree-cu12-wheel-builder"
OUTPUT_DIR="packaging/docker-wheel-builder-cu12/output"
COPY_TO_PACKAGE_DIST=1
UV_VERSION="0.7.2"
MAMBA_ROOT_PREFIX="/opt/micromamba"
