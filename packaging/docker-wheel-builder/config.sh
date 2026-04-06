#!/usr/bin/env bash

PYTHON_VERSIONS=("3.11" "3.12" "3.13" "3.14")
CUDA_ARCHS="all"
PACKAGE_DIR="packaging/jztree-cu13"
AUDITWHEEL_PLAT="manylinux_2_28_x86_64"
IMAGE_NAME="jztree-cu13-wheel-builder"
OUTPUT_DIR="packaging/docker-wheel-builder/output"
COPY_TO_PACKAGE_DIST=1
UV_VERSION="0.7.2"
