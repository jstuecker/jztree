# To create a container:
# docker run --gpus all --name leancj -it ubuntu:latest bash
# To restart it:
# docker start -ia leancj

FROM ubuntu:latest
WORKDIR /custom-jax
COPY . .
RUN apt-get update && apt-get install -y git curl g++ cmake
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN source $HOME/.local/bin/env
#RUN uv pip install -e . --no-build-isolation
RUN uv sync

# Miniforge installation instructions:
# curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
# bash Miniforge3-Linux-x86_64.sh -b
# rm Miniforge3-Linux-x86_64.sh
# eval "$(/root/miniforge3/bin/conda shell.bash hook)"
# conda init
# ... follow installation isntructions
# conda create --name cjtest -y
# conda activate cjtest
# conda install "jaxlib=0.7=*cuda12*" jax cuda-nvcc -c conda-forge
# conda install pip
# pip install .


# System CUDA installation:
# docker run --gpus all --name cjnv -it nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
# apt install nvidia-cuda-toolkit
# ...
# uv pip install --upgrade "jax[cuda13-local]"
# uv pip install .