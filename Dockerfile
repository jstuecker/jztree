FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /custom-jax
COPY . .
RUN apt-get update && apt-get install -y git curl vim g++ cmake
RUN curl -LsSf https://astral.sh/uv/install.sh | sh