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