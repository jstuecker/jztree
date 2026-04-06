FROM quay.io/pypa/manylinux_2_28_x86_64

ARG UV_VERSION=0.7.2
ENV UV_INSTALL_DIR=/opt/uv
ENV PATH="/opt/uv:$PATH"

SHELL ["/bin/bash", "-lc"]

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_VERSION="${UV_VERSION}" sh \
    && auditwheel --version \
    && patchelf --version

WORKDIR /workspace
COPY build-in-container.sh /builder/build-in-container.sh
RUN chmod +x /builder/build-in-container.sh
