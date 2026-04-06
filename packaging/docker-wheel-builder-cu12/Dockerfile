FROM quay.io/pypa/manylinux2014_x86_64

ARG UV_VERSION=0.7.2
ENV UV_INSTALL_DIR=/opt/uv
ENV PATH="/opt/uv:/usr/local/bin:$PATH"
ENV MAMBA_ROOT_PREFIX=/opt/micromamba

SHELL ["/bin/bash", "-lc"]

RUN curl -LsSf https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_VERSION="${UV_VERSION}" sh \
    && micromamba --version \
    && auditwheel --version \
    && patchelf --version

WORKDIR /workspace
COPY build-in-container.sh /builder/build-in-container.sh
RUN chmod +x /builder/build-in-container.sh
