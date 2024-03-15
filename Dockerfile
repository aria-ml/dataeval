# syntax=docker/dockerfile:1.4

ARG USER="daml"
ARG HOME="/home/$USER"
ARG PYENV_ROOT="$HOME/.pyenv"
ARG python_version="3.11"
ARG deps_image="pydeps"
ARG base_image="pybase"
ARG pyenv_enable_opt=""
ARG pyenv_with_lto=""


FROM ubuntu:22.04 as pybase
ENV DEBIAN_FRONTEND noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        libxmlsec1-dev \
        llvm \
        tk-dev \
        wget \
        xz-utils \
        zlib1g-dev
RUN useradd -m -u 1000 daml
USER daml
WORKDIR /home/daml
RUN curl https://pyenv.run | bash
# ENV PYTHON_CONFIGURE_OPTS '--enable-optimizations --with-lto'
# ENV PYTHON_CFLAGS '-march=native -mtune=native'
ARG pyenv_enable_opt
ARG pyenv_with_lto
ENV PYTHON_CONFIGURE_OPTS="${pyenv_enable_opt} ${pyenv_with_lto}"
ARG PYENV_ROOT
ENV PYENV_ROOT=${PYENV_ROOT}
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_INSTALLER_MAX_WORKERS=10
ARG python_version
RUN ${PYENV_ROOT}/bin/pyenv install ${python_version}


FROM ${base_image} as base_image
FROM base_image as pyenv
ARG PYENV_ROOT
ARG python_version
RUN ${PYENV_ROOT}/versions/${python_version}.*/bin/pip install --no-cache-dir --disable-pip-version-check poetry
RUN touch README.md
COPY --chown=daml:daml pyproject.toml poetry.lock ./
RUN ${PYENV_ROOT}/versions/${python_version}.*/bin/poetry install --no-cache --no-root --all-extras


FROM scratch as pydeps
ARG PYENV_ROOT
COPY --link --from=pyenv  ${PYENV_ROOT}/ ${PYENV_ROOT}/


# Base image for build runs and devcontainers
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        graphviz \
        libgl1
RUN useradd -m -u 1000 -s /bin/bash daml
USER daml
WORKDIR /daml
ARG PYENV_ROOT
ENV PYENV_ROOT=${PYENV_ROOT}
ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_INSTALLER_MAX_WORKERS=10
ENV TF_GPU_ALLOCATOR=cuda_malloc_async


FROM ${deps_image} as deps_image
FROM base as build
ARG PYENV_ROOT
ARG python_version
COPY --chown=daml:daml --link --from=deps_image ${PYENV_ROOT} ${PYENV_ROOT}
RUN ln -s ${PYENV_ROOT}/versions/$(${PYENV_ROOT}/bin/pyenv latest ${python_version}) ${PYENV_ROOT}/versions/${python_version}
ENV PATH ${PYENV_ROOT}/versions/${python_version}/bin:${PYENV_ROOT}/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${PYENV_ROOT}/versions/${python_version}/lib/python${python_version}/site-packages/nvidia/cudnn/lib


FROM build as versioned
RUN touch README.md
COPY --chown=daml:daml pyproject.toml poetry.lock ./
COPY --chown=daml:daml src/ src/
RUN poetry install --no-cache --all-extras --with dev --with docs


FROM versioned as run
COPY --chown=daml:daml .coveragerc ./
COPY --chown=daml:daml tests/ tests/
COPY --chown=daml:daml docs/ docs/
COPY --chown=daml:daml *.md ./
COPY --chown=daml:daml run ./
ENTRYPOINT [ "./run" ]


FROM base as devcontainer
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        gnupg2 \
        openssh-server \
        parallel
RUN addgroup --gid 1001 docker
RUN usermod -a -G docker daml
USER daml
ENV LANGUAGE=en
ENV LC_ALL=C.UTF-8
ENV LANG=en_US.UTF-8
ARG PYENV_ROOT
ENV PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
