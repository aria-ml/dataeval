# syntax=docker/dockerfile:1.4

ARG USER="dataeval"
ARG UID="1000"
ARG HOME="/home/$USER"
ARG python_version="3.11"
ARG output_dir="/dataeval/output"


######################## cuda image ########################
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS cuda
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        graphviz \
        clang \
        sudo
ARG UID
ARG USER
# Dev container tools expect non-root users to be able to sudo in a
# non-interactive context, so allow the user to use passwordless sudo.
RUN useradd -m -u ${UID} -s /bin/bash ${USER} -G sudo
RUN echo "${USER} ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/user-nopasswd
USER ${USER}
WORKDIR /${USER}
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV LANGUAGE=en
ENV LC_ALL=C.UTF-8
ENV LANG=en_US.UTF-8

######################## base image ########################
FROM cuda AS base
ARG UID
ARG python_version
ARG UV_CACHE_DIR=/home/${USER}/.cache/uv
RUN uv venv -p ${python_version}
COPY --chown=${UID} environment/requirements.txt environment/
RUN uv pip install -r environment/requirements.txt
COPY --chown=${UID} environment/requirements-dev.txt environment/
RUN uv pip install -r environment/requirements-dev.txt
ENV PATH=/${USER}/.venv/bin:${PATH}

######################## docs image ########################
FROM base AS docs
ARG UID
COPY --chown=${UID} docs/source/data.py docs/source/data.py
COPY --chown=${UID} src/dataeval/utils/datasets/*.py src/dataeval/utils/datasets/
ARG USER
RUN python -c "\
import os; \
import sys; \
sys.path.extend(['/${USER}/docs/source', '/${USER}/src']); \
import data; \
data.download(); \
"
COPY --chown=${UID} pyproject.toml poetry.lock ./
COPY --chown=${UID} src/ src/
COPY --chown=${UID} *.md ./
COPY --chown=${UID} noxfile.py ./
COPY --chown=${UID} docs/ docs/
ARG output_dir
RUN mkdir -p $output_dir
RUN mkdir -p .nox
RUN ln -s /dataeval/.venv .nox/docs
ENTRYPOINT ["nox", "-r", "-e", "docs", "--"]

######################## devcontainer ########################
FROM cuda AS devcontainer
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        git-lfs \
        gnupg2 \
        openssh-server \
        parallel \
        graphviz
RUN addgroup --gid 1001 docker
ARG USER
RUN usermod -a -G docker ${USER}
USER ${USER}
