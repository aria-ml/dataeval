# syntax=docker/dockerfile:1.4

ARG USER="dataeval"
ARG UID="1000"
ARG python_version="3.11"
ARG output_dir="/dataeval/output"


######################## data image ########################
FROM python:3.11 as data
RUN pip install --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1+cpu \
    torchvision==0.20.1+cpu \
    requests
WORKDIR /docs
COPY docs/data.py data.py
COPY src/dataeval/utils/torch/datasets.py dataeval/utils/torch/datasets.py
RUN python -c "\
from os import getcwd; \
from sys import path; \
path.append(getcwd()); \
import data; \
data.download(); \
"


######################## shared base images ########################
FROM ubuntu:22.04 as ubuntu
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends clang
ARG UID
ARG USER
RUN useradd -m -u ${UID} -s /bin/bash ${USER}
USER ${USER}
WORKDIR /${USER}
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV UV_CACHE_DIR=/home/${USER}/.cache/uv

FROM ubuntu as base
ARG UID
ARG python_version
RUN uv venv -p ${python_version}
COPY --chown=${UID} environment/requirements.txt environment/
RUN uv pip install -r environment/requirements.txt
COPY --chown=${UID} environment/requirements-dev.txt environment/
RUN uv pip install -r environment/requirements-dev.txt
ENV PATH=/${USER}/.venv/bin:${PATH}

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 as cuda
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends libgl1 clang
ARG UID
ARG USER
RUN useradd -m -u ${UID} -s /bin/bash ${USER}
USER ${USER}
WORKDIR /${USER}
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV UV_CACHE_DIR=/home/${USER}/.cache/uv
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV LANGUAGE=en
ENV LC_ALL=C.UTF-8
ENV LANG=en_US.UTF-8

FROM cuda as base-docs
ARG UID
ARG python_version
RUN uv venv -p ${python_version}
COPY --chown=${UID} environment/requirements.txt environment/
RUN uv pip install -r environment/requirements.txt
COPY --chown=${UID} environment/requirements-dev.txt environment/
RUN uv pip install -r environment/requirements-dev.txt
ENV PATH=/${USER}/.venv/bin:${PATH}
COPY --chown=${UID} --link --from=data /docs docs


######################## task layers ########################
FROM base as task-run
ARG UID
RUN touch README.md
COPY --chown=${UID} pyproject.toml poetry.lock ./
COPY --chown=${UID} src/ src/
COPY --chown=${UID} tests/ tests/
COPY --chown=${UID} noxfile.py ./
ARG output_dir
RUN mkdir -p $output_dir
RUN mkdir -p .nox

FROM base-docs as task-run-for-docs
ARG UID
RUN touch README.md
COPY --chown=${UID} pyproject.toml poetry.lock ./
COPY --chown=${UID} src/ src/
COPY --chown=${UID} tests/ tests/
COPY --chown=${UID} noxfile.py ./
COPY --chown=${UID} docs/ docs/
COPY --chown=${UID} *.md ./
ARG output_dir
RUN mkdir -p $output_dir
RUN mkdir -p .nox

FROM task-run as task-run-with-docs
ARG UID
COPY --chown=${UID} docs/ docs/
COPY --chown=${UID} *.md ./

FROM task-run as unit
ARG python_version
ENV python_version=${python_version}
RUN ln -s /dataeval/.venv .nox/test-$(echo $python_version | tr . -)
CMD nox -r -e test-${python_version}

FROM task-run as type
ARG python_version
ENV python_version=${python_version}
RUN ln -s /dataeval/.venv .nox/type-$(echo $python_version | tr . -)
CMD nox -r -e type-${python_version}

FROM task-run as deps
ARG python_version
CMD nox -e deps

FROM task-run-with-docs as lint
ARG python_version
RUN ln -s /dataeval/.venv .nox/lint
CMD nox -r -e lint

FROM task-run-with-docs as doctest
ARG python_version
RUN ln -s /dataeval/.venv .nox/doctest
CMD nox -r -e doctest

FROM task-run-for-docs as docs
RUN ln -s /dataeval/.venv .nox/docs
CMD nox -r -e docs -- clean

FROM task-run-for-docs as qdocs
RUN ln -s /dataeval/.venv .nox/docs
CMD nox -r -e docs


######################## devcontainer image ########################
FROM cuda as devcontainer
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        git-lfs \
        gnupg2 \
        openssh-server \
        parallel
RUN addgroup --gid 1001 docker
ARG USER
RUN usermod -a -G docker ${USER}
USER ${USER}
