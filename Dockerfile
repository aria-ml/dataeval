# syntax=docker/dockerfile:1.4

ARG USER="dataeval"
ARG UID="1000"
ARG HOME="/home/$USER"
ARG python_version="3.11"
ARG output_dir="/dataeval/output"


######################## data image ########################
FROM python:3.11 as data
RUN pip install --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu
WORKDIR /docs
COPY docs/data.py data.py
COPY src/dataeval/_internal/datasets.py dataeval/_internal/datasets.py
RUN python -c "\
from os import getcwd; \
from sys import path; \
path.append(getcwd()); \
import data; \
data.download(); \
"


######################## shared cuda image ########################
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as cuda
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends libgl1
ARG UID
ARG USER
RUN useradd -m -u ${UID} -s /bin/bash ${USER}
USER ${USER}
WORKDIR /${USER}
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV LANGUAGE=en
ENV LC_ALL=C.UTF-8
ENV LANG=en_US.UTF-8

FROM cuda as base
ARG UID
ARG python_version
RUN uv venv -p ${python_version}
COPY --chown=${UID} environment/requirements.txt environment/
RUN uv pip install -r environment/requirements.txt
COPY --chown=${UID} environment/requirements-dev.txt environment/
RUN uv pip install -r environment/requirements-dev.txt
ENV PATH=/${USER}/.venv/bin:${PATH}


######################## task layers ########################
# The *-run layers run individual tasks and capture the results
FROM base as task-run
ARG UID
RUN touch README.md
COPY --chown=${UID} pyproject.toml poetry.lock ./
COPY --chown=${UID} src/ src/
COPY --chown=${UID} tests/ tests/
COPY --chown=${UID} tox.ini ./
COPY --chown=${UID} capture.sh ./
ARG output_dir
RUN mkdir -p $output_dir
RUN mkdir -p .tox

FROM task-run as unit-run
ARG python_version
RUN ln -s /dataeval/.venv .tox/py$(echo ${python_version} | sed "s/\.//g")
RUN ./capture.sh unit ${python_version} tox -e py$(echo ${python_version} | sed "s/\.//g")

FROM task-run as type-run
ARG python_version
RUN ln -s /dataeval/.venv .tox/type-py$(echo ${python_version} | sed "s/\.//g")
RUN ./capture.sh type ${python_version} tox -e type-py$(echo ${python_version} | sed "s/\.//g")

FROM task-run as deps-run
ARG python_version
RUN ./capture.sh deps ${python_version} tox -e deps

FROM task-run as task-run-with-docs
ARG UID
COPY --chown=${UID} docs/ docs/
COPY --chown=${UID} *.md ./

FROM task-run-with-docs as lint-run
ARG python_version
RUN ln -s /dataeval/.venv .tox/lint
RUN ./capture.sh lint ${python_version} tox -e lint

FROM task-run-with-docs as doctest-run
ARG python_version
RUN ln -s /dataeval/.venv .tox/doctest
RUN ./capture.sh doctest ${python_version} tox -e doctest

# docs works differently than other tasks because it requires GPU access.
# The GPU requirement means that the docs image must be run as a container
# since there's no access to GPU during the build.
FROM base as task-docs
ARG UID
ARG HOME
COPY --chown=${UID} --link --from=data /docs docs
COPY --chown=${UID} pyproject.toml poetry.lock ./
COPY --chown=${UID} src/ src/
COPY --chown=${UID} docs/ docs/
COPY --chown=${UID} *.md ./
COPY --chown=${UID} tox.ini ./
COPY --chown=${UID} capture.sh ./
ARG output_dir
RUN mkdir -p $output_dir
RUN mkdir -p .tox

FROM task-docs as docs
RUN ln -s /dataeval/.venv .tox/docs
CMD tox -e docs

FROM task-docs as qdocs
RUN ln -s /dataeval/.venv .tox/qdocs
CMD tox -e qdocs


######################## results layers ########################
# These layers copy the results of the associated *-run layers into a scratch image in order to keep the created images as small as possible
FROM busybox as results
ARG output_dir
ENV output_dir=${output_dir}
CMD cat ${output_dir}/*.log && exit $(cat ${output_dir}/*-exitcode)

FROM results as unit
COPY --from=unit-run $output_dir $output_dir

FROM results as type
COPY --from=type-run $output_dir $output_dir

FROM results as lint
COPY --from=lint-run $output_dir $output_dir

FROM results as deps
COPY --from=deps-run $output_dir $output_dir

FROM results as doctest
COPY --from=doctest-run $output_dir $output_dir


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
