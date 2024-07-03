# syntax=docker/dockerfile:1.4

ARG USER="daml"
ARG UID="1000"
ARG HOME="/home/$USER"
ARG PYENV_ROOT="$HOME/.pyenv"
ARG UV_ROOT="$HOME/.cargo/bin"
ARG python_version="3.11"
ARG base_image="base"
ARG pybase_image="pybase"
ARG output_dir="/daml/output"

FROM ubuntu:22.04 as pyenv
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
ARG UID
ARG USER
RUN useradd -m -u ${UID} ${USER}
USER ${USER}
ARG HOME
WORKDIR ${HOME}
RUN curl https://pyenv.run | bash
ARG PYENV_ROOT
ENV PYENV_ROOT=${PYENV_ROOT}
ARG python_version
RUN ${PYENV_ROOT}/bin/pyenv install ${python_version}


FROM scratch as pybase
ARG PYENV_ROOT
COPY --from=pyenv ${PYENV_ROOT} ${PYENV_ROOT}


# Base image for build runs and devcontainers
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as cuda
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends curl libgl1
ARG UID
ARG USER
RUN useradd -m -u ${UID} -s /bin/bash ${USER}
USER ${USER}
WORKDIR /daml
ARG PYENV_ROOT
ENV PYENV_ROOT=${PYENV_ROOT}
ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV LANGUAGE=en
ENV LC_ALL=C.UTF-8
ENV LANG=en_US.UTF-8
RUN curl -LsSf https://astral.sh/uv/install.sh | sh


FROM ${pybase_image} as pybase_image
FROM cuda as base
ARG PYENV_ROOT
COPY --chown=${UID} --link --from=pybase_image ${PYENV_ROOT} ${PYENV_ROOT}
ARG python_version
ARG UID
COPY --chown=${UID} environment/requirements.txt environment/
ARG UV_ROOT
ARG python_version
ENV PATH ${UV_ROOT}:${PYENV_ROOT}/versions/${python_version}/bin:${PYENV_ROOT}/bin:$PATH
RUN uv venv --python=$(which ${PYENV_ROOT}/versions/${python_version}.*/bin/python) && \
    uv pip install -r environment/requirements.txt && \
    rm -rf .venv
RUN grep nvidia-cudnn-cu11 environment/requirements.txt | cut -d' ' -f1 | \
    xargs uv pip install --python=$(which ${PYENV_ROOT}/versions/${python_version}.*/bin/python) tox tox-uv
RUN ln -s ${PYENV_ROOT}/versions/$(${PYENV_ROOT}/bin/pyenv latest ${python_version}) ${PYENV_ROOT}/versions/${python_version}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${PYENV_ROOT}/versions/${python_version}/lib/python${python_version}/site-packages/nvidia/cudnn/lib


######################## Build task layers ########################
# The *-run layers run individual tasks and capture the results
FROM ${base_image} as task-run
ARG UID
RUN touch README.md
COPY --chown=${UID} pyproject.toml poetry.lock ./
COPY --chown=${UID} src/ src/
COPY --chown=${UID} tests/ tests/
COPY --chown=${UID} tox.ini ./
COPY --chown=${UID} environment/requirements-dev.txt environment/
COPY --chown=${UID} capture.sh ./
ARG output_dir
RUN mkdir -p $output_dir

FROM task-run as unit-run
ARG python_version
RUN ./capture.sh unit ${python_version} tox -e py$(echo ${python_version} | sed "s/\.//g")

FROM task-run as type-run
ARG python_version
RUN ./capture.sh type ${python_version} tox -e type-py$(echo ${python_version} | sed "s/\.//g")

FROM task-run as lint-run
ARG python_version
RUN ./capture.sh lint ${python_version} tox -e lint

FROM task-run as deps-run
ARG python_version
RUN ./capture.sh deps ${python_version} tox -e deps

# docs works differently than other tasks because it requires GPU access.
# The GPU requirement means that the docs image must be run as a container
# since there's no access to GPU during the build.
FROM task-run as docs
ARG UID
COPY --chown=${UID} docs/ docs/
COPY --chown=${UID} *.md ./
CMD tox -e docs

FROM docs as qdocs
CMD tox -e qdocs

FROM docs as docs-run
ARG python_version
RUN ./capture.sh doctest ${python_version} tox -e doctest

######################## Results layers ########################
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
COPY --from=docs-run $output_dir $output_dir

######################## Dev container layer ########################
FROM cuda as devcontainer
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
ARG USER
RUN usermod -a -G docker ${USER}
USER ${USER}
ARG PYENV_ROOT
ARG UV_ROOT
ENV PATH=${UV_ROOT}:${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
ARG UID
RUN mkdir ${HOME}/.cache
COPY --chown=${UID} --link --from=harbor.jatic.net/daml/main:pybase-3.8 ${PYENV_ROOT} ${PYENV_ROOT}
COPY --chown=${UID} --link --from=harbor.jatic.net/daml/main:pybase-3.9 ${PYENV_ROOT} ${PYENV_ROOT}
COPY --chown=${UID} --link --from=harbor.jatic.net/daml/main:pybase-3.10 ${PYENV_ROOT} ${PYENV_ROOT}
COPY --chown=${UID} --link --from=harbor.jatic.net/daml/main:pybase-3.11 ${PYENV_ROOT} ${PYENV_ROOT}
