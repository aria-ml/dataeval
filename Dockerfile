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
ARG UV_ROOT
ENV VIRTUALENV_SYSTEM_SITE_PACKAGES=1
ENV PATH ${UV_ROOT}:${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:$PATH
ARG python_version
RUN pyenv global ${python_version}
ARG UID
COPY --chown=${UID} environment/requirements.txt environment/
RUN uv pip install --system -r environment/requirements.txt
COPY --chown=${UID} environment/requirements-dev.txt environment/
RUN uv pip install --system -r environment/requirements-dev.txt
RUN uv pip install --system tox tox-uv
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${PYENV_ROOT}/versions/${python_version}/lib/python${python_version}/site-packages/nvidia/cudnn/lib


######################## Build task layers ########################
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

FROM task-run as unit
ARG python_version
RUN ./capture.sh unit ${python_version} python -m tox -e py$(echo ${python_version} | sed "s/\.//g")

FROM task-run as type
ARG python_version
RUN ./capture.sh type ${python_version} python -m tox -e type-py$(echo ${python_version} | sed "s/\.//g")

FROM task-run as lint
ARG python_version
RUN ./capture.sh lint ${python_version} python -m tox -e lint

FROM task-run as deps
ARG python_version
RUN ./capture.sh deps ${python_version} python -m tox -e deps

# docs works differently than other tasks because it requires GPU access.
# The GPU requirement means that the docs image must be run as a container
# since there's no access to GPU during the build.
FROM task-run as docs
ARG UID
COPY --chown=${UID} docs/ docs/
COPY --chown=${UID} *.md ./
CMD python -m tox -e docs

FROM docs as qdocs
CMD python -m tox -e qdocs

FROM docs as docs-run
ARG python_version
RUN ./capture.sh doctest ${python_version} python -m tox -e doctest

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
COPY --chown=${UID} --link --from=harbor.jatic.net/daml/main:pybase-3.9 ${PYENV_ROOT} ${PYENV_ROOT}
COPY --chown=${UID} --link --from=harbor.jatic.net/daml/main:pybase-3.10 ${PYENV_ROOT} ${PYENV_ROOT}
COPY --chown=${UID} --link --from=harbor.jatic.net/daml/main:pybase-3.11 ${PYENV_ROOT} ${PYENV_ROOT}
