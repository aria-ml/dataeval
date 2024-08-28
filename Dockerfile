# syntax=docker/dockerfile:1.4

ARG USER="dataeval"
ARG UID="1000"
ARG HOME="/home/$USER"
ARG PYENV_ROOT="$HOME/.pyenv"
ARG python_version="3.11"
ARG output_dir="/dataeval/output"


######################## data image ########################
FROM python:3.11 as data
RUN pip install --no-cache \
    tensorflow-cpu==2.15.1 \
    tensorflow-datasets==4.9.3 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu
WORKDIR /docs
COPY docs/data.py data.py
RUN python -c "import data; data.download();"


######################## shared cuda image ########################
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as cuda
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends libgl1
ARG UID
ARG USER
RUN useradd -m -u ${UID} -s /bin/bash ${USER}
USER ${USER}
WORKDIR /dataeval
ARG PYENV_ROOT
ENV PYENV_ROOT=${PYENV_ROOT}
ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV LANGUAGE=en
ENV LC_ALL=C.UTF-8
ENV LANG=en_US.UTF-8


######################## base task image ########################
FROM ubuntu:22.04 as pyenv-build
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
ARG USER
RUN useradd -m ${USER}
USER ${USER}
RUN curl https://pyenv.run | bash
ENV PYTHON_CONFIGURE_OPTS "--enable-optimizations --with-lto"
ARG PYENV_ROOT
ARG python_version
RUN ${PYENV_ROOT}/bin/pyenv install ${python_version}

FROM cuda as base
ARG UID
ARG PYENV_ROOT
COPY --chown=${UID} --link --from=pyenv-build ${PYENV_ROOT} ${PYENV_ROOT}
ENV PATH ${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:$PATH
ARG python_version
RUN pyenv global ${python_version}
RUN pip install tox tox-uv uv --no-cache
RUN uv venv
COPY --chown=${UID} environment/requirements.txt environment/
RUN uv pip install -r environment/requirements.txt
COPY --chown=${UID} environment/requirements-dev.txt environment/
RUN uv pip install -r environment/requirements-dev.txt


######################## pyenv image ########################
FROM scratch as pyenv
ARG PYENV_ROOT
COPY --from=pyenv-build ${PYENV_ROOT} ${PYENV_ROOT}


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
RUN ln -s /dataeval/.venv .tox/lint
RUN ./capture.sh doctest ${python_version} tox -e doctest

# docs works differently than other tasks because it requires GPU access.
# The GPU requirement means that the docs image must be run as a container
# since there's no access to GPU during the build.
FROM base as task-docs
ARG UID
ARG HOME
COPY --chown=${UID} --link --from=data /root/tensorflow_datasets ${HOME}/tensorflow_datasets
COPY --chown=${UID} --link --from=data /root/.keras ${HOME}/.keras
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
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ARG PYENV_ROOT
ENV PATH=${HOME}/.cargo/bin:${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
ARG UID
COPY --chown=${UID} --link --from=harbor.jatic.net/dataeval/pyenv:3.9 ${PYENV_ROOT} ${PYENV_ROOT}
COPY --chown=${UID} --link --from=harbor.jatic.net/dataeval/pyenv:3.10 ${PYENV_ROOT} ${PYENV_ROOT}
COPY --chown=${UID} --link --from=harbor.jatic.net/dataeval/pyenv:3.11 ${PYENV_ROOT} ${PYENV_ROOT}
