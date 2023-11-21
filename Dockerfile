ARG USER="daml"
ARG HOME="/home/$USER"
ARG PYENV_ROOT="$HOME/.pyenv"
ARG python_version="3.11"
ARG deps_image="pydeps"


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
ARG PYENV_ROOT
ENV PYENV_ROOT=${PYENV_ROOT}
ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_INSTALLER_MAX_WORKERS=10
RUN echo 'echo Installing python $1 and dependencies... \n\
${PYENV_ROOT}/bin/pyenv install $1 \n\
${PYENV_ROOT}/bin/pyenv virtualenv $1 daml-$1 \n\
${PYENV_ROOT}/versions/daml-$1/bin/pip install --no-cache-dir --disable-pip-version-check poetry \n\
${PYENV_ROOT}/versions/daml-$1/bin/poetry install --no-cache --no-root --with dev --all-extras \n\
' > install.sh && chmod +x install.sh
RUN touch README.md
COPY --chown=daml:daml pyproject.toml ./
COPY --chown=daml:daml poetry.lock    ./


FROM pybase as pyenv-3.8
RUN ./install.sh 3.8
FROM scratch as pydeps-3.8
ARG PYENV_ROOT
COPY --from=pyenv-3.8 ${PYENV_ROOT}/ ${PYENV_ROOT}/


FROM pybase as pyenv-3.9
RUN ./install.sh 3.9
FROM scratch as pydeps-3.9
ARG PYENV_ROOT
COPY --from=pyenv-3.9 ${PYENV_ROOT}/ ${PYENV_ROOT}/


FROM pybase as pyenv-3.10
RUN ./install.sh 3.10
FROM scratch as pydeps-3.10
ARG PYENV_ROOT
COPY --from=pyenv-3.10 ${PYENV_ROOT}/ ${PYENV_ROOT}/


FROM pybase as pyenv-3.11
RUN ./install.sh 3.11
FROM scratch as pydeps-3.11
ARG PYENV_ROOT
COPY --from=pyenv-3.11 ${PYENV_ROOT}/ ${PYENV_ROOT}/


FROM ${deps_image}-3.8 as pysrc-3.8
FROM ${deps_image}-3.9 as pysrc-3.9
FROM ${deps_image}-3.10 as pysrc-3.10
FROM ${deps_image}-3.11 as pysrc-3.11
FROM ${deps_image}-${python_version} as pysrc


# Base image for build runs and devcontainers
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        graphviz \
        libgl1 \
        pandoc
RUN useradd -m -u 1000 -s /bin/bash daml
USER daml
WORKDIR /daml
ARG PYENV_ROOT
ENV PYENV_ROOT=${PYENV_ROOT}
ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_INSTALLER_MAX_WORKERS=10
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV POETRY_HTTP_BASIC_JATIC_USERNAME=gitlab-ci-token


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
COPY --chown=daml:daml --from=pysrc-3.8  ${PYENV_ROOT}/ ${PYENV_ROOT}/
COPY --chown=daml:daml --from=pysrc-3.9  ${PYENV_ROOT}/ ${PYENV_ROOT}/
COPY --chown=daml:daml --from=pysrc-3.10 ${PYENV_ROOT}/ ${PYENV_ROOT}/
COPY --chown=daml:daml --from=pysrc-3.11 ${PYENV_ROOT}/ ${PYENV_ROOT}/
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc


FROM base as versioned
ARG PYENV_ROOT
ARG python_version
ENV PATH ${PYENV_ROOT}/versions/daml-${python_version}/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${PYENV_ROOT}/versions/daml-${python_version}/lib/python${python_version}/site-packages/nvidia/cudnn/lib
COPY --chown=daml:daml --from=pysrc ${PYENV_ROOT} ${PYENV_ROOT}
RUN touch README.md
COPY --chown=daml:daml pyproject.toml ./
COPY --chown=daml:daml poetry.lock    ./
COPY --chown=daml:daml run            ./
COPY --chown=daml:daml src/           src/
RUN poetry install --no-cache --only-root --all-extras


# Create list of shipped dependencies for dependency scanning
FROM versioned as deps
CMD ./run deps


# Copy tests dir
FROM versioned as tests_dir
COPY --chown=daml:daml .coveragerc ./
COPY --chown=daml:daml tests/ tests/


# Run unit tests and create coverage reports
FROM tests_dir as unit
CMD ./run unit


# Run functional tests and create coverage reports
FROM tests_dir as func
CMD ./run func


# Run typechecking
FROM tests_dir as type
CMD ./run type


# Copy docs dir
FROM tests_dir as docs_dir
COPY --chown=daml:daml docs/ docs/


# Build docs
FROM docs_dir as docs
RUN poetry install --no-cache --no-root --with dev --with docs --all-extras
ENV PYDEVD_DISABLE_FILE_VALIDATION 1
CMD ./run docs


# Run linters
FROM docs_dir as lint
CMD ./run lint
