ARG USER="daml"
ARG HOME="/home/$USER"
ARG PYENV_ROOT="$HOME/.pyenv"
ARG CACHE="$HOME/.cache"
ARG versions="3.8 3.9 3.10 3.11"
ARG python_version="3.11"

# Install pyenv and the supported python versions
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as pyenv
USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && apt-get install -y --no-install-recommends \
    curl gnupg2 git git-lfs wget nodejs parallel libgl1 graphviz pandoc

# Set the local timezone
ENV DEBIAN_FRONTEND noninteractive
RUN tz=`(wget -qO - http://geoip.ubuntu.com/lookup | sed -n -e 's/.*<TimeZone>\(.*\)<\/TimeZone>.*/\1/p')` \
 && ln -fs /usr/share/zoneinfo/$tz /etc/localtime

# Install python compiler dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

RUN addgroup --gid 1000 daml
RUN adduser  --gid 1000 --uid 1000 --disabled-password daml
USER daml
WORKDIR /daml

ARG PYENV_ROOT
ENV PYENV_ROOT=${PYENV_ROOT}
RUN curl https://pyenv.run | bash
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
ENV PATH=${PYENV_ROOT}/bin:${PATH}
ENV PATH=${PYENV_ROOT}/shims:${PATH}
ARG versions
RUN pyenv install ${versions}
RUN echo ${versions} | xargs -n1 sh -c 'pyenv virtualenv $0 daml-$0'

# Install poetry
FROM pyenv as poetry
ARG CACHE
RUN --mount=type=cache,target=${CACHE},sharing=locked,uid=1000,gid=1000 \
    echo ${versions} | xargs -n1 -P 0 sh -c '${PYENV_ROOT}/versions/daml-$0/bin/pip install poetry'

# Install daml dependencies and bootstrap pyright nodeenv
FROM poetry as base
COPY --chown=daml:daml README.md      ./
COPY --chown=daml:daml pyproject.toml ./
COPY --chown=daml:daml poetry.lock    ./

ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV POETRY_VIRTUALENVS_CREATE=false
RUN --mount=type=cache,target=${CACHE},sharing=locked,uid=1000,gid=1000 \
    echo ${versions} | xargs -n1 sh -c '${PYENV_ROOT}/versions/daml-$0/bin/poetry install --no-root --with dev --all-extras'
RUN ${PYENV_ROOT}/versions/daml-3.11/bin/pyright --version
ENV TF_GPU_ALLOCATOR cuda_malloc_async


# Install daml
FROM base as daml_installed
COPY --chown=daml:daml src/ src/
RUN --mount=type=cache,target=${CACHE},sharing=locked,uid=1000,gid=1000 \
    echo ${versions} | xargs -n1 sh -c '${PYENV_ROOT}/versions/daml-$0/bin/poetry install --only-root --all-extras'


# Set the python version for subsequent targets
FROM daml_installed as versioned
ARG python_version
ENV PATH ${PYENV_ROOT}/versions/daml-${python_version}/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${python_version}/versions/daml-${python_version}/lib/python${python_version}/site-packages/nvidia/cudnn/lib
COPY --chown=daml:daml run ./


# Create list of shipped dependencies for dependency scanning
FROM versioned as deps
CMD ./run deps


# Copy tests dir
FROM versioned as tests_dir
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
RUN --mount=type=cache,target=${CACHE},sharing=locked,uid=1000,gid=1000 \
    ${PYENV_ROOT}/versions/daml-${python_version}/bin/poetry install --no-root --with docs --all-extras
ENV PYDEVD_DISABLE_FILE_VALIDATION 1
CMD ./run docs


# Run linters
FROM docs_dir as lint
CMD ./run lint
