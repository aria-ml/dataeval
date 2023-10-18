ARG USER="daml"
ARG HOME="/home/$USER"
ARG PYENV_ROOT="$HOME/.pyenv"
ARG CACHE="$HOME/.cache"
ARG versions="3.8 3.9 3.10 3.11"
ARG python_version="3.11"


# Install pyenv and the supported python versions
FROM python:3.11.6 as pyenv
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt install -y \
    git \
    graphviz \
    pandoc

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

# Install daml dependencies
FROM poetry as base
COPY --chown=daml:daml README.md      ./
COPY --chown=daml:daml pyproject.toml ./
COPY --chown=daml:daml poetry.lock    ./

ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV POETRY_VIRTUALENVS_CREATE=false
RUN --mount=type=cache,target=${CACHE},sharing=locked,uid=1000,gid=1000 \
    echo ${versions} | xargs -n1 -P 0 sh -c '${PYENV_ROOT}/versions/daml-$0/bin/poetry install --no-root --with dev --all-extras'


# Install daml
FROM base as daml_installed
COPY --chown=daml:daml src/ src/
RUN --mount=type=cache,target=${CACHE},sharing=locked,uid=1000,gid=1000 \
    echo ${versions} | xargs -n1 -P 0 sh -c '${PYENV_ROOT}/versions/daml-$0/bin/poetry install --only-root --all-extras'


# Set the python version for subsequent targets
FROM daml_installed as versioned
ARG python_version
ENV PATH ${PYENV_ROOT}/versions/daml-${python_version}/bin:$PATH


# Create list of shipped dependencies for dependency scanning
FROM versioned as deps
RUN poetry export --extras alibi-detect --extras cuda --without-hashes --without dev --format requirements.txt --output requirements.txt


# Copy tests dir
FROM versioned as tests_dir
COPY --chown=daml:daml tests/ tests/


# Run unit tests and create coverage reports
FROM tests_dir as unit
RUN coverage run --source=daml --branch -m pytest --junitxml=junit.xml -v
RUN coverage report -m --skip-empty
RUN coverage html --skip-empty


# Run functional tests and create coverage reports
FROM tests_dir as func
RUN coverage run --source=daml --branch -m pytest --runfunctional --junitxml=junit.xml -v
RUN coverage report -m --skip-empty
RUN coverage html --skip-empty


# Run typechecking
FROM tests_dir as type
RUN pyright src/ tests/
RUN pyright --ignoreexternal --verifytypes daml


# Copy docs dir
FROM tests_dir as docs_dir
COPY --chown=daml:daml docs/ docs/


# Build docs
FROM docs_dir as docs
COPY --chown=daml:daml .devcontainer/requirements_docs.txt ./
RUN --mount=type=cache,target=${CACHE},sharing=locked,uid=1000,gid=1000 \
    pip install -r requirements_docs.txt
WORKDIR /daml/docs
RUN make html


# Run linters
FROM docs_dir as lint
RUN black --check --diff .
RUN flake8
RUN isort --check --diff .
RUN codespell
