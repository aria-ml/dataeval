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


# Install poetry
FROM pyenv as poetry
ARG CACHE
RUN --mount=type=cache,target=${CACHE},sharing=locked,uid=1000,gid=1000 \
    for py in ${versions}; do \
    pyenv global ${py}; \
    pip install poetry; \
    done


# Install daml dependencies
FROM poetry as base
COPY --chown=daml:daml README.md      ./
COPY --chown=daml:daml pyproject.toml ./
COPY --chown=daml:daml poetry.lock    ./

ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0
ENV POETRY_VIRTUALENVS_CREATE=false
RUN --mount=type=cache,target=${CACHE},sharing=locked,uid=1000,gid=1000 \
    for py in ${versions}; do \
    pyenv global ${py}; \
    poetry install --no-root --with dev --all-extras; \
    done


# Install daml
FROM base as daml_installed
COPY --chown=daml:daml src/ src/
RUN --mount=type=cache,target=${CACHE},sharing=locked,uid=1000,gid=1000 \
    for py in ${versions}; do \
    pyenv global ${py}; \
    poetry install --only-root --all-extras; \
    done


# Set the python version for subsequent targets
FROM daml_installed as versioned
ARG python_version
RUN pyenv global ${python_version}


# Create list of shipped dependencies for dependency scanning
FROM versioned as deps
RUN poetry export --extras alibi-detect --extras cuda --without-hashes --without dev --format requirements.txt --output requirements.txt


# Copy tests dir
FROM versioned as tests_dir
COPY --chown=daml:daml tests/ tests/


# Run unit tests and create coverage reports
FROM tests_dir as unit
RUN poetry run coverage run --source=daml --branch -m pytest --junitxml=junit.xml -v
RUN poetry run coverage report -m --skip-empty
RUN poetry run coverage html --skip-empty


# Run functional tests and create coverage reports
FROM tests_dir as func
RUN poetry run coverage run --source=daml --branch -m pytest --runfunctional --junitxml=junit.xml -v
RUN poetry run coverage report -m --skip-empty
RUN poetry run coverage html --skip-empty


# Run typechecking
FROM tests_dir as type
RUN poetry run pyright src/ tests/
RUN poetry run pyright --ignoreexternal --verifytypes daml


# Copy docs dir
FROM tests_dir as docs_dir
COPY --chown=daml:daml docs/ docs/


# Build docs
FROM docs_dir as docs
COPY --chown=daml:daml .devcontainer/requirements_docs.txt ./
RUN --mount=type=cache,target=${CACHE},sharing=locked,uid=1000,gid=1000 \
    poetry run pip install -r requirements_docs.txt
WORKDIR /daml/docs
RUN poetry run make html


# Run linters
FROM docs_dir as lint
RUN poetry run black --check --diff .
RUN poetry run flake8
RUN poetry run isort --check --diff .
RUN poetry run codespell
