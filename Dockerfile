ARG python_version=3.11
FROM python:${python_version}-slim as base

USER root
RUN apt-get update && apt-get install pandoc graphviz make -y
RUN pip install poetry

RUN addgroup --gid 1000 daml
RUN adduser  --gid 1000 --uid 1000 --disabled-password daml
USER daml
WORKDIR /daml

ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0

COPY --chown=daml:daml README.md      ./
COPY --chown=daml:daml pyproject.toml ./
COPY --chown=daml:daml poetry.lock    ./
RUN poetry install --no-root --with dev --all-extras

COPY --chown=daml:daml src/   src/
COPY --chown=daml:daml tests/ tests/
COPY --chown=daml:daml docs/  docs/


FROM base as daml_installed
RUN poetry install --only-root --all-extras


FROM base as deps
RUN poetry export --extras alibi-detect --extras cuda --without-hashes --without dev --format requirements.txt --output requirements.txt


FROM daml_installed as unit
RUN poetry run coverage run --source=daml --branch -m pytest --junitxml=junit.xml -v
RUN poetry run coverage report -m --skip-empty
RUN poetry run coverage html --skip-empty


FROM daml_installed as func
RUN poetry run coverage run --source=daml --branch -m pytest --runfunctional --junitxml=junit.xml -v
RUN poetry run coverage report -m --skip-empty
RUN poetry run coverage html --skip-empty


FROM daml_installed as type
RUN poetry run pyright src/ tests/
RUN poetry run pyright --ignoreexternal --verifytypes daml


FROM daml_installed as docs
COPY --chown=daml:daml .devcontainer/requirements_docs.txt ./
RUN poetry run pip install -r requirements_docs.txt
WORKDIR /daml/docs
RUN poetry run make html


FROM base as lint
RUN poetry run black --check --diff .
RUN poetry run flake8
RUN poetry run isort --check --diff .
RUN poetry run codespell
