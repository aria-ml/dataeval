ARG python_version=3.11
FROM python:${python_version} as base

USER root
RUN apt-get update && apt-get install pandoc -y
RUN pip install poetry

RUN addgroup --gid 1000 daml
RUN adduser  --gid 1000 --uid 1000 --disabled-password daml
USER daml
WORKDIR /daml

ENV POETRY_DYNAMIC_VERSIONING_BYPASS=true

COPY --chown=daml:daml pyproject.toml ./
COPY --chown=daml:daml .devcontainer/requirements.txt ./
RUN poetry run pip install -r requirements.txt

COPY --chown=daml:daml src/   src/
COPY --chown=daml:daml tests/ tests/
COPY --chown=daml:daml docs/  docs/


FROM base as daml_installed
COPY --chown=daml:daml README.md ./
RUN poetry install --only-root --all-extras


FROM daml_installed as unit
RUN poetry run coverage run --source=daml --branch -m pytest --junitxml=junit.xml -v
RUN poetry run coverage report -m --skip-empty
RUN poetry run coverage html --skip-empty


FROM daml_installed as func
RUN poetry run coverage run --source=daml --branch -m pytest --runfunctional --junitxml=junit.xml -v
RUN poetry run coverage report -m --skip-empty


FROM daml_installed as type
RUN poetry run pyright src/ tests/
RUN poetry run pyright --ignoreexternal --verifytypes daml


FROM daml_installed as docs
WORKDIR /daml/docs
RUN poetry run make html


FROM base as lint
RUN poetry run black --check --diff .
RUN poetry run flake8
RUN poetry run isort --check --diff .
RUN poetry run codespell
