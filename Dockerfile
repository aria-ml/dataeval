ARG python_version=3.10
FROM python:${python_version} as base
RUN pip install poetry

RUN addgroup --gid 1000 daml
RUN adduser  --gid 1000 --uid 1000 --disabled-password daml
USER daml
WORKDIR /daml

COPY --chown=daml:daml pyproject.toml ./
RUN poetry install --no-root --all-extras

COPY --chown=daml:daml src/   src/
COPY --chown=daml:daml tests/ tests/


FROM base as daml_installed
COPY --chown=daml:daml README.md ./
COPY --chown=daml:daml .git/     .git/
RUN poetry install --only-root --all-extras


FROM daml_installed as unit
RUN poetry run coverage run --source=daml --branch -m pytest --junitxml=junit.xml -v
RUN poetry run coverage report -m --skip-empty


FROM daml_installed as type
RUN poetry run pyright src/ tests/
RUN poetry run pyright --ignoreexternal --verifytypes daml


FROM base as lint
RUN poetry run black --check --diff . 
RUN poetry run flake8
RUN poetry run isort --check --diff .
RUN poetry run codespell
