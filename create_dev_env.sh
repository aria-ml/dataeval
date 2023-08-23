#!/bin/bash
echo "Please wait, this will take some time..."

tox devenv -e dev-py38 .venv-dev-py38
tox devenv -e dev-py39 .venv-dev-py39
tox devenv -e dev-py310 .venv-dev-py310
