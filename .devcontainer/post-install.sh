#!/bin/bash
virtualenv .venv-$VENV_SUFFIX
source .venv-$VENV_SUFFIX/bin/activate
pip install poetry tox
if ! grep -qxF "activate" ~/.bashrc
then
    echo "activate () {" >> ~/.bashrc
    echo "    . /workspaces/daml/.venv-$VENV_SUFFIX/bin/activate" >> ~/.bashrc
    echo "}" >> ~/.bashrc
    echo "activate" >> ~/.bashrc
fi
