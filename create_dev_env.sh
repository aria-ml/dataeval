#!/bin/bash
echo "Please wait, this will take some time..."

(trap 'kill 0' SIGINT; tox devenv -e dev-py38 .venv-dev-py38 & tox devenv -e dev-py39 .venv-dev-py39 & tox devenv -e dev-py310 .venv-dev-py310)

declare -A py; py[38]=3.8; py[39]=3.9; py[310]=3.10
for i in "${!py[@]}"
do
    echo "Creating symlinks for Python ${py[$i]}..."
    if [ ! -d .tox/py$i ]; then mkdir -p .tox/py$i; else rm -rf .tox/py$i/bin; fi
    ln -sT $PWD/.venv-dev-py$i/bin $PWD/.tox/py$i/bin
    if [ ! -d .tox/py$i/lib/python${py[$i]} ]; then mkdir -p .tox/py$i/lib/python${py[$i]}; else rm -rf .tox/py$i/lib/python${py[$i]}/site-packages; fi
    ln -sT $PWD/.venv-dev-py$i/lib/python${py[$i]}/site-packages $PWD/.tox/py$i/lib/python${py[$i]}/site-packages
done

echo "Initializing pyright..."
source .venv-dev-py310/bin/activate
pip install pyright
pyright

echo "All done!"
