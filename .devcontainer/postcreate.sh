#!/bin/sh

echo "\033[1;31mPlease wait for this script to complete before allowing vscode to Reload Window!\033[0m"
echo
echo "Installing daml for Python ${versions}..."
if [ ! -f ~/.cargo/bin/uv ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
echo ${versions} | tr ' ' '\n' > .python-version
echo -n ${versions} | xargs -n1 sh -c '~/.cargo/bin/uv pip install --python=$(which ${PYENV_ROOT}/versions/$0.*/bin/python) --index-strategy unsafe-best-match -r environment/requirements.txt -r environment/requirements-dev.txt && echo "Finished installing for Python $0."'
echo
echo '\033[1;37mIt is now safe to reload the window (\033[1;32mDeveloper: Reload Window\033[1;37m) and select a Python interpretor (\033[1;32mPython: Select Interpreter\033[1;37m) from the Command Palette.'
echo