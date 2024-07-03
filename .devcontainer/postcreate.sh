#!/bin/bash

echo -e "\033[1;31mPlease wait for this script to complete before allowing vscode to Reload Window!\033[0m"
echo
rm -f .python-version
available=$( pyenv versions --bare | tr ' ' '\n' | xargs -n1 echo | cut -d '.' -f 1,2 | nl | sort -u -k2 | sort -n -r | cut -f2- | xargs echo -n )
read -p "Enter space delimited desired versions of python (default: $available): " input
if [[ ! $input ]]; then
    input=$available
fi
for token in ${input}; do
    token=$(echo $token | cut -d '.' -f 1,2)
    if ( ! pyenv latest $token > /dev/null 2>&1 ); then
        versions=$( pyenv versions --bare | tail -n1 | xargs echo -n | cut -d '.' -f 1,2 )
        echo "Unrecognized input - defaulting to $versions..."
        break
    else
        versions="$versions $( echo $token | cut -d '.' -f 1,2 )"
    fi
done
versions=$( echo $versions | tr ' ' '\n' | nl | sort -u -k2 | sort -n -r | cut -f2- )
if [ -z "$versions" ]; then
    echo "Unable to find an installable pyenv installed python version. Please rebuild your development environment."
    exit 1
fi
if [ ! -f ~/.cargo/bin/uv ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
echo ${versions} | tr ' ' '\n' > .python-version
echo -n ${versions} | xargs -n1 bash -c '\
    echo "Installing in python $0..." && \
    export UV_PYTHON=$(which ${PYENV_ROOT}/versions/$0.*/bin/python) && \
    uv pip install tensorflow-datasets && \
    uv pip install \
        -e . \
        -r environment/requirements.txt \
        -r environment/requirements-dev.txt && \
    echo "Finished installing $0."'
echo
echo -e '\033[1;37mIt is now safe to reload the window (\033[1;32mDeveloper: Reload Window\033[1;37m) and select a Python interpretor (\033[1;32mPython: Select Interpreter\033[1;37m) from the Command Palette.'
echo
