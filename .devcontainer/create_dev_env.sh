#!/bin/bash

function ensure_folder() { if [[ ! -d $1 ]]; then mkdir -p "$1"; fi; }

# clear out stale tox and virtual environments
rm -rf .tox
rm -rf .venv-*

echo "Creating development environments in parallel..."
(
	trap 'kill 0' SIGINT
	tox d -e py38 .venv-py38 -x testenv.extras= &
	tox d -e py39 .venv-py39 -x testenv.extras= &
	tox d -e py310 .venv-py310 -x testenv.extras= &
	wait
)

# update docker created venv in vscode workspaces
echo "Updating workspaces virtual environments..."
cp -rn ~/.venv-* /workspaces/daml

# ensure symlinks and cuda/cudnn path variables for tox environments
declare -A py; py[38]=3.8; py[39]=3.9; py[310]=3.10
for i in "${!py[@]}"; do
	ensure_folder ".tox/py${i}"
	rm -rf ".tox/py${i}/bin"
	ln -st "${PWD}/.tox/py${i}" "${PWD}/.venv-py${i}/bin"

	ensure_folder ".tox/py${i}/lib/python${py[${i}]}"
	rm -rf ".tox/py${i}/lib/python${py[${i}]}/site-packages"
	ln -st "${PWD}/.tox/py${i}/lib/python${py[${i}]}" "${PWD}/.venv-py${i}/lib/python${py[${i}]}/site-packages"

	grep -qF 'export LD_LIBRARY_PATH' ".venv-py${i}/bin/activate" || (
		echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> ".venv-py${i}/bin/activate" &&
        echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib' >> ".venv-py${i}/bin/activate"
	)
done

echo "All done!"
echo
echo -e "\033[1;33mPlease ensure your terminal is in a virtual environment."
