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
	tox d -e docs .venv-docs -x testenv:docs.extras= &
	wait
)

# update docker created venv in vscode workspaces
echo "Updating workspaces virtual environments..."
cp -rn ~/.venv-* /workspaces/daml

# ensure symlinks and cuda/cudnn path variables for tox environments
declare -A dict; dict[py38]=3.8; dict[py39]=3.9; dict[py310]=3.10; dict[docs]=3.10;
for i in "${!dict[@]}"; do
	ensure_folder ".tox/${i}"
	rm -rf ".tox/${i}/bin"
	ln -st "${PWD}/.tox/${i}" "${PWD}/.venv-${i}/bin"

	ensure_folder ".tox/${i}/lib/python${dict[${i}]}"
	rm -rf ".tox/${i}/lib/python${dict[${i}]}/site-packages"
	ln -st "${PWD}/.tox/${i}/lib/python${dict[${i}]}" "${PWD}/.venv-${i}/lib/python${dict[${i}]}/site-packages"

	grep -qF 'export LD_LIBRARY_PATH' ".venv-${i}/bin/activate" || (
		echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> ".venv-${i}/bin/activate" &&
		echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$CUDNN_PATH/lib' >> ".venv-${i}/bin/activate"
	)
done

echo "All done!"
echo
echo -e "\033[1;33mPlease ensure your terminal is in a virtual environment."
