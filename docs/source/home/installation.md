<!-- markdownlint-disable MD004 -->

# Installation

DataEval is a library that offers powerful metric classes and dataset analysis
functions using {term}`NumPy` and {term}`PyTorch<Torch (PyTorch)>` as the
primary backends.

## Supported Python Versions

We currently support python versions `3.9` - `3.13`

## Installing DataEval

Now that you have a chosen which DataEval to install, the following methods
will show you how to install using your preferred method.

:::::{tab-set}

::::{tab-item} pip
Installing from `pip`

```bash
pip install dataeval
```

::::

::::{tab-item} conda-forge
Installing from `conda`

```bash
conda install -c conda-forge dataeval
```

::::

::::{tab-item} source (poetry)

To install DataEval from source locally on Ubuntu using poetry, begin
by ensuring poetry is installed in your Python environment.

```bash
    sudo apt-get install git-lfs
    pip install poetry
```

Pull the source down and change to the DataEval project directory.

```bash
    git clone https://github.com/aria-ml/dataeval.git
    cd dataeval
```

Install DataEval

```bash
    poetry install
```

Now that DataEval is installed, you can run commands in the Poetry virtual
environment by prefixing shell commands with poetry run, or activate the
virtual environment directly in the shell.

```bash
    poetry env activate
```

::::

::::{tab-item} source (uv)

To install DataEval from source locally on Ubuntu, you will need git-lfs to
download larger, binary source files and uv for project dependency
management.

```bash
    sudo apt-get install git-lfs
    pip install uv
```

Pull the source down and change to the DataEval project directory.

```bash
    git clone https://github.com/aria-ml/dataeval.git
    cd dataeval
```

Install DataEval with development dependencies.

```bash
    uv sync
```

Optionally, you can specify the version of Python and PyTorch CPU/CUDA
support (cpu, cu118, cu124, cu128) using -p and --extra respectively.

For example, the following command installs DataEval in a Python 3.11
environment using only PyTorch with CPU support, and no development
dependencies:

```bash
    uv sync -p 3.11 --extra cpu --no-default-groups
```

Now that DataEval is installed, you can run commands in the uv virtual
environment by prefixing shell commands with uv run, or activate the
virtual environment directly in the shell.

```bash
    source .venv/bin/activate
```

::::

:::::

<!-- code languages for text found at https://pygments.org/languages/ -->
