<!-- markdownlint-disable MD004 -->

# Installation

DataEval is a library that offers powerful metric classes and dataset analysis
functions using {term}`NumPy` and {term}`PyTorch<Torch (PyTorch)>` as the
primary backends.

## Supported Python Versions

We currently support python versions `3.9` - `3.12`

## Installing DataEval

Now that you have a chosen which DataEval to install, the following methods
will show you how to install using your preferred method.

:::::{tab-set}

::::{tab-item} pip
Installing from `pip`

```bash
pip install dataeval
```

DataEval also has an installable _extras_ `all` that provide access to
additional output formats and utility functions for analysis of your data.

```{list-table}
:header-rows: 1

* - Additional Functionality
  - Plot visualizations
* - Modules Installed
  - `matplotlib`
```

Installing from `pip` with extras

```bash
pip install dataeval[all]
```

::::

::::{tab-item} conda-forge
Installing from `conda`

```bash
conda install -c conda-forge dataeval
```

:::{note}

Conda installs all _extras_ automatically

:::

::::

::::{tab-item} source (poetry)

To install DataEval from source locally on Ubuntu, you will need git-lfs to
download larger, binary source files and uv for project dependency
management.

```bash
    sudo apt-get install git-lfs
    pip install poetry
```

Pull the source down and change to the DataEval project directory.

```bash
    git clone https://github.com/aria-ml/dataeval.git
    cd dataeval
```

Install DataEval with all extras

```bash
    poetry install --extras=all
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

Install DataEval with all extras for Python 3.X

```bash
    uv sync -p 3.X --extra=all
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
