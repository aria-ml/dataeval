<!-- markdownlint-disable MD004 -->

# Installation

DataEval is a library that offers powerful metric classes and dataset analysis
functions using {term}`NumPy` and {term}`PyTorch<Torch (PyTorch)>` as the
primary backends.

## Supported Python Versions

We currently support Python `3.10` through `3.14`.

## PyTorch Dependency

DataEval requires PyTorch to be installed. By default, `pip install dataeval`
pulls PyTorch from PyPI, which bundles CUDA support on Linux and is a much
larger download than the CPU build.

To choose a specific PyTorch variant, install `torch` from that variant's wheel
index **first**, then install DataEval — it accepts the build already present in
the environment:

```bash
# 1. Pick your PyTorch build (cpu / cu118 / cu128)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Install DataEval
pip install dataeval
```

See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for all available PyTorch installation options.

:::{warning}
Do **not** reach for `--extra-index-url` to select a CUDA variant. It *adds* an
index rather than replacing PyPI, and pip then picks the highest version across
both. The CUDA indexes lag the latest PyTorch release, so PyPI usually wins and
you silently get the default CUDA-bundled build instead of the one you asked
for — the install succeeds and nothing warns you.

`--index-url` (as above) replaces the index outright, which is why it is
reliable.
:::

For a CPU-only install there is a one-line shortcut, because the CPU index does
track the latest release:

```bash
pip install dataeval --extra-index-url https://download.pytorch.org/whl/cpu
```

:::{important}
The `cpu`, `cu118`, and `cu128` extras are **not** a way to select a PyTorch
variant with pip. All three declare exactly the same requirements (`torch` and
`torchvision`); what distinguishes them is `[tool.uv.sources]` in the project's
`pyproject.toml`, which routes those two packages to the right wheel index. That
routing is project metadata — uv applies it when resolving **from a source
checkout**, and it is not part of the published wheel. Installing
`dataeval[cu128]` from PyPI therefore only adds `torchvision`, and PyTorch still
resolves from whichever index pip is pointed at.

Select the variant with `--index-url` under pip, `--torch-backend` under
`uv pip`, and use the extras only when installing from source.
:::

## Installing DataEval

Now that you have chosen which DataEval to install, the following methods
will show you how to install using your preferred method.

:::::{tab-set}

::::{tab-item} pip
Installing from `pip`

```bash
pip install dataeval
```

To control which PyTorch build you get, see
[PyTorch Dependency](#pytorch-dependency) above.

::::

::::{tab-item} uv
Installing from PyPI with `uv`, letting uv select the PyTorch index for you:

```bash
uv pip install dataeval --torch-backend cpu   # or cu118 / cu128 / auto
```

`--torch-backend` is the `uv pip` equivalent of `--index-url`; `auto` detects
the installed CUDA driver.

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

Poetry resolves `torch` from the CPU wheel index (see `[tool.poetry.source]` in
`pyproject.toml`), so the Poetry path installs the CPU build of PyTorch. Use
`uv` if you need a CUDA variant from source.

Now that DataEval is installed, you can run commands in the Poetry virtual
environment by prefixing shell commands with poetry run, or activate the
virtual environment directly in the shell.

```bash
    poetry env activate
```

::::

::::{tab-item} source (uv)

To install DataEval from source locally on Ubuntu, you will need
[uv](https://docs.astral.sh/uv/getting-started/installation/) for
Python environment management.

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
support (cpu, cu118, cu128) using -p and --extra respectively. This is the one
place the PyTorch variant extras take effect — uv applies the `[tool.uv.sources]`
index routing when resolving from a source checkout.

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
