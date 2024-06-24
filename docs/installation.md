# Installation Guide

DAML is a lightweight toolkit that offers powerful metric classes that can be extended through additional package installations.

This guide will show you how to install the DAML that fits your needs!

**Supported Python Versions**

We currently support python versions ``3.8 - 3.11``


## Base DAML Packages


To keep DAML lightweight but powerful, only the following metrics come with the base installation

:::{list-table}
:header-rows: 1

* - Packages
* - Bayes Error Rate
* - Divergence
* - Upper bound Average Precision
:::

## Extras

However, DAML also has installable *extras* that provide a more expansive and powerful toolkit for any user.  \
These extras are **torch**, **tensorflow** and **all**. Any extra can be installed using daml[*extra*]

:::{list-table}
:header-rows: 1

* - Extras
  - Additional Packages
* - torch
  - Sufficiency
* - tensorflow
  - OOD Detection
* - all
  - Sufficiency, OOD Detection
:::

## Installation

Now that you have a chosen which DAML to install, the following methods will show you how to install using your preferred method. \
Be sure to add [*extra*] if you are not installing the base DAML 

::::{tab-set}

:::{tab-item} pip
Installing from `pip` 
```python
    pip install daml[all]
```
:::

:::{tab-item} conda-forge
Installing from `conda`
```python
    conda install daml[all]
```
:::

:::{tab-item} source

To install DAML from source locally on Ubuntu, you will need git-lfs to download larger, binary source files and poetry for project dependency management.

```pycon
    sudo apt-get install git-lfs
    pip install poetry
```

Pull the source down and change to the DAML project directory.

```pycon
    git clone https://github.com/aria-ml/daml.git
    cd daml
```

Install DAML with optional dependencies for development.

```pycon
    poetry install --all-extras --with dev
```

Alternatively, you can install with optional dependencies used to generate documentation as well.

```pycon
    poetry install --all-extras --with dev --with docs
```

Now that DAML is installed, you can run commands in the poetry virtual environment by prefixing shell commands with poetry run, or activate the virtual environment directly in the shell.

```pycon
    poetry shell
```
:::

::::

<!-- code languages for text found at https://pygments.org/languages/ -->