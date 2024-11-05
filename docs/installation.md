# Installation Guide

{term}`DataEval` is a lightweight toolkit that offers powerful metric classes that can be extended through additional package installations.

This guide will show you how to install the DataEval that fits your needs!

**Supported Python Versions**

We currently support python versions ``3.9 - 3.11``


## Base DataEval Packages


To keep {term}`DataEval` lightweight but powerful, only the following metrics come with the base installation

:::{list-table}
:header-rows: 1

* - Modules
* - [Balance](concepts/Balance.md)
* - [Bayes Error Rate](concepts/BER.md)
* - [Clusterer](concepts/Clusterer.md)
* - [Coverage](concepts/Coverage.md)
* - [Divergence](concepts/Divergence.md)
* - [Diversity](concepts/Diversity.md)
* - [Duplicates](concepts/Duplicates.md)
* - [Label Parity](concepts/LabelParity.md)
* - [Outliers](concepts/Outliers.md)
* - [Parity](concepts/Parity.md)
* - [Stats](concepts/Stats.md)
* - [Upper bound Average Precision](concepts/UAP.md)
:::

## Extras

However, {term}`DataEval` also has installable *extras* that provide a more expansive and powerful toolkit for any user.  \
These extras are **torch**, **tensorflow** and **all**. Any extra can be installed using dataeval[*extra*]

:::{list-table}
:header-rows: 1

* - Extras
  - Additional Modules
* - torch
  - [Drift](concepts/Drift.md)  
    [Sufficiency](concepts/Sufficiency.md)
* - tensorflow
  - [Out-of-Distribution Detection](concepts/OOD.md)
:::

**all** installs everything in **torch** and **tensorflow**.

## Installation

Now that you have a chosen which {term}`DataEval` to install, the following methods will show you how to install using your preferred method. \
Be sure to add [*extra*] if you are not installing the base DataEval 

::::{tab-set}

:::{tab-item} pip
Installing from `pip` 
```python
    pip install dataeval[all]
```
:::

:::{tab-item} conda-forge
Installing from `conda`
```python
    conda install dataeval[all]
```
:::

:::{tab-item} source

To install {term}`DataEval` from source locally on Ubuntu, you will need git-lfs to download larger, binary source files and poetry for project dependency management.

```pycon
    sudo apt-get install git-lfs
    pip install poetry
```

Pull the source down and change to the {term}`DataEval` project directory.

```pycon
    git clone https://github.com/aria-ml/dataeval.git
    cd dataeval
```

Install {term}`DataEval` with optional dependencies for development.

```pycon
    poetry install --all-extras --with dev
```

Alternatively, you can install with optional dependencies used to generate documentation as well.

```pycon
    poetry install --all-extras --with dev --with docs
```

Now that DataEval is installed, you can run commands in the poetry virtual environment by prefixing shell commands with poetry run, or activate the virtual environment directly in the shell.

```pycon
    poetry shell
```
:::

::::

<!-- code languages for text found at https://pygments.org/languages/ -->