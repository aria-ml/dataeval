# DataEval

To view our extensive collection of tutorials, how-to's, explanation guides,
and reference material, please visit our documentation on
**[Read the Docs](https://dataeval.readthedocs.io/)**

## About DataEval

<!-- start tagline -->

DataEval analyzes datasets and models to give users the ability to train and
test performant, unbiased, and reliable AI models and monitor data for
impactful shifts to deployed models.

<!-- end tagline -->

### Our mission

<!-- start needs -->

DataEval is an effective, powerful, and reliable set of tools for any T&E
engineer. Throughout all stages of the machine learning lifecycle, DataEval
supports model development, data analysis, and monitoring with state-of-the-art
algorithms to help you solve difficult problems. With a focus on computer
vision tasks, DataEval provides simple, but effective metrics for performance
estimation, bias detection, and dataset linting.

<!-- end needs -->

<!-- start JATIC interop -->
DataEval is easy to install, supports a wide range of Python versions, and is
compatible with many of the most popular packages in the scientific and T&E
communities.

DataEval also has native interoperability between JATIC's suite of tools when
using MAITE-compliant datasets and models.
<!-- end JATIC interop -->

## Getting Started

**Python versions:** 3.9 - 3.12

**Supported packages**: *NumPy*, *Pandas*, *Sci-kit learn*, *MAITE*, *NRTK*

Choose your preferred method of installation below or follow our
[installation guide](https://dataeval.readthedocs.io/en/v0.74.2/installation.html).

* [Installing with pip](#installing-with-pip)
* [Installing with conda/mamba](#installing-with-conda)
* [Installing from GitHub](#installing-from-github)

### **Installing with pip**

You can install DataEval directly from pypi.org using the following command.
The optional dependencies of DataEval are `all`.

```bash
pip install dataeval[all]
```

### **Installing with conda**

DataEval can be installed in a Conda/Mamba environment using the provided
`environment.yaml` file.  As some dependencies are installed from the `pytorch`
channel, the channel is specified in the below example.

```bash
micromamba create -f environment\environment.yaml -c pytorch
```

### **Installing from GitHub**

To install DataEval from source locally on Ubuntu, you will need `git-lfs` to
download larger, binary source files and `poetry` for project dependency
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

Install DataEval with optional dependencies for development.

```bash
poetry install --all-extras --with dev
```

Now that DataEval is installed, you can run commands in the poetry virtual
environment by prefixing shell commands with `poetry run`, or activate the
virtual environment directly in the shell.

```bash
poetry shell
```

## Contact Us

If you have any questions, feel free to reach out to the people below:

* **POC**: Scott Swan @scott.swan
* **DPOC**: Andrew Weng @aweng

## Acknowledgement

<!-- start acknowledgement -->

### CDAO Funding Acknowledgement

This material is based upon work supported by the Chief Digital and Artificial
Intelligence Office under Contract No. W519TC-23-9-2033. The views and
conclusions contained herein are those of the author(s) and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of the U.S. Government.

<!-- end acknowledgement -->
