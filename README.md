# Data-Analysis Metrics Library (DAML)

## About DAML

The Data-Analysis Metrics Library, or DAML, focuses on characterizing image data and its impact on model performance across classification and object-detection tasks.

<!-- start about -->

**Model-agnostic metrics that bound real-world performance**
- relevance/completeness/coverage
- metafeatures (data complexity)

**Model-specific metrics that guide model selection and training**
- dataset sufficiency
- data/model complexity mismatch

**Metrics for post-deployment monitoring of data with bounds on model performance to guide retraining**
- dataset-shift metrics
- model performance bounds under covariate shift
- guidance on sampling to assess model error and model retraining

<!-- end about -->

## Getting Started

### Requirements
- Python 3.8-3.11

### Installing DAML

You can install DAML directly from pypi.org using the following command.  The optional dependencies of DAML are `torch`, `tensorflow` and `all`.  Using `torch` enables Sufficiency metrics, and `tensorflow` enables Outlier Detection.

```
pip install daml[all]
```

### Installing DAML from GitHub

To install DAML from source locally on Ubuntu, you will need `git-lfs` to download larger, binary source files and `poetry` for project dependency management.

```
sudo apt-get install git-lfs
pip install poetry
```

Pull the source down and change to the DAML project directory.
```
git clone https://github.com/aria-ml/daml.git
cd daml
```



Install DAML with optional dependencies for development.
```
poetry install --all-extras --with test,lint
```

Alternatively, you can install with optional dependencies used to generate documentation as well.
```
poetry install --all-extras --with test,lint,docs
```

Now that DAML is installed, you can run commands in the poetry virtual environment by prefixing shell commands with `poetry run`, or activate the virtual environment directly in the shell.
```
poetry shell
```

### Documentation and Tutorials
For more ideas on getting started using DAML in your workflow, additional information and tutorials are in our Sphinx documentation hosted on [Read the Docs](https://daml.readthedocs.io/).

## Attribution
This project uses code from the [Alibi-Detect](https://github.com/SeldonIO/alibi-detect) python library developed by SeldonIO.  Additional documentation from the developers are also available [here](https://docs.seldon.io/projects/alibi-detect/en/stable/).

## POCs
- **POC**: Scott Swan @scott.swan
- **DPOC**: Andrew Weng @aweng
