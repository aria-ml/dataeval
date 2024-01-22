# Data-Analysis Metrics Library (DAML)

## About DAML

The Data-Analysis Metrics Library, or DAML, focuses on characterizing image data and its impact on model performance across classification and object-detection tasks.

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

## Getting Started

### Installing DAML

To install the package from the GitLab Pypi repository, run the following command in an environment with Python 3.8-3.11 installed:

`pip install daml`

### Additional Tutorials
For more ideas on getting started using DAML in your workflow, additional information and tutorials are in our Sphinx documentation hosted on [Read the Docs](https://daml.readthedocs.io/).

## Contributing
For steps on how to get started on contributing to the project, you can follow the steps in [CONTRIBUTING.md](CONTRIBUTING.md).

## Attribution
This project uses code from the [Alibi-Detect](https://github.com/SeldonIO/alibi-detect) python library developed by SeldonIO.  Additional documentation from the developers are also available [here](https://docs.seldon.io/projects/alibi-detect/en/stable/).

## POCs
- **POC**: Scott Swan @scott.swan
- **DPOC**: Andrew Weng @aweng
