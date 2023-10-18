[//]: # (3d456072f77d1f93b5e4cb89991638edee562978)

# DAML Change Log
## v0.27.0
- ```ae91e7e7 - Misc. pipeline fixes/enhancements```
- ```37ac9dcb - Determine which functional tests can be implemented for Dp Divergence and BER estimation```
- ```a8ba460f - Use ARiA-specific runner```
- ```61283cd7 - Add pipeline ID to build image tags```
- ```2a847999 - Update prune to run on all known runners```
- ```5c0a61a0 - Use slim versions of python images```
- ```f5b2e8d7 - Update documentation for new metrics```
## v0.26.13
- ```949e09bd - Add kNN BER implementation```
- ```5b39077c - Do not preserve docker build images```
- ```e821168c - Add MNIST to functional tests```
- ```378c2f0f - Use corrupted MNIST data for outlier tutorial```
- ```90175b36 - Remove dind service```
## v0.26.12
- ```a2fd2a9e - Add documentation to MetricClasses and outlierdetectors```
- ```29dce003 - Author Dp Divergence tutorial```
- ```59082ecb - Split requirements.txt in to multiple files to reduce cache invalidation```
- ```3efebf01 - Add dir for build output```
- ```65a578a0 - Fix rules for publish and tag```
## v0.26.11
- ```948ee5e1 - Fix nightly functional tests and prune at start```
- ```35dd5acd - Use python as the default pipeline image```
- ```2a9fa680 - Run docker prune before each run```
- ```1a000de5 - Consolidate requirements and add cuda to project toml```
- ```e5570cee - Use container-check and docker-in-docker build within the pipeline```
## v0.26.10
- ```dab0a8ff - Handle MST edge cases```
## v0.26.9
## v0.26.8
## v0.26.7
## v0.26.6
## v0.26.5
- ```45ce9971 - Refactor alibi-detect and outlier-detection base and child classes```
- ```2c2e72bc - publish and tagging should be staged after validation```
## v0.26.4
- ```bf31996f - BER lower bound capability```
## v0.26.1
- ```b526b2b9 - Add docs capability to container-check```
- ```6218d53b - Remove pipeline stages to increase parallelism```
## v0.25.14
- ```da04ddb1 - Quickfix divergence docs. Based on questions from a user at Kitware```
- ```633a4016 - Configure black format on save```
## v0.25.13
- ```3ccf6f3b - Use patch to mock long running alibi_detect calls```
## v0.25.12
- ```fcbb99dd - Enhance container-check script```
## v0.25.11
- ```dfe0bddb - Add support for python 3.11```
## v0.25.10
- ```235d5264 - First iteration demo. Trains model on 1, 4, and 9. Adds batch size to...```
## v0.25.8
- ```123433dd - Documentation overhaul```
## v0.25.7
- ```997e4dac - Make poetry dev groups optional```
- ```f7de02d5 - container-check - Use `parallel` to build containers concurrently```
- ```6069098e - Create containers for running type/lint/unit tasks```
- ```3adf3a66 - Define optional dependencies and move Output classes to related submodules```
- ```56e7580e - Refactor metric instantiation from load_metric to imports```
- ```54faef53 - Bayes error rate tutorial```
## v0.25.5
- ```2ca285cc - update BER metric to return a dataclass instead of dict```
- ```67f08b27 - Fix: Alibi-detect-models-have-fixed-architecture-shapes```
## v0.25.1
- ```db4adaff - 69 convert metric output dictionary to dataclass```
## v0.25.0
- ```9c8fae0b - Add parameter for pytest to run functional tests```
## v0.24.9
- ```3f6ca9b0 - Finalize documentation of Outlier Detection methods```
- ```55556d09 - Create gitlab folder for pipelines```
## v0.24.8
- ```79614577 - Implement Multiclass MST version of BER```
## v0.24.7
- ```569f6896 - Enable CUDA cuDNN in devcontainers```
## v0.24.5
- ```946d0f48 - Implement and unit test remaining Outlier Detection methods```
- ```70d62a7d - add pytest and python extended functionality```
- ```b75ceea8 - Write test cases for BER metrics/methods```
## v0.24.4
- ```773f8151 - Fix VS Code formatting```
- ```865985ec - Create contributing.md and update readme.md```
## v0.24.3
- ```126e27f3 - Automate changelist generation in documentation for release```
## v0.24.1
- ```84c9a16b - Move lint config to pyproject.toml```
- ```b220d4dd - Fix typecheck failure```
## v0.24.0
- ```6585856a - Write test cases for Dp Divergence```
- ```17e098e9 - Docs update readme```
- ```bc099b90 - Make tox devcontainer more efficient```
- ```773df599 - udpate devcontainer with shell```
- ```ec88ebbe - Use tox to manage dev environment and dev containers```
## v0.23.3
- ```9dc9f1cb - Setup release pipeline```
## v0.23.1
- ```99d2fd22 - Implement outlier detection metrics using the alibi-detect VAE method```
- ```6366d1c9 - Updates to tox.ini```
- ```61bc9495 - Stop duplicate pipeline execution for branch and merge requests```
- ```dae6d859 - Only tag builds when src or package changes```
## v0.23.0
- ```cec26a18 - schedule jobs based on target branch and merge req only```
- ```d095720b - Enhance devcontainers to be pre-configured```
- ```8d796cd2 - update daml __init__.py to list pointers to `list_metrics()` and `load_metric()` external functions.```
- ```a8b6bc9e - Update README.md```
- ```85eb2c1f - Implement outlier detection metrics using the alibi-detect auto-encoder method```
- ```b5b69f08 - Added a mock demo notebook for MockDataset tools```
- ```2ae9bb77 - Add Image Classification Mock Dataset Generators```
## v0.22.0
- ```7d0c9c22 - Create initial documentation for DAML```
