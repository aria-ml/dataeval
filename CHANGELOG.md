[//]: # (a28a5d75c06d3a641529bfe049e6530ec2735a2d)

# DAML Change Log
## v0.42.1
## v0.42.0
- ```b48c05ca - Release v0.42.0```
## v0.42.0
- ```5dacc855 - Update dependencies and use optional installs```
- ```a6ce3e72 - Remove UAP_MST metric```
- ```601cfae8 - Sufficiency Plotting of Multiple Metrics during one run```
- ```b46dba6e - Documentation and README.md improvements```
- ```f5172c60 - MR for discussion```
- ```3d68a6f1 - Add parameter to plot function for optional file output```
- ```e8e906d7 - Restructure src and tests to match current design```
## v0.40.2
- ```f3eddaed - Flavor 2 - Remove models from metrics entirely```
- ```db888bb7 - Remove usage of DamlDataset for ARiA metrics```
## v0.40.1
- ```69f6c16f - Refactor base AriaMetric class for better code sharing and easier future removal of model workflow```
## v0.40.0
- ```ddb798c6 - Document guidelines for sufficiency training```
## v0.38.2
- ```6bb442c3 - Refactor common minimum spanning tree code into utils```
- ```42617f43 - Enable GPU functionality in pytorch features```
## v0.38.1
## v0.38.0
- ```8fe97232 - Add export_model functionality and improve test coverage```
- ```94aa495f - Fix sufficiency tutorial and expose batch_size argument```
- ```d953fcc5 - Refactor and cleanup dead code```
- ```c9b5116e - ARiA Autoencoder as PyTorch Model```
- ```42cc77ea - Add empirical upper bound to UAP metric output```
- ```2ba149a7 - Make internal implementation of alibi-detect a subpackage of daml```
- ```da23f2f0 - Snapshot outlier detection code from alibi-detect```
- ```1c44826f - Remove JATIC token requirements for MAITE```
- ```636dfdaf - update project with __version__ metadata```
## v0.36.1
- ```7d1a599f - Implement the uap class```
- ```9ad6f904 - Create classification dataset sufficiency tutorial```
## v0.36.0
- ```0799523b - Object detection model training```
## v0.35.1
- ```e7368220 - update dependencies for maite rename```
## v0.35.0
- ```fe34408e - Refactor metrics and split models out from outlier_detection```
## v0.34.5
- ```4fd3ca77 - add devenv script for custom capabilities on devcontainer```
## v0.34.3
- ```5d858473 - Update documentation tutorials to use DamlDataset```
## v0.34.2
- ```2a0141b4 - Adds object detection to Daml Dataset. Updates jatic wrappers```
- ```4503e91c - Push composited build images to harbor to skip compositing images in each task runner```
## v0.34.1
- ```e9225fb1 - Add check for jatic_toolbox before exposing functions that have dependencies on it```
- ```d293d435 - Update dependencies using alibi-detect 0.11.5-dev```
- ```66cda4e6 - Build python with optimizations```
## v0.34.0
- ```4a59eff5 - Prepares classification sufficiency for Object Detection updates```
- ```fc5c9859 - Build dependencies from scratch if no access to harbor.jatic.net```
- ```4a76e6eb - Move to autoscaler runners and harbor image registry```
## v0.33.0
- ```fb605c56 - Fix JATIC wrapper and speed up miscellaneous test execution```
- ```cb9bff80 - Better handle milestones and sprints in auto-versioning```
## v0.29.0
- ```5c4e6e06 - Use convolutional autoencoder for BER and Divergence metrics```
- ```4d18fd36 - Switch to multi-stage build process for Dockerfile image```
- ```cc11187c - Move Gitlab token check into installJATIC function```
- ```48ced022 - Update build with help and dedupe python versions passed in```
- ```c78e5502 - Sufficiency typecheck bugfix```
- ```59c2fe4d - Remove merged branch image earlier in pipeline job and prune on build```
- ```3479dc6a - Switch to cuda image with cudnn8 and developer libraries```
## v0.28.5
- ```9d1c354c - Add fit_dataset, format_dataset to DpDivergence & BER```
## v0.28.4
- ```c39e009e - Fix typecheck issues found with pyright-1.1.333```
## v0.28.3
- ```57569af5 - Write JATIC interop test cases```
- ```5bd3c2f5 - Build and tag docker images of daml-cache by branch name```
## v0.28.2
- ```1d85c777 - Remove unused base class```
- ```b0baac49 - Print more disk usage info in pipeline```
- ```3725be9c - move token check```
- ```19deb6b7 - Use cache-from for dev container build```
- ```6c265500 - Use trap to ensure containers/images get cleaned up```
- ```413ec769 - Add standalone build job```
- ```cf87af41 - Merge devcontainer Dockerfile with build DockerFile and remove tox```
- ```46178900 - Move documentation to nightly builds```
- ```86246c21 - Update dependencies with support for torch and JATIC toolbox```
- ```9c0cf329 - matrix info and prune jobs```
## v0.28.1
- ```da14b957 - Move documentation dependencies in to pyproject.toml```
## v0.28.0
- ```f3dc7a5d - Use docker run to execute build tasks```
- ```6942b7c5 - Switch to cuda docker image for gpu support```
- ```7d610a86 - Print output in real-time for single and verbose parallel job runs```
- ```d86cf421 - use virtualenvs instead of pyenv global```
- ```1af04afe - Create a shared base image with all supported python versions and dependencies```
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
