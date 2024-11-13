[//]: # (6e55451c4cdd0d484a0ee76550b1b4026b50cee6)

# DataEval Change Log

## v0.72.2

🛠️ **Improvements and Enhancements**
- `ba52ef2e` - Refactor away _internal module

📝 **Miscellaneous**
- `6e55451c` - Integration of distribution compare and OOD MI metadata tools (continued)
- `e4f82173` - Streamlined tests
- `ac8fe3ee` - Fix type mismatch on training AEGMM
- `6289c7d0` - Add plotting helper functions to diversity and balance
- `14d0cfd4` - Integration of low-level metadata drift/OOD exploration functions

## v0.72.1

📝 **Miscellaneous**
- `32ba1f29` - Data split tests
- `76f73770` - Updated glossary and other files to use new style of links
- `20efd27e` - Add support for Python 3.12

## v0.72.0

🌟 **Feature Release**
- `14ef382c` - Update dependencies for conda compatibility

## v0.71.1

🛠️ **Improvements and Enhancements**
- `97849b01` - Update support for tensorflow >=2.16 with explicit keras v2

📝 **Miscellaneous**
- `85bafa30` - Swap brightness and darkness
- `96a30ad0` - Make optional checks more granular
- `55ca81d6` - Use native int for dict keys for Outliers
- `639e140b` - silence warnings for docs and doctest

## v0.71.0

🌟 **Feature Release**
- `cdae8a17` - Parallelize existing stats metric functions and introduce dedicated channelstats function

    Running statistical analysis functions take significant time against large datasets.  Due to the natural parallelism of analyzing individual images, we introduced parallel processing leveraging the `multiprocessing` library to accelerate processing times.
    
    Affected functions:
    
    * `datasetstats`
    * `dimensionstats`
    * `hashstats`
    * `pixelstats`
    * `visualstats`
    
    Additionally, `channelstats` was added which performs the functionality of `datasetstats` but only for the functions that support per-channel stat calculation, `pixelstats` and `visualstats`.

📝 **Miscellaneous**
- `552668a0` - Update EDA part 1 tutorial with miscellaneous changes

## v0.70.1

🛠️ **Improvements and Enhancements**
- `d1cdcda5` - API changes with supporting documentation updates

📝 **Miscellaneous**
- `5ecd4d3a` - expose datasets API
- `6c19bba7` - Make sufficiency args more permissive
- `1bc2d067` - Improving MNIST class
- `d23b3461` - Extract small-scope reusable functions from tools made for prototype Associate[Drift|OOD]withMetadataTutorial notebooks.
- `5bea9512` - remove tf-iogcs-fs

## v0.70.0

🌟 **Feature Release**
- `71e7ff06` - Integrate labelstats function
- `f40bf0e4` - Redesign stats functions for expansion to per-box, per-channel, and boxratiostats

🛠️ **Improvements and Enhancements**
- `72390edc` - Change input format of balance and diversity to be the same as parity

👾 **Fixes**
- `f598c46a` - Update pytorch to 2.2.0+

📝 **Miscellaneous**
- `b8f0d502` - Create copy on `to_numpy` by default
- `04a71337` - Fix CI docs job to load on build
- `9286f5e8` - Skip or rework MNIST based unit tests
- `704f44e3` - Investigate the use of metadata to help explain observed dataset drifts and OOD examples
- `e25f84f3` - Expose SufficiencyOutput and move class methods to output class
- `742a084c` - Adding algorithm compatability/requirements table
- `7ce85be7` - Misc concept documentation

## v0.69.4

📝 **Miscellaneous**
- `7bca6ed4` - Unified all MNIST and MNIST corrupt datasets to a single internal MNIST class
- `66ad1c92` - new drift detector: multivariate domain classifier

## v0.69.3

📝 **Miscellaneous**
- `6745e39d` - Document: Class Label Statistical Independence and Coverage Documentation
- `1f7689ac` - Adding bias tutorial (parity-balance-diversity)

## v0.69.2

📝 **Miscellaneous**
- `f7d5bac3` - Adds stats for bounding boxes
- `18be58a3` - Adding label stats
- `809d1d7a` - Always produce p-val and distance metrics for drift
- `5cd7c205` - Improving imagestats and channelstats functions
- `b379d44c` - Add dataset splitting features
- `80b68a73` - Use regex to replace markdown links
- `1d99455a` - Tag LKG at the correct commit SHA
- `ad0e368b` - Always run tasks

## v0.69.1

📝 **Miscellaneous**
- `d9068a2c` - Fix release and changelog script

## v0.69.0

📝 **Miscellaneous**
- `63ab70d7` - Remove automatic update of documentation notebooks

## v0.68.0

🌟 **Feature Release**
- `47b48e14` - Allow Duplicates and Outliers detectors to take in multiple StatsOutput objects

📝 **Miscellaneous**
- `65d8f3de` - Combine classwise bias metric outputs with non-classwise
- `ccfd72ef` - Adding clustering/coverage tutorial
- `6d09d710` - Add CONTRIBUTING.md
- `72387d9c` - Updated version replacement script to include cache files
- `5285f01b` - Prototype Performance  Estimation
- `3ae16116` - concept pages for balance and diversity, rescale Simpson diversity
- `3e16a905` - Switching documentation themes to the pydata theme

## v0.67.0

🌟 **Feature Release**
- `a0b04800` - Refactor DataEval functions and classes and update documentation

    - Changes DataEval functions and classes to be more hierarchical in modules:
      - detectors
        - drift (DriftCVS, DriftKS, DriftMMD, DriftUncertainty)
        - linters (Clusterer, Duplicates, Outliers)
        - ood (OOD_AE, OOD_AEGMM, OOD_LLR, OOD_VAE, OOD_VAEGMM)
      - flags (ImageStat)
      - metrics
        - bias (balance, coverage, diversity, parity)
        - estimators (ber, divergence, uap)
        - stats (imagestats, channelstats)
      - workflows (Sufficiency)
    - Backends have been moved from `models` to `tensorflow` and `torch`
    - Renamed following classes:
      - `Linter` -> `Outliers`
      - `parity` -> `label_parity`
      - `parity_metadata` -> `parity`
      - `DriftOutput` -> `DriftBaseOutput`
      - `DriftUnivariateOutput` -> `DriftOutput`
    - Miscellaneous fixes:
      - Documentation updated
      - Streamlined optional import checks in the `__init__.py` tree
       - Fixed misspelling in glossary

👾 **Fixes**
- `84aae760` - balance test cleanup

📝 **Miscellaneous**
- `6d09d710` - Add CONTRIBUTING.md
- `72387d9c` - Updated version replacement script to include cache files
- `5285f01b` - Prototype Performance  Estimation
- `3ae16116` - concept pages for balance and diversity, rescale Simpson diversity
- `3e16a905` - Switching documentation themes to the pydata theme
- `d50d9cd1` - Update Landing Page
- `2fd7fa59` - Author drift detection tutorial
- `49b5af42` - Use uv instead of pyenv for python deployment
- `0f6eb6b0` - Pin notebooks on release to specific version
- `4f101a4e` - Adjust imagestats and channelstats reference guides to new format
- `0ee82ede` - Only build data image in main pipeline
- `7b84ceb5` - Improve test coverage
- `d3c5258a` - Add StatsOutput as input type for linter and duplicates
- `cf73393a` - Updates drift reference guides and concept page
- `4ce5cdf7` - Adjust model reference guides to new format
- `17195a2b` - Adjust parity reference guides to new format
- `e9761b4d` - Adjust out of distribution reference guides to new format
- `eaf707a7` - Adjust uap reference guide to new format
- `335ac3be` - Adjust sufficiency reference guide to new format
- `3a866f01` - Change Optional[Type] to Type | None per 3.10+ standards

## v0.66.0

🌟 **Feature Release**
- `a0b04800` - Refactor DataEval functions and classes and update documentation

    - Changes DataEval functions and classes to be more hierarchical in modules:
      - detectors
        - drift (DriftCVS, DriftKS, DriftMMD, DriftUncertainty)
        - linters (Clusterer, Duplicates, Outliers)
        - ood (OOD_AE, OOD_AEGMM, OOD_LLR, OOD_VAE, OOD_VAEGMM)
      - flags (ImageStat)
      - metrics
        - bias (balance, coverage, diversity, parity)
        - estimators (ber, divergence, uap)
        - stats (imagestats, channelstats)
      - workflows (Sufficiency)
    - Backends have been moved from `models` to `tensorflow` and `torch`
    - Renamed following classes:
      - `Linter` -> `Outliers`
      - `parity` -> `label_parity`
      - `parity_metadata` -> `parity`
      - `DriftOutput` -> `DriftBaseOutput`
      - `DriftUnivariateOutput` -> `DriftOutput`
    - Miscellaneous fixes:
      - Documentation updated
      - Streamlined optional import checks in the `__init__.py` tree
       - Fixed misspelling in glossary

🛠️ **Improvements and Enhancements**
- `5f730baa` - Refactor ImageStats and ChannelStats as metric functions

👾 **Fixes**
- `84aae760` - balance test cleanup
- `3ebd278c` - handle float-type categorical variables in balance metric
- `066b7153` - Fixes modzscore to account for division by 0

📝 **Miscellaneous**
- `d50d9cd1` - Update Landing Page
- `2fd7fa59` - Author drift detection tutorial
- `49b5af42` - Use uv instead of pyenv for python deployment
- `0f6eb6b0` - Pin notebooks on release to specific version
- `4f101a4e` - Adjust imagestats and channelstats reference guides to new format
- `0ee82ede` - Only build data image in main pipeline
- `7b84ceb5` - Improve test coverage
- `d3c5258a` - Add StatsOutput as input type for linter and duplicates
- `cf73393a` - Updates drift reference guides and concept page
- `4ce5cdf7` - Adjust model reference guides to new format
- `17195a2b` - Adjust parity reference guides to new format
- `e9761b4d` - Adjust out of distribution reference guides to new format
- `eaf707a7` - Adjust uap reference guide to new format
- `335ac3be` - Adjust sufficiency reference guide to new format
- `3a866f01` - Change Optional[Type] to Type | None per 3.10+ standards
- `fe1e292d` - Use output dataclass with metadata
- `b3f6a027` - Unify handling of image reshaping

## v0.65.0

🛠️ **Improvements and Enhancements**
- `5f730baa` - Refactor ImageStats and ChannelStats as metric functions

👾 **Fixes**
- `3ebd278c` - handle float-type categorical variables in balance metric
- `066b7153` - Fixes modzscore to account for division by 0

📝 **Miscellaneous**
- `fe1e292d` - Use output dataclass with metadata
- `b3f6a027` - Unify handling of image reshaping

## v0.64.0

🌟 **Feature Release**
- `bea0446c` - Torch Dataset Reader

🛠️ **Improvements and Enhancements**
- `eda88822` - Refactor metrics

📝 **Miscellaneous**
- `a4b8e919` - Created new documentation issue templates
- `1028d082` - Remove is_arraylike function
- `dbcecec6` - Refactored read_dataset to handle common dataset returns
- `61b1f854` - Updated Workflow Landing Page
- `cf96c7f2` - Run doctest in CI pipeline
- `ecfcf89b` - Adjusted notebooks to work on google colab and added environment requirements
- `5f863782` - Update remaining metric output to NamedTuple
- `e58f4dba` - Add metadata parity documentation
- `6319a1d4` - Adding Duplicates concept
- `787545f5` - Adding ImageStats and ChannelStats concept document
- `7826405c` - Update Data Cleaning concept
- `50047116` - Change to Semantic Versioning
- `9e43399c` - Bayes Error Rate - explanation documentation
- `266ad738` - Updated BER docstrings with NDArray, shapes, and examples

## v0.63.0

🛠️ **Improvements and Enhancements**
- `3225cf18` - Convert remaining metrics and detectors to ArrayLike
- `5d88b82a` - Add Torch and Tensorflow interop through ArrayLike protocol and to_numpy converter
- `d3342275` - Refactor linter and duplicates to call evaluate with data
- `65d5aaa8` - Refactor metrics to call evaluate with data

## v0.61.0

🛠️ **Improvements and Enhancements**
- `cd59debb` - Release DataEval v0.61.0!

    DAML is now officially rebranded as DataEval!  New name, same great camel flavor.

## v0.56.0

🌟 **Feature Release**
- `64416675` - Update clusterer class and documentation

    * `Clusterer` detector released
    
    This class assists in exploratory data analysis of unlabeled data by identifying duplicates and outliers. Additional information on usage is available in our documentation.

## v0.55.0

🌟 **Feature Release**
- `278b4dc1` - Release Linter, Duplicates, ImageStats, ChannelStats and Parity

    `Linter`, `Duplicates` detectors and `ImageStats`, `ChannelStats`, and `Parity` metrics are now released. The existing metrics available have also been moved into different modules (`detectors` and `workflows`) that better reflect their functionality.
    
    * `detectors`
      * Drift detectors: `DriftCVM`, `DriftKS`, `DriftMMD`, `DriftUncertainty` and supporting classes
      * Out-of-distribution detectors: `OOD_AE`, `OOD_AEGMM`, `OOD_LLR`, `OOD_VAE`, `OOD_VAEGMM` and supporting classes
      * `Linter`
      * `Duplicates`
    * `metrics`
      * `BER`
      * `Divergence`
      * `Parity`
      * `ImageStats`
      * `ChannelStats`
      * `UAP`
    * `workflows`
      * `Sufficiency`

## v0.54.0

🛠️ **Improvements and Enhancements**
- `58263ac7` - Move niter param to evaluate and calculate and retain curve coefficients in output dictionary

    This change enhances the output of the `Sufficiency` metric to provide the coefficients for the learning curve by measure/class when running the metric. These parameters were previously recalculated each call to project and plot. The parameters are provided as a `Dict[str, np.ndarray]` under the `_CURVE_PARAMS_` key in the output dictionary.

## v0.53.0

🌟 **Feature Release**
- `322fc830` - Add parameter `k` to BER estimator for KNN to enable `k>1` for better consistency with ground truth in certain cases

## v0.52.0

🛠️ **Improvements and Enhancements**
- `07b12ac2` - Fully integrate outlier detection

    Outlier Detection API has been changed.  Additional details are available in our documentation.

## v0.51.0

🌟 **Feature Release**
- `2ed88a07` - Implement Drift Detection Metrics

    This change adds 4 types of Drift Detection metrics which allow for the detection of potential drift in the dataset.
    
    * Kolmogorov-Smirnov
    * Cramér-von Mises
    * Maximum Mean Discrepancy
    * Classifier Uncertainty
    
    The conceptual source is derived from [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953) and the implementation is derived from [Alibi-Detect v0.11.4](https://github.com/SeldonIO/alibi-detect/tree/v0.11.4).

## v0.45.0

🚧 **Deprecations and Removals**
- `5cc48bec` - Divergence metric naming corrected to HP Divergence

    Divergence metric output now returns a dictionary of `{ "divergence": float, "error": int }` instead of `{ "dpdivergence": float, "error": int }`.  Code, documentation and tutorials have been updated to the correct nomenclature of HP (Henze-Penrose) divergence.

## v0.44.6

🌟 **Feature Release**
- `41b20d3a` - Add rules for release label pipeline workflow and merge request release template

🛠️ **Improvements and Enhancements**
- `7ee53c9c` - Update Divergence default to MST

## v0.44.2

🛠️ **Improvements and Enhancements**
- `1468aa5c` - Switch to markdown and updated docs

## v0.43.0

🛠️ **Improvements and Enhancements**
- `670a0db5` - Add support for classwise Sufficiency metrics
- `b96ee099` - Have sufficiency train and eval functions take indices and batch size instead of a DataLoader

## v0.42.2

🛠️ **Improvements and Enhancements**
- `5225c491` - Change output classes to dictionaries
- `45040682` - Make Sufficiency a stateful class and revise SufficiencyOutput
- `7c5fdcff` - Pass method as a parameter to determine metric algorithm to use
- `2e883f6d` - Add better optimizer to find global minimum
- `c3c78680` - Expose AETrainer to public API to use model multiple times after training

👾 **Fixes**
- `93564b95` - Updating pyproject.toml and lock file to set dependency less than numpy 2.0

## v0.42.0

🛠️ **Improvements and Enhancements**
- `601cfae8` - Sufficiency Plotting of Multiple Metrics during one run
- `3d68a6f1` - Add parameter to plot function for optional file output

🚧 **Deprecations and Removals**
- `a6ce3e72` - Remove UAP_MST metric

## v0.40.2

🛠️ **Improvements and Enhancements**
- `f3eddaed` - Flavor 2 - Remove models from metrics entirely

## v0.40.1

🚧 **Deprecations and Removals**
- `db888bb7` - Remove usage of DamlDataset for ARiA metrics

## v0.38.1

🛠️ **Improvements and Enhancements**
- `42617f43` - Enable GPU functionality in pytorch features

## v0.38.0

🌟 **Feature Release**
- `c9b5116e` - ARiA Autoencoder as PyTorch Model

🛠️ **Improvements and Enhancements**
- `8fe97232` - Add export_model functionality and improve test coverage
- `42cc77ea` - Add empirical upper bound to UAP metric output

👾 **Fixes**
- `636dfdaf` - update project with __version__ metadata

## v0.36.1

🌟 **Feature Release**
- `7d1a599f` - Implement the uap class

## v0.36.0

🛠️ **Improvements and Enhancements**
- `0799523b` - Object detection model training

## v0.29.0

🌟 **Feature Release**
- `166df3b0` - Implement Dataset Sufficiency Metric

🛠️ **Improvements and Enhancements**
- `5c4e6e06` - Use convolutional autoencoder for BER and Divergence metrics

👾 **Fixes**
- `c78e5502` - Sufficiency typecheck bugfix

## v0.28.5

🛠️ **Improvements and Enhancements**
- `9d1c354c` - Add fit_dataset, format_dataset to DpDivergence & BER

## v0.28.4

👾 **Fixes**
- `c39e009e` - Fix typecheck issues found with pyright-1.1.333

## v0.26.13

🌟 **Feature Release**
- `949e09bd` - Add kNN BER implementation

## v0.26.10

🛠️ **Improvements and Enhancements**
- `dab0a8ff` - Handle MST edge cases

## v0.26.4

🛠️ **Improvements and Enhancements**
- `bf31996f` - BER lower bound capability

## v0.25.11

🛠️ **Improvements and Enhancements**
- `dfe0bddb` - Add support for python 3.11

## v0.25.4

🛠️ **Improvements and Enhancements**
- `2ca285cc` - update BER metric to return a dataclass instead of dict

## v0.25.3

👾 **Fixes**
- `67f08b27` - Fix: Alibi-detect-models-have-fixed-architecture-shapes

## v0.25.2

🛠️ **Improvements and Enhancements**
- `db4adaff` - 69 convert metric output dictionary to dataclass

## v0.24.8

🌟 **Feature Release**
- `79614577` - Implement Multiclass MST version of BER

## v0.24.6

🌟 **Feature Release**
- `2ad9fed5` - Implement BER estimate

## v0.23.1

🌟 **Feature Release**
- `99d2fd22` - Implement outlier detection metrics using the alibi-detect VAE method

## v0.23.0

🌟 **Feature Release**
- `85eb2c1f` - Implement outlier detection metrics using the alibi-detect auto-encoder method
