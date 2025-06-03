[//]: # (34e4ded2dec478bd76687b4addb91727754fb369)

# DataEval Change Log

## v0.86.8

## v0.86.7

👾 **Fixes**
- `a93ccfa2` - Hotfix: Address metadata issues for datasets with empty targets

## v0.86.6

## v0.86.5

🛠️ **Improvements and Enhancements**
- `a5154294` - Add dataframe format to StatsOutputs

👾 **Fixes**
- `980eee6f` - Hotfix: Allow for nan values in outliers and account for them correctly
- `2a484972` - Hotfix: Labelstats do not correctly account for 0 targets on an image

## v0.86.4

👾 **Fixes**
- `1a7a6ca1` - Fix metadata and labelstats regressions

## v0.86.3

🛠️ **Improvements and Enhancements**
- `3f301870` - Remove targets and switch to DataFrames for label stats calculations

## v0.86.2

🛠️ **Improvements and Enhancements**
- `5caee32e` - Update Metadata class to use Polars DataFrame

👾 **Fixes**
- `f04da307` - MILCO has trouble reading box coordinates delimited by a variable number of spaces.

## v0.86.1

🛠️ **Improvements and Enhancements**
- `28d186c5` - Adjusting dataset loaders for uniformity and deterministic behavior
- `b0873930` - Improvements to DimensionStats and plotting
- `8ea38cfe` - Add array types for inputs to dataset helpers
- `5fe8e4de` - Fix boxratiostats calculations and add missing docstrings
- `20e8fab6` - add target size and tests
- `96fd2e23` - include empty factors

👾 **Fixes**
- `897e4e0f` - Translated MILCO box corners for MAITE compliance

## v0.86.0

🌟 **Feature Release**
- `3a6c1e8b` - Port multi-variate domain classifier from NannyML

🛠️ **Improvements and Enhancements**
- `6615745b` - Add functionality for image level factors on object detection target metadata
- `485bc051` - Make pandas a required dependency

👾 **Fixes**
- `27758e18` - Fix indexing error in subselection

## v0.85.0

🌟 **Feature Release**
- `019d011d` - Enable ClassFilter for ObjectDetectionDataset
- `9d15be15` - Add concept page for Completeness

🛠️ **Improvements and Enhancements**
- `140cec9d` - Refactor data classes (Embeddings, Images, Metadata, Targets) and function (split_dataset) to be in a first level submodule
- `9eec7e96` - Add save and load functionality to embeddings

## v0.84.1

🛠️ **Improvements and Enhancements**
- `82329adc` - Move to new MAITE/NumPy ArrayLike and loosen type restriction on Datasets
- `c31915dc` - Split ClassBalance from ClassFilter
- `48bc2426` - Update drift classes to use DataEval data structures and simplify torch utility functions

## v0.84.0

🌟 **Feature Release**
- `0c41ca26` - Add factory class method to Embeddings to create from array
- `3d585e81` - Implement completeness
- `77bba6d7` - Add caching (in-memory) to embeddings

🛠️ **Improvements and Enhancements**
- `5a2dc69e` - Change split_dataset to take in Datasets and use Metadata internally
- `e07b8297` - Add plot function to Images and change coverage plot to take Images
- `656c8166` - Add user sections to welcome page

## v0.83.0

🌟 **Feature Release**
- `22206752` - Add transforms to Select dataset class
- `fca7e907` - Add metadata_ood_mi function as find_ood_predictors

🛠️ **Improvements and Enhancements**
- `54c28111` - Adjust tagline/purpose statement to mention user effort

📝 **Miscellaneous**
- `bf415bf8` - Update datatsets docstrings for completeness and move Transform type to typing module

## v0.82.1

🛠️ **Improvements and Enhancements**
- `4b6bbab8` - Intrinsic metadata using image statistics

👾 **Fixes**
- `9bae7f1a` - Perform __dict__ override hack before setting other parameters
- `a9e3fde1` - HOTFIX: remove cyclical call to getattr in Select wrapper class

📝 **Miscellaneous**
- `94b31bf1` - Create a package wide configuration for random seeds
- `36746ef9` - Switch to DeviceLike for typing of torch device
- `13ff9fd0` - Better genericize output classes and add Sequence based output collection class

## v0.82.0

🌟 **Feature Release**
- `7ad22a7c` - Switch all stats classes to use dataset inputs and change `datasetstats` to `imagestats`
- `e1da1768` - Integrate meta_distribution_compare as metadata_distance

🛠️ **Improvements and Enhancements**
- `fc8b68c2` - Spike: Define which metrics comprise completeness metrics

📝 **Miscellaneous**
- `70a8e1e2` - Create lightweight dataset wrapper factory functions
- `fef938d5` - Simplify typing and add more docstrings to dataeval.typing module

## v0.81.0

🌟 **Feature Release**
- `101fea34` - Add selection feature for datasets

📝 **Miscellaneous**
- `efc90630` - Add index based selection helper
- `666e2866` - Add collate helper functions for Dataloaders
- `4e92ae90` - Restructure typing for Datasets to allow better extensibility for other data classes

## v0.80.0

🌟 **Feature Release**
- `94a10b03` - Refactor Images, Embeddings and Metadata as a stateful classes using Dataset inputs
- `353ac6bd` - Add DataProcessor class to handle extraction of images, embeddings, targets and metadata from datasets

🛠️ **Improvements and Enhancements**
- `529e5595` - Merging current WIP state of OODdetector, based on universal embeddings from sigma-optimal-VAE, along with OOD_VAE_minimal notebook.

👾 **Fixes**
- `a2a7057e` - Adjust merge to handle numpy arrays

    Adds in a numpy array check and if true returns the array as a list prior to normal processing

📝 **Miscellaneous**
- `e55cb3f2` - Use cpu as default torch device

## v0.79.0

🌟 **Feature Release**
- `841425ff` - Release Metadata OOD function most_deviated_factors

    Adds a new explanatory function using Metadata and an OODOutput

## v0.78.0

🌟 **Feature Release**
- `bff82522` - Add collate function and convert packaged datasets to MAITE protocols

    * Changes all dataset utility classes to use `MAITE` protocol formats (`MNIST`, `CIFAR10`, and `VOCDetection`)
    * Addes `collate` to aggregate (and encode) `MAITE`datasets into images/embeddings, targets, and metadata

🛠️ **Improvements and Enhancements**
- `d9e0f8b0` - Enforce embeddings on functions/methods that take embedding inputs

## v0.77.1

🛠️ **Improvements and Enhancements**
- `9a420f7d` - Update Assess the data space tutorial to fit JATIC DR-2.3
- `3ab63f3e` - Integrate clusterer speed improvements with numba

## v0.77.0

🌟 **Feature Release**
- `a1974e41` - Add global config module to control default device and max processes

👾 **Fixes**
- `c5ca814d` - Enforce unit interval in OOD detector and coverage metric
- `41c4437b` - CoverageOutput attributes renamed for clarity

    Attributes renamed:
    - `indices` -> `uncovered_indices`
    - `radii` -> `critical_value_radii`
    - `critical_value` -> `coverage_radius`
- `99631a94` - Fix ax.hist on small ranges in NumPy 2.1+

## v0.76.1

🛠️ **Improvements and Enhancements**
- `a8a4cd4f` - Remove merge from preprocess and address metadata array length inconsistencies
- `f8061eca` - Add option to return dropped keys from metadata utility functions
- `a4ddbed1` - Add pandas dependency to `all` extras option
- `5b05981e` - Expose dropped keys from nested lists and inconsistent keys in metadata merge and preprocess

📝 **Miscellaneous**
- `a20766ec` - Updates to documentation
- `961ad923` - Miscellaneous docs changes

## v0.76.0

🌟 **Feature Release**
- `4647edca` - Expose flatten metadata function and update docstring

🛠️ **Improvements and Enhancements**
- `27d34a0c` - Incorporating NAWCAD feedback to improve the documentation for the stat functions, outliers class and coverage class

📝 **Miscellaneous**
- `c9998971` - Switch themes to sphinx-immaterial, enable graphviz and restructure documentation
- `3dedde8f` - Adding templates for auto generation of docs
- `8aaa89f3` - add deep dive prototype
- `d9d902f3` - Allow for float type bounding boxes
- `9bc0f910` - Add additional code coverage tests
- `15f1ae84` - Add logging to output metadata decorator
- `01cef92b` - Split conftest for tests and doctests
- `adc8e293` - Publish MR docs and code coverage to deployment environments
- `a3fc1f6c` - Moves document link to body to match other header titles
- `69892fd3` - Visibility enhancements to BalanceOutput.plot() heatmap
- `b6ab03a6` - Simplify docker build script for docs

## v0.75.0

🌟 **Feature Release**
- `3aa12cb3` - Refactor bias metadata helpers

    Metadata preprocessing functions have been moved from `dataeval.metrics.bias.metadata_preprocessing` to `dataeval.utils.metadata`.

🛠️ **Improvements and Enhancements**
- `ed98b6b1` - Return empty string for hashes on too small images

    `pchash` now returns empty string when attempting to perform perception hashing against images or chips that are too small to meaningfully hash. `Duplicates` also ignore empty perception hashes to avoid false positive detections.
- `b144fa1c` - Change torch to be required dependency

    PyTorch is now a required dependency and the `torch` extra is no longer required for full functionality

📝 **Miscellaneous**
- `6e4474b2` - Refactor utils and fix associated docstrings, documentation and notebooks
- `ff87cee6` - Update documentation and CI pipelines to comply with SDP DR-3
- `aa7d9205` - Updated README.md format, added tagline and cdao funding acknowledgment
- `82559846` - Replace manual markdown files with autoapi generated rst files

## v0.74.2

📝 **Miscellaneous**
- `e7a284de` - Update dataset split unit tests
- `f8731a44` - Add initial logging framework and unit test
- `771dc1d1` - Add conda tests to pipeline
- `2d9fd55a` - Update RTD yaml to use uv for installation
- `0ab99a7f` - Initial prototyping of underspecification tests

## v0.74.1

📝 **Miscellaneous**
- `102664de` - Remove tensorflow from project
- `e782dad1` - Refactor OutputMetadata and clean up set_metadata decorator
- `80aae3a6` - Just use KSOutput as a MappingOutput instance instead of extracting the dict attribute it no longer has.
- `b738e01f` - Allow docker cmds within dev container
- `16839b46` - Add MappingOutput class
- `e2cfda94` - Made metadata_tools/ks_compare compatible with new KSOutput class.

## v0.74.0

🌟 **Feature Release**
- `73c1e1be` - Implement PyTorch AutoEncoder based OOD detector

    Adds initial PyTorch based Autoencoder OOD detector available when installed with the `torch` extra.

🛠️ **Improvements and Enhancements**
- `70794b5f` - Moved discretization of metadata out of bias functions

📝 **Miscellaneous**
- `4d94e602` - Added test assertions for how_to notebooks
- `7723e242` - Introduce Pytorch OOD detector, with its new training procedure, into OOD howto notebook.
- `f5ac4bdd` - Added new KSOutput class and adapted tests and other functions accordingly
- `3a01a81a` - Introduce new Pytorch OOD detection into prototype metadata demo notebooks.
- `dc155554` - Fix torch gmm functions and enable tests
- `a715c1ef` - Adjust docs to incorporate new metadata function
- `0719bad0` - Update dependencies to remove hdbscan

## v0.73.1

👾 **Fixes**
- `cac3e2b8` - Fixes drift with pre-processing and shuffles MNIST by default

📝 **Miscellaneous**
- `bacbd0e7` - Use build script specifically for docs
- `0a87e912` - docker build for docs only
- `671b60a5` - Prototype function to infer whether a 1D sample is continuous or discrete
- `d0b8004a` - Use explicit re namespace for compile, search, sub, and MULTILINE
- `502ca2df` - Change to nox for automation test scripts
- `5b46ebea` - Add new bias functional tests and set groundwork for rediscretization

## v0.73.0

🌟 **Feature Release**
- `e055acf0` - Metadata utility function to merge, extend and flatten metadata
- `95b28ae1` - Adjust bias plotting functions to return figure

📝 **Miscellaneous**
- `532f92a2` - Minimum spanning tree and Clusterer are rewritten using numba for large code speed up
- `7377e012` - Switch jobs to use uv and tox natively
- `7af75016` - Add lazyloading for tensorflow modules

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
