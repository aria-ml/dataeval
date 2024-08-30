[//]: # (c3b759cd6af61e1c81824b6520714ac64e20527f)

# DataEval Change Log

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
