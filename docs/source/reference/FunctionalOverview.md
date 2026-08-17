# Functional Overview

The following tables summarize which computer vision tasks each algorithm in
the DataEval library supports. Each algorithm targets different types of data
or problem domains. Refer to the method-specific pages by clicking the
algorithms for more detailed information.

For what data each algorithm consumes — images, labels, metadata, embeddings,
or a model — see
[What data does each tool need?](../getting-started/input-requirements.md).

## Computer Vision Task Compatibility

The following tables show the compatible computer vision tasks that have support
in DataEval. The tables are split into categories based on usage and follow
DataEval's public API.

`````{tab-set}
:sync-group: func

````{tab-item} Metrics
:sync: metrics

```{list-table}
:widths: 40 50 5 5 5
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image Classification
  - Object Detection
  - Unsupervised
* - {func}`Bayes error rate (KNN) <.ber_knn>`  
{func}`Bayes error rate (MST) <.ber_mst>`
  - Determines feasibility of image classification by estimating the bayes error rate
  - ✔
  - 
  - 
* - {func}`Box to Image ratio statistics <.compute_ratios>`
  - Computes statistical summaries of target boxes to image ratios
  - 
  - ✔
  - 
* - {func}`Completeness <.completeness>`
  - Measures the dimensional utilization of the embedding space via eigenvalue entropy
  - ✔
  - ✔
  - ✔
* - {func}`Coverage (Adaptive) <.coverage_adaptive>`  
{func}`Coverage (Naive) <.coverage_naive>`
  - Measures how well the distribution of images in a dataset covers the input space
  - ✔
  - ✔
  - ✔
* - {func}`Divergence (FNN) <.divergence_fnn>`  
{func}`Divergence (MST) <.divergence_mst>`
  - Measures the difference between dataset distributions
  - ✔
  - ✔
  - ✔
* - {func}`Feature distance <.feature_distance>`
  - Measures the feature-wise distance between two continuous distributions
  - ✔
  - ✔
  - ✔
* - {func}`Image and Target statistics <.compute_stats>`
  - Computes statistical summaries of images and/or targets in a dataset
  - ✔
  - ✔
  - ✔
* - {func}`Label errors <.label_errors>`
  - Computes potential label errors in a dataset using embeddings
  - ✔
  - ✔
  - 
* - {func}`Label parity <.label_parity>`
  - Assesses equivalence in label frequency between datasets
  - ✔
  - ✔
  - 
* - {func}`Label stats <.label_stats>`
  - Computes statistical summaries of labels in a dataset
  - ✔
  - ✔
  - 
* - {func}`Null model metrics <.nullmodel_metrics>`
  - Calculates performance metrics for random classifiers on training and testing labels based on the class distributions
  - ✔
  - ✔
  - 
* - {func}`Parity <.parity>`
  - Detects if there is a significant relationship between the factor values and class labels
  - ✔
  - ✔
  - 
* - {func}`UAP <.uap>`
  - Determines feasibility of an object detection task by estimating upper bound on average precision
  - 
  - ✔
  -
```

````

````{tab-item} Evaluators
:sync: evaluators

```{list-table} 
:widths: 40 50 5 5 5
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image Classification
  - Object Detection  
  - Unsupervised
* - {class}`.Balance`
  - Assesses the normalized mutual information between factors
  - ✔
  - ✔
  - 
* - {class}`Chunked Drift <.ChunkedDrift>`
  - Wraps any drift detector to report drift per chunk over a stream
  - ✔
  - ✔
  - ✔
* - {class}`.Coverage`
  - Measures per-class coverage and dispersion of a dataset's embedding space
  - ✔
  - ✔
  - 
* - {class}`.Diversity`
  - Measures the distribution of metadata factors in the dataset
  - ✔
  - ✔
  - 
* - {class}`Drift Domain Classifier <.DriftDomainClassifier>`  
{class}`Drift K-Nearest Neighbors <.DriftKNeighbors>`  
{class}`Drift MMD <.DriftMMD>`  
{class}`Drift Reconstruction <.DriftReconstruction>`  
{class}`Drift Univariate <.DriftUnivariate>`  
{class}`Drift Wasserstein <.DriftWasserstein>`  
  - Detects data distribution shifts from training data
  - ✔
  - ✔
  - ✔
* - {class}`Duplicate Detection <.Duplicates>`
  - Identifies duplicate data entries
  - ✔
  - ✔
  - ✔
* - {class}`Out-of-Distribution Domain Classifier <.OODDomainClassifier>`  
{class}`Out-of-Distribution K-Nearest Neighbors <.OODKNeighbors>`  
{class}`Out-of-Distribution Reconstruction <.OODReconstruction>`
  - Detects data points that fall outside the training distribution
  - ✔
  - ✔
  - ✔
* - {class}`.Outliers`
  - Identifies anomalous data points based on deviations from mean
  - ✔
  - ✔
  - ✔
* - {class}`.Parity`
  - Detects if there is a significant relationship between the factor values and class labels
  - ✔
  - ✔
  - 
* - {class}`Prioritization <.Prioritize>`
  - Orders samples based on embeddings
  - ✔
  - ✔
  - ✔
* - {class}`.Representation`
  - Measures how well a dataset's labels cover an ontology's concept space
  - ✔
  - ✔
  - 

```

````

````{tab-item} Metadata
:sync: metadata

```{list-table}
:widths: 40 50 5 5 5
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image Classification
  - Object Detection  
  - Unsupervised
* - {func}`Factor Deviation <.factor_deviation>`
  - Computes greatest deviation in metadata features per sample
  - ✔
  - ✔
  - ✔
* - {func}`Factor Predictors <.factor_predictors>`
  - Measures the most impactful metadata factors correlated with a flagged sample 
  - ✔
  - ✔
  - ✔
```

````

````{tab-item} Workflows
:sync: workflows

```{list-table}
:widths: 40 50 5 5 5
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image Classification
  - Object Detection  
  - Unsupervised
* - {class}`.Sufficiency`
  - Determines data needs for performance standards
  - ✔
  - ✔
  - 

```

````

````{tab-item} Data Selection
:sync: data_selection

```{list-table}
:widths: 40 50 5 5 5
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image Classification
  - Object Detection  
  - Unsupervised
* - {func}`Dataset Splitter <.split_dataset>`
  - Generates train, val, and test splits based on information such 
  as labels and metadata
  - ✔
  - ✔
  - ✔
* - {class}`.View`
  - Build a dataset view from an ordered pipeline of
  filter, transform, and relabel operations
  - ✔
  - ✔
  - ✔

```

````

`````
