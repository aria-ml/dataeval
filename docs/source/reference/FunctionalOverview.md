# Functional Overview

The following tables summarize the advised use cases and technical
requirements for the algorithms provided by the DataEval library.
Each algorithm targets different types of data or problem domains.
Refer to the method-specific pages by clicking the algorithms for more detailed information.

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
  - Measures the degree to which images span the learned embedding space
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
  - Assesses the mutual information between factors
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
* - {class}`Prioritization <.Prioritize>`
  - Orders samples based on embeddings
  - ✔
  - ✔
  - ✔

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
* - {class}`.Select`
  - A set of dataset filters that enable rapid 
  development of various datasets
  - ✔
  - ✔
  - ✔

```

````

`````

## Input Requirements

The following table shows the input parameters used by each of DataEval's core functionalities.

`````{tab-set}
:sync-group: func

````{tab-item} Metrics
:sync: metrics

For more information on a specific algorithm, click the name in the table.  

```{list-table}
:widths: 50 8 8 8 8 8 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Images
  - Labels
  - Bounding Boxes
  - Metadata
  - Scores
  - Model/Extractor
* - {func}`Bayes error rate (KNN) <.ber_knn>`  
{func}`Bayes error rate (MST) <.ber_mst>`
  - ***Required***{sup}`1`
  - ***Required***
  - 
  - 
  - 
  - 
* - {func}`Box to Image Ratio statistics <.compute_ratios>`
  - ***Required***{sup}`2`
  - 
  - ***Required***
  - 
  - 
  - 
* - {func}`Completeness <.completeness>`
  - ***Required***{sup}`1`
  - 
  - 
  - 
  - 
  - 
* - {func}`Coverage (Adaptive) <.coverage_adaptive>`  
{func}`Coverage (Naive) <.coverage_naive>`
  - ***Required***{sup}`1`
  - 
  - 
  - 
  - 
  - 
* - {func}`Divergence (FNN) <.divergence_fnn>`  
{func}`Divergence (MST) <.divergence_mst>`
  - ***Required***{sup}`1`
  - 
  - 
  - 
  - 
  - 
* - {func}`Feature distance <.feature_distance>`
  - ***Required***{sup}`1`
  - 
  - 
  - 
  - 
  - 
* - {func}`Image and Target statistics <.compute_stats>`
  - ***Required***{sup}`2`
  - 
  - 
  - 
  - 
  - 
* - {func}`Label errors <.label_errors>`
  - ***Required***{sup}`1`
  - ***Required***
  - 
  - 
  - 
  - 
* - {func}`Label parity <.label_parity>`
  - 
  - ***Required***
  - 
  - 
  - 
  - 
* - {func}`Label stats <.label_stats>`
  - 
  - ***Required***
  - 
  - 
  - 
  - 
* - {func}`Null model metrics <.nullmodel_metrics>`
  - 
  - ***Required***
  - 
  - 
  - 
  - 
* - {func}`Parity <.parity>`
  - 
  - ***Required***
  - 
  - ***Required***
  - 
  - 
* - {func}`UAP <.uap>`
  - 
  - ***Required***
  - 
  - 
  - ***Required***{sup}`4`
  - 

```

````

````{tab-item} Evaluators
:sync: evaluators

For more information on a specific algorithm, click the name in the table.  

```{list-table}
:widths: 50 8 8 8 8 8 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Images
  - Labels
  - Bounding Boxes
  - Metadata
  - Scores
  - Model/Extractor
* - {class}`.Balance`
  - 
  - ***Required***
  - 
  - ***Required***
  - 
  - 
* - {class}`.Diversity`
  - 
  - ***Required***
  - 
  - ***Required***
  - 
  - 
* - {class}`Drift Domain Classifier <.DriftDomainClassifier>`  
{class}`Drift K-Nearest Neighbors <.DriftKNeighbors>`  
{class}`Drift MMD <.DriftMMD>`  
{class}`Drift Reconstruction <.DriftReconstruction>`  
{class}`Drift Univariate <.DriftUnivariate>`  
  - ***Required***
  -
  - 
  - 
  - 
  - ***Required*** (Reconstruction)  
  *Optional* (Univariate)
* - {class}`Duplicate Detection <.Duplicates>`
  - ***Required***{sup}`2`
  -
  - 
  - 
  - 
  - *Optional*
* - {class}`Out-of-Distribution Domain Classifier <.OODDomainClassifier>`  
{class}`Out-of-Distribution K-Nearest Neighbors <.OODKNeighbors>`  
{class}`Out-of-Distribution Reconstruction <.OODReconstruction>`
  - ***Required***
  -
  - 
  - 
  - 
  - ***Required*** (Reconstruction)
* - {class}`.Outliers`
  - ***Required***
  -
  - 
  - 
  - 
  - *Optional*
* - {class}`Prioritization <.Prioritize>`
  - ***Required***
  -
  - 
  - 
  - 
  - *Optional*

```

````

````{tab-item} Metadata
:sync: metadata

For more information on a specific algorithm, click the name in the table.  

```{list-table}
:widths: 50 8 8 8 8 8 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Images
  - Labels
  - Bounding Boxes
  - Metadata{sup}`3`
  - Scores{sup}`5`
  - Model/Extractor
* - {func}`Factor Deviation <.factor_deviation>`
  - 
  -
  - 
  - ***Required***
  - ***Required***
  - 
* - {func}`Factor Predictors <.factor_predictors>`
  - 
  -
  - 
  - ***Required***
  - ***Required***
  - 

```

````

````{tab-item} Workflows
:sync: workflows

For more information on a specific algorithm, click the name in the table.  

```{list-table}
:widths: 50 8 8 8 8 8 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Images
  - Labels
  - Bounding Boxes
  - Metadata
  - Scores
  - Model/Extractor
* - {class}`.Sufficiency`{sup}`2`
  - ***Required***
  - ***Required***
  - *OD Only*
  - 
  - 
  - [*Task specific*](../concepts/Embeddings.md#creating-embeddings)

```

````

````{tab-item} Data Selection
:sync: data_selection

For more information on a specific algorithm, click the name in the table.  

```{list-table}
:widths: 50 8 8 8 8 8 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Images
  - Labels
  - Bounding Boxes
  - Metadata
  - Scores
  - Model/Extractor
* - {func}`Dataset Splitter <.split_dataset>`{sup}`2`
  - 
  - *Optional*
  - 
  - *Optional*{sup}`3`
  - 
  -
* - {class}`.Select`{sup}`2`
  - *Optional*
  - *Optional*
  - 
  - *Optional*
  - 
  - 

```

````

`````

```{note}
{sup}`1` It is highly recommended to give [embeddings](../concepts/Embeddings.md)
over raw images using {class}`.Embeddings`.  
{sup}`2` Input data must be wrapped together in a `Dataset`.  
{sup}`3` When using only metadata, it must be wrapped in DataEval's {class}`.Metadata` class.  
{sup}`4` These scores are the raw outputs of a model.  
{sup}`5` These scores are retrieved by DataEval's Out Of Distribution (OOD) functions.  
```
