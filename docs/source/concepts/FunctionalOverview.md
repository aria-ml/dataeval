# Functionality Overview

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
:widths: 30 50 5 5 5
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image Classification
  - Object Detection
  - Unsupervised
* - {func}`Balance <.balance>`
  - Assesses the mutual information between factors
  - ✔
  - ✔
  - 
* - {func}`Bayes error rate <.ber>`
  - Determines feasibility of image classification by estimating the bayes error rate
  - ✔
  - 
  - 
* - {func}`Box statistics <.boxratiostats>`
  - Computes statistical summaries of target boxes
  - 
  - ✔
  - 
* - {func}`Completeness <.completeness>`
  - Measures the degree to which images span the learned embedding space
  - ✔
  - ✔
  - ✔
* - {func}`Coverage <.coverage>`
  - Measures how well the distribution of images in a dataset covers the input space
  - ✔
  - ✔
  - ✔
* - {func}`Dimension stats <.dimensionstats>`
  - Computes statistical summaries of image and target box dimensions
  - ✔
  - ✔
  - ✔
* - {func}`Divergence <.divergence>`
  - Measures the difference between dataset distributions
  - ✔
  - ✔
  - ✔
* - {func}`Diversity <.diversity>`
  - Measures the distribution of metadata factors in the dataset
  - ✔
  - ✔
  - ✔
* - {func}`Image statistics <.imagestats>`
  - Computes statistical summaries of images in a dataset
  - ✔
  - ✔
  - ✔
* - {func}`Label parity <.label_parity>`
  - Assesses equivalence in label frequency between datasets
  - ✔
  - ✔
  - 
* - {func}`Label stats <.labelstats>`
  - Computes statistical summaries of labels in a dataset
  - ✔
  - ✔
  - 
* - {func}`Null model metrics <.null_model_metrics>`
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

````{tab-item} Detectors
:sync: detectors

```{list-table} 
:widths: 30 50 5 5 5
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image Classification
  - Object Detection  
  - Unsupervised
* - {mod}`Drift <.drift>`
  - Detects data distribution shifts from training data
  - ✔
  - ✔
  - ✔
* - {class}`Duplicate Detection <.Duplicates>`
  - Identifies duplicate data entries
  - ✔
  - ✔
  - ✔
* - {mod}`Out-of-Distribution <.ood>`
  - Detects data points that fall outside the training distribution
  - ✔
  - ✔
  - ✔
* - {class}`.Outliers`
  - Identifies anomalous data points based on deviations from mean
  - ✔
  - ✔
  - ✔

```

````

````{tab-item} Metadata
:sync: metadata

```{list-table}
:widths: 30 50 5 5 5
:header-rows: 1
:class: table-text

* - Algorithm
  - Description
  - Image Classification
  - Object Detection  
  - Unsupervised
* - {func}`Most Deviated Factors <.find_most_deviated_factors>`
  - Measures the greatest deviated metadata factors for detected out of distribution samples
  - ✔
  - ✔
  - ✔
* - {func}`OOD Predictors <.find_ood_predictors>`
  - Measures the most impactful factors for detected out of distribution samples 
  - ✔
  - ✔
  - ✔
```

````

````{tab-item} Workflows
:sync: workflows

```{list-table}
:widths: 30 50 5 5 5
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
:widths: 30 50 5 5 5
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
For an overview, see the {mod}`.metrics` page.

```{list-table}
:widths: 50 10 10 10 10 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Images
  - Labels
  - Bounding Boxes
  - Metadata
  - Scores
* - {func}`Balance <.balance>`
  - 
  - ***Required***
  - 
  - ***Required***
  -
* - {func}`Bayes error rate <.ber>`
  - ***Required***{sup}`1`
  - ***Required***
  - 
  - 
  -
* - {func}`Box statistics <.boxratiostats>`
  - ***Required***
  - 
  - ***Required***
  - 
  - 
* - {func}`Completeness <.completeness>`
  - ***Required***{sup}`1`
  -
  - 
  - 
  - 
* - {func}`Coverage <.coverage>`
  - ***Required***{sup}`1`
  -
  - 
  - 
  - 
* - {func}`Dimension stats <.dimensionstats>`
  - ***Required***{sup}`2`
  -
  - 
  - 
  - 
* - {func}`Divergence <.divergence>`
  - ***Required***{sup}`1`
  -
  - 
  - 
  - 
* - {func}`Diversity <.diversity>`
  - 
  - ***Required***
  - 
  - ***Required***
  - 
* - {func}`Image statistics <.imagestats>`
  - ***Required***{sup}`2`
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
* - {func}`Label stats <.labelstats>`
  - 
  - ***Required***
  - 
  - 
  - 
* - {func}`Null model metrics <.null_model_metrics>`
  - 
  - ***Required***
  - 
  - 
  - 
* - {func}`Parity <.parity>`
  - 
  - ***Required***
  - 
  - ***Required***
  - 
* - {func}`UAP <.uap>`
  - 
  - ***Required***
  - 
  - 
  - ***Required***{sup}`4`

```

````

````{tab-item} Detectors
:sync: detectors

For more information on a specific algorithm, click the name in the table.  
For an overview, see the {mod}`.detectors` page.

```{list-table}
:widths: 50 10 10 10 10 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Images
  - Labels
  - Bounding Boxes
  - Metadata
  - Scores
* - {mod}`Drift <.drift>`
  - ***Required***
  -
  - 
  - 
  - 
* - {class}`Duplicate Detection <.Duplicates>`
  - ***Required***{sup}`2`
  -
  - 
  - 
  - 
* - {mod}`Out-of-Distribution <.ood>`
  - ***Required***
  -
  - 
  - 
  - 
* - {class}`.Outliers`
  - ***Required***
  -
  - 
  - 
  - 

```

````

````{tab-item} Metadata
:sync: metadata

For more information on a specific algorithm, click the name in the table.  
For an overview, see the {mod}`.metadata` page.

```{list-table}
:widths: 50 10 10 10 10 10
:header-rows: 1
:class: table-text

* - Algorithm
  - Images
  - Labels
  - Bounding Boxes
  - Metadata{sup}`3`
  - Scores{sup}`5`
* - {func}`Most Deviated Factors <.find_most_deviated_factors>`
  - 
  -
  - 
  - ***Required***
  - ***Required***
* - {func}`OOD Predictors <.find_ood_predictors>`
  - 
  -
  - 
  - ***Required***
  - ***Required***

```

````

````{tab-item} Workflows
:sync: workflows

For more information on a specific algorithm, click the name in the table.  
For an overview, see the {mod}`.workflows` page.

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
  - Model
* - {class}`.Sufficiency`{sup}`2`
  - ***Required***
  - ***Required***
  - *OD Only*
  - 
  - 
  - [*Task specific*](#computer-vision-task-compatibility)

```

````

````{tab-item} Data Selection
:sync: data_selection

For more information on a specific algorithm, click the name in the table.  
For an overview, see the {mod}`.data` page.

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
  - Model
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
  - *Optional*

```

````

`````

```{note}
{sup}`1` It is highly recommended to give [embeddings](Embeddings.md) over raw images using {class}`.Embeddings`.  
{sup}`2` Input data must be wrapped together in a `Dataset`.  
{sup}`3` When using only metadata, it must be wrapped in DataEval's {class}`.Metadata` class.  
{sup}`4` These scores are the raw outputs of a model.  
{sup}`5` These scores are retrieved by DataEval's {mod}`Out Of Distribution <.ood>` functions.  
```
