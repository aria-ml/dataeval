# Dataset Cleaning

## What is Data Cleaning?

Data cleaning is the process of detecting and correcting inaccurate,
incomplete, or irrelevant data from a dataset. This process involves
identifying and fixing errors, handling missing values, standardizing data
formats, and removing {term}`duplicate<Duplicates>`. The goal is to ensure
that the data is accurate, consistent, and ready for analysis.

## Why Should You Clean Your Data?

When training models, the quality of the dataset can greatly impact the
usefulness of the model. The goal of
{term}`machine learning<Machine Learning (ML)>` and AI is to create robust,
reliable, and generalizable models. A dataset contributes to the usefulness of
a model in the following ways:

1. **Accuracy**:
   Data cleaning improves the {term}`accuracy<Accuracy>` or representation of your
   dataset. Dirty data, which may include incorrect or inconsistent entries, can
   result in misleading analyses and incorrect conclusions.

2. **Reliability**:
   Clean data enhances the reliability and reproducibility of your results. When
   data is consistent and free of errors, your analysis is more likely to be
   replicable and trustworthy.

3. **Efficiency**:
   Clean data allows for more efficient processing and analysis. By removing
   irrelevant or duplicate data, you reduce the computational load and time. This
   can lead to both faster and more targeted improvements as your model is able
   to focus on the most important information.

4. **Improved Insights**:
   Data cleaning helps reveal true patterns and relationships within the data.
   With clean data, your insights are more likely to be valid and actionable.
   Explainability is an important concept in the AI world, and having pristine
   data improves the translation of how and why the model works.

## When Should You Clean Your Data?

Data cleaning should be an ongoing process throughout the data lifecycle,
but there are specific times when it becomes particularly crucial:

1. **During Data Collection**:
   Data collection and entry are key areas to ensure that you have accurate and
   reliable data. Consider implementing data validation techniques through
   built-in redundancies and multi-person verification, such as having multiple
   data recorders and/or different collection and entry people. This will greatly
   reduce the need for extensive cleaning later.

2. **After Merging Datasets**:
   When combining multiple datasets, data cleaning is essential to ensure
   consistency and avoid issues like duplicate entries or conflicting formats.

3. **Before Analysis and Reporting**:
   Always clean your data before performing any statistical analysis or model
   building. Consider having a subject matter expert verify sample statistics with
   known population dynamics to ensure that the data is representative. For
   images, the data cleaning process should include verifying image properties
   (like size and metadata) as well as the visual aspects of the image (like
   {term}`brigtness<Brightness>` and {term}`blurriness<Blur>`). This ensures that
   your results are based on accurate and relevant information.

## Inaccurate, Incomplete and/or Irrelevant Data

Having inaccurate, incomplete and/or irrelevant data leads to unreliable
conclusions which can range in the severity of the consequences. These
consequences include scenarios like your model mislabelling an animal because
you mixed up the labels or losing money because your market analysis left out a
key competitor. Some examples for each type of data error are shown below:

- Inaccurate data
  - Typographical errors such as misspelled words or incorrect entries
  - Formatting errors such as inputting the data in the wrong units
  - Representation errors such as combining two important groups or splitting
    out a subset which shouldn't be
  - Anomalies/Outliers
- Incomplete data
  - Missing values or entire groups
  - Partial records
  - Lacking enough samples in a group
- Irrelevant data
  - Including data or groups that are not related to the task at hand
  - Outdated information

Because of the variety of ways in which data quality can degrade,
the process for correcting inaccurate, incomplete and/or irrelevant data
depends greatly on the data itself and the context in which it is being used.
This can could involve simply adding in a missing group or value,
using a spell checker or a second pair of eyes to catch data entry errors,
removing a data point or group from the analysis, or something else entirely.

## Theory Behind Data Cleaning

Data cleaning is grounded in the principles of data quality and data integrity.
A guiding principle for data cleaning is _garbage in, garbage out_.

**Garbage In, Garbage Out Principle**:
This principle emphasizes that the quality of output from analysis and models
is determined by the quality of input data. If the input data is flawed, the
output will be unreliable, regardless of the sophistication of the analysis.

## Data Cleaning Metrics

DataEval's data cleaning functions and classes are:

```{list-table}
:widths: 20 80
:header-rows: 0


* - {func}`boxratiostats <.boxratiostats>`
  - Compares the statistics between a set of bounding boxes and their
    corresponding images.
* - {func}`dimensionstats <.dimensionstats>`
  - Creates dataset statistics on a per image basis.
* - {func}`hashstats <.hashstats>`
  - Creates hex-encoded image hashes on a per image basis.
* - {func}`imagestats <.imagestats>`
  - Runs `dimensionstats`, `pixelstats`, `visualstats`, and `labelstats`
    functions on a given dataset.
* - {func}`labelstats <.labelstats>`
  - Creates dataset statistics on the labels.
* - {func}`pixelstats <.pixelstats>`
  - Creates dataset statistics on a per image per channel basis.
* - {func}`visualstats <.visualstats>`
  - Creates dataset statistics on a per image per channel basis.
* - {func}`.clusterer`
  - Clusters the data and identifies data points which do not fit.
* - {class}`.Duplicates`
  - Identifies duplicate and near duplicate images.
* - {class}`.Outliers`
  - Analyzes the dataset statistics for Outliers based on the chosen
    statistical method.
```

To see data cleaning in action using DataEval, check out our
[Data Cleaning Guide](../notebooks/tt_clean_dataset.ipynb).

See the [Stats](Stats.md) concept page to learn more about the
algorithms/methods used by the functions above.

See the [Clusterer](Clustering.md) and [Outliers](Outliers.md) concept pages to
learn more about their algorithms and the stat functions used by them.

## Duplicate Detection

With the {term}`Duplicates`, exact matches are found using a byte hash of
the data information, while near matches (such as a crop of another image or a
distoration of another image) use a perception based hash.

The byte hash is achieved through the use of the
[python-xxHash](https://github.com/ifduyue/python-xxhash) Python module,
which is based on Yann Collet's [xxHash](https://github.com/Cyan4973/xxHash) C
library.

The perceptual hash is achieved on an image by resizing to a square NxN image
using the Lanczos algorithm where N is 32x32 or the largest multiple of 8 that
is smaller than the input image dimensions. The resampled image is compressed
using a discrete cosine transform and the lowest frequency component is encoded
as a bit array of greater or less than median value and returned as a hex
string.
