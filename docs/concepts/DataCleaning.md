# Cleaning Datasets

## What is Data Cleaning?

Data cleaning is the process of detecting and correcting inaccurate, incomplete, or irrelevant data from a dataset.
This process involves identifying and fixing errors, handling missing values, standardizing data formats, and removing duplicates.
The goal is to ensure that the data is accurate, consistent, and ready for analysis.

## Why Should You Clean Your Data?

When training models, the quality of the dataset can greatly impact the usefulness of the model.
The goal of machine learning and AI is to create robust, reliable, and generalizable models.
A dataset contributes to the usefulness of a model in the following ways:

1. **Accuracy**:
Data cleaning improves the accuracy or representation of your dataset.
Dirty data, which may include incorrect or inconsistent entries, can result in misleading analyses and incorrect conclusions.

2. **Reliability**:
Clean data enhances the reliability and reproducibility of your results.
When data is consistent and free of errors, your analysis is more likely to be replicable and trustworthy.

3. **Efficiency**:
Clean data allows for more efficient processing and analysis.
By removing irrelevant or duplicate data, you reduce the computational load and time.
This can lead to both faster and more targeted improvements as your model is able to focus on the most important information.

4. **Improved Insights**:
Data cleaning helps reveal true patterns and relationships within the data.
With clean data, your insights are more likely to be valid and actionable.
Explainability is an important concept in the AI world, and having pristine
data improves the translation of how and why the model works.

## When Should You Clean Your Data?

Data cleaning should be an ongoing process throughout the data lifecycle,
but there are specific times when it becomes particularly crucial:

1. **During Data Collection**:
Data collection and entry are key areas to ensure that you have accurate and reliable data.
Consider implementing data validation techniques through built-in redundancies and multi-person verification,
such as having multiple data recorders and/or different collection and entry people.
This will greatly reduce the need for extensive cleaning later.

2. **After Merging Datasets**:
When combining multiple datasets, data cleaning is essential to ensure consistency and 
avoid issues like duplicate entries or conflicting formats.

3. **Before Analysis and Reporting**:
Always clean your data before performing any statistical analysis or model building.
Consider having a subject matter expert verify sample statistics with known population dynamics to ensure that the data is representative.
For images, the data cleaning process should include verifying image properties (like size and metadata)
as well as the visual aspects of the image (like brightness and sharpness). 
This ensures that your results are based on accurate and relevant information.

## Inaccurate, Incomplete and/or Irrelevant Data

Having inaccurate, incomplete and/or irrelevant data leads to unreliable conclusions which can range in the severity of the consequences.
These consequences include scenarios like your model mislabelling an animal because you mixed up the labels or losing money because your market analysis left out a key competitor.
Some examples for each type of data error are shown below:
* Inaccurate data
    * Typographical errors such as misspelled words or incorrect entries
    * Formatting errors such as inputting the data in the wrong units
    * Representation errors such as combining two important groups or splitting out a subset which shouldn't be 
    * Anomalies/Outliers
* Incomplete data
    * Missing values or entire groups
    * Partial records
    * Lacking enough samples in a group 
* Irrelevant data
    * Including data or groups that are not related to the task at hand
    * Outdated information

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
This principle emphasizes that the quality of output from analysis and models is 
determined by the quality of input data.
If the input data is flawed, the output will be unreliable,
regardless of the sophistication of the analysis.


## Data Cleaning with DataEval

DataEval is a data analysis and monitoring library with some dedicated functions and classes for data cleaning.

DataEval's data cleaning functions and classes are:
* [dimensionstats](Stats.md#dimensionstats) function,
* [hashstats](Stats.md#hashstats) function,
* [pixelstats](Stats.md#pixelstats) function,
* [visualstats](Stats.md#visualstats) function,
* [Clusterer](Clusterer.md) class,
* [Duplicates](Duplicates.md) class, and
* [Outliers](Outliers.md) class.

These functions and classes facilitate the creation of dataset statistics and
the identification of abnormal data points and duplicates. 
The **hashstats** function creates image hashes on a per image basis.
The **dimensionstats** function creates dataset statistics on a per image basis.
The **pixelstats** and **visualstats** functions create dataset statistics on a per image per channel basis.
The **Clusterer** class clusters the data and identifies data points which do not fit into a cluster.
The **Duplicates** class identifies duplicate images.
The **Outliers** class analyzes the dataset statistics for outliers based on the chosen statistical method.

To see data cleaning in action using DataEval, check out our [Data Cleaning Guide](../tutorials/EDA_Part1.ipynb).

(data-clean-metrics)=
### Data Cleaning Metrics

Below is a list of all of the metrics available for analysis and the category the stats metric belongs to.

* hashstats
    * xxhash
    * pchash

* dimensionstats
    * width
    * height
    * size
    * aspect_ratio
    * channels
    * depth

* visualstats
    * brightness
    * darkness
    * contrast
    * sharpness
    * missing
    * zero

* pixelstats
    * mean
    * std
    * var
    * skew
    * kurtosis
    * entropy
    * percentiles
    * histogram
