---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: dataeval
  language: python
  name: python3
---

# Identify bias and correlations

This guide provides a beginner friendly introduction to dataset bias, including balance, diversity and parity.

Estimated time to complete: 15 minutes

Relevant ML stages: Data Engineering

Relevant personas: Data Engineer, T&E Engineer

+++

## What you'll do

- Use DataEval to identify bias and correlations in the 2012 VOC dataset
- Analyze the results using plots and tables

+++

## What you'll learn

- You will see how to identify bias and correlations present in a dataset.
- You will understand the potential impact on your data and ways to mitigate them.

+++

## What you'll need

- Basic familiarity with Python
- Basic understanding of your dataset structure, including but not limited to its metadata
- An environment with DataEval installed

+++

## Introduction

Identifying any biases or correlations present in a dataset is essential to accurately interpreting your model's
performance and its ability to generalize to new data. A common cause of poor generalization is shortcut learning —
where a model uses secondary or background information to make predictions — which is enabled or exacerbated by dataset
sampling biases.

### Bias and correlations

Understanding biases or correlations present in your dataset is a key component to creating meaningful data splits. Bias
in data can lead to misleading conclusions and poor model performance on operational data. There are many different
[types of bias](https://arxiv.org/abs/1908.09635). A few of these biases occur during data collection, others occur
during dataset development, others occur during model development, while others are a result of the user.

Not all forms of bias directly affect the dataset and in order to address the biases that do, you have to make a few
assumptions:

1. All desired classes are present.
1. All available metadata is provided.
1. The metadata has been recorded correctly.

If any of the above assumptions are violated, then the analysis will not be accurate. When using your own data, you
should verify the above assumptions.

This guide does not focus on eliminating all bias, rather it focuses on identifying the bias that can be found when
developing a dataset.

### DataEval metrics

DataEval has three dedicated classes for identifying and understanding the bias or correlations that may be present in a
dataset: {class}`.Balance`, {class}`.Diversity` and {class}`.Parity`.

The `Balance` evaluator measures correlational relationships between metadata factors and classes by calculating the
mutual information between the metadata factors and the labels.

The `Diversity` evaluator measures the evenness or uniformity of the sampling of metadata factors over a dataset using
the inverse Simpson index or Shannon index.

The `Parity` evaluator measures the relationship between metadata factors and classes using a chi-squared test.

These techniques help ensure that when you split the data for your projects, you minimize things like shortcut learning
and leakage between training and testing sets.

+++

## Importing the necessary libraries

You'll begin by importing the necessary libraries to walk through this guide.

```{code-cell} ipython3
---
tags: [remove_cell]
---
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    %pip install -q dataeval dataeval-plots[plotly] maite-datasets
except Exception:
    pass
```

```{code-cell} ipython3
import dataeval_plots as dep
import plotly.io as pio
from maite_datasets.object_detection import VOCDetection

# Load the functions from DataEval that are helpful for bias
# as well as the VOCDetection dataset for the tutorial
from dataeval import Metadata
from dataeval.bias import Balance, Diversity, Parity

# Use plotly to render plots
dep.set_default_backend("plotly")

# Use the notebook renderer so JS is embedded
pio.renderers.default = "notebook"
```

## Step 1: Load the data

You are going to work with the PASCAL VOC 2012 dataset. This dataset is a small curated dataset that was used for a
computer vision competition. The images were used for classification, object detection, and segmentation. This dataset
was chosen because it has multiple classes and a variety of images and metadata.

If this data is already on your computer you can change the file location from `"./data"` to wherever the data is
stored. Remember to also change the download value from `True` to `False`.

For the sake of ensuring that this tutorial runs quickly on most computers, you are going to analyze only the training
dataset, which is a little under 6000 images.

```{code-cell} ipython3
# Download the 2012 train dataset and verify the size of the loaded dataset
ds = VOCDetection(root="./data", download=True, image_set="train", year="2012")
len(ds)
```

Before moving on, verify that the above code cell printed out 5717 for the size of the
[dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/dbstats.html).

This ensures that everything is working as needed for the tutorial.

+++

## Step 2: Structure the metadata

This guide focuses on evaluating labels and metadata of the dataset rather than the images themselves. As each dataset
has its own image and metadata formats, you will need to understand how your particular metadata is structured.

Start by taking a look at the metadata structure of the VOC 2012 dataset by creating a `Metadata` class from the
dataset.

```{code-cell} ipython3
# Extract the Metadata from the dataset
metadata = Metadata(ds)
```

The metadata in the dataset is provided as a dictionary entry for each datum, such that the aggregated data is a
collection of _N_ metadata dictionaries each with a nested list of _M_ objects in the image. Start by inspecting the raw
metadata of the first image.

```{code-cell} ipython3
metadata.raw[0]
```

:::\{note} `Metadata` is unable to process nested lists. For this dataset, _part_ is a factor that describes certain
parts of a _person_ object (such as _head_, _foot_ and _hand_), each with separate bounding box coordinates. You will
ignore this information for this example. :::

+++

:::\{note} The nested objects _horse_ and _person_ from the first metadata entry will be expanded to a complete metadata
entry for each object. :::

+++

Next you will want to select the factors to include for bias analysis as well as the continuous factor bins for any
continuous data.

```{code-cell} ipython3
metadata.include = [
    "image_width",
    "image_height",
    "segmented",
    "pose",
    "truncated",
    "difficult",
]

metadata.continuous_factor_bins = {
    "image_width": 5,
    "image_height": 5,
}
```

Now that the `Metadata` is ready to go, you can begin analyzing the dataset for bias!

+++

## Step 3: Assess dataset balance

+++

The {class}`.Balance` class measures correlational relationships between metadata factors and classes in a dataset. It
analyzes the metadata factors against both the classes and other factors to identify relationships.

The results can be retrieved using the _balance_ and _factors_ attributes of the output.

```{code-cell} ipython3
bal = Balance().evaluate(metadata)
```

The information provided by `Balance` may be visually understood with a heat map.

```{code-cell} ipython3
dep.plot(bal)
```

The heatmap shows that the greatest correlations are in the bounding box locations (_xmin_ with _xmax_ and _ymin_ with
_ymax_) and the image dimensions (_height_ and _width_).

Also the _ymax_ of the bounding box location is correlated with the _height_ of the image. It is not surprising that
_height_ and _width_ have correlation since many of the images are similarly sized.

The correlations between _xmin_ and _xmax_ and between _ymin_ and _ymax_ suggests that there is repetition in bounding
box width and height across the objects. However, the fact that _pose_ has a value of 0.08 with _class_ means that a few
of the classes have specific poses across a fair percentage of the images for that class. An example of this would be
most _pottedplant_ images having the same _pose_ value.

In addition to analyzing class and other factors, the balance function also analyzes metadata factors with individual
classes to identify relationships between only one class and secondary factors.

You can visualize the classwise results for balance by setting the _plot_classwise_ parameter to _True_.

```{code-cell} ipython3
dep.plot(bal, plot_classwise=True)
```

The classwise heatmap shows that factors other than _class_ do not have any significant correlation with a specific
class.

Classwise balance shows correlation of individual classes with all class labels, indicating relative class imbalance. In
this case the _person_ class is over-represented relative to most other classes.

This means that a model might learn a bias towards the _person_ class label due to its frequency in the training set,
which becomes a problem if the test/operational dataset doesn't have the same imbalance.

+++

## Step 4: Assess dataset diversity

+++

The {class}`.Diversity` evaluator measures the evenness or uniformity of the sampling of metadata factors over a
dataset. Values near 1 indicate uniform sampling, while values near 0 indicate imbalanced sampling, e.g. all values
taking a single value. For more information see the [Diversity](../concepts/Diversity.md) concept page.

The results can be retrieved using the _diversity_index_ attribute of the output.

```{code-cell} ipython3
div = Diversity().evaluate(metadata)
```

Again, it's often easiest to see the differences between the different factors when visualizing them using a bar chart
to show the factor-class analysis.

```{code-cell} ipython3
dep.plot(div)
```

In the results above, the factors _truncated_ and _occluded_ have values near 1, meaning that there is relatively little
or no bias in these factors.

The categories of most interest are those that are between 0.4 and 0.1 because this region represents skewed value
distributions for the factor.

The following factors fall into this category:

- _class_
- _width_
- _height_
- _segmented_
- _difficult_

These factors contain bias that should be addressed either by adding or removing data to even out the sampling. For
instance, the _class_ factor highlights that there is unevenness in the number of data points per class.

In addition to analyzing class, the diversity function also analyzes metadata factors with individual classes to assess
uniformity of metadata factors within a class. You can visualize the classwise results by setting the `plot_classwise`
parameter to True.

```{code-cell} ipython3
dep.plot(div, plot_classwise=True)
```

These results expand the above results on a classwise basis.

Things to look for here are large variances for a given factor across the different classes. For example, _pose_ has
values ranging from 0.01 to 0.84, which means that a few classes have almost uniform selection of the different _pose_
values while other classes essentially only have one _pose_ value. This makes sense as the _bottle_ or _pottedplant_
class does not have multiple _pose_ directions, while the _person_ class does.

What needs to be further investigated are things like whether the _sofa_ class should have a _pose_ direction, because a
diversity value of 0.4 means that a few of the images do while others do not.

Also, the _cat_ class has a low score signifying that most of the images fall into one or two categories rather than
being spread even across the categories. This highlights an error in the data collection process — the value was not
specified for most _cat_ images and therefore defaulted to "Unspecified".

An alternative error would be a dataset in which the _cat_ images have most cats facing a specific direction, which
would require additional data to overcome the bias, but that is not the case for this dataset. It has plenty of cats
facing each direction, but only a few of them contain a _pose_ value.

+++

## Step 5: Assess dataset parity

+++

The {class}`.Parity` evaluator measures the relationship between metadata factors and classes using a chi-squared test.
A high score with a low p-value suggests that a metadata factor is strongly correlated with a class label.

The results can be retrieved using the _score_ and _p_value_ attributes of the output.

```{code-cell} ipython3
par = Parity().evaluate(metadata)
```

The warning above states that the metric works best when there are more than 5 samples in each value-label combination.
However, because of the large number of total samples, the difference between 1 and 5 samples does not significantly
affect the results.

When evaluating the results of parity for a large number of factors, it may be easier to understand the results in a
DataFrame.

The {class}`.ParityOutput` class contains a `to_dataframe` function to format the results of the diversity function as a
DataFrame.

```{code-cell} ipython3
par.factors
```

According to the results, all metadata are correlated with _class_ labels. However, `parity` is based on the idea of an
expected frequency and how the observed differs from what is expected. The expected frequencies are determined by sums
of the values for each metadata category.

This function works best when the expected frequencies for a given factor for each individual class are known _a
priori_. For the case above, the expected frequency for the _pose_ metadata category shouldn't be the same for all
classes. The _diningtable_, _pottedplant_, and _bottle_ classes only have a single value for _pose_ which automatically
throws off the metric because not all of the classes have an identical expected frequency for _pose_.

+++

## Conclusion

+++

Having analyzed the dataset for bias with multiple metrics, the conclusion is that this dataset has bias. Training a
model on this dataset has the potential to learn shortcuts and underperform on operational data if the biases are not
representative of biases in the operational dataset.

The metadata categories identified by the `Balance`, `Diversity` and `Parity` evaluators contain issues such as
imbalanced classes and imbalanced parameters per class. DataEval isn't able to tell you exactly why they are imbalanced,
but it highlights the categories that you need to check.

As you can see, the DataEval methods are here to help you gain a deep understanding of your dataset and all of its
strengths and limitations. It is designed to help you create representative and reliable datasets.

Good luck with your data!

______________________________________________________________________

+++

## What's next

In addition to identifying bias and correlations in a dataset, DataEval offers additional tutorials to help you learn
about dataset analysis:

- To clean a dataset use the [Data Cleaning Guide](tt_clean_dataset.ipynb).
- To identify coverage gaps and outliers use the [Assessing the Data Space Guide](tt_assess_data_space.ipynb).
- To monitor data for shifts during operation use the [Data Monitoring Guide](tt_monitor_shift.ipynb).

To learn more about the balance, diversity and parity evaluators, see the [Balance](../concepts/Balance.md),
[Diversity](../concepts/Diversity.md) and [Parity](../concepts/Parity.md) concept pages.

## On your own

Once you are familiar with DataEval and dataset analysis, you will want to run this analysis on your own dataset. When
you do, make sure that you analyze all of your data and not just the training set.
