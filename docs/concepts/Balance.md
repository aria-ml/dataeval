# Balance

## What is it

{term}`Balance` and classwise balance are metrics that measure correlational
relationships between metadata factors and class labels.  Balance and classwise
balance can indicate opportunities for shortcut learning and disproportionate
dataset sampling, with respect to individual classes or between metadata factors, that
leads to poor generalization or overestimation of model performance.

Balance metrics compute the {term}`mutual information<Mutual Information (MI)>` between class labels and metadata
factors which may include intrinsic metadata, such as image statistics or
bounding box statistics; or extrinsic metadata factors such as environmental
information, sensor information, and operational information.  Mutual
information measures the information gained about one variable, e.g. class
label, by observing another variable, e.g. location.  Balance metrics provide a
T&E engineer or model developer insight into dataset relationships which should
be investigated or accounted for during model training or model evaluation.

The {term}`balance<Balance>` metric returns mutual information computed between metadata factors
and class labels but also mutual information between all pairs of metadata
factors to characterize inter-factor relationships.  The `balance_classwise`
metric returns mutual information between individual class labels and metadata
factors to identify relationships between only one class and secondary factors.


## When to use it

For both model and dataset development it is important to understand
correlational relationships that underlie the dataset.  Often, opportunities for
data collection are sparse, available only in non-operational locations and
conditions, with limited target {term}`diversity<Diversity>`, etc.  A model trained on these
realistic datasets could learn to use secondary information to perform the
primary learning task, reducing the model's ability to generalize to new domains
or to perform unexpectedly when presented with new data.  {term}`balance<Balance>` metrics
provide a method for identifying relationships between dataset factors and class
labels _a priori_.  A T&E engineer or model developer should then use that information to
design tests for model generalization or data augmentation to mitigate the
opportunity for shortcut learning or sampling imbalance.

In order to use {term}`balance<Balance>`, the user must supply their metadata in a DataEval
specific format. Because of this requirement, DataEval has a `preprocess` function
that will take in user [metadata](Metadata.md) and format it into DataEval's format. The balance function takes
in the output of the {func}`.preprocess` function for its analysis.

### Identifying opportunity for shortcut learning

The literature contains many [examples](https://arxiv.org/pdf/2004.07780) of
shortcut learning and adversarial perturbations that cause a model to fail to
generalize.  For instance, an image classifier may learn to detect cows in a
grassy field but internally cues on the grassy field (background) rather than the
properties of the cow.  When presented with an image of a cow at a sandy beach,
the model fails to identify the cow because it was able to previously use
secondary information about location or visual context to reliably identify
cows.  {term}`Balance` metrics are one way to identify the *potential* for learning such
shortcuts.

### Identifying disproportionate dataset sampling
In addition to identifying possible shortcuts, {term}`balance<Balance>` metrics may identify
issues where data are sampled disproportionately with respect to a particular
factor.  For instance, in the example above where the model is trained on images
where cows nearly always appear in a grassy field, classwise balance, would show
a strong relationship between the `cow` class label and grassy field
environment, provided that background information is encoded in the metadata.
Given the apparent correlation between the `cow` class label and grassy field
background, a model developer or T&E engineer should first assess whether the
correlation is problematic and whether the dataset should be resampled, further
data collected in other environments, or augmentation techniques used to
mitigate the apparent {term}`bias<Bias>`.

Not all dataset correlation and sampling biases are problematic, however.  For
instance, it may be expected that elevation of the sun correlates with day of
the year, and we do not expect this relationship to bias our model performance.
Or, consider a case where different sensors are available in different
geographic regions, leading to a correlational relationship between
location/region and sensor.  A subject matter expert could determine, given
properties of the data and sensors, whether this relationship is problematic and
whether data need to be augmented for training and evaluation.

It is important to note that correlational relationships within a dataset
measured by balance metrics only indicate *opportunity* for shortcut learning;
balance and other metrics within {term}`DataEval` do not measure whether shortcut
learning has occurred.  It is important to interrogate potential biases
exhibited by the trained model and to assess the need for further data
augmentation to mitigate or compensate observed biases.

## Theory behind it

{term}`Mutual information<Mutual Information (MI)>` is a metric that is often used for measuring the quality of
dataset clustering or for feature selection, and there are several formulations
to measure relationships between two Categorical Variables, between categorical
and continuous variables, and between two continuous variables.  We consider
class label a categorical variable, as there is typically no presumed ordering
between classes; however, other metadata factors, such as latitude and longitude
or time stamps may take continuous (ordered) values.

The implementation of mutual information within {term}`DataEval` draws on multiple
implementations within the scikit-learn package including
[`mutual_info_classification`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)
and
[`mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html).
For categorical or discrete target variables, `mutual_info_classif` computes the
mutual information with respect to both discrete/categorical and continuous
factors.  DataEval attempts to infer whether a variable is continuous or
discrete by the fraction of unique values present&mdash;i.e. whether the data
may be binned uniquely with a relatively small number of bins.

{term}`Mutual information<Mutual Information (MI)>` between categorical/discrete variables is computed from
contingency tables which measure co-occurence of each variable, while mutual
information involving continuous (ordered) data is computed using the k-nearest
neighbor (KNN) graph as in Refs. [1] and [2].

### Normalization

Raw {term}`mutual information<Mutual Information (MI)>` scores are difficult for a human to contextualize, so
{term}`balance<Balance>` metrics normalize the mutual information by the arithmetic mean of
marginal entropies of each variable.  Given that some variables could have a
marginal entropy of zero (all values the same), the arithmetic mean is somewhat
preferable over the geometric mean in those cases.

Currently, entropies are computed over unique values for categorical variables
and over binned values for continuous variables.  Since the KNN representation
used to compute mutual information is not necessarily consistent with the
histogram representation used to compute marginal entropies it is possible for
`balance` to return normalized mutual information greater than 1.  However, most
values will lie in the interval [0, 1].  A value near or above 1 indicates a
high degree of correlation, and a value near zero indicates little measured correlation.

Normalized mutual information is not adjusted for chance and may lead to larger
values than might be expected.  In particular, the normalized mutual information
associated with random label assignments is not in general 0 and may lead to
overestimated normalized mutual information [3].  Adjusted forms of mutual
information [are implemented](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score) for the categorical-categorical case with
probabilities computed from a contingency table but, since we admit continuous
variables as well, normalized mutual information is the value reported by
`balance` and `balance_classwise`.


## References

[1] [Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating
mutual information." _Physical Review E—Statistical, Nonlinear, and Soft Matter
Physics_ 69, no. 6 (2004): 066138.](https://arxiv.org/abs/cond-mat/0305641)

[2] [Ross, Brian C. "Mutual information between discrete and continuous data sets." _PloS one_ 9, no. 2 (2014): e87357.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357)

[3] [Vinh, N. X., J. Epps, and J. Bailey. "Information Theoretic Measures for
Clusterings Comparison: Variants: Properties, Normalization and Correction for
Chance" _Journal of Machine Learning Research_ 11 (2010).](https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf)