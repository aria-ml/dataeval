# Diversity

## What is it

{term}`Diversity` indices and classwise diversity indices measure the evenness
or uniformity of the sampling of metadata factors over a dataset. Diversity
indices identify whether intrinsic or extrinsic metadata factors are sampled
disproportionately to others, which may indicate sources of sampling
{term}`bias<Bias>`. Even, or uniform, sampling with respect to class labels is
often referred to as stratification, but it may also be important to understand
whether a dataset is sampled uniformly with respect to metadata and contextual
variables.

Diversity indices are normalized measures of uniformity taking values on the
interval [0, 1]. DataEval offers two diversity indices&mdash;the inverse
Simpson diversity index and the normalized Shannon entropy. Values near 1
indicate uniform sampling, while values near 0 indicate imbalanced sampling,
e.g. all values taking a single value.

Classwise diversity indices measure uniformity of metadata factors among
samples within a class and may be a useful diagnostic metric for assessing
classwise sampling bias in a dataset.

## When to use it

{term}`Diversity` indices can be used as a diagnostic metric during dataset
development and during model development and evaluation. Much like class
imbalance, sampling imbalance with respect to environment, context, or other
factors could lead to poor generalization or otherwise poor model performance
and are important to understand for both model developers and test and
evaluation (T&E) engineers.

For dataset development and test and evaluation, diversity indices may inform
{term}`dataset splits<Dataset Splits>` or measure the quality of the splits by
identifying sources of sampling {term}`bias<Bias>` across training, validation,
and test splits. Diversity indices may also be used during T&E in order to
contextualize model performance results by identifying sources of sampling
biases encoded in the model through training and validation datasets.
Similarly, model developers may use diversity indices to develop and sample
their training data or to contextualize results during iterative model
development.

In order to use {term}`diversity<Diversity>`, the user must supply their
metadata in a DataEval specific format. Because of this requirement, DataEval
has a `Metadata` class that will take in user [metadata](Metadata.md) and
format it into DataEval's format. The `diversity` function takes in the
`Metadata` class for its analysis.

## Theory behind it

### Simpson diversity index

The inverse Simpson diversity index is given by

$$
d = \frac{1}{N \sum_i^N p_i^2},
$$

where $p_i$ are discrete probabilities for bin $i$ where $p_i \neq 0$. When the
data take $N$ unique values and are uniformly sampled, $p_i = 1/N$ for $ i =
1\ldots N$, which makes $d = 1 / \left( \sum_i^N 1/N\right) = 1$. The minimum
value of the Simpson diversity index is $d = 1/N$ when $p_i = 1, p_j = 0 \,
\forall j \neq i$. The metric reported by DataEval, $d'$, is rescaled linearly
to the interval $[0, 1]$ using

$$
d' = \frac{d - (1/N)}{1 - (1/N)} = \frac{d N - 1}{ N - 1}.
$$

For data with few unique factors, the unscaled diversity index can take
relatively high minimum values, e.g. 0.5 for $N = 2$. Linear rescaling expands
the range of values to the more intuitive unit interval, removes dependence on
number of classes or unique values, and enables consistent treatment of
limiting values for both Shannon and Simpson diversity indices.

### Shannon diversity index

The Shannon diversity index is given by

$$
d = - \frac{1}{\log N}\sum_i^N p_i \log p_i,
$$

where the typical Shannon entropy has been normalized by its maximum value,
$\log N$. Like the Simpson diversity index and its rescaled version, the
Shannon diversity index takes a maximum value of 1 when all $p_i$ are equal.
In particular, for $p_i = 1/N$,

$$
d  = \frac{1}{\log N} \sum_i^N \frac{\log N}{N} = 1
$$

Strongly asymmetric distributions take values close to 0, and the minimum
entropy distribution where $p_i = 1, p_j = 0 \,\forall j \neq i$ leads to a
scaled entropy of 0.

### Edge cases

Real-world datasets often result in strange distributions of metadata factors,
resulting in diversity indices that can take somewhat counterintuitive values.
For instance, in the case of one unique value or a single class, it may not be
obvious that diversity indices should convey perfect uniformity (1) or strong
asymmetry (0). The value of the unscaled inverse Simpson diversity index when
all data take a single value is 1; however, the rescaling instead maps the
index to 0.

The Shannon entropy approaches 0 as the PDF approaches a single bin, i.e. $1
\log(1)= 0$. Variables have a single value with probability 1 indicating no
uncertainty. Normalizing the entropy by its maximum value of $\log N$
could lead to ambiguity in this case due to an indeterminate form; however, in
DataEval the ambiguity is resolved by assigning the limit of the unnormalized
entropy&mdash;i.e. not reversing the interpretation of the single-bin entropy
through a normalization artifact.

When distributions collapse to a single bin, data take a single value with
probability 1. There is no uncertainty (entropy) and no diversity, which is
reflected by the metric.
