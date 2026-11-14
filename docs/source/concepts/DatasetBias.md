<!-- markdownlint-disable MD051 -->

# Dataset bias

This page describes how DataEval measures dataset bias with a dataset's metadata factors and class labels. It covers
the association between class labels and factors, the evenness of sampling across a factor, the test of that
association and the correction of an imbalance. It also states what each result does and does not establish. Balance,
evenness and the association between labels and factors describe the dataset's own sampling. By themselves they do not
show that the dataset matches the intended operational population, which the practitioner defines.

Dataset readiness assessment considers five properties with respect to a dataset's known factors. Each is judged
against the operational population and is not an ideal that every dataset must meet.

- Balanced: class labels are statistically independent of metadata factors, so that no condition, sensor, or context
  is systematically associated with a particular class. Independence is a reference point and not a requirement: an
  association that also holds in the operational population is not by itself evidence of bias.
- Covered: samples populate the relevant regions of the feature space, leaving no gap that the model will encounter in
  deployment but has not seen in training.
- Complete: the dataset spans the full dimensionality of the embedding space;
  samples are not so similar to each other that they effectively occupy only a
  fraction of the representational capacity available.
- Representative: the sampling distribution reflects the operational
  distribution; the conditions, frequencies, and contexts present in the dataset
  match those the model will encounter after deployment.
- Relevant: every sample belongs to the problem domain; the dataset contains no
  off-domain images, misassigned sources, or annotation artifacts that introduce
  class structure the model should not learn.

The tools described on this page assess the first property. Balance and Diversity assess whether the dataset
is balanced, and [Parity](Parity.md) tests it for each factor level and class. [Dataset coverage](DatasetCoverage.md)
describes the tools that assess whether it is covered and complete. No tool on either page computes
representativeness or relevance. They are judged by combining metadata analysis with the operational context that
the practitioner supplies.

## What dataset bias is

Dataset bias occurs when a systematic property of how data was collected or labeled causes the training distribution
to diverge from the true operational distribution in a way that can reduce model performance on that distribution.

Bias is systematic. Random noise in a large dataset tends to average out, whereas bias shifts the model's decision
boundaries in a consistent direction. A model trained on biased data can perform well on data that resembles the
training set and poorly on data that does not.

[Friedman & Nissenbaum (1996)](#ref2) grouped bias into: pre-existing bias
(inherited from social or institutional practice), technical bias (introduced by
design choices in the system), and emergent bias (arising from deployment in a
context the system was not designed for). This is a useful framework for
describing the origins of bias. This page uses a distinction based on where in the
data pipeline the bias entered:

**Sampling bias** arises when the mechanism of data collection systematically
favors certain conditions over others. A dataset assembled from a single proving
ground, a single sensor platform, or a single collection window can oversample
those conditions and undersample everything else, because of what was collected
and not because of any flaw in what was labeled.
[Torralba & Efros (2011)](#ref9) showed that this effect appears across
widely used benchmark datasets: classifiers trained on one dataset perform well
within it but transfer poorly to others. This is consistent with the collection
conditions acting as a latent variable that the model exploits.

**Representation bias** is the distributional consequence of sampling choices,
but it is distinct enough to treat separately because it can arise without any
flawed sampling mechanism. Certain classes, conditions, or groups may be
under-represented because the target phenomenon is rare, because collection
constraints limited access, or because systematic gaps in what was deemed worth
collecting were never recognized as gaps. A dataset can be sampled correctly
from a convenience population and still be unrepresentative of the operational
distribution. [Fabbrizzi et al. (2022)](#ref1) survey a taxonomy of
representation gaps in visual datasets and document how frequently they appear
even in widely used benchmarks. [Merler et al. (2019)](#ref7) illustrate the
downstream consequences in face recognition, where representation gaps across
demographic groups produced systematic performance disparities that aggregate
accuracy metrics can hide.

**Label bias** arises when the annotation process introduces systematic errors.
Human annotators may apply labels inconsistently across classes, across
annotators, or across contextual conditions. When certain classes are
consistently annotated more carefully than others, or when annotator
disagreement correlates with a metadata factor (time of day, image quality,
object size), the learned decision boundary can reflect annotation practice and not
ground truth.

Any form or combination of bias can produce **shortcut learning** in the trained
model. When a metadata factor, such as background environment, sensor platform,
weather condition, or time of day, is consistently associated with a class label
in training data, the model can achieve low training loss by learning the
metadata correlation rather than the intended feature.
[Geirhos et al. (2020)](#ref3) treat this failure mode in computer vision, with
the example of models that classify "cow" by detecting grass rather than the
animal. In real world datasets, analogous shortcuts can occur: target detection
models that learn sensor platform as a proxy for target class, or that perform
well only in the lighting conditions of the original collection campaign.

## Theory

### Five questions about factors and classes

An analysis of a dataset's known factors addresses one of five questions. The table gives the tool for each.

:::{list-table}
:widths: 24 76
:header-rows: 1

- - Question
  - Where it is answered
- - Class association: does a factor tell you the class?
  - The `balance` table of {class}`.Balance`, which scores each factor against the class label, and {class}`.Parity` ⚠️,
    which tests each factor and each combination of a level and a class.
- - Within-class sampling: within one class, are the factor's values sampled evenly?
  - The `classwise` tables of Balance and {class}`.Diversity`, and the cells of Parity.
- - Operational mismatch: do the dataset's factors match the conditions of deployment?
  - None of these tools. They read one dataset, and the comparison needs the operational values. See
    [Distribution shift](DistributionShift.md).
- - Confounds: is an association with the class shared with another factor?
  - The `factors` table of Balance, which scores each pair of factors. It does not say which of two associated factors
    the class depends on.
- - Missingness: is a factor unrecorded for some rows, and does that track the class?
  - A missing value in a binned factor gets a bin of its own, so missingness that tracks the class shows as
    association with that level. See
    [Missing values get a bin with a position but no meaning](Binning.md#missing-values-get-a-bin-with-a-position-but-no-meaning).

:::

### Measuring bias: mutual information and evenness

{class}`.Balance` and {class}`.Diversity` measure bias using metadata, the contextual variables attached to each
sample (sensor platform, weather, time of day, location, annotator ID, and so on). Balance tests whether class labels
are independent of these factors. Diversity measures how evenly samples are spread across the values of each factor.

Balance measures the statistical dependence between class labels and
metadata factors or two metadata factors using
{term}`mutual information (MI) <Mutual Information (MI)>`. Mutual information
quantifies how much knowing one variable reduces uncertainty about another:

$$I(Y; M) = \sum_{y, m} P(Y=y, M=m) \log \frac{P(Y=y, M=m)}{P(Y=y)\, P(M=m)}$$

where $Y$ is the class label (or the second metadata factor) and $M$ is a metadata
factor. When $Y$ and $M$ are independent, $I(Y; M) = 0$. When knowing $M$ completely
determines $Y$, $I(Y; M)$ equals the entropy of $Y$.

Raw MI values are difficult to interpret, so DataEval provides normalized MI
(NMI), a value between 0 and 1. The type of normalization that
DataEval applies depends on the type of data passed in.

If both factors are unbinned discrete factors, then DataEval divides by the
smaller of the two marginal entropies:

$$I_\text{norm}(M_1; M_2) = \frac{I(M_1; M_2)}{\min(H(M_1), H(M_2))}$$

The mutual information can never exceed the smaller of the two discrete
entropies, so dividing by the minimum keeps the result in $[0, 1]$ and reaches 1
if and only if either factor determines the other.

If one factor is unbinned discrete and the other continuous, DataEval normalizes
by the entropy of the discrete variable.

If neither variable is unbinned discrete (that is, both are continuous, both are binned discrete, or one of each),
the mutual information is not bounded and cannot be normalized, so DataEval returns a transformation of the MI onto the
interval [0, 1]. For two jointly normal variables, the transformed value is the square of their correlation
coefficient ([Linfoot, 1957](#ref11)).

Between the class label and a factor the question is directed (how much of the
class label does this factor account for), and every factor is measured against
one common denominator, the class label's own entropy:

$$I_\text{norm}(Y; M) = \frac{I(Y; M) - \mathbb{E}[I(Y; M)]}{H(Y) - \mathbb{E}[I(Y; M)]}$$

Dividing every factor by the same class entropy is what makes these values
comparable to each other, so the result can be read as a ranking of the factors.
Dividing each factor by its own entropy instead would reward a factor for having
few categories rather than for being informative: a coarse grouping of the class
label reaches 1.0 under the minimum while accounting for only a fraction of what
the class label conveys, and would outrank a factor that explains more.

The correction below is subtracted from the denominator as well, and it grows
with a factor's cardinality, so the denominator is not identical across factors.
The effect is small over the range a binned factor occupies and grows from
there, which means the ranking holds between factors of comparable width and a
very wide factor receives a lower score than an uncorrected measure would give.

$\mathbb{E}[I(Y; M)]$ is the mutual information expected under a random
assignment holding the same margins, and subtracting it corrects for chance.
Mutual information estimated from a contingency table grows with the number of
categories even when the two variables are independent: every extra category is
another opportunity for the counts to line up by luck, and with finite data some
of them do. Uncorrected, an identifier column taking a distinct value per sample
can outrank a genuine effect. With the correction an unrelated factor scores
near zero regardless of its cardinality, and the score of an informative factor
changes little. The correction can make the numerator negative for an unrelated
pair, and the reported value is clipped to [0, 1]. The same correction is
applied between two metadata factors, over the smaller of their two entropies
rather than the class entropy. Without it, a factor binned finely enough can
report a correlation with everything it is measured against.
[Vinh et al. (2010)](#ref10) analyze normalization schemes for
information-theoretic association measures, including their behavior under
chance and the conditions under which each is most appropriate.

Values close to 1 indicate high dependence; values close to 0 indicate near-independence. When one variable has zero
entropy (a degenerate factor that takes a single value), DataEval sets the mutual information and its normalized value
to zero.

{class}`.Balance` computes NMI at two levels. The **global** output measures the
dependence between each metadata factor and the class labels across the full
dataset. The **classwise** output measures the dependence between each metadata
factor and each individual class, identifying cases where a factor is associated
with one class but not others. This distinction matters: a factor like "time of
day" might show low global NMI while still being strongly associated with a
specific rare class whose samples were all collected at dawn.

{class}`.Balance` also computes **inter-factor NMI**, the pairwise dependence
between metadata factors themselves. Highly correlated factors (for example, if
location and sensor platform are nearly perfectly associated) provide redundant
information and complicate the interpretation of individual factor effects.

MI is computed differently for a factor read as codes and for one read as measured values, and
{class}`.Balance` can read either. Codes, which are bin and category indices, are tabulated and corrected for chance.
Measured values go to the neighbor-based estimators of scikit-learn ([Kraskov et al., 2004](#ref6);
[Ross, 2014](#ref8)), which need no bins and are not corrected for chance.
`factor_source` selects the reading. Its default, `"auto"`, reads a factor as codes where someone chose its cut and as
measured values otherwise, and always reads a category as codes. The `scored_as` column of `factors` records the regime
of each pair, and [Choosing bin edges](BinEdges.md#binning-reaches-the-three-outputs-differently) compares them.
DataEval infers discreteness from the proportion of unique values in a factor. Discreteness is a statement about a
factor's support and does not imply that the support is small, so a numeric
factor that reads as discrete but takes more distinct values than the sample can
fill, such as pixel areas, timestamps, or a non-reserved identifier-like column,
is binned as well, against the same budget a histogram would use. Without that,
the contingency table would hold more cells than there are observations to fill
them. A non-numeric factor has no line to place edges on, so it keeps one
category per distinct value however many there are; a near-unique string column
that is not a reserved id is best dropped from the metadata rather than
measured. The datum's own `id` never reaches this question: it is a reserved
identifier carried on the `item_id` column, validated unique, and excluded from
the factor set.

```{note}
Both directions are adjusted for chance, so a factor with many unique values no
longer reports inflated association for being wide. What the correction cannot
do is put the two directions on one scale: a class-to-factor value is a share of
the class entropy and an inter-factor value is a share of the smaller of two
factor entropies, so compare values within a table rather than between them.
```

Diversity measures the evenness of sampling across each metadata factor,
independently of class labels. Where Balance asks "is class label correlated
with this factor?", Diversity asks "are samples spread uniformly across this
factor's values?"

DataEval supports two diversity indices, both drawn from ecological diversity
measurement ([Hill, 1973](#ref5); [Heip et al., 1998](#ref4)). The **inverse
Simpson index** is:

$$d = \frac{1}{N \sum_i p_i^2}$$

where $p_i$ are the discrete probabilities of each bin and $N$ is the number of
unique values. This is linearly rescaled to $[0, 1]$:

$$d' = \frac{dN - 1}{N - 1}$$

The rescaling removes dependence on $N$, making values comparable across factors
with different numbers of categories.

The **normalized Shannon entropy** is:

$$d = -\frac{1}{\log N}\sum_i p_i \log p_i$$

Both indices reach 1 when samples are uniformly distributed and 0 when all
samples belong to a single category. The Simpson index is more sensitive to
dominant categories; the Shannon index weights all categories more equally. The
default is Simpson.

Diversity is computed both globally across the full dataset and classwise within
each class. Low classwise diversity for a particular class and factor (for
example, all "aircraft" samples collected at the same location) can indicate
collection bias for that class.

### Testing association: Parity

{class}`.Parity` ⚠️ tests the factor–class dependence that Balance measures.
For each factor it tests association with the class labels, and for each
combination of a factor level and a class it tests whether the cell holds more
or fewer rows than independence predicts. Its p-values account for rows grouped
into images or sequences. Its `underrepresented` cells are collection
candidates, and its `overrepresented` cells point to possible shortcuts. It
does not measure coverage. The method, its assumptions and its calibration are
described on the [Parity](Parity.md) page.

### Correcting for imbalance: ClassBalance

{class}`.ClassBalance` is a dataset selection tool rather than a measurement
tool. Once Balance or Diversity analysis has revealed class imbalance,
`ClassBalance` produces a rebalanced subset for training or evaluation without
requiring new data collection.

Two strategies are available. The **global** method samples images with
probability proportional to the inverse square root of class frequency, giving
rare classes higher weight while still allowing common classes to appear:

$$w_c = \max\!\left(1,\, \sqrt{\frac{\alpha}{f_c}}\right)$$

where $f_c$ is the frequency of class $c$ and $\alpha$ is an `oversample_factor`
controlling the degree of reweighting.

The **interclass** method samples an equal number of images from each class,
regardless of their original frequencies. This is appropriate when strict class
parity is required (for example, when constructing a balanced evaluation set),
but will require sampling with replacement for minority classes in imbalanced
datasets.

For multi-label datasets (object detection, segmentation), images containing
multiple classes receive a weight derived by aggregating the repeat factors of
all classes they contain, using either the mean or maximum.

A rebalanced subset serves training and evaluation differently. For training, it changes the class mix that the
model learns from. For evaluation, the class frequencies set the prevalence that accuracy, precision and similar
metrics depend on. An unweighted metric on a rebalanced evaluation subset estimates performance under the rebalanced
mix and not under the operational mix, unless the estimate is weighted back to the operational frequencies or the
estimand is stated. The composition and the weights of an evaluation set belong to the evaluation protocol.

## Interpretation

Each measure describes the dataset's own statistics. None of them compares the dataset with the operational
population, so none establishes that a dataset is biased or unrepresentative in that sense.

- Balance reports normalized mutual information between each metadata factor and the class label. A value near 1
  indicates that the factor accounts for much of the class label's entropy, and a value near 0 indicates
  near-independence. The value measures association. It does not establish that the association is a defect, that it
  is causal, or that a trained model uses it. Compare values within a table and not between the class-to-factor and
  inter-factor tables.
- Diversity reports the evenness of sampling across the values of each factor. A low value indicates that samples
  concentrate in a few values. Whether that departs from the operational distribution depends on the operational
  distribution, which Diversity does not measure.

To decide what to do after assessing for bias, see
[Dataset bias and coverage findings](ActingOnResults.md#dataset-bias-and-coverage-findings) in Acting on results.

## Limitations

Mutual information estimates from {class}`.Balance` depend on a random seed and
are consistent to $O(10^{-4})$ but not identical across runs without fixing the
seed. See the
[Configuring the seed](../notebooks/h2_configure_defaults.py#set-the-seed)
how-to for an example of how to set and use seeds in DataEval. For continuous
metadata factors, KNN-based MI estimation introduces additional variance that
falls as the number of neighbors $k$ grows, but a $k$ that is large relative to
sample count inflates the estimate for a factor associated with the class.

Balance and Diversity measure the potential for shortcut learning and sampling
bias. They do not measure whether a trained model has actually learned a
shortcut. A high NMI between a metadata factor and a class label is a signal to
investigate, not a guarantee that the model will exploit the correlation.
Conversely, a low NMI does not rule out shortcuts that operate below the
resolution of the available metadata.

Rows are not independent observations. Balance and Diversity count each row of the level they read as one observation.
At the detection level, a factor measured once for each image is copied onto each of its detections, so its
distribution is weighted by detections per image. Reading the factor at its own level, with {meth}`.Metadata.at`,
removes the weighting (see [Propagation](MetadataLevels.md#propagation-values-move-down-never-up)). The rows of one
track share an object, its scene and usually its class, so they carry less information than their number suggests.
Balance corrects its chance level for the repeated rows of a factor defined at a coarser level, and Parity groups rows
by item (see [Why rows are not independent](Parity.md#why-rows-are-not-independent)).

Scores from two datasets compare only when both were cut at the same edges, and each dataset otherwise derives its own
from its own values. Pass the first dataset's {meth}`.Metadata.encoding` mapping, or the descriptor that
{meth}`.Metadata.export_encoding` writes, as `encoding=` to the second. Two results with the same `encoding_digest`
were encoded identically. See [Control and reuse a factor's binning](../notebooks/h2_control_factor_binning.py).

Parity is experimental. Its p-values depend on independent groups and enough
effective groups at each level; see
[Assumptions and their limits](Parity.md#assumptions-and-their-limits).

## When to use it

Balance and Diversity should be run on any dataset before training begins,
provided metadata is available. They are most valuable when the dataset was
assembled from heterogeneous collection events or merged from multiple sources,
because those conditions can produce systematic
metadata-class correlations. They are also useful before constructing
train/validation/test splits, since split quality depends on metadata
distributions being comparable across splits.

Parity applies when you need to know where a factor and the class depart from
independence, cell by cell, with p-values that account for grouped rows. See
[When to use it](Parity.md#when-to-use-it) on the Parity page.

ClassBalance is a remediation tool, not a diagnostic. Use it after Balance or
Diversity analysis has identified imbalance, to construct a training or
evaluation subset that corrects for it.

## Related concept pages

- [Parity](Parity.md): the hypothesis test of factor–class association, with
  results for each factor level and class
- [Binning](Binning.md): Balance, Diversity and Parity all read binned factors,
  so the cut points decide what these evaluators can see
- [Dataset coverage](DatasetCoverage.md): coverage and completeness in the embedding space
- [Data integrity](DataIntegrity.md): when your data has sample-level quality
  problems rather than population-level bias
- [Performance limits](PerformanceLimits.md): when bias problems have been
  addressed but performance is still bounded by the task itself
- [Acting on results](ActingOnResults.md): how bias findings map to decisions and actions

## See this in practice

### How-to guides

- [Add intrinsic factors to Metadata](../notebooks/h2_add_intrinsic_factors.py)
- [Find crowded, truncated and conflicting annotations](../notebooks/h2_find_annotation_review_candidates.py)
- [Measure label independence](../notebooks/h2_measure_label_independence.py)
- [Configure global DataEval defaults](../notebooks/h2_configure_defaults.py)
- [Build a MetadataLike object from a DataFrame](../notebooks/h2_metadata_from_dataframe.py)
- [Build dataset views with View and Operation](../notebooks/h2_build_dataset_views.py)

### Tutorials

- [Identify bias and correlations](../notebooks/tt_identify_bias.py): end-to-end
  walkthrough of balance and diversity analysis on a realistic dataset

## References

1. [Fabbrizzi, S., Papadopoulos, S., Ntoutsi, E., & Kompatsiaris, I. (2022). A
   survey on bias in visual datasets. *Computer Vision and Image Understanding*,
   223, 103552. [paper](https://doi.org/10.1016/j.cviu.2022.103552)]{#ref1}

2. [Friedman, B., & Nissenbaum, H. (1996). Bias in computer systems. *ACM
   Transactions on Information Systems*, 14(3), 330–347.]{#ref2}

3. [Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge,
   M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks.
   *Nature Machine Intelligence*, 2(11), 665–673.
   [paper](https://arxiv.org/abs/2004.07780)]{#ref3}

4. [Heip, C. H. R., Herman, P. M. J., & Soetaert, K. (1998). Indices of
   diversity and evenness. *Océanis*, 24(4), 61–87.]{#ref4}

5. [Hill, M. O. (1973). Diversity and evenness: A unifying notation and its
   consequences. *Ecology*, 54(2), 427–432.]{#ref5}

6. [Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
   information. *Physical Review E*, 69(6), 066138.
   [paper](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138)]{#ref6}

7. [Merler, M., Ratha, N., Feris, R. S., & Smith, J. R. (2019). Diversity in
   faces. *arXiv preprint arXiv:1901.10436*.
   [paper](https://arxiv.org/abs/1901.10436)]{#ref7}

8. [Ross, B. C. (2014). Mutual information between discrete and continuous data
   sets. *PLOS ONE*, 9(2), e87357.
   [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357)]{#ref8}

9. [Torralba, A., & Efros, A. A. (2011). Unbiased look at dataset bias. In
   *Proceedings of the IEEE Conference on Computer Vision and Pattern
   Recognition* (pp. 1521–1528).
   [paper](https://ieeexplore.ieee.org/document/5995347)]{#ref9}

10. [Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures
    for clusterings comparison: Variants, properties, normalization and
    correction for chance. *Journal of Machine Learning Research*, 11,
    2837–2854. [paper](https://jmlr.org/papers/v11/vinh10a.html)]{#ref10}

11. [Linfoot, E. H. (1957). An informational measure of correlation.
    *Information and Control*, 1(1), 85–89.
    [paper](https://www.sciencedirect.com/science/article/pii/S001999585790116X)]{#ref11}
