<!-- markdownlint-disable MD051 -->
# Dataset Bias and Coverage

A model is only as good as the data it learned from. Two distinct but related
problems determine whether a dataset will produce a model that generalizes to
its operational environment: **bias**, where the statistical properties of the
training set systematically misrepresent the real-world distribution, and
**coverage**, where the dataset fails to populate enough of the relevant
feature space for the model to perform reliably across the conditions it will
encounter.

These are not the same problem and they do not have the same solution, but they
often co-occur. A dataset collected exclusively from a single sensor platform
in fair-weather conditions may be both biased (overrepresenting those
conditions relative to the operational environment) and lacking in coverage
(leaving large regions of the feature space — night, rain, alternative sensor
modalities — entirely empty). Understanding the distinction, and measuring
both, is a core part of dataset readiness assessment for AI development.

A dataset that is ready for training should satisfy five properties with respect
to its known factors:

- **Balanced** — class labels are statistically independent of metadata factors;
  no condition, sensor, or context is systematically associated with a particular
  class.
- **Covered** — samples populate the relevant regions of the feature space;
  there are no gaps the model will encounter in deployment but has never
  seen in training.
- **Complete** — the dataset spans the full dimensionality of the embedding
  space; samples are not so similar to each other that they effectively occupy
  only a fraction of the representational capacity available.
- **Representative** — the sampling distribution reflects the operational
  distribution; the conditions, frequencies, and contexts present in the dataset
  match those the model will encounter after deployment.
- **Relevant** — every sample belongs to the problem domain; the dataset
  contains no off-domain images, misassigned sources, or annotation artifacts
  that introduce class structure the model should not learn.

The tools described in this page address each of these properties directly.
Balance and Diversity assess whether the dataset is balanced. Coverage and
Completeness assess whether it is covered and complete. Representativeness
and relevance are assessed through the combination of metadata analysis and
the domain context the practitioner brings to interpreting the results.

## What they are

### Bias

Dataset bias occurs when a systematic property of *how data was collected or
labeled* causes the training distribution to diverge from the true operational
distribution in a way that will predictably harm model performance.

The key word is *systematic*. Random noise in a large dataset averages out.
Bias does not — it pushes the model's decision boundaries in a consistent
direction, producing a model that performs well on data that resembles the
training set and fails on data that does not.

[Friedman & Nissenbaum (1996)](#ref3) grouped bias into: pre-existing bias
(inherited from social or institutional practice), technical bias (introduced
by design choices in the system), and emergent bias (arising from deployment
in a context the system was not designed for) — a useful framework for understanding
the origins of bias. However, to better understand *where in the data pipeline*
the bias entered, a more actionable distinction is:

**Sampling bias** arises when the mechanism of data collection systematically
favors certain conditions over others. A dataset assembled from a single proving
ground, a single sensor platform, or a single collection window will oversample
those conditions and undersample everything else — not because of any flaw in
what was labeled, but because of what was collected in the first place.
[Torralba & Efros (2011)](#ref12) showed that this effect is pervasive across
widely used benchmark datasets — classifiers trained on one dataset perform well
within it but transfer poorly to others, because the collection conditions
themselves are a latent variable the model has learned to exploit.

**Representation bias** is the distributional consequence of sampling choices,
but it is distinct enough to treat separately because it can arise without any
flawed sampling mechanism. Certain classes, conditions, or groups may be
under-represented because the target phenomenon is genuinely rare, because
collection constraints limited access, or because systematic gaps in what was
deemed worth collecting were never recognized as gaps. A dataset can be sampled
correctly from a convenience population and still be unrepresentative of the
operational distribution. [Fabbrizzi et al. (2022)](#ref2) survey the full
taxonomy of representation gaps in visual datasets and document how frequently
they appear even in widely used benchmarks. [Merler et al. (2019)](#ref8)
illustrate the downstream consequences in face recognition, where representation
gaps across demographic groups produced systematic performance disparities
invisible to aggregate accuracy metrics.

**Label bias** arises when the annotation process introduces systematic errors.
Human annotators may apply labels inconsistently across classes, across
annotators, or across contextual conditions. When certain classes are
consistently annotated more carefully than others, or when annotator
disagreement correlates with a metadata factor (time of day, image quality,
object size), the learned decision boundary reflects annotation practice rather
than ground truth.

Any form or combination of bias can produce **shortcut learning** in
the trained model. When a metadata factor — background environment, sensor
platform, weather condition, time of day — is consistently associated with a
class label in training data, the model can achieve low training loss by
learning the metadata correlation rather than the intended feature.
[Geirhos et al. (2020)](#ref4) provide the canonical treatment of this failure
mode in computer vision, with the well-known example of models that classify
"cow" by detecting grass rather than the animal. In real world datasets,
analogous shortcuts are common: target detection models that learn sensor
platform as a proxy for target class, or that perform well only in the lighting
conditions of the original collection campaign.

### Coverage

Coverage is a geometric concept. It describes how well a dataset populates
the relevant regions of the {term}`embedding <Embeddings>` space — the
high-dimensional representation space in which the model "sees" the data.

A dataset with high coverage has samples distributed across the full range of
conditions the model will encounter in deployment. A dataset with poor coverage
has dense clusters in some regions and empty regions — "cold spots" —
elsewhere. A model trained on low-coverage data performs well on inputs that
resemble the dense regions and poorly on inputs that fall in the cold spots,
regardless of how many total samples the dataset contains.

Coverage and sample count are independent. A dataset of 100,000 images
collected from a single afternoon at a single location may have far worse
coverage of the operational distribution than a dataset of 5,000 images
deliberately collected across diverse conditions, sensors, times, and
environments.

{term}`Completeness <Completeness>` is a related but distinct concept.
Coverage asks whether the existing samples populate the feature space;
completeness measures how effectively those samples utilize all available
dimensions of the embedding space. A dataset can have good coverage of the
conditions it represents while still being incomplete — collapsed into a
lower-dimensional subspace — if the collected samples are too similar to each
other in their high-level features.

Consider a vehicle detection dataset assembled from three collection campaigns:
urban daylight, suburban daylight, and rural daylight. A coverage analysis might
show that the data adequately populate the embedding regions corresponding
to each of those three conditions — no cold spots within them. But a completeness
analysis would reveal that all three campaigns share the same sensor, lighting
band, and weather profile, so the dataset's embeddings collapse into a narrow
subspace. Night conditions, adverse weather, and alternative sensor modalities
are not merely undersampled; they are entirely absent from the space the dataset
spans. A model trained on this dataset would have high confidence in the
represented conditions and no meaningful representation of anything outside them
— a failure that neither sample count nor within-distribution coverage metrics
would surface.

## Theory

### Measuring bias: mutual information

{class}`.Balance` and {class}`.Diversity` both measure bias through the lens
of metadata — the contextual variables attached to each sample (sensor
platform, weather, time of day, location, annotator ID, and so on). The
fundamental question they answer is: *are class labels independent of these
factors?*

**Balance** measures the statistical dependence between class labels and
metadata factors using {term}`mutual information (MI) <Mutual Information
(MI)>`. Mutual information quantifies how much knowing one variable reduces
uncertainty about another:

$$I(Y; M) = \sum_{y, m} P(Y=y, M=m) \log \frac{P(Y=y, M=m)}{P(Y=y)\, P(M=m)}$$

where $Y$ is the class label and $M$ is a metadata factor. When $Y$ and $M$
are independent, $I(Y; M) = 0$. When knowing $M$ completely determines $Y$,
$I(Y; M)$ equals the entropy of $Y$.

Raw MI values are difficult to interpret, so DataEval normalizes by the
arithmetic mean of the marginal entropies of each variable:

$$I_\text{norm}(Y; M) = \frac{I(Y; M)}{\frac{1}{2}[H(Y) + H(M)]}$$

Values close to 1 indicate high dependence; values close to 0 indicate near-
independence. The arithmetic mean is preferred over the geometric mean because
it remains well-defined when one variable has zero entropy (a degenerate factor
that takes a single value). [Vinh et al. (2010)](#ref13) provide a comprehensive
analysis of normalization schemes for information-theoretic association measures,
including their behavior under chance and the conditions under which each is
most appropriate.

{class}`.Balance` computes MI at two levels. The **global** output measures
the dependence between each metadata factor and the class labels across the
full dataset. The **classwise** output measures the dependence between each
metadata factor and each individual class, identifying cases where a factor is
associated with one class but not others. This distinction matters: a factor
like "time of day" might show low global MI while still being strongly
associated with a specific rare class whose samples were all collected at dawn.

{class}`.Balance` also computes **inter-factor MI** — the pairwise dependence
between metadata factors themselves. Highly correlated factors (for example,
if location and sensor platform are nearly perfectly associated) provide
redundant information and complicate the interpretation of individual factor
effects.

MI is computed differently depending on whether variables are discrete or
continuous. For categorical class labels, DataEval uses
`mutual_info_classif` from scikit-learn for both discrete and continuous
metadata factors, falling back to KNN-based estimation ([Kraskov et al., 2004](#ref7);
[Ross, 2014](#ref9)) for continuous variables. DataEval infers discreteness from the
proportion of unique values in a factor.

```{note}
Normalized MI is not adjusted for chance. For small datasets or factors with
many unique values, MI estimates can be inflated relative to what would be
expected under independence. Treat low but nonzero MI values with caution,
particularly when sample counts per factor category are small.
```

**Diversity** measures the *evenness* of sampling across each metadata factor,
independently of class labels. Where Balance asks "is class label correlated
with this factor?", Diversity asks "are samples spread uniformly across this
factor's values?"

DataEval supports two diversity indices, both drawn from ecological diversity
measurement ([Hill, 1973](#ref6); [Heip et al., 1998](#ref5)).
The **inverse Simpson index** is:

$$d = \frac{1}{N \sum_i p_i^2}$$

where $p_i$ are the discrete probabilities of each bin and $N$ is the number
of unique values. This is linearly rescaled to $[0, 1]$:

$$d' = \frac{dN - 1}{N - 1}$$

The rescaling removes dependence on $N$, making values comparable across
factors with different numbers of categories.

The **normalized Shannon entropy** is:

$$d = -\frac{1}{\log N}\sum_i p_i \log p_i$$

Both indices reach 1 when samples are uniformly distributed and 0 when all
samples belong to a single category. The Simpson index is more sensitive to
dominant categories; the Shannon index weights all categories more equally.
The default is Simpson.

Diversity is computed both globally across the full dataset and classwise
within each class. Low classwise diversity for a particular class and factor
— for example, all "aircraft" samples collected at the same location — is
a direct indicator of collection bias for that class.

**Parity** takes a complementary approach to balance, using Bias-Corrected
Cramér's V rather than mutual information to measure association between
metadata factors and class labels. This is an experimental evaluator.

Cramér's V is derived from the chi-squared statistic on a contingency
table of factor values against class labels:

$$V = \sqrt{\frac{\chi^2 / n}{\min(r-1, c-1)}}$$

where $n$ is the number of samples and $r$, $c$ are the number of rows and
columns in the contingency table. DataEval applies the [Bergsma (2013)](#ref1) bias
correction, which provides a more accurate estimate of association strength
for large contingency tables and finite samples than the standard Cramér's V correction.

Statistical significance is assessed with the G-test (log-likelihood ratio)
rather than Pearson's chi-squared, and both a score threshold (default: 0.3)
and a p-value threshold (default: 0.05) must be exceeded before a factor is
flagged as correlated.

```{important}
Parity requires a minimum of 5 samples per cell in the contingency table
(factor value × class label combination) for reliable chi-squared
approximations. Factors with insufficient cell counts are flagged in the
output. Results for those factors should be treated as indicative only.
```

### Measuring coverage: geometry in embedding space

Coverage analysis works in {term}`embedding <Embeddings>` space. Images are
first projected into a compact vector representation using a pre-trained
feature extractor, so that geometric distance between vectors corresponds to
perceptual or semantic similarity between images. Coverage then asks: given
this geometry, which images are in well-populated regions of the space, and
which are isolated?

DataEval provides two coverage algorithms.

**Naive coverage** defines a theoretical coverage radius based on the
dimensionality $d$ of the embedding space and the required number of
observations $k$ per covered region:

$$r = \frac{1}{\sqrt{\pi}} \left(\frac{2k\, \Gamma(d/2 + 1)}{n}\right)^{1/d}$$

A sample is considered *uncovered* if fewer than $k$ other samples fall within
radius $r$ of it. This radius is derived from the volume of a $d$-dimensional
ball and assumes uniform distribution across the unit hypercube. It provides
an absolute criterion: either a region of the space meets the minimum density
requirement or it does not.

**Adaptive coverage** uses a data-driven radius instead. The critical value
radius for each sample is the distance to its $k$-th nearest neighbor. The
adaptive threshold is set so that a specified percentage of samples — those
with the largest critical value radii — are classified as uncovered. This
approach is more robust when embeddings are not uniformly distributed across
the unit hypercube, which is almost always the case in practice. The adaptive
method identifies the most isolated samples relative to the actual distribution
of the data, rather than against a theoretical uniform baseline.

Both methods are based on [Ting et al. (2022)](#ref11).

**Completeness** measures dimensional utilization of the embedding space using
eigenvalue entropy. The embedding matrix is decomposed via singular value
decomposition (SVD), and the eigenvalues of the covariance matrix are computed
from the singular values:

$$\lambda_i = \frac{s_i^2}{n-1}$$

The entropy of the normalized eigenvalue distribution gives the **effective
dimensionality** of the data:

$$H = -\sum_i \hat{\lambda}_i \log \hat{\lambda}_i, \qquad
\hat{\lambda}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

$$d_\text{eff} = e^H$$

Completeness is then the ratio of effective to total dimensions:

$$\text{completeness} = \frac{d_\text{eff}}{d}$$

A completeness score of 1.0 means the dataset uses all available embedding
dimensions equally. A low completeness score means the dataset is effectively
collapsed into a lower-dimensional subspace — the samples are too similar to
each other in high-level structure to provide diverse training signal across
the full capacity of the embedding model.

Completeness also returns **nearest neighbor pairs** sorted by decreasing
distance, identifying the samples that are most isolated from any neighbor.
These are the dataset's rarest examples — either valuable hard cases or
collection artifacts, depending on context.

### Correcting for imbalance: ClassBalance

{class}`.ClassBalance` is a dataset selection tool rather than a measurement
tool. Once Balance or Diversity analysis has revealed class imbalance,
`ClassBalance` produces a rebalanced subset for training or evaluation
without requiring new data collection.

Two strategies are available. The **global** method samples images with
probability proportional to the inverse square root of class frequency,
giving rare classes higher weight while still allowing common classes to
appear:

$$w_c = \max\!\left(1,\, \sqrt{\frac{\alpha}{f_c}}\right)$$

where $f_c$ is the frequency of class $c$ and $\alpha$ is an `oversample_factor`
controlling the degree of reweighting.

The **interclass** method samples an equal number of images from each class,
regardless of their original frequencies. This is appropriate when strict
class parity is required (for example, when constructing a balanced evaluation
set), but will require sampling with replacement for minority classes in
imbalanced datasets.

For multi-label datasets (object detection, segmentation), images containing
multiple classes receive a weight derived by aggregating the repeat factors
of all classes they contain, using either the mean or maximum.

### Prioritization as coverage-driven data selection

When coverage analysis identifies cold spots in the embedding space, the
question becomes which new samples to collect or label next.
{class}`.Prioritize` ranks unlabeled or unevaluated samples by their
informational value, so that collection and annotation effort is directed
toward the regions of the feature space that need it most.

The connection to coverage is direct: samples ranked highest by difficulty-
based policies (KNN, HDBSCAN distance) are those that fall in sparse regions
of the embedding space — exactly the regions that coverage analysis flags as
uncovered. The full treatment of Prioritization, including policy selection
and ordering, is in the
[Acting on Results](ActingOnResults.md) concept page.

## When to use it

**Balance and Diversity** should be run on any dataset before training begins,
provided metadata is available. They are most valuable when the dataset was
assembled from heterogeneous collection events or merged from multiple sources,
because those are the conditions most likely to produce systematic metadata-
class correlations. They are also useful before constructing train/validation/
test splits, since split quality depends on metadata distributions being
comparable across splits.

**Parity** is most appropriate when the metadata factors are categorical and
the dataset is large enough to meet the minimum cell-count requirement. It
provides a statistically grounded p-value alongside the association score,
which is useful when a formal threshold is required for a test report.

**Coverage** should be run when you have a feature extractor available and
want to understand where the dataset is thin relative to the operational
distribution. The naive method is appropriate when you have a prior on the
minimum density required per region; the adaptive method is appropriate when
you want to identify the most isolated samples relative to the dataset as it
actually is. Coverage is particularly valuable after merging datasets or before
a targeted collection campaign.

**Completeness** is most useful as a quick signal on whether a dataset's
embedding structure is degenerate — that is, whether the collected samples are
so similar to each other that they effectively span only a fraction of the
embedding space. A very low completeness score is a flag that the dataset
lacks diversity in its high-level features, even if sample count is high.

**ClassBalance** is a remediation tool, not a diagnostic. Use it after Balance
or Diversity analysis has confirmed imbalance, to construct a training or
evaluation subset that corrects for it.

To better understand what to do after assessing for bias, review the
[Dataset Bias and Coverage section in the Acting on Results explanation page](ActingOnResults.md#dataset-bias-and-coverage-findings).

## Limitations

Mutual information estimates from {class}`.Balance` depend on a random seed
and are consistent to $O(10^{-4})$ but not exactly reproducible across runs
without fixing the seed. See the
[Configuring the seed](../notebooks/h2_configure_hardware_settings.md#configuring-the-global-seed)
how-to for an example of how to set and use seeds in DataEval.
For continuous metadata factors, KNN-based MI
estimation introduces additional variance that grows with the number of
neighbors $k$ relative to sample count.

All coverage and completeness analyses are contingent on embedding quality.
The same data projected through a weak or domain-inappropriate feature
extractor will produce coverage and completeness scores that reflect the
extractor's limitations rather than the data's properties. Before
interpreting coverage results, verify that the chosen embedding model produces
meaningful geometric structure for the target domain (see the
[Embeddings](Embeddings.md) concept page). An additional concern is that
pretrained feature extractors can themselves carry biases absorbed during
pretraining — systematic geometric distortions that cause certain groups or
conditions to cluster more tightly or more loosely than their true diversity
warrants. [Steed & Caliskan (2021)](#ref10) demonstrate that representations
learned through unsupervised pretraining encode human-like social biases,
which can propagate silently into coverage and completeness assessments
downstream.

Balance and Diversity measure the *potential* for shortcut learning and
sampling bias. They do not measure whether a trained model has actually learned
a shortcut. A high MI between a metadata factor and a class label is a signal
to investigate — not a guarantee that the model will exploit the correlation.
Conversely, a low MI does not rule out shortcuts that operate below the
resolution of the available metadata.

Parity is marked experimental and may change in future releases. Results
should be treated with caution when cell counts are below 5, which the
output flags explicitly.

## Related concept pages

- [Embeddings](Embeddings.md) — understanding the representation space that
  coverage and completeness operate in
- [Data Integrity](DataIntegrity.md) — when your data has sample-level quality
  problems rather than population-level bias
- [Performance Limits](PerformanceLimits.md) — when bias and coverage problems
  have been addressed but performance is still bounded by the task itself
- [Acting on Results](ActingOnResults.md) — how to use Prioritization to close
  coverage gaps through targeted data collection

## See this in practice

### How-to guides

- [How to add intrinsic metadata factors](../notebooks/h2_add_intrinsic_factors.md)
- [How to measure label independence](../notebooks/h2_measure_label_independence.md)
- [How to detect undersampling](../notebooks/h2_detect_undersampling.md)
- [How to perform cluster analysis](../notebooks/h2_cluster_analysis.md)
- [How to configure global hardware configuration defaults in DataEval](../notebooks/h2_configure_hardware_settings.md)

### Tutorials

- [Identifying bias tutorial](../notebooks/tt_identify_bias.md) — end-to-end
  walkthrough of balance and diversity analysis on a realistic dataset
- [Assessing data space tutorial](../notebooks/tt_assess_data_space.md) —
  coverage and completeness analysis in practice

## References

1. [Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T.
*Journal of the Korean Statistical Society*, 42(3), 323–328. [paper](https://www.sciencedirect.com/science/article/abs/pii/S1226319212001032)]{#ref1}

2. [Fabbrizzi, S., Papadopoulos, S., Ntoutsi, E., & Kompatsiaris, I. (2022).
A survey on bias in visual datasets. *Computer Vision and Image Understanding*,
223, 103552. [paper](https://doi.org/10.1016/j.cviu.2022.103552)]{#ref2}

3. [Friedman, B., & Nissenbaum, H. (1996). Bias in computer systems. *ACM
Transactions on Information Systems*, 14(3), 330–347.]{#ref3}

4. [Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W.,
Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural
networks. *Nature Machine Intelligence*, 2(11), 665–673. [paper](https://arxiv.org/abs/2004.07780)]{#ref4}

5. [Heip, C. H. R., Herman, P. M. J., & Soetaert, K. (1998). Indices of
diversity and evenness. *Océanis*, 24(4), 61–87.]{#ref5}

6. [Hill, M. O. (1973). Diversity and evenness: A unifying notation and its
consequences. *Ecology*, 54(2), 427–432.]{#ref6}

7. [Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
information. *Physical Review E*, 69(6), 066138. [paper](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138)]{#ref7}

8. [Merler, M., Ratha, N., Feris, R. S., & Smith, J. R. (2019). Diversity in
faces. *arXiv preprint arXiv:1901.10436*. [paper](https://arxiv.org/abs/1901.10436)]{#ref8}

9. [Ross, B. C. (2014). Mutual information between discrete and continuous data
sets. *PLOS ONE*, 9(2), e87357. [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357)]{#ref9}

10. [Steed, R., & Caliskan, A. (2021). Image representations learned with
unsupervised pretraining contain human-like biases. In *Proceedings of the
2021 ACM Conference on Fairness, Accountability, and Transparency* (pp.
701–713). ACM. [paper](https://doi.org/10.1145/3442188.3445932)]{#ref10}

11. [Ting, K. M., Wells, J. R., & Washio, T. (2022). Isolation kernel: The X
factor in efficient and effective large scale online kernel learning.
*Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining*, 1775–1784. [paper](https://arxiv.org/abs/1907.01104)]{#ref11}

12. [Torralba, A., & Efros, A. A. (2011). Unbiased look at dataset bias. In
*Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
(pp. 1521–1528). [paper](https://ieeexplore.ieee.org/document/5995347)]{#ref12}

13. [Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures
for clusterings comparison: Variants, properties, normalization and correction
for chance. *Journal of Machine Learning Research*, 11, 2837–2854. [paper](https://jmlr.org/papers/v11/vinh10a.html)]{#ref13}
