<!-- markdownlint-disable MD051 -->

# Dataset coverage

This page describes how DataEval measures coverage and completeness, two properties of how a dataset occupies the
embedding space. It also describes how DataEval ranks samples to fill a gap, and states what each result does and
does not establish.

Coverage and bias are different problems with different remedies, and they often co-occur. A dataset collected
exclusively from a single sensor platform in fair-weather conditions may be both biased (overrepresenting those
conditions relative to the operational environment) and lacking in coverage (leaving large regions of the feature
space, such as night, rain, and alternative sensor modalities, empty). [Dataset bias](DatasetBias.md) lists the five
dataset readiness properties, of which covered and complete are two, and describes the measures of bias.

## What coverage is

Coverage is a geometric concept. It describes how well a dataset populates the
relevant regions of the {term}`embedding <Embeddings>` space, the
high-dimensional space in which a model represents the data.

A dataset with high coverage has samples distributed across the full range of
conditions the model will encounter in deployment. A dataset with poor coverage
has dense clusters in some regions and empty regions ("cold spots") elsewhere. A
model trained on low-coverage data can perform well on inputs that resemble the
dense regions and poorly on inputs that fall in the cold spots, regardless of
how many total samples the dataset contains.

Coverage is not determined by sample count. A dataset of 100,000 images collected
from a single afternoon at a single location may have far worse coverage of the
operational distribution than a dataset of 5,000 images deliberately collected
across diverse conditions, sensors, times, and environments.

{term}`Completeness <Completeness>` is a related but distinct concept. Coverage
asks whether the existing samples populate the feature space; completeness
measures how effectively those samples utilize all available dimensions of the
embedding space. A dataset can have good coverage of the conditions it
represents while still being incomplete (collapsed into a lower-dimensional
subspace) if the collected samples are too similar to each other in their
high-level features.

Consider a hypothetical vehicle detection dataset assembled from three collection campaigns: urban daylight,
suburban daylight, and rural daylight. A coverage analysis could show that the data populate the embedding regions
corresponding to each of those three conditions, with no cold spots within them. A completeness analysis could show
that the embeddings occupy a narrow subspace, because all three campaigns share the same sensor, lighting band, and
weather profile. Night conditions, adverse weather, and alternative sensor modalities would then be absent from the
space the dataset spans and not only undersampled. A model trained on this dataset can be expected to perform well in
the represented conditions and to have no basis for performance outside them. Neither sample count nor
within-distribution coverage would indicate this.

## Theory

### Measuring coverage: geometry in embedding space

Coverage analysis works in {term}`embedding <Embeddings>` space. Images are
first projected into a compact vector representation using a pre-trained feature
extractor, so that geometric distance between vectors corresponds to perceptual
or semantic similarity between images. Coverage then identifies which images are
in well-populated regions of the space and which are isolated.

DataEval provides two coverage algorithms.

**Naive coverage** defines a theoretical coverage radius based on the
dimensionality $d$ of the embedding space and the required number of
observations $k$ per covered region:

$$r = \frac{1}{\sqrt{\pi}} \left(\frac{2k\, \Gamma(d/2 + 1)}{n}\right)^{1/d}$$

A sample is considered **uncovered** if fewer than $k$ other samples fall within
radius $r$ of it. This radius is derived from the volume of a $d$-dimensional
ball and assumes uniform distribution across the unit hypercube. It provides an
absolute criterion: either a region of the space meets the minimum density
requirement or it does not.

**Adaptive coverage** uses a data-driven radius instead. The critical value
radius for each sample is the distance to its $k$-th nearest neighbor. The
adaptive threshold is set so that a specified percentage of samples (those with
the largest critical value radii) are classified as uncovered. This approach is
better suited to embeddings that are not uniformly distributed across the unit
hypercube, which is typical of embeddings from a trained feature extractor. The
adaptive method identifies the most isolated samples relative to the actual
distribution of the data, and not against a theoretical uniform baseline.

Both methods are based on [Ting et al. (2022)](#ref2).

Completeness measures dimensional utilization of the embedding space using
eigenvalue entropy. The embedding matrix is rank-normalized before
decomposition: each column is replaced by its within-column rank, scaled to the
interval $(-\frac{1}{2}, \frac{1}{2})$. This makes the metric invariant to
monotonic transformations of individual embedding dimensions. The
rank-normalized matrix is then decomposed via singular value decomposition
(SVD), and the eigenvalues of the covariance matrix are computed from the
singular values:

$\lambda_i = \frac{s_i^2}{n-1}$

The entropy of the normalized eigenvalue distribution gives the **effective
dimensionality** of the data:

$$
H = -\sum_i \hat{\lambda}_i \log \hat{\lambda}_i, \qquad
\hat{\lambda}_i = \frac{\lambda_i}{\sum_j \lambda_j}
$$

$d_\text{eff} = e^H$

Completeness is the ratio of effective to total dimensions:

$\text{completeness} = \frac{d_\text{eff}}{d}$

A completeness score of 1.0 means the dataset uses all available embedding
dimensions equally. A low completeness score means the dataset is effectively
confined to a lower-dimensional subspace. This is consistent with samples that
are similar to each other in high-level structure.

**Isotropy** measures how evenly the data spans the dimensions it actually
occupies. Rather than comparing effective dimensionality to the full ambient
space, isotropy compares it to the rank of the data, the number of dimensions
the data truly reaches:

$\text{isotropy} = \frac{d_\text{eff}}{r}$

where $r = \text{rank}(X)$ is computed using a numerically stable threshold.
Isotropy is always in $[0, 1]$: a value of 1.0 means variance is distributed
evenly across all occupied dimensions; a low value means a few directions
dominate and the occupied subspace is underutilized.

One implementation detail matters when comparing the two numbers directly:
isotropy's $d_\text{eff}$ is computed from the raw centered embeddings, while
completeness's is computed from the rank-normalized matrix described above.
Completeness is therefore invariant to monotone rescaling of individual
dimensions and isotropy is not; isotropy reports the variance geometry of the
embedding space as the extractor actually produced it.

Completeness and isotropy are complementary. Completeness is penalized by both
unused ambient dimensions and uneven variance within the occupied subspace.
Isotropy is penalized only by uneven variance within the occupied subspace,
independent of how many ambient dimensions are unused. A dataset can have low
completeness with high isotropy (data confined to a low-dimensional subspace but
evenly spread within it), or high completeness with low isotropy (data reaching
many dimensions but with variance concentrated in a few).

Both metrics rest on the effective dimensionality $d_\text{eff} = e^H$, where
$H$ is the Shannon entropy of the covariance eigenvalues normalized to sum to
one. This estimator is not taken from a single source. It is closely related to
the **effective rank** of [Roy & Vetterli (2007)](#ref3),
which applies the same $e^H$ construction to the normalized singular values.
Taking the entropy over eigenvalues instead makes $d_\text{eff}$ the exponential
of the von Neumann entropy of the trace-normalized covariance matrix, the form
[Kim et al. (2023)](#ref5) use to regularize representation rank and isotropy,
and [Zhuo et al. (2023)](#ref6) apply to feature-correlation eigenvalues. It is
the entropy-based sibling of the **participation ratio**
$\left(\sum_i \lambda_i\right)^2 / \sum_i \lambda_i^2$ commonly used for the
effective dimensionality of neural representations ([Gao et al., 2017](#ref4));
both summarize a spectrum by how evenly its mass is spread, the participation
ratio through a second-moment ratio and $d_\text{eff}$ through entropy, which
weights the long tail of small eigenvalues more heavily.

Beyond the choice of spectrum summary, what DataEval adds is the normalization:
the two metrics differ only in what $d_\text{eff}$ is divided by. Completeness
divides by the ambient dimension $d$, yielding the fraction of the
representation space the data occupies. Isotropy divides by the numerical rank
$r$, yielding directional evenness within the occupied subspace. Neither
normalization is inherited from the works above, and neither has been validated
against an external implementation; see
[Validation and trust](ValidationAndTrust.md#core-functions).

Completeness also returns **nearest neighbor pairs** sorted by decreasing
distance, identifying the samples that are most isolated from any neighbor.
These are the dataset's rarest examples: either valuable hard cases or
collection artifacts, depending on context.

The {class}`.Coverage` evaluator reports three variety signals for each class.
`dispersion` is how far the class spreads, relative to the median class.
`isotropy` is how evenly it spreads across the directions it occupies.
`near_duplicate_fraction` is the share of its images with an unusually close
neighbor. Each signal has a flag in the same table. `is_clustered` is true when
`dispersion` is below `dispersion_factor`. `is_one_dimensional` is true when
`isotropy` is below `isotropy_factor` times the median isotropy of the classes.
`has_near_duplicates` is true when `near_duplicate_fraction` is above
`near_duplicate_threshold`. The flags are independent, so a class can trip more
than one.

The defaults are 0.5 for both factors and 0.1 for the threshold. They are
conservative. At these values a flag is consistent with a class that has
collapsed, and a class a little below its peers is not flagged. You can tighten
a factor to catch a partial collapse when your classes are of similar size,
because isotropy rises with class size. A flag is null when it cannot judge, which happens when the
class is too small to measure or when the median it is compared with is zero.
The defaults are tuned for the deep embeddings of a pretrained extractor. On an
embedding of fewer than about 8 dimensions, healthy classes can have a
near-duplicate share above 0.1, so you should raise the threshold there.

### Prioritization as coverage-driven data selection

When coverage analysis identifies cold spots in the embedding space, the
question becomes which new samples to collect or label next.
{class}`.Prioritize` ranks unlabeled or unevaluated samples by their
informational value, so that collection and annotation effort is directed toward
the regions of the feature space that need it most.

The connection to coverage is direct: samples ranked highest by difficulty-based
policies (KNN, HDBSCAN distance) are those that fall in sparse regions of the
embedding space, the regions that coverage analysis flags as uncovered. The full
treatment of Prioritization, including policy selection and ordering, is in the
[Acting on results](ActingOnResults.md) concept page.

## Interpretation

Each measure describes the dataset's own statistics. None of them compares the dataset with the operational
population, so none establishes that a dataset is unrepresentative in that sense.

- Coverage identifies the samples that are isolated relative to a radius. The naive radius assumes a uniform
  distribution across the unit hypercube. The adaptive radius is set so that a specified percentage of samples is
  uncovered, so it flags that percentage by construction and ranks isolation within the dataset. Neither establishes
  that an operational condition is missing, because both work in the embedding space of one feature extractor.
- Completeness reports the ratio of effective to total embedding dimensions, and isotropy reports the evenness of
  variance across the occupied dimensions. A low completeness is consistent with data confined to a lower-dimensional
  subspace. It does not identify which conditions are absent.

To decide what to do after finding gaps in coverage, see
[Dataset bias and coverage findings](ActingOnResults.md#dataset-bias-and-coverage-findings) in Acting on results.

## Limitations

All coverage and completeness analyses are contingent on embedding quality. The
same data projected through a weak or domain-inappropriate feature extractor
can produce coverage and completeness scores that reflect the extractor's
limitations and not the data's properties. Before interpreting coverage
results, verify that the chosen embedding model produces meaningful geometric
structure for the target domain (see the [Embeddings](Embeddings.md) concept
page). An additional concern is that pretrained feature extractors can
themselves carry biases absorbed during pretraining: systematic geometric
distortions that cause certain groups or conditions to cluster more tightly or
more loosely than their true diversity warrants.
[Steed & Caliskan (2021)](#ref1) demonstrate that representations learned
through unsupervised pretraining encode human-like social biases. Such biases
may carry into coverage and completeness assessments.

## When to use it

Coverage should be run when you have a feature extractor available and want to
understand where the dataset is thin relative to the operational distribution.
The naive method is appropriate when you have a prior on the minimum density
required per region; the adaptive method is appropriate when you want to
identify the most isolated samples relative to the dataset as it actually is.
Coverage is particularly valuable after merging datasets or before a targeted
collection campaign.

Completeness is most useful as a quick signal on whether a dataset's embedding
structure is degenerate, that is, whether the collected samples are so similar
to each other that they effectively span only a fraction of the embedding space.
A very low completeness score can indicate that the dataset lacks diversity in its
high-level features, even if sample count is high.

## Related concept pages

- [Dataset bias](DatasetBias.md): the association between class labels and metadata factors, and the
  evenness of sampling
- [Embeddings](Embeddings.md): understanding the representation space that
  coverage and completeness operate in
- [Performance limits](PerformanceLimits.md): when coverage problems
  have been addressed but performance is still bounded by the task itself
- [Acting on results](ActingOnResults.md): how to use Prioritization to close
  coverage gaps through targeted data collection

## See this in practice

### How-to guides

- [Run clustering analysis](../notebooks/h2_cluster_analysis.py)

### Tutorials

- [Assess an unlabeled data space](../notebooks/tt_assess_data_space.py):
  coverage and completeness analysis in practice
- [Identify gaps in a labeled dataset](../notebooks/tt_identify_dataset_gaps.py): label-space and embedding-space
  gaps in a labeled dataset

## References

1. [Steed, R., & Caliskan, A. (2021). Image representations learned with
   unsupervised pretraining contain human-like biases. In *Proceedings of the
   2021 ACM Conference on Fairness, Accountability, and Transparency* (pp.
   701–713). ACM. [paper](https://doi.org/10.1145/3442188.3445932)]{#ref1}

2. [Ting, K. M., Wells, J. R., & Washio, T. (2022). Isolation kernel: The X
   factor in efficient and effective large scale online kernel learning.
   *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and
   Data Mining*, 1775–1784. [paper](https://arxiv.org/abs/1907.01104)]{#ref2}

3. [Roy, O., & Vetterli, M. (2007). The effective rank: A measure of effective
   dimensionality. *Proceedings of the 15th European Signal Processing
   Conference (EUSIPCO)*, 606–610.
   [paper](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf)]{#ref3}

4. [Gao, P., Trautmann, E., Yu, B., Santhanam, G., Ryu, S., Shenoy, K., &
   Ganguli, S. (2017). A theory of multineuronal dimensionality, dynamics and
   measurement. *bioRxiv*, 214262.
   [paper](https://doi.org/10.1101/214262)]{#ref4}

5. [Kim, J., Kang, S., Hwang, D., Shin, J., & Rhee, W. (2023). VNE: An effective
   method for improving deep representation by manipulating eigenvalue
   distribution. *Proceedings of the IEEE/CVF Conference on Computer Vision and
   Pattern Recognition (CVPR)*, 3799–3810.
   [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_VNE_An_Effective_Method_for_Improving_Deep_Representation_by_Manipulating_CVPR_2023_paper.html)]{#ref5}

6. [Zhuo, Z., Wang, Y., Ma, J., & Wang, Y. (2023). Towards a unified theoretical
   understanding of non-contrastive learning via rank differential mechanism.
   *International Conference on Learning Representations (ICLR)*.
   [paper](https://arxiv.org/abs/2303.02387)]{#ref6}
