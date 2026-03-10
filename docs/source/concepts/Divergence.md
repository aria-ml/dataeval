<!-- markdownlint-disable MD051 -->

# Divergence

{term}`Drift` detectors answer a binary question: has the [distribution shifted](DistributionShift.md)
enough to reject the null hypothesis? That answer is appropriate for continuous
monitoring, where you need a clear signal to act on. But it leaves a question
unanswered: _how far apart_ are two datasets, and is the gap growing over time?

HP divergence answers that question. It is a nonparametric, quantitative
measure of distributional distance — not a hypothesis test, but a number on a
meaningful scale from 0 to 1. This makes it the right tool when you need to
characterize, compare, or track the magnitude of distributional difference
rather than merely detect its presence.

## What is it

{term}`HP divergence <Divergence>` measures the distance between two datasets
in their feature space. The scale is fixed and interpretable:

- **0**: the two datasets are approximately identically distributed
- **1**: the two datasets are completely separable — no overlap between their
  distributions

Unlike formal drift tests, which return a p-value and a binary flag, HP
divergence returns a single continuous score that can be tracked across
time, compared across dataset pairs, or used as an input to downstream
decision logic.

## When to use it

Use HP divergence when you need to _quantify_ distributional distance rather
than _test_ for it. Specific situations where it is the right choice:

**Comparing training and operational distributions before deployment.** A
divergence near 0 confirms that the operational data is well-covered by the
training distribution. A divergence approaching 1 is a warning that the model
will be operating largely outside its training envelope. This is a pre-deployment
check, not a monitoring signal — you run it once to characterize the gap before
a model enters service.

**Tracking distributional drift over time.** Running divergence on successive
operational batches against a fixed reference produces a time series of
distance scores. A rising trend identifies gradual, accumulating drift that
might never trigger a single-batch drift test but is nonetheless moving the
operational distribution away from the training distribution.

**Comparing data collection campaigns.** When new data is collected to augment
an existing dataset, divergence quantifies how much the new data adds
distributionally versus how much it overlaps with existing data. Low divergence
means the new collection is redundant; high divergence means it covers new
territory.

**Prioritizing investigation after drift detection.** When a formal drift test
flags a shift, HP divergence on subsets of the batch — by collection source,
geographic region, time window, or sensor — can localize which subpopulation
is driving the detected shift, without the multiple-testing burden of running
individual drift tests on each subpopulation.

HP divergence is _not_ the right tool when you need a formal statistical
decision: use the {term}`drift <Drift>` detectors in {class}`.DriftUnivariate`,
{class}`.DriftMMD`, or {class}`.DriftDomainClassifier` when you need a p-value
or a defensible pass/fail determination.

## Theory

HP divergence is defined as:

$$
D_p(f_0, f_1) = \frac{1}{4pq} \int \frac{(pf_0(x) - qf_1(x))^2}{pf_0(x) + qf_1(x)} \, dx - (p - q)^2
$$

with $0 \leq p \leq 1$ and $q = 1 - p$, where $f_0$ and $f_1$ are the
probability density functions of the two datasets and $p$, $q$ are their
mixing weights.

When $p = q = 0.5$ (equal-sized datasets), the expression simplifies to:

$$
D_{0.5}(f_0, f_1) = \frac{1}{2} \int \frac{(f_0(x) - f_1(x))^2}{f_0(x) + f_1(x)} \, dx
$$

This is an $f$-divergence — a member of the same family as KL divergence and
total variation distance — with the useful property that it is bounded $[0, 1]$
regardless of the distributions being compared.

### Estimation methods

DataEval provides two nonparametric estimators for HP divergence. Both operate
on embeddings rather than raw images, so the quality of the divergence estimate
is contingent on the embedding capturing the features relevant to the
distributional difference.

**MST-based estimator** ([Sekeh et al., 2020](#ref2)): constructs a minimum spanning
tree over the pooled set of both datasets and estimates divergence from the
fraction of edges that cross the dataset boundary. An edge that crosses from
a point in dataset A to a point in dataset B is evidence of distributional
overlap at that point in feature space; a tree dominated by within-dataset
edges indicates separation. The MST estimator is more accurate for small
samples but scales quadratically with dataset size.

**KNN-based estimator** ([Cover & Hart, 1967](#ref1)): estimates divergence from the
proportion of each point's $k$ nearest neighbors that come from the same
dataset. In regions of distributional overlap, a randomly chosen point's
neighbors will come from both datasets roughly equally; in regions of
separation, neighbors will predominantly come from the same dataset. The KNN
estimator is computationally lighter and preferred for large datasets. Empirical
comparisons suggest no decisive performance difference between the two
estimators on practical datasets.

## Relationship to other evaluators

HP divergence and the drift detectors are complementary, not redundant. They
operate at different levels of the same question:

| Tool                    | Question                                          | Output               | When to use                                          |
| ----------------------- | ------------------------------------------------- | -------------------- | ---------------------------------------------------- |
| HP divergence           | How far apart are these distributions?            | Score in [0, 1]      | Quantification, tracking, comparison                 |
| `DriftUnivariate`       | Which features have shifted significantly?        | Per-feature p-values | Monitoring, diagnosing which features drifted        |
| `DriftMMD`              | Has the joint distribution shifted significantly? | Single p-value       | Monitoring, sensitive to complex multivariate shifts |
| `DriftDomainClassifier` | Can a classifier distinguish the datasets?        | AUROC                | Monitoring, feature importance for diagnosed shift   |

HP divergence is also closely related to {func}`.ber` — both the MST and KNN
estimators for HP divergence are derived from the same foundational papers as
the BER estimators ([Renggli et al., 2021](#ref3)). Where BER estimates the irreducible classification error
_within_ a dataset (overlap between classes), HP divergence estimates the
distributional distance _between_ two datasets. The two metrics use shared
infrastructure and are natural companions in pre-deployment analysis.

## Limitations

**Embedding dependence.** Like all DataEval evaluators that operate in feature
space, HP divergence is only as informative as the embedding is discriminative.
A divergence near 0 in a poor embedding does not confirm distributional
similarity — it may reflect the embedding's failure to represent the relevant
variation.

**No significance threshold.** HP divergence does not produce a p-value. There
is no principled threshold above which divergence is "too high." Interpretation
requires a reference point: a baseline divergence from a previous comparison,
an expected value from domain knowledge, or a trend over time. A divergence of
0.3 is only interpretable in relation to something.

**Sample size sensitivity.** Both estimators can be biased with very small
samples. The KNN estimator in particular can overestimate divergence when
samples are sparse relative to the dimensionality of the embedding space.

## Related concept pages

- [Distribution Shift](DistributionShift.md) — formal hypothesis tests for
  drift detection
- [Performance Limits](PerformanceLimits.md) — BER shares the same estimators
  and is the within-dataset analogue of HP divergence
- [Embeddings](Embeddings.md) — the feature representation both estimators
  depend on
- [Acting on Results](ActingOnResults.md) — how to use divergence scores in
  practice

## See this in practice

### How-to guides

- [How to measure distributional divergence](../notebooks/h2_measure_divergence.py)

## References

1. [Cover, T. & Hart, P. (1967). Nearest neighbor pattern classification.
   _IEEE Transactions on Information Theory_, 13(1), 21–27.
   doi: 10.1109/TIT.1967.1053964 [paper](https://ieeexplore.ieee.org/document/1053964)]{#ref1}

2. [Sekeh, S. Y., Oselio, B., & Hero, A. O. (2020). Learning to bound the
   multi-class Bayes error. _IEEE Transactions on Signal Processing_, 68, 3793–3807.
   doi: 10.1109/TSP.2020.2994807 [paper](https://arxiv.org/abs/1811.06419)]{#ref2}

3. [Renggli, C., Rimanic, L., Hollenstein, N., & Zhang, C. (2021). Evaluating Bayes
   Error Estimators on real-world datasets with FeeBee. _arXiv preprint arXiv:2108.13034_.
   [paper](https://arxiv.org/abs/2108.13034)]{#ref3}
