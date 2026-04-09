<!-- markdownlint-disable MD051 -->

# Distribution Shift

A model trained and validated on one dataset is deployed into a world that does
not hold still. Operational environments shift — seasons change, sensors degrade,
mission parameters evolve, target populations shift. When the statistical
properties of data encountered during deployment diverge from the properties of
training data, model performance degrades. The model's learned decision
boundaries were optimized for a distribution that no longer describes what it
sees.

This is **distribution shift**: the gap between training distribution and
operational distribution. It is not a rare edge case. For systems with long
fielded lifespans, it is a planning assumption.

DataEval provides two related but distinct capabilities for managing this gap.
**Drift detection** operates at the population level — it monitors whether a
batch of incoming data as a whole is statistically consistent with the training
reference. **Out-of-distribution (OOD) detection** operates at the instance
level — it scores individual samples for how anomalous they are relative to
what the model was trained on. Both are necessary; neither is sufficient alone.

## Taxonomy of shift

Understanding which kind of shift is occurring guides both the choice of
detection method and the appropriate response. For a classification model
predicting $Y$ from inputs $X$, the joint distribution $P(X, Y)$ can be
decomposed two ways:

$$P(X, Y) = P(Y \mid X)\, P(X) = P(X \mid Y)\, P(Y)$$

Shift occurs when the training joint distribution $P_t(X, Y)$ differs from the
deployment distribution $P_d(X, Y)$. Three distinct failure modes follow from
which component has changed.

**Covariate shift** (also called population shift or virtual drift) occurs when
the input distribution changes but the class-conditional relationship is stable:

$$P_t(Y \mid X) = P_d(Y \mid X), \quad P_t(X) \neq P_d(X)$$

The model's decision boundaries are still correct — if only it could see the
right inputs. The problem is that it encounters inputs in regions of the feature
space it saw rarely or never during training. For a vehicle detection system
trained predominantly on clear daytime imagery, a deployment in heavy fog
represents covariate shift: the mapping from image to vehicle label has not
changed, but the image distribution has moved into a region where training data
was sparse.

**Label shift** (prior-probability shift) occurs when the class distribution
changes but the class-conditional input distribution is stable:

$$P_t(X \mid Y) = P_d(X \mid Y), \quad P_t(Y) \neq P_d(Y)$$

If a model was trained on a dataset where target and non-target images appeared
at equal frequency but is deployed in a scenario where targets are rare, the
prior has shifted. A model that was not calibrated for this imbalance will
over-predict the previously common class.

**Concept drift** (posterior-probability shift or real drift) occurs when the
input distribution is stable but the relationship between inputs and labels has
changed:

$$P_t(X) = P_d(X), \quad P_t(Y \mid X) = P_d(Y \mid X)$$

The learned decision boundaries are now wrong for the data they were trained on.
In long-lifecycle systems, concept drift can arise from changes in the targets
themselves (a new vehicle variant that resembles an existing class but should be
classified differently), changes in adversarial behavior, or changes in labeling
standards.

These three types are not mutually exclusive. Covariate shift can induce label
shift; real-world shift events often involve all three simultaneously. The
taxonomy is useful not because real shifts are pure examples of one type, but
because it identifies what has changed and therefore what can be done about it.

## Drift detection

Drift detection answers a population-level question: _is this batch of incoming
data drawn from the same distribution as my training data?_ All of DataEval's
drift detectors follow the same two-phase pattern: a **fit** phase on reference
(training) data, followed by repeated **predict** calls on operational batches,
each returning a p-value and a binary drift flag.

The choice of detector depends on the data representation and the nature of the
expected shift. DataEval provides univariate tests for feature-by-feature
analysis, multivariate tests for joint distributional comparison, and two
additional detectors that use reconstruction error or nearest-neighbor distance
as the drift signal.

### Univariate tests: `DriftUnivariate`

{class}`.DriftUnivariate` applies a statistical two-sample test to each feature
independently, then aggregates the resulting p-values across features to produce
a single drift decision. Two correction methods are available for the
multiple-testing problem: **Bonferroni** correction controls the probability of
any false positive (conservative, appropriate when any drifting feature is
actionable) and **False Discovery Rate (FDR)** correction controls the
proportion of false positives among all flagged features (less conservative,
appropriate when you want to identify which features drifted).

Five test statistics are supported. They differ in which part of the
distribution they are most sensitive to, and therefore in which kinds of shift
they detect best.

**Kolmogorov-Smirnov (KS)** measures the maximum absolute difference between
two empirical CDFs:

$$\text{KS} = \sup_x \left| F(x) - F_\text{ref}(x) \right|$$

The supremum statistic makes KS most sensitive to shifts in the central
(modal) region of the distribution — where the CDF is steepest and the
difference is largest. It is the standard baseline test: well-understood, widely
supported, and appropriate when shift is expected to move the bulk of the
distribution. It is less sensitive to changes in the tails.

**Cramér-von Mises (CVM)** measures the sum of squared CDF differences over the
full joint sample:

$$W = \sum_{z \in k} \left| F(z) - F_\text{ref}(z) \right|^2$$

Integrating over all points rather than taking a maximum gives CVM higher power
than KS for detecting variance changes and subtle shifts spread across the
distribution. It is the preferred test when camera calibration drift, gradual
sensor aging, or other slow diffuse shifts are the expected failure mode.

**Mann-Whitney U (MWU)** is a rank-based test that compares the tendency of
one sample to produce larger values than the other:

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$$

where $R_1$ is the sum of ranks for the reference sample. Because it operates
on ranks rather than raw values, MWU is robust to outliers and heavy-tailed
distributions, and is most sensitive to shifts in the median. Its limitation is
the converse: if the median is stable but the variance has changed — for
example, a sensor producing wider spread without shifting the mean — MWU will
not detect it.

**Anderson-Darling (AD)** modifies the CVM weighting to emphasize the tails:

$$A^2 = -n - \sum_{i=1}^n \frac{2i-1}{n} \left[\ln F(z_i) + \ln(1 - F(z_{n+1-i}))\right]$$

The $1/F(1-F)$ weighting makes AD more powerful than KS for tail differences,
which corresponds to detecting rare but extreme events — unusual lighting
conditions, rare target classes, weather extremes. The cost is sensitivity:
AD may trigger on tail noise that does not affect model performance.

**Baumgartner-Weiss-Schindler (BWS)** is a modern rank-weighted test that
combines the location sensitivity of MWU with the tail sensitivity of AD:

$$B = \frac{1}{n_1 n_2} \sum_{i,j} \psi(F(x_i), F(y_j))$$

where $\psi$ is a weighting function emphasizing tail regions. BWS provides
higher statistical power than KS, CVM, or MWU across a range of shift types,
making it the strongest general-purpose choice when computational cost is
not a constraint.

**Test selection guidance:**

| Expected shift type                           | Recommended test |
| --------------------------------------------- | ---------------- |
| Central distribution shift (location, scale)  | KS (baseline)    |
| Subtle variance / higher-order moment changes | CVM              |
| Median shift, data with outliers              | MWU              |
| Tail / extreme-value shifts, rare events      | AD               |
| Unknown or mixed shift type, high stakes      | BWS              |

### Multivariate tests

Univariate tests examine features independently and can miss shift that
manifests only in the _relationships_ between features — for example, a sensor
that now produces correlated noise across channels where noise was previously
independent. Two multivariate approaches are available.

**Maximum Mean Discrepancy (`DriftMMD`)** measures the distance between
distribution mean embeddings in a reproducing kernel Hilbert space (RKHS):

$$\text{MMD}^2(p, q) = \| \mu_p - \mu_q \|_\mathcal{F}^2$$

With the RBF kernel $k(x,y) = \exp(-\|x-y\|^2 / 2\sigma^2)$, MMD can detect
any distributional difference (it is a _universal_ kernel). Statistical
significance is assessed by a permutation test: the reference and test samples
are pooled, randomly split many times, and the observed MMD is compared to the
permutation distribution. MMD is the natural choice for detecting shift in
high-dimensional {term}`embedding <Embeddings>` spaces from deep networks, where
univariate feature-by-feature testing would require thousands of individual
tests. Its limitation is quadratic scaling in sample size and the need to
select a kernel bandwidth $\sigma$.

**Domain Classifier (`DriftDomainClassifier`)** takes a discriminative
approach: train a binary classifier to distinguish reference from analysis data,
then measure how well it succeeds:

$$\text{DC} = \text{AUROC}\!\left(C(X_\text{ref},\, X_\text{analysis})\right)$$

An AUROC near 0.5 means the classifier cannot tell the two distributions apart —
no drift. An AUROC approaching 1.0 means the distributions are highly
distinguishable. DataEval uses LightGBM with stratified k-fold
cross-validation to prevent overfitting. The domain classifier is particularly
effective at detecting subtle non-linear shifts in joint feature distributions
that MMD or univariate tests might miss, and it provides an intuitive magnitude
score. Its cost is computational: it trains multiple classifiers per call.

### K-nearest neighbor drift detection (`DriftKNeighbors`)

{class}`.DriftKNeighbors` detects drift by comparing the distances of test
samples to their nearest neighbors in the reference set against the baseline
distribution of reference-to-reference distances. In-distribution test samples
should have similar nearest-neighbor distances to reference samples; OOD or
drifted samples will be farther from any reference neighbor.

During fit, the scorer indexes the full reference set and computes
**self-distances** for each reference sample: the mean distance to its $k+1$
nearest neighbors excluding itself (leave-one-out). These self-distances form
the baseline distribution.

During predict, each test sample's mean distance to its $k$ nearest reference
neighbors is computed and the full per-sample distance distributions are
compared with a **Mann-Whitney U test** (one-sided, `alternative="greater"`):
are test distances stochastically greater than reference self-distances?

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$$

A p-value below the threshold (default: 0.05) flags drift. The MWU test is
appropriate here because it is non-parametric, robust to outliers, and compares
the full distribution of distances rather than just means — making it less
susceptible to false positives from a few distant test points.

This detector is lightweight and training-free beyond fitting the neighbor
index. It is particularly effective for detecting covariate shift in embedding
spaces: if the test embeddings have moved to a different region, their distances
to the reference index will increase. Both `cosine` and `euclidean` distance
metrics are supported (default: `euclidean`; default $k$: 10). Inputs are
flattened to 2D before indexing, so multi-dimensional arrays are handled
automatically.

**Chunked mode** is available for both `DriftKNeighbors` and `DriftReconstruction`.
Rather than testing a single batch against the reference, chunked mode splits
the data into windows, computes a per-chunk metric (mean KNN distance or mean
reconstruction error), and uses a threshold strategy (default: z-score bounds
derived from reference chunks) to flag individual chunks. This is appropriate
for streaming deployments where drift may onset gradually and you want to
localize when it began.

### Reconstruction-based drift detection (`DriftReconstruction`)

{class}`.DriftReconstruction` detects drift using the same autoencoder
architecture as [`OODReconstruction` (AE, VAE, or AE/VAE with GMM)](#reconstruction-based-detection-oodreconstruction),
but applies it at the batch level rather than per-instance. The model is trained on
reference data; reconstruction error on test data is compared to the reference baseline.

In non-chunked mode, the test is a **one-sided z-test** on the mean
reconstruction error:

$$z = \frac{\bar{e}_\text{test} - \bar{e}_\text{ref}}{\hat{\sigma}_\text{ref} / \sqrt{n_\text{eff}}}$$

where $n_\text{eff} = \min(|\text{test}|, 75)$. The effective-sample cap is
deliberate: without it, the test becomes overpowered at large batch sizes,
flagging trivially small differences (Cohen's $d \ll 0.2$) as statistically
significant. The cap requires approximately 0.2 standard deviations of mean
shift for detection. The resulting p-value is one-sided (higher reconstruction
error in the test set indicates drift).

`DriftReconstruction` and `OODReconstruction` share the same underlying
`ReconstructionScorer` and accept the same model architectures (AE, VAE, GMM
variants). They are separate classes, each taking a model as a constructor
argument and training it independently during `fit`. In practice, deployments
that need both population-level drift monitoring and instance-level anomaly
scoring train two instances on the same reference data.

### Uncertainty-based drift detection

The detectors above operate on input features and are blind to whether detected
shift affects model predictions. **Uncertainty-based drift detection**
(`DriftUnivariate`, via `ClassifierUncertaintyExtractor`) takes a
model-aware approach: it monitors changes in the model's prediction confidence
rather than the inputs directly.

For each image, a trained classifier produces class probabilities. The entropy
of those probabilities measures how uncertain the model is:

$$H(p) = -\sum_{i=1}^k p_i \log p_i$$

Low entropy means the model is confident; high entropy means it is not. If the
entropy distribution of incoming data is significantly higher than that of the
reference set — detected by applying a univariate test (typically KS) to the
entropy scores — the model is encountering data in its uncertainty regions,
which directly predicts performance degradation.

This approach has a specific, important advantage: it is **insensitive to
irrelevant shift**. A lighting change that modifies pixel values but does not
move the model's predictions into uncertain territory will not trigger an alert.
Feature-based detectors would flag it. The trade-off is that uncertainty-based
detection requires a trained classifier and will miss shift that the model
confidently misclassifies — the model remains certain, but wrong.

### Label parity

All the drift detectors above operate on input features $P(X)$ or model
confidence scores. A separate but complementary question is whether the
_class frequency distribution_ has changed between two datasets — training vs.
operational, or one collection period vs. another. This is label shift in its
simplest measurable form, and it is the specific question that
{func}`.label_parity` addresses.

Label parity compares label frequency distributions between two datasets using
a chi-squared test. When the test returns a significant result, the class prior
$P(Y)$ has shifted: certain classes appear more or less frequently in the
operational data than in the reference. A model calibrated on training class
frequencies will be miscalibrated for the new distribution — it will
over-predict previously common classes and under-predict now-more-common ones.

Label parity is particularly useful as a lightweight, human-interpretable
complement to feature-based drift detection. It requires only class labels,
not embeddings or a trained model, and its output is immediately actionable:
if class frequencies have shifted, rebalancing training data or adjusting
decision thresholds is a concrete response. If feature drift is detected but
label parity holds, the shift is in the input distribution without a change in
class prevalence — characteristic of covariate shift rather than label shift.

## Out-of-distribution detection

While drift detection asks whether a _batch_ has shifted, OOD detection asks
whether a _specific sample_ is anomalous relative to the training distribution.
The two capabilities are complementary: while a batch may pass drift detection,
it might contain a handful of genuine anomalies whose effect is diluted below
the detection threshold of a drift test, or contain instances that will cause
confident model failures.

DataEval provides three OOD detection approaches that differ in their
computational requirements, interpretability, and the type of anomaly they
detect best.

### Reconstruction-based detection (`OODReconstruction`)

Reconstruction-based detection trains an autoencoder-family model on
reference data and uses the model's failure to reconstruct a test sample as the
anomaly signal.

The core assumption: a model trained to compress and reconstruct _in-distribution_
images learns the manifold of normal data. When it encounters an OOD sample —
one that lies off that manifold — it cannot reconstruct it accurately, producing
high reconstruction error. The per-pixel squared error serves as both an
instance-level score (mean error across the image) and a **feature-level
anomaly map** (the spatial distribution of error), which identifies _where_ in
the image the anomaly is located.

DataEval supports three reconstruction architectures, selected automatically or
explicitly via `model_type`:

**Autoencoder (AE)** learns a deterministic encoding. Each input maps to a
single point in the latent space. The OOD score is mean squared error between
input and reconstruction. AEs are most effective for _structural_ anomalies —
local defects, sensor corruption, image artifacts — where the abnormality is a
specific localized departure from normal image structure.

**Variational Autoencoder (VAE)** adds a probabilistic constraint: the latent
representation is a distribution (mean and variance) rather than a point. The
training objective includes an Evidence Lower Bound (ELBO) term that penalizes
latent distributions deviating from a standard Normal. VAEs are more effective
for _statistical_ anomalies — samples that are structurally plausible but
represent a rare or unlikely combination of features.

**AE or VAE with Gaussian Mixture Model (GMM)** extends either architecture
by modeling the latent space as a mixture of $K$ Gaussians rather than a single
distribution. This is appropriate when the reference data is **multimodal** —
when normal data clusters into distinct subpopulations (day vs. night imagery,
multiple sensor modes, different operational environments). A GMM can learn each
mode separately; a unimodal model would have to compromise between them.

When GMM is used, the GMM energy is calculated in addition to the reconstruction
error. The GMM energy for a latent point $z$ measures how unlikely it is under the
mixture:

$$E(z) = -\log \sum_{k=1}^K \phi_k \cdot \mathcal{N}(z;\, \mu_k, \Sigma_k)$$

implemented in $logsumexp$ form for numerical stability.

The OOD score for that sample combines the reconstruction error and the GMM energy.
DataEval provides two fusion modes for combining the reconstruction error score
with the GMM energy score: **standardized** and **percentile**. The `standardized`
method performs a z-score normalization of each score (difference from mean
divided by standard deviation) before a weighted combination. The `gmm_weight` parameter
controls how heavily the GMM score is weighted during standardized fusion (default 0.5,
range [0, 1]). The `percentile` method computes the cumulative distribution function
(CDF) for the batch for each score and then computes the joint tail probability
for each sample.

### Distance-based detection (`OODKNeighbors`)

Distance-based OOD detection requires no model training. It stores reference
{term}`embeddings <Embeddings>` and at inference time computes the distance
from a test sample's embedding to its $k$ nearest neighbors in the reference
set. Samples far from their nearest neighbors — in sparse regions of the
embedding space — receive high OOD scores.

The mean distance to $k$ nearest neighbors is the OOD score. A threshold
calibrated at fit time (typically the $p$-th percentile of reference-set
self-distances) converts this to a binary in/out-of-distribution prediction.

Distance-based detection is **training-free** beyond the feature extractor. It
is fast to deploy, easy to update (adding new reference samples requires only
re-indexing, not retraining), and interpretable — the score directly corresponds
to geometric distance in a meaningful space. The limitation is that it inherits
the strengths and weaknesses of the embedding model: if the feature extractor
does not represent the anomaly dimension well, the distance will not be
informative. Distance-based detection also produces only instance-level scores,
not feature-level maps.

### Domain Classifier OOD detection (`OODDomainClassifier`)

{class}`.OODDomainClassifier` applies the same discriminative principle as
`DriftDomainClassifier` — but at the instance level rather than the batch
level. Where the drift detector produces a single AUROC score for an entire
batch, the OOD detector produces a **per-sample class-1 prediction rate**: how
consistently does the classifier identify this specific sample as "not
reference"?

The fit phase establishes a null distribution. The reference set is split
randomly into two halves, labeled pseudo-class-0 and pseudo-class-1, and
repeated stratified k-fold cross-validation is run on this internal split.
Because both halves come from the same reference distribution, the resulting
class-1 rates represent what the classifier sees when there is no real signal
to learn — they capture baseline discrimination due to noise alone. The OOD
threshold is set at $\mu_{null} + n_\sigma \cdot \sigma_{null}$
(default $n_\sigma = 2.0$), placing it approximately two standard deviations
above the null mean.

The predict phase concatenates reference and test data, labels reference
samples 0 and test samples 1, and runs the same repeated k-fold procedure.
Each test sample's average class-1 probability across folds is its OOD score.
Samples whose scores exceed the threshold are flagged as out-of-distribution.

This approach is particularly sensitive to **semantic** anomalies — samples
representing classes or object types not present in reference data — rather
than structural defects. Because LightGBM operates on features directly, it
works well on both raw low-dimensional features and on pre-computed embeddings
where distance-based methods may be less discriminative due to the curse of
dimensionality. The cost is the same as `DriftDomainClassifier`: multiple
classifier training runs per call, scaling with `n_folds × n_repeats`.

### Choosing between the three approaches

|                           | Reconstruction (`OODReconstruction`)                      | Distance (`OODKNeighbors`)                     | Domain Classifier (`OODDomainClassifier`)        |
| ------------------------- | --------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| Training required         | Yes — autoencoder on reference data                       | No — only indexes embeddings                   | No — fits classifier each call                   |
| Input                     | Raw images                                                | Pre-computed embeddings                        | Raw features or embeddings                       |
| Anomaly score granularity | Instance + feature map                                    | Instance only                                  | Instance only                                    |
| Best for                  | Structural defects, learned task-specific representations | Fast deployment, strong pre-trained embeddings | Semantic anomalies, complex feature interactions |
| Multimodal reference data | Yes — via GMM extension                                   | Only if embedding separates modes              | Yes — classifier learns any decision boundary    |
| Update cost               | Retrain autoencoder                                       | Re-index reference set                         | Re-run fit on new reference data                 |
| Interpretability          | Feature-level error map                                   | Distance score                                 | Class-1 probability score                        |

In practice, reconstruction and distance-based methods are often complementary:
distance-based detection provides fast initial screening; reconstruction-based
detection provides detailed feature-level analysis for flagged samples. The
domain classifier is most useful when neither reconstruction error nor
embedding distance cleanly separates normal from anomalous samples — for
example, when anomalies are semantically coherent images of unseen object types
that an autoencoder can reconstruct plausibly and that are not spatially distant
in embedding space.

## When to use it

**Drift detection** should begin as soon as a model enters operational
deployment. Every batch of data the model processes is a candidate for drift
monitoring. The monitoring cadence depends on operational tempo — for
high-throughput real-time systems, monitoring may run on statistical samples;
for batch processing systems, each batch should be tested.

For choosing among drift detectors: start with a univariate test (KS as a
baseline) on pre-computed embedding dimensions or image statistics for
computational efficiency. Add MMD or the domain classifier when embedding-space
correlations are important or when univariate tests fail to detect known drift.
Add uncertainty-based detection when you have a deployed classifier and want to
monitor specifically for performance-degrading shift.

**OOD detection** is most valuable at two points: during post-training data ingestion
(to flag anomalous samples before they reach a model) and during operational
monitoring (to identify individual predictions that should be flagged for human
review because the input is outside the training distribution). Reconstruction-
based detection is preferred when the reference data has complex structure that
must be learned and when feature-level localization is needed. Distance-based
detection is preferred for fast deployment, frequent reference updates, or when
high-quality pre-trained embeddings are available. The domain classifier is
preferred when anomalies are semantically coherent samples of unseen types that
neither reconstruction error nor embedding distance readily separates.

To better understand what to do after detecting drift or OOD samples, review the
[Distribution Shift section in the Acting on Results explanation page](ActingOnResults.md#distribution-shift-findings).

## Limitations

All drift detectors require a stable and representative reference dataset. If
the reference is itself biased or unrepresentative of the operational range, the
detector will have a miscalibrated baseline. See the [Dataset bias and
coverage](DatasetBias.md) concept page for guidance on assessing reference
quality.

Statistical drift tests produce p-values, not certainties. A very large batch
will produce statistically significant drift detections for differences that
have no practical impact on model performance — the tests become sensitive to
trivial distributional variation at scale. Conversely, small batches reduce
statistical power and may miss real shift. The operational significance of a
drift detection always requires judgment about the magnitude and nature of the
shift, not just the p-value.

Univariate tests are independent per feature and cannot detect shift that
manifests only in feature correlations. Multivariate tests address this but at
higher computational cost and with reduced interpretability.

Uncertainty-based detection is blind to shift that the model confidently
mishandles. It detects _uncertain_ failure modes, not _confident wrong_
failure modes — the latter require performance monitoring with ground truth
labels, which is a separate capability.

Reconstruction-based OOD detection requires training a model on reference data.
If the reference distribution changes substantially over time, the model must be
retrained. The reconstruction model's quality is also architecture-dependent:
a model with insufficient capacity may fail to reconstruct even in-distribution
samples accurately, producing high false positive rates.

Distance-based OOD detection is subject to the **curse of dimensionality** in
high-dimensional embedding spaces: as dimensionality grows, distances between
points become increasingly uniform, reducing the discriminability of the OOD
score. Dimensionality reduction before KNN scoring may be necessary for very
high-dimensional embeddings.

## Related concept pages

- [Embeddings](Embeddings.md) — the representation space that MMD, domain
  classifier, distance-based OOD, and uncertainty-based drift all operate in
- [Divergence](Divergence.md) — the quantitative distance metric measuring
  distribution gaps
- [Data Integrity](DataIntegrity.md) — when anomalous samples are a
  collection artifact rather than an operational distribution shift
- [Dataset Bias and Coverage](DatasetBias.md) — when the training reference
  distribution is itself unrepresentative, drift detection baselines are
  miscalibrated
- [Performance Limits](PerformanceLimits.md) — when persistent shift indicates
  the model must be retrained rather than monitored

## See this in practice

### How-to guides

- [How to encode with ONNX](../notebooks/h2_encode_with_onnx.py)

### Tutorials

- [Monitoring distribution shift tutorial](../notebooks/tt_monitor_shift.py) —
  end-to-end walkthrough of drift detection on operational data
- [Identifying out-of-distribution samples tutorial](../notebooks/tt_identify_ood_samples.py) —
  comparison of reconstruction and distance-based OOD detection methods

## References

1. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a
   practical and powerful approach to multiple testing. _Journal of the Royal
   Statistical Society: Series B_, 57(1), 289–300. [paper](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x)

2. Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A.
   (2012). A kernel two-sample test. _Journal of Machine Learning Research_,
   13(1), 723–773. [paper](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html)

3. Kuan, J., & Mueller, J. (2022). Back to the basics: Revisiting out-of-
   distribution detection baselines. _arXiv preprint arXiv:2207.03061._ [paper](https://arxiv.org/abs/2207.03061)

4. Lipton, Z., Wang, Y. X., & Smola, A. (2018). Detecting and correcting for
   label shift with black box predictors. _Proceedings of ICML_, 3122–3130. [paper](https://arxiv.org/abs/1802.03916)

5. Rabanser, S., Günnemann, S., & Lipton, Z. (2019). Failing loudly: An
   empirical study of methods for detecting dataset shift. _Advances in Neural
   Information Processing Systems_, 32. [paper](https://arxiv.org/abs/1810.11953)

6. Sethi, T. S., & Kantardzic, M. (2017). On the reliable detection of concept
   drift from streaming unlabeled data. _Expert Systems with Applications_, 82,
   77–99. [paper](https://arxiv.org/abs/1704.00023)

7. Van Looveren, A., Klaise, J., Vacanti, G., Cobb, O., Scillitoe, A.,
   Samoilescu, R., & Athorne, A. (2024). Alibi Detect: Algorithms for outlier,
   adversarial and drift detection. Seldon Technologies. [documentation](https://docs.seldon.ai/alibi-detect)
