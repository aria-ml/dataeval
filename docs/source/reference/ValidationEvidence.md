# Validation evidence by evaluator

This page lists, for each DataEval evaluator and core function, the basis of its evidence, the level of validation
behind it and its contraindications. [Validation and trust](../concepts/ValidationAndTrust.md) explains the evidence
levels used in the tables
([Validation evidence levels](../concepts/ValidationAndTrust.md#validation-evidence-levels)), the difference between
evaluators and core functions
([Core functions vs. evaluators](../concepts/ValidationAndTrust.md#core-functions-vs-evaluators)), and the dependencies
shared by every evaluator.

## Evaluators

### Data quality

See {doc}`Data integrity <../concepts/DataIntegrity>` for what these measure.

:::{list-table}
:widths: 16 26 19 39
:header-rows: 1

- - Evaluator
  - Evidence basis
  - Validation evidence
  - Contraindications
- - {class}`.Duplicates`
  - Perceptual hashing ([Zauner, 2010](../concepts/DataIntegrity.md#references)); clustering
    via scikit-learn KMeans or DataEval's HDBSCAN. See
    {doc}`Clustering <../concepts/Clustering>`.
  - Anchored (hashes); Delegated (KMeans); Ported (HDBSCAN)
  - Hash and embedding modes fail on disjoint transformation classes.
    Neither establishes semantic identity: a hash collision is visual
    similarity, not the same object.
- - {class}`.Outliers`
  - Robust statistics (Double-MAD, z-score, IQR fences) over the image
    statistics vocabulary.
  - Cross-checked (skew, kurtosis, entropy verified against SciPy);
    Anchored (thresholds)
  - The default tolerates skew and heavy tails but assumes one mode: on
    multimodal data (night plus day, mixed sensors) it flags a legitimate
    second population. The z-score `cluster_threshold` is not robust there
    either. Outputs are linting warnings for review, never a deletion list.

:::

Both sets of defaults are tuned for general-purpose imagery. If your dataset is
domain-specific, treat them as a starting point to calibrate, not as a
standard.

```{note}
The two detection modes in {class}`.Duplicates` are complementary, not
competing. Hashing is fast and survives photometric change; embedding-based
clustering survives geometric change. Datasets with augmentation pipelines
generally need both.
```

```{warning}
The specific transformation-by-transformation behavior of each mode is
demonstrated empirically in the
[Detect common augmentations as duplicates](../notebooks/tt_augmentation_duplicates.py)
tutorial. Consult that notebook for the current results rather than relying on
a summary table here: the behavior depends on hash size, cluster sensitivity,
and the extractor in use, and a static table would go stale silently.
```

### Dataset bias

See {doc}`Dataset bias <../concepts/DatasetBias>` for what these measure.

:::{list-table}
:widths: 16 26 19 39
:header-rows: 1

- - Evaluator
  - Evidence basis
  - Validation evidence
  - Contraindications
- - {class}`.Balance`
  - Normalized mutual information ([Linfoot, 1957](../concepts/DatasetBias.md#references);
    [Vinh et al., 2010](../concepts/DatasetBias.md#references)). Estimators from
    scikit-learn.
  - Delegated (MI estimation); Internal only (normalization)
  - A factor whose cut was declared, requested by count or accepted is read as
    codes, so the result depends on that cut, which you may not have chosen. See
    [Metadata binning](../concepts/ValidationAndTrust.md#metadata-binning-a-policy-applied-to-every-factor).
    A value scored from codes is corrected for chance. A value that reads a
    measured factor is not, and the `scored_as` column of `factors` records which
    regime scored each pair.
- - {class}`.Diversity`
  - Shannon and inverse-Simpson indices
    ([Hill, 1973](../concepts/DatasetBias.md#references);
    [Heip et al., 1998](../concepts/DatasetBias.md#references)).
  - Anchored
  - Discrete and categorical variables only. Over binned continuous data the
    score measures the binning as much as the data. See
    [Metadata binning](../concepts/ValidationAndTrust.md#metadata-binning-a-policy-applied-to-every-factor).
- - {class}`.Parity` ⚠️
  - Pearson's chi-squared on independent rows; the second-order Rao–Scott F
    with Kish effective groups on grouped rows ([Rao & Scott, 1984](../concepts/Parity.md#references);
    [Thomas & Rao, 1987](../concepts/Parity.md#references)). Exact hypergeometric and
    Korn–Graubard cell tests, the grouped cells' variance bounded below by a
    Kish cluster-model variance whose within-group correlation for each group
    is the largest of the class's correlation pooled over the other levels, its
    largest at another single level, the level's correlation in the other
    classes, and the level's largest correlation in another single class (the
    model is Kish's; how its correlation is estimated is this package's choice,
    not a cited method), Bergsma's V with its null bias
    matched to the design effect,
    Cochran's screen, Benjamini–Hochberg adjustment; all cited in the
    docstrings. See
    [Parity](../concepts/Parity.md).
  - Cross-checked (statistic, p-value and cell p-values on independent rows
    against SciPy); Anchored (grouped: sandwich covariance against a dense
    implementation of its definition, the cluster-model cell variance against a
    dense implementation of its definition, calibration under the null in eight
    simulated designs within three Monte Carlo standard errors, and no cell
    flagged beyond that bound in eleven null designs with bursty classes: up to
    40 × 40 and 100 classes, two and three classes, group sizes and burst
    patterns that differ by level, classes bursting in blocks of levels at
    20 × 20 and 40 × 40; one further test holds the known limit below at its
    measured rate)
  - Experimental: the API may change or be removed. Dependence between
    groups, such as consecutive sequences from one flight, is not corrected and
    makes p-values too small. With `group_source="declared"`, the p-values hold
    only if the declaration is true. The reference distribution is asymptotic in
    effective groups: calibration was measured in the scenarios on the concept
    page, not proven in general, and power is low with few groups. A grouped
    cell's variance is never below the cluster-model one, which does not shrink
    when the cell is short. In two- and three-level tables of four classes this
    cut the power to find a planted deficit by three quarters to nine tenths
    at 60 to 100 groups (none lost at 200 to 300). In a 20 × 20 table of
    bursty classes over 2,000 images, about five bursts per cell, a class at a
    tenth of its share was never found; over 20,000 images a class at 30% of
    its share always was. A deficit held by few groups is therefore often
    reported as not significant rather than `underrepresented`. Contraindication: a class that comes in bursts at
    one level of the factor only, while that level's other classes do not burst
    (with two classes there are no others), such as a rare class seen in one or
    a few crowded images at one level. A cell of that class at that level that
    holds none of its bursts looks like a real shortage, and no variance
    estimated from the data tells them apart; Parity does not detect it. In
    simulated nulls, 11–17% of tables had a falsely flagged cell with two
    classes and about two bursty images at one of 3 to 10 levels, 3–10% for one
    such cell in a 2 × 2 to 3 × 4 table, and 26–33% and 49–54% when each of 20
    or 40 levels burst in its own class, against the nominal 5%. The
    contraindication is repeated in the `Parity` and `parity` docstrings.
    `inconclusive` and `untestable` are not passes. Nominal: ordered bins are
    read as unordered, and the binning decides cell occupancy. See
    [Metadata binning](../concepts/ValidationAndTrust.md#metadata-binning-a-policy-applied-to-every-factor).
    Measures association, not coverage against a mission.

:::

### Dataset scope

See {doc}`Dataset coverage <../concepts/DatasetCoverage>` for coverage, and
{doc}`Acting on results <../concepts/ActingOnResults>` for how to act on a prioritization.

:::{list-table}
:widths: 16 26 19 39
:header-rows: 1

- - Evaluator
  - Evidence basis
  - Validation evidence
  - Contraindications
- - {class}`.Coverage`
  - Coverage radius from
    [Asudeh et al. (2021)](https://dl.acm.org/doi/abs/10.1145/3448016.3457315);
    the per-class dispersion, isotropy, and near-duplicate signals it adds are
    DataEval's own. `isotropy` is {func}`.completeness`'s effective-dimension
    estimate applied per class; see that row below for its lineage. See
    [Measuring coverage](../concepts/DatasetCoverage.md#measuring-coverage-geometry-in-embedding-space).
  - Internal only
  - Coverage of a poor embedding space is not coverage of the data. Small
    datasets return degenerate radii. `isotropy` is null, not zero, for any
    class below its sample floor, which on a high-dimensional extractor is most
    classes; reduce dimensionality first if you need it.
    The per-class flag defaults are conservative and tuned for deep embeddings.
    Below about 8 dimensions the near-duplicate flag can trip on healthy classes.
- - {class}`.Prioritize`
  - Coreset and cluster-complexity ranking
    ([Sorscher et al., 2022](https://arxiv.org/abs/2206.14486);
    [Abbas et al., 2024](https://arxiv.org/abs/2401.04578);
    [Zheng et al., 2023](https://arxiv.org/abs/2210.15809)). See
    [Prioritization](../concepts/DatasetCoverage.md#prioritization-as-coverage-driven-data-selection).
  - Internal only
  - Ranks by embedding geometry alone, with no notion of label correctness,
    annotation cost, or operational importance. See
    [What Prioritize does not do](../concepts/ActingOnResults.md#what-prioritize-does-not-do).
- - {class}`.Representation`
  - Observed label mass over an {class}`.Ontology`, judged against an expected
    distribution. See {doc}`Ontology <../concepts/Ontology>`.
  - Internal only
  - Only as good as the ontology and expectation supplied. The uniform default
    is wrong for any domain with a skewed operational prior.

:::

### Distribution shift

See {doc}`Distribution shift <../concepts/DistributionShift>` for what these measure and
how to choose among them.

:::{list-table}
:widths: 16 26 19 39
:header-rows: 1

- - Evaluator
  - Evidence basis
  - Validation evidence
  - Contraindications
- - {class}`.DriftUnivariate`
  - [Rabanser et al. (2019)](../concepts/DistributionShift.md#references) methodology.
    All five tests (`ks`, `cvm`, `mwu`, `anderson`, `bws`) are computed by
    `scipy.stats`. Corrections: Bonferroni,
    [Benjamini-Hochberg (1995)](../concepts/DistributionShift.md#references).
  - Delegated (test statistics); Ported (workflow, from alibi-detect 0.11.4)
  - Do not apply to raw high-dimensional input. Univariate tests over
    thousands of correlated dimensions lose power even with correction; reduce
    dimensionality first. Feature-wise tests cannot see shift that exists only
    in the joint distribution.
- - {class}`.DriftMMD`
  - [Gretton et al.](../concepts/DistributionShift.md#references) kernel two-sample test;
    permutation-based significance.
  - Ported (alibi-detect 0.11.4)
  - Permutation testing scales quadratically in sample size: unsuitable for
    per-frame or real-time use without chunking.
    `sigma=None` selects a bandwidth heuristic that is not universally
    appropriate, and `n_permutations` bounds the smallest reachable p-value.
- - {class}`.DriftDomainClassifier`
  - Domain-classifier drift detection; cross-validated discriminability.
  - Internal only
  - AUROC is optimistic under severe imbalance between reference and test sets.
    A negative result is weak evidence: it may mean no drift, or an
    underpowered proxy classifier.
    The verdict is one AUROC threshold.
- - {class}`.DriftKNeighbors`
  - Nearest-neighbor distance-based detection. See
    [K-nearest neighbor drift detection](../concepts/DistributionShift.md#k-nearest-neighbor-drift-detection-driftkneighbors).
  - Internal only
  - Depends on the embedding metric being meaningful. Sensitive to `k` and to
    reference-set size.
- - {class}`.DriftReconstruction`
  - Autoencoder reconstruction error. The detector is DataEval's own; the
    scoring math it calls is shared with {class}`.OODReconstruction`.
  - Internal only (detector); Ported (shared scorer, from alibi-detect 0.11.4)
  - Inherits every bias of the autoencoder it trains or is handed.
    Reconstruction error conflates "novel" with "hard to reconstruct."
- - {class}`.DriftWasserstein`
  - Wasserstein distance against a validation baseline. See
    [Wasserstein-based drift detection](../concepts/DistributionShift.md#wasserstein-based-drift-detection-driftwasserstein).
  - Internal only
  - Requires a held-out validation split to set the baseline, and the baseline
    is one train/validation comparison. Unusable on constant or discrete
    features and on infinite-variance tails. The mean does not see a shift
    confined to a few of many features (1 of 512 was not detected). In chunked
    mode only the default constant threshold is calibrated.
- - {class}`.ChunkedDrift`
  - Wrapper applying any detector over sequential chunks.
  - Internal only
  - Chunk size trades detection latency against statistical power; small chunks
    give unstable per-chunk verdicts. A configuration choice, not a property of
    the wrapped detector. The chunk size is also applied to the reference, and
    every threshold except `ConstantThreshold` sets its bounds from the spread
    of the reference chunks, so `fit` rejects a reference that splits into
    fewer than 3 chunks. A few more than 3 still gives a rough spread.
- - {class}`.OODReconstruction`
  - Autoencoder / VAE reconstruction-based OOD scoring.
  - Ported (alibi-detect 0.11.4); Cross-checked against scikit-learn in tests
  - Inherits every bias of the autoencoder/VAE it trains or is handed.
    Per-instance scores are not calibrated probabilities; the threshold needs a
    trusted in-distribution reference set.
    The default `threshold_perc=95.0` assumes the reference set is 5% contaminated.
- - {class}`.OODKNeighbors`
  - Distance-based OOD scoring. See
    [Distance-based detection](../concepts/DistributionShift.md#distance-based-detection-oodkneighbors).
  - Cross-checked (against scikit-learn in tests)
  - Same embedding dependence as `DriftKNeighbors`. Threshold requires
    calibration on trusted in-distribution data.
    The default `threshold_perc=95.0` carries the same 5% contamination assumption.
- - {class}`.OODDomainClassifier`
  - Domain-classifier OOD detection.
  - Internal only
  - Same imbalance caveat as `DriftDomainClassifier`.

:::

```{note}
Drift detection answers "has the input distribution changed?" It does not
answer "has model performance degraded?" DataEval measures the first. See
[Taxonomy of shift](../concepts/DistributionShift.md#taxonomy-of-shift) for why a detected
shift may be harmless and an undetected one may not be.
```

### Performance

See {doc}`Performance limits <../concepts/PerformanceLimits>`.

:::{list-table}
:widths: 16 26 19 39
:header-rows: 1

- - Evaluator
  - Evidence basis
  - Validation evidence
  - Contraindications
- - {class}`.Sufficiency`
  - Power-law learning curves
    ([Hestness et al., 2017](../concepts/PerformanceLimits.md#references)); curve fitting
    via basin-hopping ([Wales & Doye, 1997](../concepts/PerformanceLimits.md#references)).
  - Internal only
  - Extrapolation assumes future data is drawn from the same distribution as
    the pilot data. If new collection introduces shift, the projection is
    void. Projections far beyond the largest measured substep are speculative.
    A single run, the default, yields no usable confidence interval; raise `runs` before quoting one.

:::

## Core functions

Core functions carry no thresholds or policy. Their contraindications are
statistical and computational.

### Feasibility and performance limits

:::{list-table}
:widths: 20 26 18 36
:header-rows: 1

- - Function
  - Evidence basis
  - Validation evidence
  - Contraindications
- - {func}`.ber_knn`, {func}`.ber_mst`
  - "Learning to Bound the Multi-class Bayes Error", Theorems 3 and 4
    ([arXiv:1811.06419](https://arxiv.org/abs/1811.06419)), cited in both
    docstrings. See
    [BER estimation](../concepts/PerformanceLimits.md#ber-estimation-mst-and-knn-bounds).
  - Ground-truth checked (bounds must contain, and stay tight around, the
    analytic Bayes error of two Gaussians at four separations)
  - Only as meaningful as the embedding: a weak extractor inflates apparent
    class overlap and overstates difficulty. `ber_knn` is sensitive to `k`; MST
    construction is costly at scale. Classification only.
- - {func}`.uap` ⚠️
  - Empirical upper-bound average precision
    ([Borji & Iranmanesh, 2019](../concepts/PerformanceLimits.md#references)).
  - Anchored
  - Experimental: the API may change or be removed. Reduces detection to
    classification over crops; ignores localization error entirely.
- - {func}`.nullmodel_metrics` and the `nullmodel_*` family
  - Dummy-classifier baselines.
  - Ground-truth checked (uniform, proportional, and dominant-class accuracies
    matched against `1/C`, `sum(p_i * q_i)`, and `p[argmax(q)]`)
  - Baselines only: what a trivial model achieves on your class distribution,
    not what is achievable.

:::

### Divergence and embedding geometry

:::{list-table}
:widths: 20 26 18 36
:header-rows: 1

- - Function
  - Evidence basis
  - Validation evidence
  - Contraindications
- - {func}`.divergence_mst`, {func}`.divergence_fnn`
  - Henze-Penrose style divergence via multivariate run statistics. See
    {doc}`Divergence <../concepts/Divergence>`.
  - Ground-truth checked (returns 0 for identical distributions and 1 for
    disjoint ones, and increases monotonically with separation)
  - Embedding-dependent, like BER. A divergence near 0 in a poor embedding
    space is not evidence that two datasets match.
- - {func}`.minimum_spanning_tree`, {func}`.compute_neighbors`
  - Standard graph and neighbor construction.
  - Internal only
  - Geometric primitives with no statistical interpretation on their own.
- - {func}`.cluster`, {func}`.compute_cluster_stats`
  - scikit-learn KMeans; DataEval's HDBSCAN implementation. See
    {doc}`Clustering <../concepts/Clustering>`.
  - Delegated (KMeans); Ported (HDBSCAN, from fast_hdbscan);
    Cross-checked (clusterer tests use scikit-learn)
  - Cluster counts are a modeling choice, not a discovered fact. See
    [Practical considerations](../concepts/Clustering.md#practical-considerations).
- - {func}`.completeness`
  - Effective dimensionality $d_\text{eff} = e^H$ over the covariance
    eigenvalues: the eigenvalue form of the effective rank of
    [Roy & Vetterli (2007)](../concepts/DatasetCoverage.md#references), also used by
    [Kim et al. (2023)](../concepts/DatasetCoverage.md#references) and
    [Zhuo et al. (2023)](../concepts/DatasetCoverage.md#references), and the entropy-based
    sibling of the participation ratio
    ([Gao et al., 2017](../concepts/DatasetCoverage.md#references)). All four are cited in the
    docstring. The two normalizations are DataEval's own: $d_\text{eff}/d$
    (completeness) and $d_\text{eff}/r$ (isotropy). See
    [Completeness and isotropy](../concepts/DatasetCoverage.md#measuring-coverage-geometry-in-embedding-space).
  - Internal only
  - Measures how the data fills the embedding's dimensions, not whether the
    embedding is semantically right. The two scores are not directly comparable
    as a ratio: completeness decomposes the rank-normalized matrix, isotropy the
    raw centered one. `isotropy` is degenerate unless samples exceed dimensions.
- - {func}`.coverage_naive`, {func}`.coverage_adaptive`
  - Radius-based coverage of
    [Asudeh et al. (2021)](https://dl.acm.org/doi/abs/10.1145/3448016.3457315),
    with the 20-50 observation guidance from Sudman (1976), both cited in the
    docstrings. See
    [Measuring coverage](../concepts/DatasetCoverage.md#measuring-coverage-geometry-in-embedding-space).
  - Internal only
  - Undefined below a minimum sample count. The naive and adaptive radii
    answer different questions; they are not interchangeable.
- - {func}`.feature_distance`
  - Feature-wise distance between two continuous distributions.
  - Internal only
  - Feature-wise only; blind to joint-distribution differences.

:::

### Ranking and selection

:::{list-table}
:widths: 20 26 18 36
:header-rows: 1

- - Function
  - Evidence basis
  - Validation evidence
  - Contraindications
- - {func}`.rank_kmeans_distance`, {func}`.rank_hdbscan_distance`
  - Self-supervised prototypicality metric of
    [Sorscher et al. (2022)](https://arxiv.org/abs/2206.14486): cluster the
    embeddings, rank each sample by distance to its centroid.
  - Internal only
  - Favors sparse regions: on non-uniform manifolds it can oversample sparse
    outlier regions and undersample dense, decision-boundary-rich ones.
- - {func}`.rank_kmeans_complexity`, {func}`.rank_hdbscan_complexity`
  - Concept-cluster complexity of
    [Abbas et al. (2024)](https://arxiv.org/abs/2401.04578): complexity is the
    product of intra- and inter-cluster distance, and the sampling budget is
    allocated across clusters in proportion to it.
  - Internal only
  - Complexity weighting mitigates but does not eliminate the above. Depends on
    cluster structure actually existing in the embedding.
- - {func}`.rank_knn`
  - Nearest-neighbor distance profiling.
  - Internal only
  - Sensitive to `k` and to local density variation.
- - {func}`.rank_result_class_balanced`, {func}`.rank_result_stratified`
  - `rank_result_stratified` implements the score-bin stratification of
    [Zheng et al. (2023)](https://arxiv.org/abs/2210.15809), which preserves
    distribution coverage at high pruning rates.
    `rank_result_class_balanced` has no identified source.
  - Internal only
  - These impose a selection policy on an observation. Using them means
    accepting a balance or stratification objective that may conflict with the
    ranking's original criterion.

:::

### Image statistics and hashing

:::{list-table}
:widths: 20 26 18 36
:header-rows: 1

- - Function
  - Evidence basis
  - Validation evidence
  - Contraindications
- - {func}`.image_stats`, {func}`.combine_stats_results`,
    {func}`.compute_ratios`
  - Image statistics vocabulary. See
    [Image statistics in integrity checks](../concepts/DataIntegrity.md#image-statistics-in-integrity-checks).
  - Cross-checked (skew, kurtosis, and entropy verified against SciPy in CI)
  - Descriptive only. A statistic being unusual is not evidence a sample is
    bad. Every pixel counts, so a part of the vehicle in frame shifts the
    statistics. On real drone footage, a hood covering about a fifth of the
    frame lowered sharpness to 89% of the clean value, against 93% for lens
    vignetting alone. Compare absolute values only within footage from one
    platform. Within one video, a part did not change which frames a sharpness
    gate dropped.
- - {func}`.image_stats` (`ImageStats.BLUR`) and {func}`.video_stats` (`VideoStats.BLUR`)
  - Edge width after Liang et al. (2010), with a percentile summary in place of the paper's own score.
  - Internal only (the vectorized edge widths match a slower per-point loop kept in the tests, from the team's earlier
    prototype, so not independent; tests on synthetic images check the trends, the NaN cases and determinism; nothing
    is compared with a published implementation)
  - Experimental: the column names and values may change. Checked on six real frames with synthetic degradations and on
    11 clips with simulated blur, with no human-rated blur. Widths are in pixels, so compare only at one resolution. Not
    a calibrated level of defocus and not a motion blur detector. Noise and JPEG artifacts move it by amounts and in
    directions that depend on the scene, and a soft 8-bit image at low contrast reads sharper than at full contrast. See
    [Blur statistics](../concepts/BlurStatistics.md).
- - {func}`.image_stats` (`ImageStats.SPATIAL`)
  - The geometry of a box against its peers and the image: the distance between box centers over the box's own
    diagonal, the share of the box inside the image, the share the other boxes cover, and the largest overlap with a
    box of the same class and of another.
  - Anchored (hand-computed fixtures; the union is also compared with a raster reference written for the tests, which
    is not independent of it. A run on the 118,127 real boxes of one aerial dataset, VisDrone2019-MOT validation, with
    a stand-in picture of the real shape for each frame, gave no NaN, no invalid box and no warning. The overlaps
    agreed with a brute-force calculation to 3e-8, and the covered fraction agreed with a sampled-grid estimate to
    0.011 at most. That estimate was written by the statistic's author, so it checks the arithmetic and not the
    definition, and the run has no ground truth for what the statistics are used for)
  - Descriptive geometry only: the values do not predict a model's behavior or measure occlusion. Boxes are
    rectangles, and an object nobody annotated covers nothing, so the covered fraction and the overlaps describe
    annotated boxes and not objects. The distance is directed and center to center, not a gap. The fraction reads 1.0
    on datasets whose annotation tools clip boxes to the image, and after a {class}`.Resize` or {class}`.Crop`. The
    class overlaps read labels, and where a missing label could hide the largest overlap the value is NaN, which is
    not a guess. Where another box of the image is corrupt, every value is computed from the valid boxes and flagged
    `incomplete_geometry`, so the largest distances are the least reliable and the covered fraction and the overlaps
    can be too small. On that aerial footage the covered fraction ran opposite to the dataset's own `occlusion` field (a
    mean of 0.11 on the boxes the field marks 1 and 0.38 on those it marks 0), and the export does not document that
    field, so do not read the covered fraction as occlusion. Most of the highest other-class overlaps paired a person
    class with a bicycle or motor class, which the class names suggest is a rider on the vehicle. No image was
    inspected, so the run does not show that the largest other-class overlap finds label conflicts. Boxes that mark
    areas the annotators skipped, passed as a class, raised the covered fraction of the ordinary boxes, so drop them
    first.
- - {func}`.image_stats` (`ImageStats.CONTEXT`)
  - The intensity of the pixels around a box: the mean and the population standard deviation, over a frame a quarter of
    the box's width and height wide, less the box and less every other valid box, with the intensity mapped to [0, 1]
    from the image's interval and averaged over the channels.
  - Anchored (hand-computed fixtures for the overlap, the edge and the encodings; a per-pixel check written from the
    same definition, so not independent of it, agrees on 200 awkward boxes; constructed images only, plus a run on the
    real boxes of one aerial dataset that counts the statuses and times the statistic, and checks no value)
  - Descriptive only: the values do not say whether a target is camouflaged or in clutter, which are the readings built
    on them. An object that no box annotates counts as surround. A box that contains another box's window, such as an
    area marked as ignored, leaves that box no surround, so it reads `insufficient_support`, and a small box can fall
    under the 32-pixel floor at the defaults. On the 114,133 boxes of VisDrone2019-MOT validation that are not ignored
    regions, 373 read `insufficient_support`: 257 had nothing left of their surround after the other boxes, and 116 had
    fewer than 32 measurable pixels left. With the ignored regions kept as a class, 818 of 118,127 did, 686 of them with
    nothing left. A box near the image edge reads from a ring that the edge has cut. An image of integers with no
    declared range is read against the interval its own maximum decodes to, so compare images only where the range is
    declared. The intensity is a mean over the channels, so hue does not show in it. The padding of a quarter and the
    floor of 32 pixels were chosen by no study.
- - {func}`.track_stats`
  - Per-track statistics for video sequences.
  - Anchored
  - Assumes track identity is already resolved: detections carrying no track id
    take no part, and nothing here re-associates them. `max_gap_length` and `net_heading_rad`
    are tested on constructed tracks only. The heading is that of the chord from the first to
    the last observed center, so it says nothing about the path and is set by box jitter where
    a track barely moved.
- - {func}`.ego_stats`
  - Per-frame camera ego-motion, from a robust affine fit to cheap optical flow,
    with declared roll-ups to the sequence level. See
    [Camera motion as a per-frame factor](../concepts/MetadataLevels.md#camera-motion-as-a-per-frame-factor).
  - Ground-truth checked (kinematics recovered from analytically constructed
    flow fields)
  - Measures apparent motion, so it cannot distinguish a moving camera from
    a scene moving coherently against a still one. A pair is refused rather than
    guessed when the camera cannot be told apart from an independent mover, a
    mover captures the fit, or the fit settles between the camera's motion and a
    large mover's, which the samples read better as two motions; when the fitted
    motion does not explain the new frame
    (a hard cut, a large field-of-view switch, a misread fast pan) or nothing in
    the frame confirms it (a nearly textureless noisy frame); when the pair
    cannot be read (the frames differ in size, a processed side is under 32 px,
    or a frame holds a non-finite pixel or is a single gray level); when the flow
    samples cannot pin an affine motion or, with a `mask`, too few remain or they
    or the fit's inliers do not spread across the frame; or when the pair departs
    from the camera state the rest of the sequence agrees on (the temporal gate,
    which assumes a roughly steady camera). The per-frame `ego_trusted` flag
    reports which frames were refused, and a sequence roll-up answers null rather
    than from too little evidence once fewer than a quarter of its frames were
    trusted. The affine model is first-order and carries no depth, so a camera
    translating through a scene with strong parallax is fitted only
    approximately. `pan_speed` and `mover_speed` are in the pixels of the frames
    they were measured on and are not comparable across resolutions without
    normalizing; the rates are. A reading is attributed to the later frame of
    the pair it was measured from, so the leading `gap` frames of every sequence
    carry no reading. A burned-in overlay (a HUD, telemetry text) reads as scene
    content: a light one gets pans refused, and a dense one drags a pan's reading
    toward a still camera, which stays trusted when the drag is mild and is
    refused when it splits the samples into two motions. Passing the overlay as
    `mask=` excludes it. A part of the vehicle in view, such as a hood or a bow,
    reads as scene content the same way. On real drone footage, an unmasked hood
    covering about a fifth of the frame read about one pair in nine wrong while
    trusting it, and a mask brought that under one in a hundred. A mover pulling
    the fit toward its own motion is refused once its pull stands well above the
    flow noise, so a slow mover in noisy footage can still read as a trusted
    zoom of a pixel or two. A mover holding most of the samples the fit sees
    (with a mask, most of what the mask leaves) can be read as the camera; one
    pair cannot tell the two apart, and the temporal gate catches it only when
    it is brief.
- - {func}`.frame_stats` ⚠️
  - Per-frame time since the previous frame and whether a frame repeats the one before it exactly, with declared
    roll-ups to the sequence level.
  - Anchored (clocks and frames constructed with known steps and repeats)
  - Experimental: the names may change. Reads the timestamps a decoder reports and does not check them against the
    capture. A missing, repeated or restarted timestamp leaves the step NaN rather than guessed, and a clip that starts
    later is one long step. A repeat is exact equality, so a re-encoded or noisy copy of a frame reads as not repeated.
    Tested on constructed inputs only, not on real footage.
- - {func}`.track_steps` ⚠️
  - Per-detection step from the previous observation of its track: distance in diagonals of the previous box, that
    distance per frame and per second, overlap and scale change, with a status for every row that cannot be measured.
  - Anchored (boxes constructed with known steps, scales and overlaps)
  - Experimental: the names may change. A step is not conditioned on shot boundaries or camera motion, so a cut or a
    pan reads as a large step, and the function does not judge identity, so an id switch reads as one too. Frame sizes
    are compared as the stream reports them, so a lazy stream hides a resolution change inside one file. Tested on
    constructed inputs only, not on real footage.
- - {func}`.video_stats` (`VideoStats.PERSISTENCE`)
  - Per-pixel persistence while the scene moves behind the camera, and a
    flat-field estimate of how much each part of the frame softens the same
    content, from frame pairs aligned by a similarity transform.
  - Anchored (constructed videos with a known burned-in patch and known
    blurred regions)
  - Experimental: the map shape, column names and keywords may change.
    Checked only on synthetic and NRTK-simulated defects, never on real
    damaged footage. Needs camera motion and texture: a still or flat video
    reads NaN, and so does a flat region fixed to the camera. A frame pair
    counts only when one camera motion fitted to what changed explains more of
    the frame than "nothing moved" does, and a pixel only where that motion
    moved it at least 2 px at a 360 px short side, so a still camera over
    movers, or one moving slower than that, reads NaN rather than reading its
    scenery as fixed (see [Studies on real footage](../concepts/ValidationAndTrust.md#studies-on-real-footage)).
    One mover that fills most of the frame and moves as one can still pass for
    the camera. The blur maps are relative to each video's average, so softness
    that is the same everywhere is invisible, and a loss of sharpness that
    grows at a steady rate across the frame reads 0. `blur_map` removes blur
    wider than about `blur_sigma` cells. `blur_map_raw` keeps it, but on clean
    real footage the raw map of one video read large areas as soft that no lens
    defect caused, so read it across many videos from one camera. `ego_map`
    does not tell a burned-in overlay from a part of the vehicle. Frame pairs
    are 10 and 30 frames apart, so they span different times at different frame
    rates.
- - {func}`.phash`, {func}`.phash_d4`, {func}`.dhash`, {func}`.dhash_d4`,
    {func}`.xxhash`
  - Perceptual hashing ([Zauner, 2010](../concepts/DataIntegrity.md#references)); xxHash
    for exact matching.
  - Anchored
  - `xxhash` detects exact byte-level duplicates only. Perceptual hashes
    survive photometric change but not arbitrary geometric warping; the `_d4`
    variants add invariance to the eight square symmetries, not to arbitrary
    rotation.
- - {func}`.hash_neighbors` (`partition`)
  - Every pair of packed hashes within a Hamming radius, by multi-index hashing.
    With `partition`, only the pairs whose two labels differ, found without
    expanding a pair inside one label.
  - Cross-checked (the pairs and distances equal those of an independent
    brute-force reference kept to the pairs whose labels differ, under both of
    its search strategies)
  - Experimental: the argument may change. No study on real footage backs it.
    The search is not narrowed, so distinct hashes under one label are still
    compared. {class}`.Duplicates` passes it in its video segment search and
    raises no {class}`~dataeval.exceptions.ExperimentalWarning` there.
- - {func}`.hash_neighbors` (`k`), {class}`.Duplicates` (`segment_neighbors`)
  - With `k`, each position keeps its `k` nearest neighbors among the hashes
    within a Hamming radius, and a pair is reported when either of its
    positions keeps it. `segment_neighbors` passes it to the video segment
    search, as an opt-in cap on the pairs each frame contributes.
  - Cross-checked (against an independent brute-force reference, every pair
    reported is among the nearest `k` of one of its positions and every
    position keeps its `k` nearest, up to ties; the rule that chooses among
    equally near neighbors is pinned against an independent implementation of
    it). One {class}`.Duplicates` test on a single synthetic fixture of two
    fixed-camera videos.
  - Experimental: the arguments may change. No real fixed-camera footage was
    studied. Pairs within the radius are left out by design, so a cap can
    shrink or move a segment's extent, boundaries and `mean_distance`, and can
    drop the segment the static runs of two videos make while keeping the
    stretch of distinct frames they share. 10 is a provisional starting point
    for `segment_neighbors`, chosen on simulated static runs, for shared
    stretches inside runs of hundreds to thousands of frames; a larger value is
    not safer. Among equally near neighbors the result depends on the order of
    the input. The search for pairs of distinct hashes is not capped, so many
    distinct hashes can still exceed the pair budget, and a pair of videos that
    does exceed it loses its segments with a warning.

:::

### Metadata and bias

:::{list-table}
:widths: 20 26 18 36
:header-rows: 1

- - Function
  - Evidence basis
  - Validation evidence
  - Contraindications
- - {func}`.parity` ⚠️
  - Pearson's chi-squared, or the second-order Rao–Scott F over `groups`, with
    exact cell tests; Bergsma, Cochran, Rao & Scott, Thomas & Rao, Korn &
    Graubard, Kish and Benjamini & Hochberg are cited in the docstrings. See
    [Parity](../concepts/Parity.md).
  - Cross-checked (independent rows, against SciPy); Anchored (grouped)
  - Experimental. Needs `groups` or `assume_independent=True`, and raises
    without one. Same grouping, calibration and binning caveats as
    {class}`.Parity` ⚠️. An untestable factor or cell is NaN, not independent:
    read `testable` and `insufficient` before using a score.
- - {func}`.label_parity`
  - Pearson's chi-squared test of homogeneity on the 2 × K table of class
    counts; the second-order Rao–Scott F when groups are given. See
    [Label parity](../concepts/DistributionShift.md#label-parity).
  - Cross-checked (statistic and p-value without groups against
    `scipy.stats.chi2_contingency`); Anchored (calibration at unequal sample
    sizes, grouped calibration with unequal detections per image, symmetry)
  - Detection and tracking labels need `expected_groups` and `observed_groups`:
    read as independent labels, two detection datasets with one class
    distribution were declared different 84–91% of the time in simulation.
    Calling without groups or `assume_independent=True` warns in 1.2 and raises
    from 2.0. Detection-label frequency only, not image-level class presence.
    Both inputs need one class vocabulary. "No shift observed" is not evidence
    of parity. Says nothing about feature-space differences between the two
    sets.
- - {func}`.mutual_info`, {func}`.mutual_info_classwise`
  - [Linfoot (1957)](https://www.sciencedirect.com/science/article/pii/S001999585790116X),
    cited in the docstring.
  - Delegated (estimators from scikit-learn)
  - Handles continuous factors natively in the factor-to-factor block:
    {func}`.is_continuous` selects `mutual_info_regression` per column and the
    Linfoot transform maps the result to [0, 1], so binning is a
    {class}`.Metadata` convention, not a requirement here, but
    {class}`.Balance` applies it anyway, see
    [Metadata binning](../concepts/ValidationAndTrust.md#metadata-binning-a-policy-applied-to-every-factor). The
    class-to-factor row instead divides by the class entropy and subtracts the
    mutual information expected under a random assignment with the same
    margins, so it is not biased upward at small n the way that block is.
- - {func}`.is_continuous`
  - DataEval-original heuristic. Uses the Wasserstein distance from
    `scipy.stats`; the normalized near-neighbor construction and the two
    corroborating signals are DataEval's own. See
    [Metadata binning](../concepts/ValidationAndTrust.md#metadata-binning-a-policy-applied-to-every-factor).
  - Internal only
  - A tuned classifier, not a test with a stated error rate: the five constants
    that decide its verdict have no derivation and no measured
    misclassification rate. Returns `False` below 20 samples or 3 distinct
    values, so on a small dataset every numeric factor is called discrete.
    Its answer decides whether a factor gets binned, so a wrong call is
    invisible in the metric that follows.
- - {func}`.factor_deviation`, {func}`.factor_predictors`
  - Metadata attribution for flagged samples. See
    [Diagnosing findings with metadata](../concepts/ActingOnResults.md#diagnosing-findings-with-metadata).
  - Internal only
  - Associational, not causal. A factor that predicts flagged samples is a
    lead to investigate, not an established cause.

:::

### Labels and ontology

:::{list-table}
:widths: 20 26 18 36
:header-rows: 1

- - Function
  - Evidence basis
  - Validation evidence
  - Contraindications
- - {func}`.label_errors`
  - Embedding-geometry label error detection. See
    [Label error detection](../concepts/DataIntegrity.md#label-error-detection-embedding-geometry).
  - Internal only
  - Produces candidates for review, not confirmed errors. Precision depends
    on embedding quality; ambiguous samples are flagged as readily as
    genuine mistakes.
- - {func}`.label_stats`
  - Label distribution accounting.
  - Anchored
  - Descriptive.
- - {func}`.label_coverage`
  - Observed label mass over an {class}`.Ontology`. See
    {doc}`Ontology <../concepts/Ontology>`.
  - Internal only
  - Reports observed distribution only. Judging that distribution against an
    expectation is {class}`.Representation`'s job, not this function's.
- - {func}`.label_reconciliation`, {func}`.label_alignment`
  - Ontology reconciliation and matching. See
    [Reconciliation](../concepts/Ontology.md#reconciliation-checking-labels-against-the-ontology)
    and [Alignment](../concepts/Ontology.md#alignment-relating-two-vocabularies).
  - Internal only
  - Matcher output is a proposal requiring human confirmation. Confidence
    scores are matcher-relative and not comparable across matchers. See
    [Confidence and abstention](../concepts/Ontology.md#confidence-and-abstention).
- - {func}`.ontology_validation`
  - Structural and naming facts about an ontology artifact. See
    [Validation](../concepts/Ontology.md#validation-checking-the-ontology-artifact).
  - Internal only
  - Checks structure, not semantic correctness. A well-formed ontology can
    still be the wrong ontology for your problem.

:::

## Related concept pages

- [Validation and trust](../concepts/ValidationAndTrust.md): when to trust a result, the evidence levels, how DataEval
  validates its own implementations, and the known gaps
- [Acting on results](../concepts/ActingOnResults.md): what to do with a finding once it is trusted
