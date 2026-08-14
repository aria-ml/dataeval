<!-- markdownlint-disable MD051 -->

# Validation and Trust

## When should you trust a DataEval result?

Use this page as a pre-flight check before citing any DataEval metric. Every
evaluator relies on specific assumptions about your data, so the validity of a
measurement depends on whether those assumptions hold. This page makes them
explicit.

DataEval's evidence for its own correctness is
[real and checkable](#how-dataeval-validates-its-own-implementations), but it is
not uniform across the library — so every evaluator and core function is rated
individually rather than covered by a blanket claim. Each table below carries
the entry's name plus these columns:

- **Evidence basis** — the source of the method: a published result, a DataEval
  concept page where the derivation is written down, or a third-party library.
  Where the implementation delegates, that library is named, because its
  correctness is the relevant evidence.
- **Validation evidence** — how DataEval verifies its own implementation in CI.
  One of six levels, [defined below](#validation-evidence-levels).
- **Policy levers** — the parameters that gate the evaluator's decisions, with
  their defaults. **Evaluator tables only**, because core functions carry no
  policy. A default is a choice someone made, not a property of the math.
- **Contraindications** — conditions under which the output is invalid or
  meaningless. These are not performance caveats; they are cases where the number
  is wrong or undefined and should not be reported.

Check your intended metric's row before you cite it: verify that the
contraindications do not apply to your data, and use the validation level to
calibrate the strength of your claim. Where DataEval's own evidence is thin,
[Known gaps](#known-gaps-in-this-pages-evidence) says so directly.

```{important}
DataEval is not a certification tool. It produces measurements, not pass/fail
judgments. If a result surprises you, check the policy behind it (for
evaluators), the extractor that produced its input (for anything
embedding-based), or the
[binning applied to its factors](#metadata-binning-a-policy-applied-to-every-factor)
(for anything metadata-based).
```

### Core functions vs. evaluators

The library is split into two layers &mdash; [Evaluators](#evaluators) and
[Core functions](#core-functions) &mdash; each with its own potential failure
modes.

- Core functions ({doc}`dataeval.core <../reference/autoapi/dataeval/core/index>`)
  are stateless and observe raw data — a bound, a count, a distance,
  a distribution, with no threshold, prior, or recommendation attached.
  The trust question is: **"is the math right, and do my data satisfy its
  assumptions?"**
- **Evaluators** ({doc}`dataeval.quality <../reference/autoapi/dataeval/quality/index>`,
  {doc}`dataeval.bias <../reference/autoapi/dataeval/bias/index>`,
  {doc}`dataeval.scope <../reference/autoapi/dataeval/scope/index>`,
  {doc}`dataeval.shift <../reference/autoapi/dataeval/shift/index>`,
  {doc}`dataeval.performance <../reference/autoapi/dataeval/performance/index>`)
  wrap core observations in policy: a
  threshold, a default distribution, a significance level, a ranking rule. The
  evaluator tables name the policy each one carries, because that policy — not
  the underlying math — is the most common source of results that do not transfer
  between domains. The trust question is: **"is the math right, *and is the built-in
  policy appropriate for my problem?*"**

## Feature extractors: a dependency of both layers

Extractors are not a third kind of feature. They produce the embeddings that
many of the functions and classes rely on. **This makes extractor choice the
single largest uncontrolled variable in a DataEval assessment.** BER,
divergence, coverage, prioritization, label-error detection, and every
embedding-based drift detector inherit the geometry of whatever extractor
produced their input, so an extractor problem shows up as a wrong number in a
table below rather than as an error.

{doc}`Embeddings <Embeddings>` covers how to choose one. Extractors carry no
validation-evidence rating of their own, because they produce inputs rather than
measurements: for the model-backed ones, the evidence is whatever you have about
the model you supplied. What follows is every extractor DataEval ships, what it
actually embeds, and the conditions under which it will mislead you.

### Extractors that embed the data

These describe the images. Distances between their outputs are claims about the
data itself, which is what every geometric metric in this library assumes.

:::{list-table}
:widths: 20 32 48
:header-rows: 1

- - Extractor
  - What it produces
  - What it inherits, and when it misleads
- - {class}`.FlattenExtractor`
  - Raw pixels flattened to `(n_images, C x H x W)`. No framework, no weights,
    no randomness.
  - The only extractor with nothing to version and nothing to seed — no model,
    no fitted state — but pixel space is not semantic space. Distances are
    dominated by
    brightness, alignment, and background rather than content, and the
    dimensionality is the pixel count, so coverage radii, completeness, and
    isotropy are being estimated in tens of thousands of dimensions from a few
    thousand samples. A baseline, not an assessment.
- - {class}`.BoVWExtractor`
  - L2-normalized histogram of SIFT visual words, `(n_images, vocab_size)`.
    Requires the `opencv` extra.
  - Invariant to rotation, scale, and minor viewpoint change, which is why it is
    the option for finding near-duplicates that survived a geometric transform.
    Two hazards: it is the **only stateful extractor** — `__call__` fits the
    vocabulary on the first data it sees and reuses it thereafter, so two
    separately constructed instances produce embeddings that are not comparable
    (fit once, share the instance) — and the vocabulary is a k-means fit, so
    reproducibility depends on the
    [global seed](../notebooks/h2_configure_defaults.py#configuring-the-global-seed).
    Images with no SIFT keypoints (uniform or untextured) become all-zero rows,
    which read as identical to each other.
- - {class}`.TorchExtractor`
  - A `torch.nn.Module`'s output, or a hooked intermediate layer
    (`layer_name`, `use_output`).
  - Inherits the biases and training distribution of the model you supply. A
    model pre-trained on natural imagery may embed overhead or infrared data
    poorly, and nothing in DataEval will warn you that it has. Two additional
    choices change the geometry silently: an intermediate layer and the head
    output are different spaces with different metrics, and `transforms` that do
    not match the model's training preprocessing degrade the embedding without
    raising anything. Device and precision differences perturb values slightly.
- - {class}`.OnnxExtractor`
  - The same, executed through ONNX Runtime, with lazy model loading and
    automatic provider selection (CUDA, then CoreML, then CPU).
  - Everything above, plus: on a multi-output model `output_name` decides which
    tensor becomes the embedding, and picking the wrong one produces a valid
    array rather than an error; `image_size` overrides the model's native input
    size by bilinear resize. Because the execution provider is selected from
    what is available, the same model on two machines can differ numerically —
    do not treat runs as bitwise comparable across hardware.

:::

### Extractors that embed a model's predictions

These describe what a **model does**, not what the data looks like. All of
them re-run the supplied model and use only its predictions, ignoring any
ground-truth targets carried by the dataset. Three consequences follow, and they
apply to every row in this table:

- Drift found on these features is drift in **model behavior**. It may be caused
  by the input distribution, but the measurement cannot separate the two.
- Changing the model or its weights changes the features. Results are not
  comparable across model versions, only across data for a fixed model.
- For detection models, the number of rows is itself a model output, so sample
  counts move with the model's confidence and calibration.

:::{list-table}
:widths: 20 32 48
:header-rows: 1

- - Extractor
  - What it produces
  - What it inherits, and when it misleads
- - {class}`.ScoresExtractor`
  - Per-row class scores from a MAITE model: `(n_images, n_classes)` for
    classification, `(n_detections, n_classes)` for detection.
  - The rawest of these — the model's output surface with no reduction applied.
    Useful when you want drift measured on the decision the model is actually
    making rather than on features it might use. It is also the score producer
    the two uncertainty extractors are usually wrapped around.
- - {class}`.DetectionGeometryExtractor`
  - `[center_x, center_y, width, height, area, aspect]` per detection, from
    normalized boxes, so it is resolution-independent.
  - Carries **no appearance information at all**: it can tell you that object
    size, position or shape distribution shifted, never why. The `confidence`
    parameter (default `0.0`, keeping everything) drops detections below a score,
    so once set, both the rows and how many there are depend on the model's
    calibration — a recalibrated model looks like changed data.
- - {class}`.UncertaintyExtractor`
  - `(n_samples, 1)` Shannon entropy of the predicted class distribution, from a
    wrapped score producer. Suited to {class}`.DriftUnivariate`.
  - `preds_type` must match what the score producer emits — `"logits"` applies a
    softmax first, `"probs"` does not — and getting it wrong computes entropy on
    the wrong scale with no error raised. `normalize=True` divides by the maximum
    entropy for the number of classes present, which makes scores comparable
    across class counts but not across models.
- - {class}`.ClasswiseUncertaintyExtractor`
  - A `dict` mapping class index to that class's array of detection
    uncertainties.
  - **Not a drift feature extractor**, despite passing an `isinstance` check
    against the `FeatureExtractor` protocol at runtime (that check only looks for
    `__call__`). Select a single class's array and pass that to a detector.
    `threshold=0.99` is a policy lever, not a detail: it decides how many classes
    each detection is counted under, with `1.0` meaning winner-take-all.
- - {class}`.ClassifierUncertaintyExtractor`
  - Superseded. **Deprecated as of v1.1**, removal scheduled for v1.2.
  - Use {class}`.UncertaintyExtractor` for per-instance uncertainty or
    {class}`.ClasswiseUncertaintyExtractor` for the per-class breakdown.

:::

```{note}
{class}`.UncertaintyExtractor` and {class}`.ClasswiseUncertaintyExtractor` both
wrap a score producer rather than a model, so any `FeatureExtractor` can supply
their input — including {class}`.TorchExtractor`, {class}`.OnnxExtractor`, or an
{class}`.Embeddings`. Running both over the same data invokes that producer
twice; wrapping it in a caching {class}`.Embeddings` and sharing the one instance
avoids paying for inference again.
```

No result object records the extractor that produced it, so **record which
extractor and which weights produced any assessment you intend to cite.**
Two DataEval runs with different extractors are not comparable.

## Metadata binning: a policy applied to every factor

Binning is not a preprocessing step, it is a policy choice. Many of the functions
and classes that consume a {class}`.Metadata` class require factor discretization.
How the data is discretized matters, and **that discretization is a hidden
policy choice: applied by default, decisive for the data, and recorded nowhere
in the output.** A {class}`.Balance` score
describes your factors *as binned*, not your factors. This is the metadata-side
counterpart to the extractor problem above: where embeddings are the largest
uncontrolled variable in the geometric half of the library, binning is the
largest one in the metadata half.

### What the default does

With `continuous_factor_bins=None` — the default — {class}`.Metadata` decides
per column, without being asked:

- **Non-numeric columns** are ordinal-encoded, one category per distinct value.
- **Numeric columns** are classified by {func}`.is_continuous`, a heuristic
  combining three signals: the Wasserstein distance of the normalized
  near-neighbor distribution from uniform (threshold `0.5 / sqrt(n)`), the
  fraction of exact duplicate values, and a GCD test for values sitting on a
  regular lattice. Columns it judges discrete are ordinal-encoded like
  categoricals; columns it judges continuous are binned by `auto_bin_method`,
  default `"uniform_width"`. It is public, so you can run it on a factor and
  see the call it will make before you build the {class}`.Metadata`.
- **Below 20 samples the heuristic returns "discrete" unconditionally**, so on
  a small dataset every distinct float becomes its own category and the
  resulting factor is nearly one-to-one with the sample index.

On object detection metadata, a factor recorded at the **image** level is
classified on its per-image values rather than on the copy held by each
detection, so a continuous factor is not mistaken for a discrete one by its own
repetition. A factor recorded at the **target** level is classified on every
detection, because there the repeats are real observations. The 20-sample floor
applies to whichever of the two the factor is judged on, so an image-level
factor on fewer than 20 images is always called discrete no matter how many
detections those images carry.

`"uniform_width"` does not use a fixed bin count. It starts from NumPy's
`histogram(bins="auto")`, then *reduces* the count — at most 20 times — while
any non-empty bin holds fewer than 10 samples. That rule is aggressive: 500
draws from a standard normal come out in **4 to 6 bins depending only on the
draw**, because the tails keep tripping the 10-sample floor. There is no lower
guard, so a small or lopsided
factor can be reduced to a **single bin**, at which point it is constant and
carries no information into any metric that reads it — silently.

The count is therefore a function of the data, not a setting: **the same factor
measured on two datasets can be cut into different numbers of bins**, so binned
factor values are not comparable across runs. `"uniform_count"` keeps that same
count but moves the edges to quantiles; `"clusters"` derives edges from
DataEval's HDBSCAN port and inherits its behavior. All three set the outermost
edges to ±∞, so the tails are absorbed into the end bins instead of forming
their own.

Passing an explicit entry in `continuous_factor_bins` overrides all of this —
and also marks that factor continuous downstream regardless of what it actually
contains.

### Missing values

A `NaN` is not a small value, a large value, or a value between two edges, so
it is **given a bin of its own** above the bins holding observed values, on
every path: automatic or explicit binning, classification or object detection.
Two consequences worth knowing:

- **Edges are placed on the observed values alone**, so a missing value does
  not shift where the cuts fall. Infinities are observed extremes and land in
  the end bins, not the missing bin.
- **The missing bin has a position but no meaning.** It sits above the highest
  observed bin because the codes have to be contiguous, not because missing is
  large. Anything that treats a binned factor as ordinal — {class}`.Balance`
  passes bin indices to `mutual_info_regression` — reads that position as a
  value. Where the missing rate is material, check whether a finding is about
  the factor or about its absence.

The continuous/discrete test drops `NaN` before deciding, and the 20-sample
floor then counts observed values only, so a factor that is mostly missing is
called discrete.

### Which results depend on it

:::{list-table}
:widths: 24 18 58
:header-rows: 1

- - Consumer
  - Binning is
  - What that means
- - {class}`.Diversity`, {func}`.parity` ⚠️, {class}`.Parity` ⚠️
  - **Required**
  - The diversity indices and the contingency-table test are defined over
    discrete categories, so a continuous factor must be cut somewhere. Changing
    the cut changes the score, and no setting is correct independent of the
    question being asked. For `parity` the cut also decides cell occupancy —
    coarser bins avoid the sparse cells reported in `insufficient_data`, at the
    cost of resolution.
- - {class}`.Balance`
  - **A convention**
  - {func}`.mutual_info` handles continuous factors natively — `is_continuous`
    selects `mutual_info_regression` per column and the Linfoot transform maps
    the result to [0, 1]. But `Balance` always passes
    `Metadata.factor_data`, which is already binned, so the estimator receives
    bin indices rather than raw values even for factors it has labeled
    continuous. The information lost at the binning step cannot be recovered
    downstream, and nothing in the output says it happened.
- - {func}`.split_dataset` with `split_on`
  - **Applied**
  - Groups are formed from binned factor values, so grouping granularity is the
    binning's. A factor the automatic path reduced to one bin puts every sample
    in a single group, quietly weakening the split constraint you asked for.
- - Anything reading `Metadata.factor_data`
  - **Applied**
  - `factor_data` is the binned, integer-valued view and is what the bias
    evaluators and grouping code consume. `Metadata.rows_at(md.view)` carries
    the unbinned values, and `Metadata.factor_info` reports per factor whether it
    was binned or digitized.
- - {func}`.factor_deviation`, {func}`.factor_predictors`
  - **Not applied**
  - These take plain mappings of factor name to raw array, so they are
    unaffected unless you build those mappings out of binned columns yourself.

:::

### Working with it

- **Set `continuous_factor_bins` explicitly** for any factor you intend to
  report on. It is the difference between a binning you chose and one chosen
  for you.
- **Re-run at a second setting** — a different bin count, or `"uniform_width"`
  against `"uniform_count"`. A conclusion that survives both is about the data;
  one that does not is about the binning. DataEval does not perform this check
  and will not warn you when a result is binning-sensitive.
- **The automatic path announces itself only in logs.** Each auto-binned factor
  emits a WARNING on the `dataeval.metadata` logger, and DataEval attaches a
  `NullHandler`, so nothing is printed unless you configure logging — call
  {func}`dataeval.log` with `logging.WARNING` to see them.
- **Record the binning alongside the extractor.** Two runs with different
  binning are no more comparable than two runs with different extractors.

## Validation evidence levels

These describe DataEval's verification of its own code, not the standing of the
underlying method in the literature. A method can be well established in the
literature and still be `Internal only` here — that combination means the theory
is sound and the implementation has not been checked against a second
implementation.

:::{list-table}
:widths: 18 82
:header-rows: 1

- - Level
  - Meaning
- - **Delegated**
  - The statistic is computed by an established third-party library
    (SciPy, scikit-learn). DataEval supplies the surrounding workflow, not the
    estimator. Correctness of the statistic is that library's.
- - **Ported**
  - Implementation derived from a published reference implementation, carried
    over together with that project's test suite.
- - **Cross-checked**
  - DataEval's own implementation, verified in CI against an independent
    implementation of the same quantity.
- - **Ground-truth checked**
  - Verified in CI against a closed-form value. The test constructs a
    distribution whose target quantity is known analytically and asserts the
    estimator recovers it, so the test fails when the estimate is wrong rather
    than merely when it changes.
- - **Anchored**
  - Verified against hand-computed or analytically known values on constructed
    inputs, plus behavioral tests. No second implementation is compared.
- - **Internal only**
  - Verified by unit and property tests (invariants, edge cases, shape, and
    error handling) written against the intended behavior. Correctness relative
    to the cited derivation rests on code review.

:::

## Evaluators

### Data quality

See {doc}`Data Integrity <DataIntegrity>` for what these measure.

:::{list-table}
:widths: 13 21 15 21 30
:header-rows: 1

- - Evaluator
  - Evidence basis
  - Validation evidence
  - Policy levers
  - Contraindications
- - {class}`.Duplicates`
  - Perceptual hashing ([Zauner, 2010](DataIntegrity.md#references)); clustering
    via scikit-learn KMeans or DataEval's HDBSCAN. See
    {doc}`Clustering <Clustering>`.
  - Anchored (hashes); Delegated (KMeans); Ported (HDBSCAN)
  - `cluster_sensitivity` — what counts as "near". `None` by default, so
    **cluster detection is off unless set together with an `extractor`**.
    `merge_near_duplicates=True`; `cluster_algorithm="hdbscan"`.
  - Hash and embedding modes fail on **disjoint** transformation classes.
    Neither establishes semantic identity: a hash collision is visual
    similarity, not the same object.
- - {class}`.Outliers`
  - Robust statistics (Double-MAD, z-score, IQR fences) over the image
    statistics vocabulary.
  - Cross-checked (skew, kurtosis, entropy verified against SciPy);
    Anchored (thresholds)
  - `outlier_threshold` — what counts as "extreme"; defaults to
    `AdaptiveThreshold` (tail-weighted Double-MAD, asymmetric bounds).
    `cluster_threshold` defaults to `ZScoreThreshold(upper_multiplier=2.5)`.
    `flags` selects the statistics scored.
  - The default tolerates skew and heavy tails but assumes **one** mode: on
    multimodal data — night plus day, mixed sensors — it flags a legitimate
    second population. The z-score `cluster_threshold` is not robust there
    either. Outputs are **linting warnings for review, never a deletion list.**

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
[Detecting common augmentations as duplicates](../notebooks/tt_augmentation_duplicates.py)
tutorial. Consult that notebook for the current results rather than relying on
a summary table here — the behavior depends on hash size, cluster sensitivity,
and the extractor in use, and a static table would go stale silently.
```

### Dataset bias

See {doc}`Dataset Bias and Coverage <DatasetBias>` for what these measure.

:::{list-table}
:widths: 13 21 15 21 30
:header-rows: 1

- - Evaluator
  - Evidence basis
  - Validation evidence
  - Policy levers
  - Contraindications
- - {class}`.Balance`
  - Normalized mutual information ([Linfoot, 1957](DatasetBias.md#references);
    [Vinh et al., 2010](DatasetBias.md#references)). Estimators from
    scikit-learn.
  - Delegated (MI estimation); Internal only (normalization)
  - `num_neighbors=5` sets the MI estimator's neighborhood.
    `class_imbalance_threshold=0.3` and `factor_correlation_threshold=0.5`
    decide what is reported as imbalanced or entangled.
  - Runs on binned factors even though {func}`.mutual_info` does not require
    it, so the result depends on a binning you may not have chosen — see
    [Metadata binning](#metadata-binning-a-policy-applied-to-every-factor).
    MI is biased upward at small n, so small datasets look more entangled than
    they are.
- - {class}`.Diversity`
  - Shannon and inverse-Simpson indices
    ([Hill, 1973](DatasetBias.md#references);
    [Heip et al., 1998](DatasetBias.md#references)).
  - Anchored
  - `method="simpson"` chooses inverse-Simpson over Shannon; the two weight
    rare categories differently. `threshold=0.5` decides what counts as low
    diversity.
  - Discrete and categorical variables only. Over binned continuous data the
    score measures the binning as much as the data — see
    [Metadata binning](#metadata-binning-a-policy-applied-to-every-factor).
- - {class}`.Parity` ⚠️
  - G-test with Bergsma (2013) bias correction for Cramér's V; cited in the
    docstring.
  - Anchored
  - `score_threshold=0.3` (Cramér's V) and `p_value_threshold=0.05` decide which
    factors are reported as associated with the label.
  - **Experimental — the API may change or be removed.** Cells with fewer than
    5 samples break the asymptotic approximation; they are listed in
    `insufficient_data`, which **must** be checked before the score is used.
    Categorical factors only — continuous factors reach it already binned, and
    the binning decides cell occupancy. See
    [Metadata binning](#metadata-binning-a-policy-applied-to-every-factor).

:::

### Dataset scope

See {doc}`Dataset Bias and Coverage <DatasetBias>` for coverage, and
{doc}`Acting on Results <ActingOnResults>` for how to act on a prioritization.

:::{list-table}
:widths: 13 21 15 21 30
:header-rows: 1

- - Evaluator
  - Evidence basis
  - Validation evidence
  - Policy levers
  - Contraindications
- - {class}`.Coverage`
  - Coverage radius from
    [Asudeh et al. (2021)](https://dl.acm.org/doi/abs/10.1145/3448016.3457315);
    the per-class dispersion, isotropy, and near-duplicate signals it adds are
    DataEval's own. `isotropy` is {func}`.completeness`'s effective-dimension
    estimate applied per class — see that row below for its lineage. See
    [Measuring coverage](DatasetBias.md#measuring-coverage-geometry-in-embedding-space).
  - Internal only
  - `method="adaptive"` and `num_observations=20` set the radius (the latter is
    Sudman's 20-50 guidance). `percent=0.01` is the fraction reported as
    uncovered. `min_class_samples=20` and `near_duplicate_factor=0.5` gate the
    per-class and duplicate signals. `isotropy_min_samples=None` gates
    `isotropy` at one more than the embedding dimensionality.
  - Coverage of a poor embedding space is not coverage of the data. Small
    datasets return degenerate radii. `isotropy` is null — not zero — for any
    class below its sample floor, which on a high-dimensional extractor is most
    classes; reduce dimensionality first if you need it.
- - {class}`.Prioritize`
  - Coreset and cluster-complexity ranking
    ([Sorscher et al., 2022](https://arxiv.org/abs/2206.14486);
    [Abbas et al., 2024](https://arxiv.org/abs/2401.04578);
    [Zheng et al., 2023](https://arxiv.org/abs/2210.15809)). See
    [Prioritization](DatasetBias.md#prioritization-as-coverage-driven-data-selection).
  - Internal only
  - `method="knn"` selects the ranking criterion; `policy="difficulty"` and
    `order="easy_first"` the selection rule on top of it; `num_bins=50` the
    stratification granularity. Each is a separate objective — changing one
    changes what "prioritized" means.
  - Ranks by embedding geometry alone — no notion of label correctness,
    annotation cost, or operational importance. See
    [What Prioritize does not do](ActingOnResults.md#what-prioritize-does-not-do).
- - {class}`.Representation`
  - Observed label mass over an {class}`.Ontology`, judged against an expected
    distribution. See {doc}`Ontology <Ontology>`.
  - Internal only
  - `expected=None` — the entire policy. `None` is a uniform expectation over
    leaf classes; pass a mapping to assert per-class minimum shares.
  - Only as good as the ontology and expectation supplied. The uniform default
    is **wrong** for any domain with a genuinely skewed operational prior.

:::

### Distribution shift

See {doc}`Distribution Shift <DistributionShift>` for what these measure and
how to choose among them.

:::{list-table}
:widths: 13 21 15 21 30
:header-rows: 1

- - Evaluator
  - Evidence basis
  - Validation evidence
  - Policy levers
  - Contraindications
- - {class}`.DriftUnivariate`
  - [Rabanser et al. (2019)](DistributionShift.md#references) methodology.
    All five tests (`ks`, `cvm`, `mwu`, `anderson`, `bws`) are computed by
    `scipy.stats`. Corrections: Bonferroni,
    [Benjamini-Hochberg (1995)](DistributionShift.md#references).
  - Delegated (test statistics); Ported (workflow, from alibi-detect 0.11.4)
  - `method="ks"`, `p_val=0.05`, `correction="bonferroni"`,
    `alternative="two-sided"`. Bonferroni is the conservative choice; `"fdr"`
    trades false positives for power across many features.
  - **Do not apply to raw high-dimensional input.** Univariate tests over
    thousands of correlated dimensions lose power even with correction; reduce
    dimensionality first. Feature-wise tests cannot see shift that exists only
    in the joint distribution.
- - {class}`.DriftMMD`
  - [Gretton et al.](DistributionShift.md#references) kernel two-sample test;
    permutation-based significance.
  - Ported (alibi-detect 0.11.4)
  - `p_val=0.05`; `sigma=None` selects the bandwidth heuristic, which is not
    universally appropriate; `n_permutations=100` bounds the smallest
    reachable p-value.
  - Permutation testing scales quadratically in sample size — unsuitable for
    per-frame or real-time use without chunking.
- - {class}`.DriftDomainClassifier`
  - Domain-classifier drift detection; cross-validated discriminability.
  - Internal only
  - `threshold=0.55` AUROC is the entire drift verdict; `n_folds=5` sets how
    much data each fold sees.
  - AUROC is optimistic under severe imbalance between reference and test sets.
    A negative result is weak evidence: it may mean no drift, or an
    underpowered proxy classifier.
- - {class}`.DriftKNeighbors`
  - Nearest-neighbor distance-based detection. See
    [K-nearest neighbor drift detection](DistributionShift.md#k-nearest-neighbor-drift-detection-driftkneighbors).
  - Internal only
  - `k=10`, `p_val=0.05`, `distance_metric="euclidean"` — note this differs
    from {class}`.OODKNeighbors`, which defaults to `"cosine"`.
  - Depends on the embedding metric being meaningful. Sensitive to `k` and to
    reference-set size.
- - {class}`.DriftReconstruction`
  - Autoencoder reconstruction error. The detector is DataEval's own; the
    scoring math it calls is shared with {class}`.OODReconstruction`.
  - Internal only (detector); Ported (shared scorer, from alibi-detect 0.11.4)
  - `p_val=0.05`; `epochs=20` and `batch_size=64` govern the autoencoder
    trained in `fit()`; `gmm_weight=0.5` and `gmm_score_mode="standardized"`
    fuse reconstruction and density scores.
  - Inherits every bias of the autoencoder it trains or is handed.
    Reconstruction error conflates "novel" with "hard to reconstruct."
- - {class}`.DriftWasserstein`
  - Wasserstein distance against a validation baseline. See
    [Wasserstein-based drift detection](DistributionShift.md#wasserstein-based-drift-detection-driftwasserstein).
  - Internal only
  - `ratio_threshold=1.4` — drift is flagged when the train/test distance
    exceeds 1.4x the train/validation baseline. A pure convention.
  - Requires a held-out validation split to calibrate the baseline. Without a
    representative baseline the threshold is arbitrary.
- - {class}`.ChunkedDrift`
  - Wrapper applying any detector over sequential chunks.
  - Internal only
  - `chunker` / `chunk_size` / `chunk_count` — **one must be supplied; there is
    no default chunking.** `threshold=None` falls back to the wrapped
    detector's own chunk threshold.
  - Chunk size trades detection latency against statistical power; small chunks
    give unstable per-chunk verdicts. A configuration choice, not a property of
    the wrapped detector.
- - {class}`.OODReconstruction`
  - Autoencoder / VAE reconstruction-based OOD scoring.
  - Ported (alibi-detect 0.11.4); Cross-checked against scikit-learn in tests
  - `threshold_perc=95.0` declares the top 5% of reference scores OOD — it
    assumes your reference set is 5% contaminated. `epochs=20`,
    `batch_size=64`, `gmm_weight=0.5` govern the model and score fusion.
  - Inherits every bias of the autoencoder/VAE it trains or is handed.
    Per-instance scores are not calibrated probabilities; the threshold needs a
    trusted in-distribution reference set.
- - {class}`.OODKNeighbors`
  - Distance-based OOD scoring. See
    [Distance-based detection](DistributionShift.md#distance-based-detection-oodkneighbors).
  - Cross-checked (against scikit-learn in tests)
  - `k=10`, `distance_metric="cosine"`, `threshold_perc=95.0` — same
    contamination assumption as {class}`.OODReconstruction`.
  - Same embedding dependence as `DriftKNeighbors`. Threshold requires
    calibration on trusted in-distribution data.
- - {class}`.OODDomainClassifier`
  - Domain-classifier OOD detection.
  - Internal only
  - `n_folds=5`, `n_repeats=5`, `n_std=2.0`. `threshold_perc=None` by default,
    so the cutoff comes from `n_std` above the reference mean rather than a
    percentile.
  - Same imbalance caveat as `DriftDomainClassifier`.

:::

```{note}
Drift detection answers "has the input distribution changed?" It does **not**
answer "has model performance degraded?" DataEval measures the first. See
[Taxonomy of shift](DistributionShift.md#taxonomy-of-shift) for why a detected
shift may be harmless and an undetected one may not be.
```

### Performance

See {doc}`Performance Limits <PerformanceLimits>`.

:::{list-table}
:widths: 13 21 15 21 30
:header-rows: 1

- - Evaluator
  - Evidence basis
  - Validation evidence
  - Policy levers
  - Contraindications
- - {class}`.Sufficiency`
  - Power-law learning curves
    ([Hestness et al., 2017](PerformanceLimits.md#references)); curve fitting
    via basin-hopping ([Wales & Doye, 1997](PerformanceLimits.md#references)).
  - Internal only
  - `runs=1` and `substeps=5` decide how much evidence the curve is fitted to.
    **A single run yields no usable confidence interval** — raise it before
    quoting one.
  - **Extrapolation assumes future data is drawn from the same distribution as
    the pilot data.** If new collection introduces shift, the projection is
    void. Projections far beyond the largest measured substep are speculative.

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
  - *Learning to Bound the Multi-class Bayes Error*, Theorems 3 and 4
    ([arXiv:1811.06419](https://arxiv.org/abs/1811.06419)) — cited in both
    docstrings. See
    [BER estimation](PerformanceLimits.md#ber-estimation-mst-and-knn-bounds).
  - Ground-truth checked (bounds must contain, and stay tight around, the
    analytic Bayes error of two Gaussians at four separations)
  - Only as meaningful as the embedding: a weak extractor inflates apparent
    class overlap and overstates difficulty. `ber_knn` is sensitive to `k`; MST
    construction is costly at scale. Classification only.
- - {func}`.uap` ⚠️
  - Empirical upper-bound average precision
    ([Borji & Iranmanesh, 2019](PerformanceLimits.md#references)).
  - Anchored
  - **Experimental — the API may change or be removed.** Reduces detection to
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
    {doc}`Divergence <Divergence>`.
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
    {doc}`Clustering <Clustering>`.
  - Delegated (KMeans); Ported (HDBSCAN, from fast_hdbscan);
    Cross-checked (clusterer tests use scikit-learn)
  - Cluster counts are a modeling choice, not a discovered fact. See
    [Practical considerations](Clustering.md#practical-considerations).
- - {func}`.completeness`
  - Effective dimensionality $d_\text{eff} = e^H$ over the covariance
    eigenvalues — the eigenvalue form of the effective rank of
    [Roy & Vetterli (2007)](DatasetBias.md#references), also used by
    [Kim et al. (2023)](DatasetBias.md#references) and
    [Zhuo et al. (2023)](DatasetBias.md#references), and the entropy-based
    sibling of the participation ratio
    ([Gao et al., 2017](DatasetBias.md#references)). All four are cited in the
    docstring. **The two normalizations are DataEval's own**: $d_\text{eff}/d$
    (completeness) and $d_\text{eff}/r$ (isotropy). See
    [Completeness and isotropy](DatasetBias.md#measuring-coverage-geometry-in-embedding-space).
  - Internal only
  - Measures how the data fills the embedding's dimensions, not whether the
    embedding is semantically right. The two scores are not directly comparable
    as a ratio: completeness decomposes the rank-normalized matrix, isotropy the
    raw centered one. `isotropy` is degenerate unless samples exceed dimensions.
- - {func}`.coverage_naive`, {func}`.coverage_adaptive`
  - Radius-based coverage of
    [Asudeh et al. (2021)](https://dl.acm.org/doi/abs/10.1145/3448016.3457315),
    with the 20-50 observation guidance from Sudman (1976) — both cited in the
    docstrings. See
    [Measuring coverage](DatasetBias.md#measuring-coverage-geometry-in-embedding-space).
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
    [Sorscher et al. (2022)](https://arxiv.org/abs/2206.14486) — cluster the
    embeddings, rank each sample by distance to its centroid.
  - Internal only
  - Favors sparse regions: on non-uniform manifolds it can oversample sparse
    outlier regions and undersample dense, decision-boundary-rich ones.
- - {func}`.rank_kmeans_complexity`, {func}`.rank_hdbscan_complexity`
  - Concept-cluster complexity of
    [Abbas et al. (2024)](https://arxiv.org/abs/2401.04578) — complexity is the
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
  - These impose a **selection policy** on an observation. Using them means
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
- - {func}`.compute_stats`, {func}`.combine_stats_results`,
    {func}`.compute_ratios`
  - Image statistics vocabulary. See
    [Image statistics as a linting vocabulary](DataIntegrity.md#image-statistics-as-a-linting-vocabulary).
  - Cross-checked (skew, kurtosis, and entropy verified against SciPy in CI)
  - Descriptive only. A statistic being unusual is not evidence a sample is
    bad. Note `compute_stats` carries a deprecation notice on a parameter
    default that changed in v1.1.
- - {func}`.track_stats`
  - Per-track statistics for video sequences.
  - Anchored
  - Single-sequence scope; assumes track identity is already resolved.
- - {func}`.phash`, {func}`.phash_d4`, {func}`.dhash`, {func}`.dhash_d4`,
    {func}`.xxhash`
  - Perceptual hashing ([Zauner, 2010](DataIntegrity.md#references)); xxHash
    for exact matching.
  - Anchored
  - `xxhash` detects exact byte-level duplicates only. Perceptual hashes
    survive photometric change but not arbitrary geometric warping; the `_d4`
    variants add invariance to the eight square symmetries, not to arbitrary
    rotation.

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
  - Bergsma (2013), cited in the docstring. See
    {doc}`Dataset Bias and Coverage <DatasetBias>`.
  - Anchored
  - **Experimental.** Same sparse-cell and binning caveats as
    {class}`.Parity` ⚠️; check `insufficient_data` before using the score.
- - {func}`.label_parity`
  - Chi-square test of label distribution parity.
  - Anchored
  - Compares label distributions only. Says nothing about feature-space
    differences between the two sets.
- - {func}`.mutual_info`, {func}`.mutual_info_classwise`
  - [Linfoot (1957)](https://www.sciencedirect.com/science/article/pii/S001999585790116X),
    cited in the docstring.
  - Delegated (estimators from scikit-learn)
  - Handles continuous factors natively: {func}`.is_continuous` selects
    `mutual_info_regression` per column and the Linfoot transform maps the
    result to [0, 1], so binning is a {class}`.Metadata` convention, not a
    requirement here — but {class}`.Balance` applies it anyway, see
    [Metadata binning](#metadata-binning-a-policy-applied-to-every-factor).
    Upward-biased at small n.
- - {func}`.is_continuous`
  - DataEval-original heuristic. Uses the Wasserstein distance from
    `scipy.stats`; the normalized near-neighbor construction and the two
    corroborating signals are DataEval's own. See
    [Metadata binning](#metadata-binning-a-policy-applied-to-every-factor).
  - Internal only
  - A tuned classifier, not a test with a stated error rate: the five constants
    that decide its verdict have no derivation and no measured
    misclassification rate. Returns `False` below 20 samples or 3 distinct
    values, so on a small dataset every numeric factor is called discrete.
    Its answer decides whether a factor gets binned, so a wrong call is
    invisible in the metric that follows.
- - {func}`.factor_deviation`, {func}`.factor_predictors`
  - Metadata attribution for flagged samples. See
    [Diagnosing findings with metadata](ActingOnResults.md#diagnosing-findings-with-metadata).
  - Internal only
  - **Associational, not causal.** A factor that predicts flagged samples is a
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
    [Label error detection](DataIntegrity.md#label-error-detection-embedding-geometry).
  - Internal only
  - Produces **candidates for review**, not confirmed errors. Precision depends
    on embedding quality; genuinely ambiguous samples are flagged as readily as
    genuine mistakes.
- - {func}`.label_stats`
  - Label distribution accounting.
  - Anchored
  - Descriptive.
- - {func}`.label_coverage`
  - Observed label mass over an {class}`.Ontology`. See
    {doc}`Ontology <Ontology>`.
  - Internal only
  - Reports observed distribution only. Judging that distribution against an
    expectation is {class}`.Representation`'s job, not this function's.
- - {func}`.label_reconciliation`, {func}`.label_alignment`
  - Ontology reconciliation and matching. See
    [Reconciliation](Ontology.md#reconciliation-checking-labels-against-the-ontology)
    and [Alignment](Ontology.md#alignment-relating-two-vocabularies).
  - Internal only
  - Matcher output is a proposal requiring human confirmation. Confidence
    scores are matcher-relative and not comparable across matchers. See
    [Confidence and abstention](Ontology.md#confidence-and-abstention).
- - {func}`.ontology_validation`
  - Structural and naming facts about an ontology artifact. See
    [Validation](Ontology.md#validation-checking-the-ontology-artifact).
  - Internal only
  - Checks structure, not semantic correctness. A well-formed ontology can
    still be the wrong ontology for your problem.

:::

## How DataEval validates its own implementations

### Provenance

Several drift and out-of-distribution detectors in `dataeval.shift` —
{class}`.DriftMMD`, {class}`.DriftUnivariate`, {class}`.OODReconstruction`, the
shared OOD base, and the reconstruction scorer those detectors call — are
derived from
[alibi-detect v0.11.4](https://github.com/SeldonIO/alibi-detect/tree/v0.11.4)
(Copyright © 2023 Seldon Technologies Ltd, Apache 2.0), together with that
project's tests. `dataeval.utils.losses` shares that lineage.

{class}`.DriftReconstruction` is the exception in that group: it is
conceptually inspired by alibi-detect's reconstruction detector, but the
detector code and its tests are DataEval's own. It reaches alibi-derived code
only through the shared reconstruction scorer, which is why its row above
carries two ratings.

DataEval's HDBSCAN implementation (`dataeval.core._fast_hdbscan`) is adapted
from [fast_hdbscan](https://github.com/TutteInstitute/fast_hdbscan) v0.2.0-0.2.2
(Copyright © 2020 Leland McInnes, BSD 2-Clause). Everything reached through
`cluster_algorithm="hdbscan"` — in {class}`.Duplicates`, {class}`.Outliers`,
{func}`.cluster`, and the `rank_hdbscan_*` functions — rests on that port.

Where an established implementation of a statistic exists, DataEval calls it
rather than reimplementing it. {class}`.DriftUnivariate` delegates all five of
its tests to `scipy.stats`; clustering delegates to scikit-learn's `KMeans`;
mutual information estimation uses scikit-learn's estimators.

### Test suite

The suite is organized to mirror the package (`tests/core`, `tests/shift`,
`tests/bias`, `tests/quality`, `tests/scope`, `tests/performance`, ...) and CI
enforces a **90% coverage floor** (`fail_under = 90`). Two hand-written paths
are excluded from that measurement — `core/_fast_hdbscan/*` and
`_warm_cache.py` — alongside the generated `_version.py`. The ported HDBSCAN
code therefore does not count toward the floor, which is worth knowing because
`cluster_algorithm="hdbscan"` is the default in {class}`.Duplicates` and
{class}`.Outliers`.

The suite contains three kinds of evidence, and they are not equally strong:

- **Regression tests** pin an estimator's output against a value recorded from
  a previous run. They catch unintended change, but they pass just as happily
  when an estimator has been wrong from the start — which is why the two kinds
  below, not this one, carry the weight of the ratings in the tables above.
- **Cross-implementation checks** compare against an independent library:
  image statistics (skew, kurtosis, entropy) against SciPy, and several
  clustering and OOD tests against scikit-learn.
- **Ground-truth checks** (`tests/core/test_validation_benchmarks.py`)
  construct a distribution whose target quantity has a closed-form solution and
  assert the estimator recovers it. BER bounds are checked against the analytic
  Bayes error of two Gaussians — both that the interval contains it and that it
  stays tight enough to be informative, since containment alone would be
  satisfied by returning `[0, 1]`. Divergence is checked at its 0 and 1
  endpoints, and the null-model baselines against their closed forms.

A separate test (`tests/test_references.py`) pins the set of symbols that must
cite their sources, so citation coverage cannot regress.

What the suite does **not** do is reproduce published benchmark results on
public datasets. See [Known gaps](#known-gaps-in-this-pages-evidence).

### API stability discipline

Two mechanisms make stability observable rather than a matter of trust:

- Experimental features raise
  {class}`~dataeval.exceptions.ExperimentalWarning` at runtime and are marked in
  their docstrings — you cannot use one without being told.
- Deprecated symbols raise {class}`~dataeval.exceptions.DeprecatedWarning`, and
  a dedicated CI test (`tests/test_deprecated_docs.py`) fails the build if any
  deprecated public object lacks a `.. deprecated::` directive in its docstring.
  A deprecated symbol cannot ship silently. Transitional warnings for changed
  *parameter defaults* are a separate mechanism and use a plain `FutureWarning`
  — see [Changed defaults](#stability-status) below.

## Stability status

As of v{{ release }}:

**Experimental** ⚠️ (API may change or be removed):

- {func}`.parity`, {class}`.Parity`, {func}`.uap` and `ParityOutput` raise
  {class}`~dataeval.exceptions.ExperimentalWarning` on first call or
  instantiation.
- `ParityResult` carries the same experimental status but raises nothing on its
  own — it is a `TypedDict` describing {func}`.parity`'s return value, and the
  warning comes from the call that produced it.

**Deprecated** (raises `DeprecatedWarning`; scheduled for removal in v1.2):

- {class}`.ClassifierUncertaintyExtractor` → use
  {class}`.UncertaintyExtractor` or {class}`.ClasswiseUncertaintyExtractor`
- `clip_and_pad` → use {func}`.crop_with_fill`, passing `fill=np.nan`
  and taking the first tuple value

**Changed defaults** (the symbol is not deprecated; only the transitional
warning goes away):

- {func}`.compute_stats` — the `normalize_pixel_values` default changed to
  `False` in v1.1. Calling without it raises a plain `FutureWarning`, not
  {class}`~dataeval.exceptions.DeprecatedWarning`, so a warning filter narrowed
  to that class will not catch it. Pass the argument explicitly to silence it.
  The function itself is stable and is not scheduled for removal.

Everything else in the public API is considered stable for the v1.x series.

## When not to use DataEval

Being explicit about non-applicability is part of the trust argument.

**DataEval is not applicable when:**

- **You need a model evaluation.** DataEval characterizes datasets and the
  limits they impose. It does not evaluate a trained model's performance,
  fairness, or robustness. `nullmodel_*` and {class}`.Sufficiency` touch models
  only to establish data-side baselines and curves.
- **You need a certification or an accreditation artifact.** No DataEval output
  constitutes evidence of fitness for deployment. The outputs are inputs to a
  human assessment.
- **Your data is not amenable to embeddings.** Most of the library operates in
  embedding space. Without a feature extractor that represents your domain,
  BER, divergence, coverage, prioritization, and embedding-based drift are all
  measuring the extractor rather than the data.
- **Your sample sizes are very small.** Non-parametric estimators, mutual
  information, contingency-table tests, and radius-based coverage all degrade
  at small n — several silently, by returning a plausible number.
- **You need real-time, per-frame monitoring.** {class}`.DriftMMD` permutation
  testing and MST construction scale poorly. Use chunking, sampling, or a
  cheaper detector.
- **You need causal explanations.** Metadata attribution
  ({func}`.factor_deviation`, {func}`.factor_predictors`) and bias metrics are
  associational. They generate hypotheses.

**Additionally, treat any single number with suspicion when:**

- It comes from an experimental feature (see [above](#stability-status)).
- It was produced by an extractor you have not validated on your domain.
- It is a bias metric over continuous metadata whose binning you did not set
  yourself (see
  [Metadata binning](#metadata-binning-a-policy-applied-to-every-factor)).
- The evaluator's default policy was not reviewed against your operational
  prior.

## Known gaps in this page's evidence

Stating these is the point of the page. Each is a real, current limitation of
DataEval's evidentiary record, not a research frontier.

1. **Most public symbols carry no literature citation in their docstring.** A
   minority of public functions and classes have a `References` section, though
   the count has grown substantially. `REQUIRES_REFERENCES` in
   `tests/test_references.py` is the current list and the source of truth for
   it; the test pins that set so it cannot regress, and adding a symbol to the
   list is how the number goes up. For everything else the citation lives on a
   concept page or nowhere — where an entry above cites a concept page rather
   than a docstring, that is why. Much of the remainder is `*Result`
   TypedDicts, which describe output shapes and need no citation.
   The genuine unknowns are now few: {func}`.label_errors`,
   {func}`.feature_distance`, and {func}`.rank_result_class_balanced`. Those
   need someone to either identify the source or state plainly that the method
   is DataEval-original. {func}`.completeness` was on that list and is now
   resolved in the second sense: its effective-dimensionality estimate is cited
   to the effective-rank and participation-ratio literature, and the two
   normalizations that turn it into `completeness` and `isotropy` are stated in
   the docstring as DataEval's own rather than left implicitly sourced.

2. **No reproduction of published benchmarks on real datasets.** The validation
   benchmark suite (`tests/core/test_validation_benchmarks.py`) checks BER,
   divergence, and the null-model baselines against closed-form values on
   constructed distributions — those tests fail when an estimate is wrong, not
   merely when it changes. What is missing is evidence on *real* data: nothing
   in CI demonstrates that DataEval's estimators reproduce published results on
   public datasets. The FeeBee BER benchmark
   ([Renggli et al., 2021](Divergence.md#references)) is the obvious candidate.
   Until that exists, every rating on this page rests on synthetic or internal
   evidence.

3. **Threshold and policy defaults have no documented provenance.** The
   `Outliers` thresholds, `Duplicates` cluster sensitivity, drift `p_val`, and
   the expected distributions in `Representation` are all policy. None of them
   records who chose the value, on what data, or what the alternatives were.
   Users inheriting a default deserve to know whether it was derived or
   inherited.

4. **Coreset selection bounds are unverified on foundation-model embeddings.**
   The geometric arguments behind distance-based selection were established on
   classical CNN features. Whether they hold in the latent spaces of modern
   vision transformers and multimodal encoders is an open question, and one
   DataEval does not currently benchmark.

5. **The domain-classifier detectors are unbenchmarked under severe
   imbalance.** Both {class}`.DriftDomainClassifier` and
   {class}`.OODDomainClassifier` rely on AUROC, which is optimistic when the
   negative class dominates — the regime of rare-event operational monitoring.
   PR-AUC or stratified validation folds would be the fix; neither has been
   evaluated.

6. **The binning policy is documented but unvalidated, and unsupported by
   tooling.** Which evaluators depend on binning, and whether they require it
   or merely inherit it, is now stated above under
   [Metadata binning](#metadata-binning-a-policy-applied-to-every-factor).
   What remains missing is evidence for the specific choices and any means of
   testing them:
   - **The automatic thresholds have no documented provenance.** The
     continuous/discrete heuristic's `0.5 / sqrt(n)` Wasserstein cutoff, its
     0.85 lattice fraction, its 0.005 duplicate tolerance, the 20-sample floor
     below which every factor is called discrete, and `"uniform_width"`'s
     10-samples-per-bin target are all tuned constants. None records who chose
     the value or on what data, and no test establishes the misclassification
     rate of the heuristic on known-continuous and known-discrete factors. The
     bin-reduction loop also has no lower guard, so it can collapse a factor to
     one bin and report nothing about it.
   - **No sensitivity analysis exists.** Nothing in DataEval re-runs a bias
     metric across binnings or reports how much of a score is attributable to
     the cut, so a user cannot distinguish a finding from an artifact without
     scripting the comparison themselves.
   - **Bin counts are not stable across datasets.** Because `"uniform_width"`
     derives the count from the data, the same factor can be binned differently
     in a reference and an operational set. DataEval neither detects this nor
     provides a way to pin a binning learned on one dataset and reapply it to
     another, short of passing explicit edges.
   - **The auto-binning notice is easy to miss.** It is a log record on a
     logger with a `NullHandler` attached, not a `Warning`, so the default
     experience is silent discretization. A `Warning` would be caught by the
     same filters users already apply to
     {class}`~dataeval.exceptions.ExperimentalWarning` and
     {class}`~dataeval.exceptions.DeprecatedWarning`.

## Related concept pages

- {doc}`Embeddings <Embeddings>` — the largest uncontrolled variable in any
  DataEval assessment; read this before trusting any embedding-based result
- {doc}`Acting on Results <ActingOnResults>` — what to do with a finding, and
  which findings do not warrant action
- {doc}`Data Integrity <DataIntegrity>`,
  {doc}`Dataset Bias and Coverage <DatasetBias>`,
  {doc}`Performance Limits <PerformanceLimits>`,
  {doc}`Distribution Shift <DistributionShift>`,
  {doc}`Divergence <Divergence>`, {doc}`Clustering <Clustering>`,
  {doc}`Ontology <Ontology>` — the theory behind each feature, with each
  page's own references
- {doc}`Data Leakage <Leakage>` — why an assessment can be internally valid and
  still not generalize

## References

Method-level citations live on the concept page for each method, linked from
the tables above. The sources specific to this page are:

1. Seldon Technologies Ltd. (2023). *alibi-detect v0.11.4.*
   [repository](https://github.com/SeldonIO/alibi-detect/tree/v0.11.4)
   (Apache 2.0; origin of DataEval's MMD, univariate, and reconstruction-based
   detectors.)

2. Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for
   scientific computing in Python. *Nature Methods*, 17, 261–272.
   [paper](https://www.nature.com/articles/s41592-019-0686-2)
   (Provides the univariate drift test statistics and the reference values for
   DataEval's image statistics tests.)

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
   *Journal of Machine Learning Research*, 12, 2825–2830.
   [paper](https://www.jmlr.org/papers/v12/pedregosa11a.html)
   (Provides KMeans clustering and mutual information estimation.)

4. Sekeh, S. Y., Oselio, B., & Hero, A. O. (2020). Learning to bound the
   multi-class Bayes error. *IEEE Transactions on Signal Processing*, 68,
   3793–3807. doi: 10.1109/TSP.2020.2994807
   [paper](https://arxiv.org/abs/1811.06419)
   (Theorems 3 and 4 provide the MST and KNN bounds; cited directly in the
   {func}`.ber_knn` and {func}`.ber_mst` docstrings.)

5. Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T.
   *Journal of the Korean Statistical Society*, 42(3), 323–328.
   [paper](https://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf)
   (Cited directly in the {func}`.parity` docstring.)

6. Asudeh, A., Shahbazi, N., Jin, Z., & Jagadish, H. V. (2021). Identifying
   insufficient data coverage for ordinal continuous-valued attributes. In
   *Proceedings of the 2021 International Conference on Management of Data*
   (SIGMOD '21). doi: 10.1145/3448016.3457315
   [paper](https://dl.acm.org/doi/abs/10.1145/3448016.3457315)
   (Source of the coverage radius; cited in the `coverage_*` docstrings.)

7. Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., & Morcos, A. S. (2022).
   Beyond neural scaling laws: beating power law scaling via data pruning.
   *Advances in Neural Information Processing Systems*, 35.
   [paper](https://arxiv.org/abs/2206.14486)
   (Source of the distance-to-centroid prototypicality ranking.)

8. Zheng, H., Liu, R., Lai, F., & Prakash, A. (2023). Coverage-centric coreset
   selection for high pruning rates. *International Conference on Learning
   Representations*. [paper](https://arxiv.org/abs/2210.15809)
   (Source of the score-bin stratification in
   {func}`.rank_result_stratified`.)

9. Abbas, A., Rusak, E., Tirumala, K., Brendel, W., Chaudhuri, K., & Morcos,
   A. S. (2024). Effective pruning of web-scale datasets based on complexity of
   concept clusters. *International Conference on Learning Representations*.
   [paper](https://arxiv.org/abs/2401.04578)
   (Source of the cluster complexity measure used by the `*_complexity`
   ranking functions.)
