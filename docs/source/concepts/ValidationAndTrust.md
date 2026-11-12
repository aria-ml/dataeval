<!-- markdownlint-disable MD051 -->

# Validation and trust

## When to trust a DataEval result

Use this page as a pre-flight check before citing any DataEval metric. Every
evaluator relies on specific assumptions about your data, so the validity of a
measurement depends on whether those assumptions hold. This page makes them
explicit.

DataEval's evidence for its own correctness is
[documented and checkable](#how-dataeval-validates-its-own-implementations), but it is
not uniform across the library, so every evaluator and core function is rated
individually and not covered by a blanket claim. Each table on
[Validation evidence by evaluator](ValidationEvidence.md) carries the entry's name plus these columns:

- **Evidence basis** is the source of the method: a published result, a DataEval
  concept page where the derivation is written down, or a third-party library.
  Where the implementation delegates, that library is named, because its
  correctness is the relevant evidence.
- **Validation evidence** is how DataEval verifies its own implementation in CI.
  It takes one of six levels, [defined below](#validation-evidence-levels).
- **Policy levers** are the parameters that gate the evaluator's decisions, with
  their defaults. They appear in evaluator tables only, because core functions
  carry no policy. A default is a choice someone made, not a property of the
  math.
- **Contraindications** are conditions under which the output is invalid or
  meaningless. These are not performance caveats; they are cases where the number
  is wrong or undefined and should not be reported.

Check your intended metric's row before you cite it: verify that the
contraindications do not apply to your data, and use the validation level to
calibrate the strength of your claim. Where DataEval's own evidence is thin,
[Known gaps](#known-gaps-in-this-pages-evidence) says so.

```{important}
DataEval is not a certification tool. It produces measurements and rule-based classifications of them, not
certifications. If a result surprises you, check the policy behind it (for
evaluators), the extractor that produced its input (for anything
embedding-based), or the
[binning applied to its factors](#metadata-binning-a-policy-applied-to-every-factor)
(for anything metadata-based).
```

### Core functions vs. evaluators

The library is split into two layers, [Evaluators](#evaluators) and
[Core functions](#core-functions), each with its own potential failure
modes.

- Core functions ({doc}`dataeval.core <../reference/autoapi/dataeval/core/index>`)
  are stateless and observe the data they are given, which can be images, metadata or
  embeddings: a bound, a count, a distance,
  a distribution, with no threshold, prior, or recommendation attached.
  The trust question is: "is the math right, and do my data satisfy its
  assumptions?"
- Evaluators ({doc}`dataeval.quality <../reference/autoapi/dataeval/quality/index>`,
  {doc}`dataeval.bias <../reference/autoapi/dataeval/bias/index>`,
  {doc}`dataeval.scope <../reference/autoapi/dataeval/scope/index>`,
  {doc}`dataeval.shift <../reference/autoapi/dataeval/shift/index>`,
  {doc}`dataeval.performance <../reference/autoapi/dataeval/performance/index>`)
  wrap core observations in policy: a
  threshold, a default distribution, a significance level, a ranking rule. The
  evaluator tables name the policy each one carries, because that policy, not
  the underlying math, is a common source of results that do not transfer
  between domains. The trust question is: "is the math right, and is the built-in
  policy appropriate for my problem?"

## Feature extractors: a dependency of both layers

Extractors are not a third kind of feature. They produce the embeddings that
many of the functions and classes rely on. This makes extractor choice the
single largest uncontrolled variable in a DataEval assessment. BER,
divergence, coverage, prioritization, label-error detection, and every
embedding-based drift detector inherit the geometry of whatever extractor
produced their input, so an extractor problem shows up as a wrong number in a
table on [Validation evidence by evaluator](ValidationEvidence.md) and not as an error.

{doc}`Embeddings <Embeddings>` covers how to choose one. Extractors carry no
validation-evidence rating of their own, because they produce inputs rather than
measurements: for the model-backed ones, the evidence is whatever you have about
the model you supplied. What follows is every extractor DataEval ships, what it
actually embeds, and the conditions under which it will mislead you.

### Extractors that embed the data

These describe the images. Distances between their outputs are claims about the
data itself, which is what every geometric metric in this library assumes.

{class}`.Embeddings` accepts an extractor only when it returns one row for each
dataset item. Every extractor in this table does, except {class}`.TorchExtractor`
with `postprocess_fn`.

:::{list-table}
:widths: 20 32 48
:header-rows: 1

- - Extractor
  - What it produces
  - What it inherits, and when it misleads
- - {class}`.FlattenExtractor`
  - Raw pixels flattened to `(n_images, C x H x W)`. No framework, no weights,
    no randomness.
  - The only extractor with nothing to version and nothing to seed (no model,
    no fitted state), but pixel space is not semantic space. Distances are
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
    Two hazards: it is the only stateful extractor, learning its
    vocabulary from the images you give to `fit`, so two separately constructed
    instances produce embeddings that are not comparable (fit once, share the
    instance); and the vocabulary is a k-means fit, so reproducibility depends
    on the
    [global seed](../notebooks/h2_configure_defaults.py#set-the-seed).
    Fit it yourself on the images that should define the vocabulary. An unfitted
    instance fits on the data of its first call and raises a `UserWarning`. Inside
    {class}`.Embeddings` or {class}`.Duplicates` that call covers one batch, so
    the vocabulary comes from `batch_size` images and not from the dataset.
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
    With `postprocess_fn` set, the extractor returns one row for each detection
    and does not fit {class}`.Embeddings`. See
    [Keep one row for each item](Embeddings.md#keep-one-row-for-each-item).
- - {class}`.OnnxExtractor`
  - The same, executed through ONNX Runtime, with lazy model loading and
    automatic provider selection (CUDA, then CoreML, then CPU).
  - Everything above, plus: on a multi-output model `output_name` decides which
    tensor becomes the embedding, and picking the wrong one produces a valid
    array rather than an error; `image_size` overrides the model's native input
    size by bilinear resize. Because the execution provider is selected from
    what is available, the same model on two machines can differ numerically. Do
    not treat runs as bitwise comparable across hardware.
- - {class}`.VideoTorchExtractor`
  - One vector for each video. The extractor splits a video into non-overlapping
    clips, embeds each clip with a pre-trained video model, and combines the clip
    vectors with `clip_aggregation` (`"mean"` by default, or `"max"`).
  - Everything above for {class}`.TorchExtractor`, plus: `pooling` (`"first"` by
    default) decides how the sequence output of a clip becomes one vector, and
    `clip_aggregation` decides how clips become a video. Each choice changes the
    geometry without raising an error. The extractor needs a frame count from
    `num_frames` or from `model.config.num_frames`. Without a `processor`, the
    videos must already be preprocessed.

:::

### Extractors that embed a model's predictions

These describe what a model does, not what the data looks like. All of
them re-run the supplied model and use only its predictions, ignoring any
ground-truth targets carried by the dataset. Three consequences follow, and they
apply to every row in this table:

- Drift found on these features is drift in model behavior. It may be caused
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
  - The rawest of these: the model's output surface with no reduction applied.
    Useful when you want drift measured on the decision the model is actually
    making rather than on features it might use. It is also the score producer
    the two uncertainty extractors are usually wrapped around.
- - {class}`.DetectionGeometryExtractor`
  - `[center_x, center_y, width, height, area, aspect]` per detection, from
    normalized boxes, so it is resolution-independent.
  - Carries no appearance information at all: it can tell you that object
    size, position or shape distribution shifted, never why. The `confidence`
    parameter (default `0.0`, keeping everything) drops detections below a score,
    so once set, both the rows and how many there are depend on the model's
    calibration. A recalibrated model looks like changed data.
- - {class}`.UncertaintyExtractor`
  - `(n_samples, 1)` Shannon entropy of the predicted class distribution, from a
    wrapped score producer. Suited to {class}`.DriftUnivariate`.
  - `preds_type` must match what the score producer emits (`"logits"` applies a
    softmax first, `"probs"` does not), and getting it wrong computes entropy on
    the wrong scale with no error raised. `normalize=True` divides by the maximum
    entropy for the number of classes present, which makes scores comparable
    across class counts but not across models.
- - {class}`.ClasswiseUncertaintyExtractor`
  - A `dict` mapping class index to that class's array of detection
    uncertainties.
  - Not a drift feature extractor, despite passing an `isinstance` check
    against the `FeatureExtractor` protocol at runtime (that check only looks for
    `__call__`). Select a single class's array and pass that to a detector.
    `threshold=0.99` is a policy lever, not a detail: it decides how many classes
    each detection is counted under, with `1.0` meaning winner-take-all.

:::

```{note}
{class}`.UncertaintyExtractor` and {class}`.ClasswiseUncertaintyExtractor` both
wrap a score producer rather than a model, so any `FeatureExtractor` can supply
their input, including {class}`.TorchExtractor`, {class}`.OnnxExtractor`, or an
{class}`.Embeddings`. Running both over the same data invokes that producer
twice. For a classification model, wrapping the producer in a caching
{class}`.Embeddings` and sharing the one instance avoids paying for inference
again. A detection model returns one row for each detection, which
{class}`.Embeddings` does not accept, so both extractors run the model.
```

No result object records the extractor that produced it, so record which
extractor and which weights produced any assessment you intend to cite.
Two DataEval runs with different extractors are not comparable.

## Metadata binning: a policy applied to every factor

Binning encodes an analytical policy. Many of the functions
and classes that consume a {class}`.Metadata` class require factor discretization.
How the data is discretized matters, and that discretization is a hidden
policy choice: applied by default, decisive for the data, and recorded in the
output only as a fingerprint. Under the default `factor_source="auto"`, a
{class}`.Balance` score describes a factor as binned where someone chose or
accepted its cut, and as measured otherwise. This is the metadata-side
counterpart to the extractor problem above: where embeddings are the largest
uncontrolled variable in the geometric half of the library, binning is the
largest one in the metadata half.

### What the default does

[Binning](Binning.md) explains the mechanism in full: the {func}`.is_continuous`
heuristic and its tuned constants, the three `auto_bin_method` strategies and
what each trades away, and the treatment of missing values. The properties that
bear on trusting a result:

- The bin count is a function of the data, not a setting. `"uniform_width"`
  derives its count from the draw, so 500 samples from a standard normal come
  out in anywhere from 3 to 10 bins. The same factor measured on two datasets
  can be cut into different numbers of bins, so binned factor values are not
  comparable across runs.
- A factor can collapse to two bins. The reduction floors there, so a factor
  cannot come back constant, but a small sample with a far-off tail reaches that
  floor reliably and arrives as a binary split with nothing in the output saying
  so.
- The continuous/discrete verdict is a heuristic, reporting no p-value and
  no error rate, resting on five tuned constants. Below 20 observations it
  returns "discrete" unconditionally, so on a small dataset every distinct float
  becomes its own category and the factor is nearly one-to-one with the sample
  index.
- A discrete verdict does not mean the factor is left as it was. No numeric
  factor carries more levels than `max(20, sqrt(n))`, the budget a histogram
  would use, so a discrete column taking more distinct values than that is
  binned rather than ordinal-encoded: a contingency table with more cells than
  observations reports on which cells were hit rather than on the factor. The
  cap applies to numeric columns only; a non-numeric column keeps one category
  per distinct value however many there are.
- Missing values get a bin above the observed ones, whose position is an
  artifact of keeping the codes contiguous rather than a statement that missing
  is large. Anything treating a binned factor as ordinal reads that position as
  a value. {class}`.Balance` does not where it reads codes, since it tabulates
  them as categories, and neither does {class}`.Parity`, whose test is nominal: it reads each code as
  an unordered category, never as a magnitude. For Parity the missing bin is
  still a level, so a factor whose missingness goes with the class shows up as
  an associated missing-value level. Where the missing rate is material, check
  whether a finding is about the factor or about its absence.
- An explicit `continuous_factor_bins` entry marks a factor continuous
  downstream regardless of what it actually contains.

### Which results depend on it

:::{list-table}
:widths: 24 18 58
:header-rows: 1

- - Consumer
  - Binning is
  - What that means
- - {class}`.Diversity`, {func}`.parity` ⚠️, {class}`.Parity` ⚠️
  - Required
  - The diversity indices and the contingency-table test are defined over
    discrete categories, so a continuous factor must be cut somewhere. Changing
    the cut changes the score, and no setting is correct independent of the
    question being asked. For `parity` the cut also decides cell occupancy:
    coarser bins avoid the sparse cells reported as `too_thin` in
    {attr}`.ParityOutput.cells`, at the cost of resolution.
- - {class}`.Balance`
  - Applied where a cut was chosen
  - Under `factor_source="auto"`, a factor whose cut was declared, requested by
    count or accepted is read from `Metadata.factor_data`, which is bin and
    category indices, so the information lost at the cut cannot be recovered
    downstream. A factor whose cut nobody chose is read from its measured
    values, and the `scored_as` column of `factors` records which regime scored
    each pair. See [Binning](Binning.md#which-consumers-require-it) for the rule.
    Binning reaches the three DataFrames unequally: `balance` and `classwise` divide by the class
    label's entropy, which is never binned, so their scores converge as a factor
    is cut more finely; `factors` compares two possibly-binned factors and is
    scored so its denominator no longer grows with a bin count you did not
    choose, though the mutual information above the line still shifts with the
    binning. See [Choosing bin edges](BinEdges.md#binning-reaches-the-three-outputs-differently).
- - {func}`.split_dataset` with `split_on`
  - Floats and declared cuts only
  - Groups are formed from recorded factor values, except for a float factor or
    one with a declared cut, whose grouping granularity is the binning's. A float
    factor the automatic path reduced to two bins gives you two groups, silently
    weakening the split constraint you asked for.
- - Anything reading `Metadata.factor_data`
  - Applied
  - `factor_data` is the binned, integer-valued view and is what the bias
    evaluators and the grouping of a float factor consume.
    `Metadata.rows_at(md.view)` carries the unbinned values, and
    `Metadata.factor_info` reports per factor whether it was binned or digitized.
- - {func}`.factor_deviation`, {func}`.factor_predictors`
  - Not applied
  - These take plain mappings of factor name to raw array, so they are
    unaffected unless you build those mappings out of binned columns yourself.

:::

### What no output records

- The extractor does not appear in a result, and the binning appears only as a
  fingerprint. Every bias result carries
  {attr}`~dataeval.Metadata.encoding_digest` in its execution metadata, so two
  runs can be told apart, but the digest says whether the cuts matched, not
  what they were, and nothing records the extractor at all. Keep
  {meth}`.Metadata.export_encoding`'s descriptor alongside the result for the
  binning; comparability of an embedding-based result still rests entirely on
  what was recorded outside DataEval. A record that supports an audit of an
  embedding-based result names the extractor's architecture and version, its
  checkpoint or an artifact digest, the preprocessing and resizing, the
  normalization, the random seeds, the sampling unit, the metadata mapping and the
  DataEval version. A digest detects that an artifact changed and does not
  reconstruct the environment that produced it.
- The automatic path announces itself. Structuring raises a `UserWarning`
  naming every factor it binned unasked, because the per-factor WARNING it also
  writes to the `dataeval.metadata` logger is silenced by the `NullHandler`
  DataEval attaches there. {attr}`.FactorInfo.encoding` keeps the same fact
  afterwards, in `provenance`.

[Binning](Binning.md#what-the-choice-determines) covers the rest: what
{attr}`.Metadata.factor_info` reports and what it does not, the thresholds no
automatic method can find, and the sense in which a conclusion can be about the
cut rather than about the data.

## Validation evidence levels

These describe DataEval's verification of its own code, not the standing of the
underlying method in the literature. A method can be well established in the
literature and still be `Internal only` here: that combination means the theory
is sound and the implementation has not been checked against a second
implementation. A level also does not establish external validity: verification
against constructed inputs and known values does not show that a result is valid for a
given mission, operating population or embedding extractor.

:::{list-table}
:widths: 18 82
:header-rows: 1

- - Level
  - Meaning
- - Delegated
  - The statistic is computed by an established third-party library
    (SciPy, scikit-learn). DataEval supplies the surrounding workflow, not the
    estimator. Correctness of the statistic is that library's.
- - Ported
  - Implementation derived from a published reference implementation, carried
    over together with that project's test suite.
- - Cross-checked
  - DataEval's own implementation, verified in CI against an independent
    implementation of the same quantity.
- - Ground-truth checked
  - Verified in CI against a closed-form value. The test constructs a
    distribution whose target quantity is known analytically and asserts the
    estimator recovers it, so the test fails when the estimate is wrong rather
    than merely when it changes.
- - Anchored
  - Verified against hand-computed or analytically known values on constructed
    inputs, plus behavioral tests. No second implementation is compared.
- - Internal only
  - Verified by unit and property tests (invariants, edge cases, shape, and
    error handling) written against the intended behavior. Correctness relative
    to the cited derivation rests on code review.

:::

## Evaluators

Every evaluator's evidence basis, validation evidence, policy levers and contraindications are listed in
[Validation evidence by evaluator](ValidationEvidence.md#evaluators), grouped as data quality, dataset bias, dataset
scope, distribution shift and performance.

## Core functions

Core functions carry no thresholds or policy, and their contraindications are statistical and computational. They are
listed in [Validation evidence by evaluator](ValidationEvidence.md#core-functions) in six groups: feasibility and
performance limits, divergence and embedding geometry, ranking and selection, image statistics and hashing, metadata and
bias, and labels and ontology.

## How DataEval validates its own implementations

### Provenance

Several drift and out-of-distribution detectors in `dataeval.shift`
({class}`.DriftMMD`, {class}`.DriftUnivariate`, {class}`.OODReconstruction`, the
shared OOD base, and the reconstruction scorer those detectors call) are derived
from
[alibi-detect v0.11.4](https://github.com/SeldonIO/alibi-detect/tree/v0.11.4)
(Copyright © 2023 Seldon Technologies Ltd, Apache 2.0), together with that
project's tests. `dataeval.utils.losses` shares that lineage.

{class}`.DriftReconstruction` is the exception in that group: it is
conceptually inspired by alibi-detect's reconstruction detector, but the
detector code and its tests are DataEval's own. It reaches alibi-derived code
only through the shared reconstruction scorer, which is why its row on
[Validation evidence by evaluator](ValidationEvidence.md#distribution-shift) carries two ratings.

DataEval's HDBSCAN implementation (`dataeval.core._fast_hdbscan`) is adapted
from [fast_hdbscan](https://github.com/TutteInstitute/fast_hdbscan) v0.2.0-0.2.2
(Copyright © 2020 Leland McInnes, BSD 2-Clause). Everything reached through
`cluster_algorithm="hdbscan"` (in {class}`.Duplicates`, {class}`.Outliers`,
{func}`.cluster`, and the `rank_hdbscan_*` functions) rests on that port.

Where an established implementation of a statistic exists, DataEval calls it
rather than reimplementing it. {class}`.DriftUnivariate` delegates all five of
its tests to `scipy.stats`; clustering delegates to scikit-learn's `KMeans`;
mutual information estimation uses scikit-learn's estimators.

### Test suite

The suite is organized to mirror the package (`tests/core`, `tests/shift`,
`tests/bias`, `tests/quality`, `tests/scope`, `tests/performance`, ...) and CI
enforces a 90% coverage floor (`fail_under = 90`). One hand-written path
(`core/_fast_hdbscan/*`) is excluded from that measurement, alongside the
generated `_version.py`. The ported HDBSCAN code therefore does not count
toward the floor, which is worth knowing because `cluster_algorithm="hdbscan"`
is the default in {class}`.Duplicates` and {class}`.Outliers`.

The suite contains three kinds of evidence, and they are not equally strong:

- **Regression tests** pin an estimator's output against a value recorded from
  a previous run. They catch unintended change, but they pass just as happily
  when an estimator has been wrong from the start, which is why the two kinds
  below, not this one, carry the weight of the ratings in the evidence tables.
- **Cross-implementation checks** compare against an independent library:
  image statistics (skew, kurtosis, entropy) against SciPy, and several
  clustering and OOD tests against scikit-learn.
- **Ground-truth checks** (`tests/core/test_validation_benchmarks.py`)
  construct a distribution whose target quantity has a closed-form solution and
  assert the estimator recovers it. BER bounds are checked against the analytic
  Bayes error of two Gaussians: both that the interval contains it and that it
  stays tight enough to be informative, since containment alone would be
  satisfied by returning `[0, 1]`. Divergence is checked at its 0 and 1
  endpoints, and the null-model baselines against their closed forms.

A separate test (`docs/tests/test_references.py`) pins the set of symbols that must
cite their sources, so citation coverage cannot regress.

What the suite does not do is reproduce published benchmark results on
public datasets. See [Known gaps](#known-gaps-in-this-pages-evidence).

### Studies on real footage

Internal testing in October 2026, on the development code that followed v1.1.4,
demonstrated the findings below for {func}`.ego_stats`, the sharpness of
{func}`.image_stats` and the `PERSISTENCE` maps of {func}`.video_stats`. The
scripts, saved outputs and footage checksums are available on request.

The footage is eleven public drone clips of 120 to 600 frames each: seven
low-altitude urban sequences from the VisDrone2019-MOT validation set and four
landscape and town clips from Wikimedia Commons. Burned-in overlays, parts of
the vehicle and, for the maps, a smudge and a scratch on the lens are rendered
onto it with a known mask. Ego motion is scored only where the clean
footage's reading is trusted. A pair is wrong while trusted when
`ego_stats` trusts it and reads more than 0.5 px from the clean reading.

| Study | Result |
| --- | --- |
| `ego_stats` on 3,489 frame pairs, before and after the two-motion check | The same 3,363 pairs are trusted before and after. |
| Four overlay layouts covering 3% to 20% of the frame, on 1,957 pairs | The densest layout, with telemetry in all four corners and two scrolling tapes, read 135 pairs (6.9%) wrong while trusted, and 21 (1.1%) with the overlay passed as `mask=`. The three lighter layouts read at most 2 pairs wrong unmasked and at most 6 masked. A mask reduced refusals in all four. |
| Four parts of the vehicle, on 1,557 pairs | An unmasked hood (about a fifth of the frame) read 173 pairs (11%) wrong while trusted, and a bow (15%) read 73 (4.7%). Masked, each read 14 or fewer. A wing (5%) and skids (4%) read at most 5 wrong, masked or not. |
| `image_stats` sharpness on unblurred frames, with and without each part | A hood with lens vignetting lowered sharpness to 89% of the clean value, and vignetting alone to 93%. |
| A gate that drops the lowest 10% of sharpness, on 60 frames per clip with six blurred by 5, 9 or 21 px | With or without a part, the gate dropped all six blurred frames on 10 clips and none of them on the eleventh. |
| Constructed scenes: the two-motion check on 2,095 cases of still, panning, zooming and rolling cameras, with and without movers | Still-camera readings that were wrong while trusted fell from 255 to 3, each with a mover covering 71% to 85% of the samples. One correct camera motion was newly refused. |
| Constructed scenes: a half-frame mover in front of a still camera, pixel noise with a standard deviation of 3, five seeds | The pair read as a trusted zoom in every seed at 1 and 2 px per frame and in four of five at 3 px. It was refused at 4 and 5 px. |
| `video_stats` `PERSISTENCE` maps on 99 videos: each clip's first 150 frames, clean, under four overlay layouts, under a hood and a bow, and behind a smudge and a scratch | Cells read as fixed outside any overlay averaged 0.36%. The near-still clip (VisDrone `uav0000268`, about 1.25 px of camera motion per pair at a 360 px short side) reads NaN. Overlay and part recall, the share of an overlay's or part's cells read as fixed, averaged 24.7% over its 66 videos. |

### API stability discipline

Two mechanisms make stability observable rather than a matter of trust:

- Experimental features raise
  {class}`~dataeval.exceptions.ExperimentalWarning` at runtime and are marked in
  their docstrings: you cannot use one without being told.
- Deprecated symbols raise {class}`~dataeval.exceptions.DeprecatedWarning` when
  used, and a dedicated CI test (`docs/tests/test_deprecated_docs.py`) fails the build
  if any deprecated public object lacks a `.. deprecated::` directive in its
  docstring. A deprecated symbol cannot ship silently. The symbols deprecated in
  v1.1 (notably `ClassifierUncertaintyExtractor`) were removed in v1.2. Deprecated
  in v1.2 and removed in v2.0: `compute_stats` (renamed {func}`.image_stats`)
  and `ClassFilter`. {func}`.compute_ratios` is deprecated in v1.2 and removed
  in v1.3. See the [v1.2 migration guide](../migration/v1.2.md). Transitional
  warnings for changed parameter defaults are a separate mechanism and use a
  plain `FutureWarning`. See [Changed defaults](#stability-status) below.

## Stability status

As of v{{ release }}:

Experimental ⚠️ (API may change or be removed):

- {func}`.parity`, {class}`.Parity`, {func}`.uap`, {class}`.Feasibility`,
  {class}`.DriftKnownDomains`, `ParityOutput` and `FeasibilityOutput` raise
  {class}`~dataeval.exceptions.ExperimentalWarning` on first call or
  instantiation.
- {func}`.video_stats` with any `VideoStats.PERSISTENCE` flag
  (`PERSISTENCE_EGO`, `PERSISTENCE_BLUR`, `PERSISTENCE_BLUR_RAW`, selected
  directly, through the group or through `ALL`) raises
  {class}`~dataeval.exceptions.ExperimentalWarning` on the first such call in
  a process. Every other {func}`.video_stats` statistic is stable, except the
  `BLUR` statistic in the next item.
  {class}`~dataeval.protocols.SharpnessMeasure`, the protocol a `blur_measure`
  follows, is experimental with the persistence statistics and raises nothing
  on its own.
- {func}`.video_stats` with `VideoStats.BLUR_LEVEL`, selected directly, through the `BLUR` group or through `ALL`,
  raises {class}`~dataeval.exceptions.ExperimentalWarning` on the first such call in a process. The default,
  `VideoStats.BASIC`, never raises it. The column name and the `blur_samples` keyword may change. See
  [Blur statistics](BlurStatistics.md).
- {func}`.image_stats` with any `ImageStats.BLUR` flag (`VISUAL_BLUR`, `VISUAL_BLUR_DIRECTIONALITY`) raises
  {class}`~dataeval.exceptions.ExperimentalWarning` on the first such call in a process, and so does a
  {class}`.QualityGate` whose `stat` is one of their columns. `ImageStats.ALL` includes the group and the default of
  {func}`.image_stats` does not, so a default call never raises it. See [Blur statistics](BlurStatistics.md).
- {func}`.image_stats` with the `labels` keyword. The keyword may change. It raises no warning. The two class
  statistics of `ImageStats.SPATIAL` read it, and no other statistic does.
- {func}`.hash_neighbors` with the `partition` argument or the `k` argument
  raises {class}`~dataeval.exceptions.ExperimentalWarning` on the first call
  that passes each in a process. Every other use of {func}`.hash_neighbors` is
  stable.
- {class}`.Duplicates` with the `segment_neighbors` argument, or built from a
  `Duplicates.Config` that sets it, raises
  {class}`~dataeval.exceptions.ExperimentalWarning` when the first
  {class}`.Duplicates` that carries the option is constructed in a process.
  Building the `Config` alone raises nothing. {class}`.Duplicates` passes `k`
  to {func}`.hash_neighbors` itself and raises nothing there.
- {func}`.frame_stats` raises {class}`~dataeval.exceptions.ExperimentalWarning` on the first call in a process, whichever
  {class}`.FrameStats` flags it selects. The flag names, the column and detail names, and the roll-up names may change.
- The inputs of {class}`.Feasibility` ({class}`.TargetSpec` and
  {class}`.Allowances`), the core
  functions behind its bounds ({func}`.ap_lower_staircase`,
  {func}`.ap_upper_envelope`, {func}`.binomial_endpoint`,
  {func}`.consensus_max`, {func}`.cost_consensus`, {func}`.cost_consensus_max`,
  {func}`.disjoint_panels`, {func}`.fit_partition`, {func}`.kink_costs`,
  {func}`.local_panel_bound`, {func}`.mean_endpoint`, {func}`.ratio_endpoint`,
  {func}`.units_to_cross`, {func}`.witness_thresholds` and
  {func}`.zero_one_consensus`) and the `fold_scores` method of
  {class}`.Relabel` are experimental in the same way.
- `ParityResult` and `ParityCells` carry the same experimental status but raise
  nothing on their own: they are `TypedDict`s describing {func}`.parity`'s
  return value, and the warning comes from the call that produced it.

Changed defaults (the symbol is not deprecated; only the transitional
warning goes away):

- {class}`.Duplicates`: the image `hash_radius` default of `0` will change in a
  future major release, so that it finds perceptual near-duplicates rather than
  only bit-identical images. Leaving it unset when `evaluate` reads image data
  raises a plain `FutureWarning`, not
  {class}`~dataeval.exceptions.DeprecatedWarning`, so a warning filter narrowed
  to that class will not catch it. Pass the argument explicitly to silence it.
  Video already defaults to `6` and does not warn.
  The class itself is stable and is not scheduled for removal.

Everything else in the public API is considered stable for the v1.x series. A group of statistics can gain members in
a release, and `ImageStats.ALL` and `VideoStats.ALL` can gain experimental ones. `ImageStats.BASIC`,
`ImageStats.STANDARD` and `VideoStats.BASIC` change only after a deprecation. See {class}`.ImageStats` and
{class}`.VideoStats`.

## When not to use DataEval

Being explicit about non-applicability is part of the trust argument.

DataEval is not applicable when:

- You need a model evaluation. DataEval characterizes datasets and the
  limits they impose. It does not evaluate a trained model's performance,
  fairness, or robustness. `nullmodel_*` and {class}`.Sufficiency` touch models
  only to establish data-side baselines and curves.
- You need a certification or an accreditation artifact. No DataEval output
  constitutes evidence of fitness for deployment. The outputs are inputs to a
  human assessment.
- Your data is not amenable to embeddings. Most of the library operates in
  embedding space. Without a feature extractor that represents your domain,
  BER, divergence, coverage, prioritization, and embedding-based drift are all
  measuring the extractor rather than the data.
- Your sample sizes are very small. Non-parametric estimators, mutual
  information, contingency-table tests, and radius-based coverage all degrade
  at small n, several silently, by returning a plausible number.
- You need real-time, per-frame monitoring. {class}`.DriftMMD` permutation
  testing and MST construction scale poorly. Use chunking, sampling, or a
  cheaper detector.
- You need causal explanations. Metadata attribution
  ({func}`.factor_deviation`, {func}`.factor_predictors`) and bias metrics are
  associational. They generate hypotheses.

Additionally, treat a single number with caution when:

- It comes from an experimental feature (see [above](#stability-status)).
- It was produced by an extractor you have not validated on your domain.
- It is a bias metric over continuous metadata whose binning you did not set
  yourself (see
  [Metadata binning](#metadata-binning-a-policy-applied-to-every-factor)).
- The evaluator's default policy was not reviewed against your operational
  prior.

## Known gaps in this page's evidence

Each gap below is a current limitation of DataEval's evidentiary record and not
a research frontier.

1. Most public symbols carry no literature citation in their docstring. A
   minority of public functions and classes have a `References` section, though
   the count has grown substantially. `REQUIRES_REFERENCES` in
   `docs/tests/test_references.py` is the current list and the source of truth for
   it; the test pins that set so it cannot regress, and adding a symbol to the
   list is how the number goes up. For everything else the citation lives on a
   concept page or nowhere. Where an entry above cites a concept page rather
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

2. No reproduction of published benchmarks on real datasets. The validation
   benchmark suite (`tests/core/test_validation_benchmarks.py`) checks BER,
   divergence, and the null-model baselines against closed-form values on
   constructed distributions. Those tests fail when an estimate is wrong, not
   merely when it changes. What is missing is evidence on real data: nothing
   in CI demonstrates that DataEval's estimators reproduce published results on
   public datasets. The FeeBee BER benchmark
   ([Renggli et al., 2021](Divergence.md#references)) is the obvious candidate.
   Until that exists, every rating on this page rests on synthetic or internal
   evidence.

3. Threshold and policy defaults have no documented provenance. The
   `Outliers` thresholds, `Duplicates` cluster sensitivity, drift `p_val`, and
   the expected distributions in `Representation` are all policy. None of them
   records who chose the value, on what data, or what the alternatives were.
   Users inheriting a default deserve to know whether it was derived or
   inherited.

4. Coreset selection bounds are unverified on foundation-model embeddings.
   The geometric arguments behind distance-based selection were established on
   classical CNN features. Whether they hold in the latent spaces of modern
   vision transformers and multimodal encoders is an open question, and one
   DataEval does not currently benchmark.

5. The domain-classifier detectors are unbenchmarked under severe
   imbalance. Both {class}`.DriftDomainClassifier` and
   {class}`.OODDomainClassifier` rely on AUROC, which is optimistic when the
   negative class dominates, the regime of rare-event operational monitoring.
   PR-AUC or stratified validation folds would be the fix; neither has been
   evaluated.

6. The binning policy is documented but unvalidated, and unsupported by
   tooling. Which evaluators depend on binning, and whether they require it
   or merely inherit it, is now stated above under
   [Metadata binning](#metadata-binning-a-policy-applied-to-every-factor).
   What remains missing is evidence for the specific choices and any means of
   testing them:
   - The automatic thresholds have no documented provenance. The
     continuous/discrete heuristic's `0.5 / sqrt(n)` Wasserstein cutoff, its
     0.85 lattice fraction, its 0.005 duplicate tolerance, the 20-sample floor
     below which every factor is called discrete, and `"uniform_width"`'s
     10-samples-per-bin target are all tuned constants. None records who chose
     the value or on what data, and no test establishes the misclassification
     rate of the heuristic on known-continuous and known-discrete factors. The
     same holds for the `max(20, sqrt(n))` level budget above which a discrete
     factor is binned rather than left alone, and for the two-bin floor the
     reduction stops at, which a small lopsided factor reaches without comment.
   - No sensitivity analysis exists. Nothing in DataEval re-runs a bias
     metric across binnings or reports how much of a score is attributable to
     the cut, so a user cannot distinguish a finding from an artifact without
     scripting the comparison themselves.
   - Bin counts are not stable across datasets. Because `"uniform_width"`
     derives the count from the data, the same factor can be binned differently
     in a reference and an operational set. {attr}`.FactorInfo.encoding` now
     records the edges that were applied, so the discrepancy is detectable by
     comparing two runs' specs, and {meth}`.Metadata.export_encoding` writes them
     out for `Metadata(dataset, encoding=...)` to reapply unchanged. What is still
     missing is a way to tell whether a pinned cut should have moved: a locked
     encoding reports the bins the new data no longer fills, but nothing measures
     how much of a score the cut is responsible for.

## Related concept pages

- [Validation evidence by evaluator](ValidationEvidence.md): the evidence basis, validation level, policy levers and
  contraindications of every evaluator and core function
- {doc}`Embeddings <Embeddings>`: the largest uncontrolled variable in any
  DataEval assessment; read this before trusting any embedding-based result
- {doc}`Acting on results <ActingOnResults>`: what to do with a finding, and
  which findings do not warrant action
- {doc}`Data integrity <DataIntegrity>`,
  {doc}`Dataset bias <DatasetBias>`, {doc}`Dataset coverage <DatasetCoverage>`,
  {doc}`Performance limits <PerformanceLimits>`,
  {doc}`Distribution shift <DistributionShift>`,
  {doc}`Divergence <Divergence>`, {doc}`Clustering <Clustering>`,
  {doc}`Ontology <Ontology>`: the theory behind each feature, with each
  page's own references
- {doc}`Data leakage <Leakage>`: why an assessment can be internally valid and
  still not generalize

## References

Method-level citations live on the concept page for each method, linked from
the tables on [Validation evidence by evaluator](ValidationEvidence.md). The sources specific to this page are:

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
