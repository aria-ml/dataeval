# New in v1.2

v1.2 evaluates image and video datasets in context. You can investigate duplicate footage, unusual frames and objects,
and metadata bias. You can choose the level of analysis and trace quality findings back to the source video, frame,
track or detection.

This page is for teams deciding which capabilities to bring into a test and evaluation platform. It says what works on
a multi-object tracking (MOT) dataset, what takes one more step, and what does not work yet. The
[Functional overview](../reference/FunctionalOverview.md) lists each tool's support by task, and the
[API reference](../reference/autoapi/dataeval/index.rst) documents each name.

## What works on a tracking dataset

These tools take a tracking dataset, one video for each item:

- {class}`.Duplicates` finds exact and near copies among frames, segments and whole videos, and reports footage that
  crosses a split. {class}`.Outliers` judges whole videos and frames, and with `per_target=True` tracks and detections.
  Both also take videos that carry no tracking targets or no metadata.
- {class}`.Balance`, {class}`.Diversity` and {class}`.Parity` read the dataset's {class}`.Metadata`. `Parity` groups
  its rows by the video they came from. `Balance` counts a video once for a factor that is constant within it, so
  repeated frames do not read as correlation.
- {func}`.split_dataset` splits by video. {func}`.video_stats`, {func}`.track_stats` and {func}`.ego_stats` measure
  each video, each track and the camera motion of each frame.
- {class}`.SequenceFrames` selects the frames that other tools analyze and records how many source frames each one
  stands for. {class}`.VideoTorchExtractor` embeds a whole video, so the drift and out-of-distribution detectors can
  compare videos.

## What takes one more step

Some tools work on one image, one object or one array at a time. The
[Adapting your data](../reference/FunctionalOverview.md#adapting-your-data) section of the Functional overview explains
each adapter.

- Tools that read images or frame embeddings, such as {func}`.image_stats`, the embedding metrics and the drift
  detectors over frames, take `SequenceFrames(dataset)`. Pass a frame selector, because the number of frames sets the
  cost.
- Tools that need one embedding per label, such as {class}`.Coverage`, {class}`.Prioritize` and
  {func}`.label_errors`, take `DetectionCrops(SequenceFrames(dataset))`.
- Tools that read each row as independent, such as {func}`.label_parity` and {func}`.ber_knn`, take `groups` so that
  the frames of one video count as one unit.
- The label functions, such as {func}`.label_stats` and {func}`.label_cooccurrence`, take arrays. Read the labels of
  every detection from `md = Metadata(dataset)`, as in `label_stats(md.class_labels, md.item_indices)`.

## Analyze your data by level

{class}`.Metadata` describes a tracking dataset at four levels: `sequence` (a video), `unit` (a frame), `track` and
`instance` (one detection). A value recorded for a video applies to each of its frames. {meth}`.Metadata.aggregate`
rolls finer values up, such as the mean, variance, trend or longest run of a frame statistic for each video. A
statistics output can also declare its own roll-ups. {meth}`.Metadata.add_factors` applies them when it adds the
output, as it does for the output of {func}`.ego_stats`.

- {meth}`.Metadata.classed_by` makes any factor the class axis that the bias evaluators read.
- `Outliers` judges a value against the rest of its level. It can judge within a stratum of any categorical factor
  (`strata=`), flag an improbable combination of ordinary values (`combinations=`), or compare a value with its
  neighbors in the video (`neighbors=`). The last two take a `Metadata`.
- `Outliers` keys a finding on a frame, track or detection with a {class}`.SourceIndex`, which names one row at one
  level. {class}`.SourceLocator` returns the frame, track or detection it names.

In this example, `dataset` is a tracking dataset of 25 videos, one of them shot in the dark:

```python
from dataeval.data import SourceLocator
from dataeval.quality import Outliers

result = Outliers().evaluate(dataset)

result.sequences  # the dark video, judged against the other videos
result.frames  # each of its frames, judged against every frame in the dataset

address = next(iter(result.frames.outliers))  # SourceIndex(7, 0, 'unit'): frame 0 of video 7
frame = SourceLocator(dataset)[address]  # that frame, with its pixels and the video it came from
```

## Other additions

v1.2 adds two experimental evaluators. {class}`.Feasibility` decides whether a classification target, such as an
accuracy, AP or recall, is reachable on a dataset. It reports whether the dataset demonstrates the target, rules it out
under recorded assumptions, or does neither, and it returns the evidence behind each answer. {class}`.DriftKnownDomains`
tells which of several known domains, such as sensors, sites or seasons, new data is closest to. It flags data that
belongs to none of them.

{class}`.Metadata` checks and repairs its inputs. It checks factors with a fixed meaning, such as `timestamp` and
`latitude`, against that meaning. {meth}`.Metadata.repair` declares how a column is read, such as a sentinel value or
a unit. {meth}`.Metadata.unusable_rows` names the rows behind a factor that could not be used, and
{meth}`.Metadata.add_factors` reports where new values disagree with existing ones.

## What does not work yet

v1.2 ships with these limits. The ROADMAP's
[Known limits at release](https://github.com/aria-ml/dataeval/blob/v1.2.0/ROADMAP.md#known-limits-at-release) states
each one in full.

- No evaluator reads the sample weights that frame selection records, so a result over selected frames describes the
  selection, not the whole video.
- The drift detectors read each row as independent. Run over every frame of a few videos, they report a shift as more
  certain than the data supports. Compare one row per video instead.
- On a tracking dataset, `Outliers` judges against baselines pooled over the whole dataset. It refuses cluster mode,
  the factor checks and several datasets in one call, with an error that names the alternative.
- Grouped splits count groups, not frames, so a few videos give uneven split fractions.
- Videos from one flight or collection can still depend on each other, and there is no public way yet to declare that
  grouping across evaluators.

v1.2 does not include codec and container metadata, video quality statistics such as compression and noise,
annotation diagnostics for tracks, or splitting that keeps a video's clips together by default. The ROADMAP lists them
under v1.3.

## Experimental in v1.2

These APIs are experimental and may change or be removed without a deprecation cycle. The
[Stability status](../concepts/ValidationAndTrust.md#stability-status) section says how each one warns.

- {class}`.Parity` and {func}`.parity`
- {func}`.uap`
- {class}`.Feasibility`, its evidence and target types, and the functions it builds on
- {class}`.DriftKnownDomains`
- {meth}`.Relabel.fold_scores`
- `VideoStats.PERSISTENCE` and the {class}`.SharpnessMeasure` protocol

## Upgrade from v1.1

v1.2 requires Python 3.11 or later and removes the functionality that v1.1 deprecated. Some calls now raise, and some
results report different numbers. [Migrating from v1.1 to v1.2](../migration/v1.2.md) lists each change and what to do
about it.
