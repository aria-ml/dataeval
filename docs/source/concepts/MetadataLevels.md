<!-- markdownlint-disable MD051 -->

# Metadata levels

A dataset's metadata does not fit cleanly in one table. Brightness is a property of an image, a bounding box is a
property of one detection inside that image, and a frame rate is a property of the whole video. Each fact is recorded
once, about a different kind of thing.

Flattening them into a single table forces a choice between two imperfect options. Replicating coarse values down
(copying an image's brightness onto each of its detections) means an image with twelve detections contributes twelve
readings and an image with one contributes a single reading, so any statistic computed over that column is silently
weighted by detection count. Aggregating fine values up (averaging the box areas per image) removes the individual
detections as units of analysis.

{class}`.Metadata` avoids the choice. It keeps every fact at the granularity it was measured, in one data frame with
rows of several kinds, and records how those kinds relate. Those kinds are called **levels**, and this page explains
what they are, the graph that connects them, and why much of what {class}`.Metadata` does follows from that graph. The
behavior of each method is documented on the method; this page covers the ideas behind them.

## What a level is

A **level** is a granularity at which one row means one entity.

For an {term}`object detection <Object Detection>` dataset there are two: one
row per image, and one row per detection. A `unit`-level row (which represents
the image) holds the facts that are true of the whole image: the weather it was
shot in, its altitude, its capture time. An `instance`-level row holds the facts
true of one detection: its class, its score, its box.

{attr}`.Metadata.dataframe` holds rows at every level simultaneously. Each row
carries a `level` column naming which kind it is, and {meth}`.Metadata.rows_at`
filters to one:

```python
md = Metadata(dataset)

md.levels  # ('unit', 'instance')
md.level_counts  # {'unit': 50, 'instance': 93}

md.rows_at("unit").height  # 50  — one row per image
md.rows_at("instance").height  # 93  — one row per detection
```

A level is not a subset of the data, and it is not a grouping applied after the
fact. It is a statement about what a row is. Two rows at different levels are
not comparable records with different filters applied; they describe different
things.

## The level graph

Levels are related by containment: a detection sits inside a frame, a frame sits
inside a video. DataEval declares those relationships once, as a **directed
acyclic graph** over four level names.

```{graphviz}

   digraph levels {
      rankdir=TB
      node [shape=box, style="rounded,filled", width=1.5, height=0.5, fontsize=11]
      edge [arrowsize=0.7]

      sequence [label="sequence"]
      unit     [label="unit"]
      track    [label="track"]
      instance [label="instance"]

      sequence -> unit
      sequence -> track
      unit     -> instance
      track    -> instance
   }
```

Read an arrow as "contains", and therefore as the direction values travel. Each
name has one job:

- `sequence`: an ordered run of units held by a single dataset item. A video is
  a sequence of frames.
- `unit`: one element of media. An image, a video frame, an audio clip, a text
  document, a row of a table. This is the level whose name used to be `image`;
  [Level names](#level-names) explains why it is not.
- `track`: one identity a tracker assigns, of which each observation is one
  detection.
- `instance`: one labeled thing. A detection for object detection and tracking;
  the image itself for whole-image {term}`classification <Classification>`.
  Every supervised task has this level.

Each task keeps the subset of the four levels that it needs, and the graph
collapses edges through whatever it omits, so an image-based task, which has
neither `sequence` nor `track`, reports `unit` as `instance`'s only
parent.

| Task                                       | Levels                                  | Item level   | Label level        | `unit_type` |
| ------------------------------------------ | --------------------------------------- | ------------ | ------------------ | ----------- |
| `IC` (image classification)                | `unit`, `instance`                      | `unit`       | `instance`         | `image`     |
| `OD` (object detection)                    | `unit`, `instance`                      | `unit`       | `instance`         | `image`     |
| `MOT` (multi-object tracking)              | `sequence`, `unit`, `track`, `instance` | `sequence`   | `instance`         | `frame`     |
| `factors` ({meth}`.Metadata.from_factors`) | as requested, `unit` by default         | as requested | same as item level | `item`      |

The `factors` row describes the bare-array form. A `from_factors` call given a
`source_index` that names both items and labels takes object detection's shape:
`unit`, `instance`, labels at `instance`.
That is what lets {func}`.image_stats` output be reimported without its dataset.

{attr}`.Metadata.levels` enumerates the names an instance accepts, in this
order, coarsest first. {data}`.FactorLevel` types them, and
{class}`.FactorLevelSchema` is the graph itself.

```{note}
The graph is fixed. It is a closed vocabulary of four names, not an extension
point, and a new kind of dataset is expected to reuse it rather than add to it.
That constraint is what keeps evaluators task-generic: {class}`.Balance` does not
know whether it is looking at images or audio clips, only that it has rows at
some level and factors defined on them.
```

## Propagation: values move down, never up

A factor is stored once, at its own level, and read from descendant rows by
propagation. A `unit`-level factor is written on the 50 image rows; asking for
it on the 93 detection rows gathers each detection's image value.

Three rules follow, and together they explain most of the observed behavior:

1. Values propagate downwards. A factor defined at `unit` is readable at
   `instance`, because every instance has a unit ancestor.
2. Values never propagate upwards, and are never aggregated. A factor defined at
   `instance` is null on `unit` rows. DataEval does not average the box
   areas in an image to produce a unit-level value. To obtain that summary,
   compute it and add it as a factor in its own right.
3. A row with no ancestor at a factor's level holds null, even on a branch where
   propagation would otherwise apply.

Propagation manages the replication problem described at the top of this page and does not remove it. A `unit`-level
factor is replicated when read from detection rows, and its marginal distribution there is weighted by
detections-per-image. Levels make it possible to read the factor at its own level instead: {meth}`.Metadata.at` reads
the same factor once per image, and the weighting is gone.

Reading a factor at its own level does not make the rows at a finer level independent observations. Frames that carry
the value of one video-level source share it, so a significance test or a count over frames treats correlated rows as
independent. Choose the sampling unit of the analysis and group the rows that belong to one clip, sequence, sensor or
platform. [Parity](Parity.md#effective-groups) does this with its `groups` argument.

### Propagation is what lets levels be compared

Downward propagation also makes a factor at one level comparable against a factor at another, because both are
readable from the finer level's rows.

Background statistics are the clearest case. `image_stats(per_background=True)` measures each image's pixels outside its
bounding boxes: the scene an image was captured in. That is a property of the image and of nothing inside it, so it is
stored once per image, at `unit`, beside the per-detection values:

```python
md.factor_names
# ['unit_background_brightness', 'unit_brightness', 'instance_brightness', ...]
```

Read at `instance`, `unit_background_brightness` gathers each detection's own image's background, so a single frame
holds each object's brightness beside the brightness of the scene behind it. Read at `unit`, via `md.at("unit")`, it is
one value per image, unweighted by detection count. No `instance_background_brightness` factor exists: a detection has
no background of its own, and DataEval does not copy its image's value into a new instance-level factor.

{func}`.image_stats` returns one row for each image and one for each detection. A statistic that applies to only one
kind arrives with NaN on the other kind's rows: a background statistic is NaN on every detection row, and a statistic
of a detection alone is NaN on every full-image row. When
{meth}`.Metadata.add_factors` splits the result into `unit_<name>` and `instance_<name>`, it drops the half that holds
no value at all, and it records the dropped name in {attr}`.Metadata.dropped_factors` with the reason
`no_values_at_level`. An instance-only statistic therefore appears as `instance_<name>` alone, and a background
statistic as `unit_<name>` alone.

## Levels in tracking datasets

For image-based tasks the graph is a chain, and the rules above are all there is. Tracking makes it a DAG rather than a
tree.

In a tracking dataset, one dataset item is a video, a `sequence`. Inside it, `unit` means a frame. A detection belongs
to two things at once: it was observed in a frame, and it is an observation of a track. `unit` and `track` are
therefore siblings under `sequence`, and `instance` descends from both. The graph is a diamond. Two consequences
follow, and neither arises for classification or detection.

Siblings do not propagate to each other. A per-frame factor, such as the frame's motion blur, has no value on a track
row, because a track spans many frames and no single value is the right one. Likewise a per-track factor, such as the
track's total lifetime, has no value on a frame row. Such a factor stays in {attr}`.Metadata.dataframe`, but it is
excluded from factor analysis at a view that cannot see it and is not filled with a default value.

A row may have one parent and not the other. A detection that no tracker linked, conventionally `track_id == -1`, sits
in a frame but belongs to no track. Per-track factors are null on it, and that null states that the row has no such
ancestor and does not record a missing measurement. A factor that cannot describe every row of a view is left out of
factor analysis at that view. Read at its own level, via `md.at("track")`, it is complete, and it can be rolled up into
`sequence` (below) whatever the current view is, because that is a question about the data and not about the view.
Both consequences are visible: an excluded factor is absent from {attr}`.Metadata.factor_names` at that view, and the
reason is logged at debug level. It is never replaced by a default.

## Item level, label level and view

Three of {class}`.Metadata`'s accessors name a level, and they answer different questions.

- {attr}`.Metadata.item_level` is where one row is one dataset item: `unit` (an image) for image tasks and `sequence` (a
  video) for tracking. It is a structural fact about the task and is read-only.
- {attr}`.Metadata.label_level` is where the class labels live, which is `instance` on every dataset task. It is always
  distinct from the item level, so an item with no label, such as an image with no detections, still has an item row
  but no instance rows. {meth}`.Metadata.from_factors` is the exception: with no dataset to impose a shape, it places
  both levels at the level the caller asked for, `unit` by default.
- {attr}`.Metadata.view` determines which rows the array-shaped accessors, such as `factor_data` and `factor_names`,
  describe. It defaults to `label_level`, which keeps the projection aligned with `class_labels` row for row. The
  dataframe always holds every level.

Changing the view changes the question a tool answers. The first view below directs analysis toward variation across
detections, for example "across all detections, how do these factors co-vary?". The second directs it "across all
images" instead.

```python
md.factor_data.shape[0]  # 93 — one row per detection
md.at("unit").factor_data.shape[0]  # 50 — one row per image
```

```{important}
Assigning to {attr}`.Metadata.view` mutates the instance, and evaluators hold a reference to the metadata they were
given and not a snapshot, so a view assigned after construction changes what an already-built evaluator reads.
{meth}`.Metadata.at` returns an independent copy, which lets two evaluators read two levels of one dataset at once.
```

### Choosing which factors to analyze: `inherited`

The view chooses which rows are projected, and `inherited` chooses which factors are analyzed on them. By default a
view analyzes every factor it can read: its own plus everything propagated down from coarser levels. With
`inherited=False` it analyzes only the factors defined at the view itself.

On an object detection dataset with image metadata plus per-image and per-detection brightness from
{func}`.image_stats`:

| view                       | `inherited=True` (default)                                       | `inherited=False`     |
| -------------------------- | ---------------------------------------------------------------- | --------------------- |
| `at("unit")` (50 rows)     | `angle`, `location`, `time_of_day`, `unit_brightness`, `weather` | the same five         |
| `at("instance")` (93 rows) | all five above, plus `instance_brightness`                       | `instance_brightness` |

The row count is set by the view alone, and `inherited` changes only how many columns describe the rows. At the
coarsest level the two columns agree, because nothing sits above `unit` to inherit. `inherited=False` asks a question
about one level's own measurements without replicated context: "do the detections in this dataset vary among
themselves?" and not "do detections vary, including by which image they came from?".

## Aggregation: moving values up

Rule 2 above says values never propagate upwards and are never aggregated implicitly. {meth}`.Metadata.agg` aggregates
explicitly: the call states which rows to roll up, into what, and by what summary, and the result becomes an ordinary
factor at the coarser level:

```python
rolled = md.agg(
    "instance",
    "unit",
    pl.len().alias("n_detections"),
    pl.col("instance_brightness").mean().alias("mean_bright"),
)
```

`n_detections` is now a per-image fact stored once per image, and every level rule applies to it unchanged: it bins at
`unit`, propagates down to detections, and is analyzed at `unit` with one vote per image.

Two ideas govern what a roll-up accepts. First, a row with no ancestor at the target level takes no part, so an
untracked detection is neither counted by a track nor averaged into one, and a target row with nothing beneath it
answers null and not zero, because nothing was measured there. Second, rolling up a factor defined above the source
level repeats the replication problem from the top of this page in the opposite direction: averaging a per-image value
over the detections in a track weights each image by how many detections it contributed, which is usually not the
intended question. Such a roll-up is refused unless it states what to count once (`unique_by`), for example one reading
per frame. Counting rows never trips this, because a count reads no columns.

### Named reductions

{meth}`.Metadata.agg` takes an expression and leaves its meaning to the caller. {meth}`.Metadata.aggregate` takes the
name of a reduction, and a name carries what an expression cannot: which value types the reduction applies to, and what
an empty group answers. `mean` of a class label is refused before it is evaluated, and `count` of an empty frame is
zero where `mean` of one is null. The source level is inferred from where each factor is defined, so the call states
what to roll up and where to, not where each number lives.

```python
rolled = md.aggregate("instance_brightness", level="sequence", how="mean")
# adds `instance_brightness_mean`, one value per video
```

Some reductions are **temporal**: they are functions of the ordered series and not of the bag of values, so they need an
ordering. `variability` is the mean absolute change per unit of the ordering key, and `trend` is the least-squares slope
against it. Neither is a variance: a slow illumination drift and a strobing feed can have identical variance and very
different variability. The ordering comes from the level itself, and a level with no ordering is refused rather than
run against row order, which is an artifact of the walk and not time. A roll-up can also be declared as a value, an
{class}`.Aggregator`, that carries its route, its uniqueness constraint and its coverage requirement and can be
validated against the level graph without a dataset.

### Choosing a route in a branched level graph

In a chain there is one way up. In the tracking level graph a detection reaches its sequence two ways, through its frame
and through its track, and the two disagree about untracked detections because one branch has no answer for them. By
default a roll-up takes every route, so it reaches a row wherever any branch does, and it can be restricted to one
branch (`via="track"`).

These are different questions. The default asks "how much of this, per video", and the `via="track"` form asks "how much
of this per video, among things a tracker committed to". They differ by the untracked detections and nothing else, and
roll-up is not associative across a branch that stops short: going up in two hops, `instance` to `track` and then
`track` to `sequence`, gives the second answer. The route appears in the output name whenever it is not the default.
Where both branches have an answer they must agree, and a detection whose track sits in one video while its frame sits
in another is a contradiction that raises an error.

## Narrowing the population: `where` and `having`

{meth}`.Metadata.where` and {meth}`.Metadata.having` both take a predicate at some level and return a new metadata over
fewer rows. They differ in which direction the cut travels, which is easiest to see with the same predicate:

```python
md.level_counts  # {'unit': 50, 'instance': 93}
md.where(pl.col("class_label") == 0, "instance").level_counts  # {'unit': 50, 'instance': 20}
md.having(pl.col("class_label") == 0, "instance").level_counts  # {'unit': 17, 'instance': 38}
```

`where` keeps the matching rows: the 20 detections of class 0, and every image, because `where` never filters upwards.
`having` keeps the entities that have a match: the 17 images holding at least one class-0 detection, and then all 38
detections in those images, not just the class-0 ones. It is the "images containing a person" filter, and it travels up
first and then back down.

One rule decides both cases:

```{important}
A row survives only if every ancestor it actually has survives.

Applied downwards from wherever the filter seeds it. A row with no ancestor at some level is not failing this test: it
has nothing there to lose. That is why an untracked detection is not swept away by a filter over tracks.
```

A filter can therefore only add factors to the analysis, never remove one: dropping the rows that lacked an ancestor can
turn a partly-null column into a total one, and a total column can be binned. And `where` at one level leaves that
level's siblings untouched, because filtering frames does not remove tracks: a track is not beneath a frame.

### A filtered metadata still holds its whole dataset

Filtering removes rows. It does not remove items from the dataset the metadata is bound to, so anything computed from
that dataset, embeddings above all, still describes the original population, and pairing the two row-for-row is a
misalignment that raises nothing. {attr}`.Metadata.is_filtered` records that this happened, evaluators that take both a
metadata and something dataset-shaped refuse a filtered one, and {meth}`.Metadata.selected_items` returns the surviving
item indices when the filter cut along item boundaries. A filter that kept some of an image's detections has no
item-level answer, because a dataset can hand back an image and not three of its detections.

## Factors only some rows declare

A dataset does not always record everything for everything: one image's metadata omits a key the others carry, or one
frame of a video declares no timestamp. By default DataEval is **all-or-nothing**: such a factor is dropped for every
row, and {attr}`.Metadata.dropped_factors` says so. That is the conservative reading. A balance result over "the rows
that happened to have a value" describes a population nobody chose. It also discards the part that was recorded, which
for a large dataset with one incomplete entry is the whole factor.

`partial_factors=True` asks for the other reading. The rows that recorded nothing get a **missing value**, which
binning places in a bin of its own, so "not recorded" never becomes one of the factor's values and cannot be mistaken
for a category the data contains. The policy applies wherever structuring meets an incompletely declared value: a
metadata key some items omit, and a per-frame timing some frames omit. A factor that no row declares is dropped either
way, because it is a factor the dataset does not carry and an all-null column says nothing that its absence does not.

The difference matters for aggregation. A sequence-level mean over the 999 frames that did declare a timestamp is a
valid number, and under the default the factor was gone before anything could ask for it.

## Binning happens at a factor's own level

{class}`.Metadata` discretizes continuous factors into bins so that categorical analyses can consume them (see
[Binning](Binning.md)). Each factor is binned at its own level, over the rows that hold one value per entity, and the
resulting bin assignments propagate downwards like any other value.

Consider altitude, recorded once per image, on a detection dataset. Binned at `unit`, its cut points are computed over
50 altitude readings, the distribution of altitudes across the images. Binned at `instance`, they would be computed over
93 readings, each image's altitude repeated once per detection it contains, and crowded images would pull the cut
points toward their own altitudes.

Because binning happens at the factor's own level, an entity's bin number is the same wherever it is read: the altitude
bin on an image row is the altitude bin on every detection inside it. Changing the view changes which rows are counted
and never what any of them says. See {ref}`binning-levels` for a worked example with real numbers.

## Choosing a level for a video measurement

Video datasets have four structural levels: a sequence, its frames, its tracks, and its detections. The core rule is to
store a value at the level where it was measured. Store a per-frame metric at `unit` even if a sequence-level summary is
wanted, and compute the summary with {meth}`.Metadata.agg`, which keeps both the raw measurements and the summaries.

| Measured entity | Level | Which column names a row |
| --- | --- | --- |
| Property of the whole video (e.g., codec, resolution, camera motion) | `sequence` | None (the item names it) |
| Property of one frame (e.g., brightness, blur, scene cuts) | `unit` | `unit_index` |
| Property of one track (e.g., mean speed, total displacement) | `track` | `track_id` |
| Property of one detection (e.g., box area, confidence, IoU) | `instance` | `target_index` |

Each key names a row within its item, because `unit_index`, `track_id` and `target_index` all restart in every sequence,
so DataEval matches on `(item_index, key)`. A sequence needs no key, because there is one per item. Detections are
keyed by `target_index` and not `instance_index`, which numbers detections within a frame and so repeats across the
frames of a sequence.

These `*_index` columns are DataEval's own positional identity. They are distinct from the dataset's own identifiers,
which are **reserved** and never treated as factors: `item_id` carries the datum's own `id` on every row so that any row
can be traced to its source item, and `track_id` does the same for a tracking observation. An identifier names a row and
does not measure the data, so it is never binned, correlated, or reported on.

### Naming a row: keys or addresses

A row is named by an item index plus a key within the item, and a {class}`.SourceIndex` is that address: `(item, key,
level)`. The bulk form of {meth}`.Metadata.add_factors` passes the key column alongside the values, and the scalar form
passes addresses, which is how an evaluator's findings come back. An address that states no level is the
**task-generic** reading, the item level with no key and the label level under one, so `SourceIndex(3)` names image 3 on
an image dataset and video 3 on a tracking one. That is why {func}`.image_stats` output places correctly without knowing
which kind of dataset it measured. An address names a row without saying what it sits inside, so addresses can place
values into rows that exist but cannot build rows: construct from the dataset, then place.

### The impact of misaligned storage levels

Storing a measurement away from its native level changes how the data behaves in analysis.

- Coarser than measured (aggregation): frame-level brightness stored at `sequence` loses the individual frame
  variation. Store readings at their native level and roll them up, which preserves both.
- Finer than measured (replication): do not replicate a coarser value, such as a `sequence` property, across `unit` or
  `instance` rows. DataEval propagates values down the hierarchy, and manual replication adds redundant rows that
  over-represent the replicated values in analysis and binning.
- Sibling branches: `unit` and `track` are siblings, so a per-frame value cannot be stored at `track`. Roll it up from
  `instance` to `track` with `unique_by="unit"`, so that frames with several detections do not over-weight the track's
  average.

### Handling empty frames and untracked detections

Video datasets contain structures that standard image datasets do not:

- Empty frames: A frame with no detections still has a `unit` row, but has no
  `instance` rows. Storing frame-level measurements at the `instance` level
  silently discards data for all empty frames. Storing them at `unit` keeps the
  data complete.
- Untracked detections: A detection with no active track (e.g.,
  `track_id == -1`) belongs to a frame (`unit`) but not to any `track`.
  Track-level factors will be null on it. If you need to analyze untracked
  detections, measure and store those factors at the `instance` level.

### Camera motion as a per-frame factor

How the camera moved is a property of each frame, so {func}`.ego_stats` measures it there and its rows attach at the
`unit` level. Camera motion is a shortcut hazard: a detector can learn "the camera was panning" in place of the object
whenever the two correlate, and storing these measurements as factors lets a bias or balance analysis condition on the
camera. The factors sit on three axes, kept apart so that each can be conditioned on independently: the camera's own
motion, the independent motion of other objects in frame (contamination), and whether the frame's camera estimate can
be relied on (trust). `ego_stats` also declares how each per-frame factor rolls up to a sequence summary, a rate by
median and a label by mode, and the roll-ups land at `sequence`, so they propagate to every frame, track and detection
beneath it.

```{warning}
Read `ego_trusted` first. Where it is `False`, that frame's camera fields are NaN, and a sequence-level roll-up answers
only when enough of the sequence's frames were trusted. See {func}`.ego_stats` for the factors and their roll-ups.
```

### Volatility and drift of a per-frame statistic

Brightness, blur and every other {func}`.image_stats` value are measured on one frame, so they belong at `unit`. Two
questions about a video ask for more than the mean of that column: how much the value jumps from frame to frame, and
whether it moves steadily in one direction. Both depend on frame order, so they are the temporal reductions
`variability` and `trend` (see [Named reductions](#named-reductions)). A flickering or strobing feed has
high variability whatever its average brightness, and a sunset or an auto-exposure ramp has a trend far from zero.

{class}`.SequenceFrames` presents a video as a dataset of frames, so {func}`.image_stats` can measure each frame, and
{meth}`.SequenceFrames.to_source` translates its addresses into the video's own. Each per-frame value is then rolled up
to its video:

```python
frames = SequenceFrames(dataset)
stats = image_stats(frames, stats=ImageStats.VISUAL_BRIGHTNESS, per_target=False)
md.add_factors(stats, source_index=frames.to_source(stats["source_index"]))
rolled = md.aggregate("brightness", level="sequence", how="variability")  # adds `brightness_variability`
```

{func}`.video_stats` answers a neighboring question with one row per video, and the two split the work: a per-video
summary screens the dataset, and the per-frame values localize the problem inside one flagged video.

## Saving and reading Metadata

Building metadata reads every item of the dataset, and everything after that is arithmetic over rows that are small
next to it. {meth}`.Metadata.save` and {meth}`.Metadata.load` cache those rows so the walk runs once. The file is a
cache and not an interchange format: it holds DataEval's internal per-level layout, which may change in any release,
and a stale file is refused with a {class}`.MetadataFormatError`. To keep metadata as a record or for another tool,
write {attr}`.Metadata.dataframe` to a parquet file.

## Level names

The media-unit level is called `unit` and not `image`, which is the one piece of the vocabulary chosen for what it
enables and not for what it describes today. Every level name describes a structural role: `sequence` is "an ordered
run" of `units`, `instance` is "one labeled thing", and `track` is "one identity across time", an ordered run of
`instances`. The familiar term for the thing at each level, which differs by medium, is carried separately in
{attr}`.Metadata.unit_type` (`'image'`, `'frame'` or `'item'`). It is descriptive only: nothing in structuring, binning
or projection consults it, and it is a plain string and not part of {data}`.FactorLevel`, so a new modality supplies a
new value and edits no type, while the level vocabulary stays closed and task-independent.

As a result, every {class}`.Metadata` writes `"unit"` into its `level` column regardless of modality. A pipeline that
concatenates metadata from an image dataset and an audio dataset can filter both the same way, and {class}`.Balance`,
{class}`.Diversity` and {class}`.Parity` need no modality awareness. The earlier names `image` and `target` no longer
resolve; see the [migration notes](../migration/v1.2.md).

## Levels and ontologies are different graphs

DataEval describes data with two directed acyclic graphs (DAGs), and they differ in what they relate. An
{class}`.Ontology` ([Ontology](Ontology.md)) is a graph of concepts related by "is-a": a sedan is a land vehicle, and a
land vehicle is a vehicle. It describes what labels mean. The level graph is a graph of granularities related by
containment: a detection sits in a frame, and a frame sits in a video. It describes where each fact lives. They are
orthogonal, and the same detection has a position in both: it is an `instance`-level row (level graph) carrying a class
label that resolves to some concept (ontology). A question like "is this label valid?" is an ontology question, and "at
what granularity was this measured?" is a level question.

## Related concept pages

- [Binning](Binning.md): what the discretization on this page's factors
  does to them, and why the cut points are a choice rather than a detail.
- [Image statistics](ImageStatistics.md): more factors that can be calculated
  from raw data and added to Metadata on demand.
- [Dataset bias](DatasetBias.md): the evaluators that consume
  {class}`.Metadata`, and where the choice of view changes what they measure.
- [Ontology](Ontology.md): the other graph, over label concepts rather than
  granularities.
- [Validation and trust](ValidationAndTrust.md): how the level a factor is judged
  at affects the confidence you can place in a result.
- [Acting on results](ActingOnResults.md): mapping findings back onto the rows
  they came from.

## See this in practice

### Tutorials

- [Analyze a dataset across its levels](../notebooks/tt_analyze_across_levels.py):
  everything on this page worked end to end on an aerial detection dataset: the same
  factor reporting two distributions, `at` and `inherited`, `agg`, `where` versus
  `having`, and the bias question that can only be asked once a level is chosen.
- [Compare objects to their surroundings](../notebooks/tt_compare_objects_to_surroundings.py): the background is
  stored at the unit level and reaches the instance rows beneath it.

### How-to guides

- [Filter a dataset by metadata factors](../notebooks/h2_filter_by_metadata.py): keeping the images and detections
  that match a predicate over factors.
- [Keep metadata in sync with its dataset and cache it](../notebooks/h2_sync_and_cache_metadata.py):
  realigning filtered metadata with its dataset, and saving it so it is built
  once.
- [Trace a finding back to its source](../notebooks/h2_trace_findings_to_source.py): retrieving the image,
  frame, track or detection that a finding's `SourceIndex` names.
