<!-- markdownlint-disable MD051 -->

# Metadata Levels

A dataset's metadata is not one table. Brightness is a property of an image; a
bounding box is a property of one detection inside that image; a frame rate is a
property of the whole video the image came from. Each of those facts is recorded
once, about a different kind of thing, and they are not interchangeable.

Flattening them into a single table forces a choice, and both options are lossy.
Replicate the coarse values down — copy an image's brightness onto each of its
detections — and an image holding twelve detections now contributes twelve
brightness readings while an image holding one contributes a single reading. Any
statistic computed over that column is silently weighted by detection count.
Aggregate the fine values up instead — average the box areas per image — and the
individual detections stop existing as units of analysis at all.

{class}`.Metadata` avoids the choice. It keeps every fact at the granularity it
was actually measured, in one dataframe with rows of several kinds at once, and
records how those kinds relate. Those kinds are called **levels**, and this page
explains what they are, the graph that connects them, and why nearly everything
{class}`.Metadata` does is a consequence of that graph.

## What a level is

A **level** is a granularity at which one row means one entity.

For an {term}`object detection <Object Detection>` dataset there are two: one row
per image, and one row per detection. A `unit`-level row (which represents the
image) holds the facts that are true of the whole image — the weather it was shot
in, its altitude, its capture time. An `instance`-level row holds the facts true
of one detection — its class, its score, its box.

{attr}`.Metadata.dataframe` holds rows at **every** level simultaneously. Each row
carries a `level` column naming which kind it is, and
{meth}`.Metadata.rows_at` filters to one:

```python
md = Metadata(dataset)

md.levels  # ('unit', 'instance')
md.level_counts  # {'unit': 50, 'instance': 93}

md.rows_at("unit").height  # 50  — one row per image
md.rows_at("instance").height  # 93  — one row per detection
```

A level is not a subset of the data, and it is not a grouping applied after the
fact. It is a statement about what a row *is*. Two rows at different levels are
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

- **`sequence`** — an ordered run of units held by a single dataset item. A video
  is a sequence of frames.
- **`unit`** — one element of media. An image, a video frame, an audio clip, a
  text document, a row of a table. This is the level whose name used to be
  `image`; [Units, not images](#units-not-images) explains why it is not.
- **`track`** — one identity a tracker assigns, of which each observation is one
  detection.
- **`instance`** — one labelled thing. A detection for object detection and
  tracking; the image itself for whole-image {term}`classification <Classification>`.
  Every task has this level, so the same object keeps the same level name whichever
  task produced it.

No task uses all four. A task keeps the subset it needs, and the graph collapses
edges through whatever it omits — so an image-based task, which has neither
`sequence` nor `track`, correctly reports `unit` as `instance`'s only parent.

| Task | Levels | Item level | Label level | `unit_type` |
| --- | --- | --- | --- | --- |
| `IC` — image classification | `unit`, `instance` | `unit` | `instance` | `image` |
| `OD` — object detection | `unit`, `instance` | `unit` | `instance` | `image` |
| `MOT` — multi-object tracking | `sequence`, `unit`, `track`, `instance` | `sequence` | `instance` | `frame` |

{attr}`.Metadata.levels` enumerates the names an instance accepts, in this order —
coarsest first. {data}`.FactorLevel` types them, and {class}`.FactorLevelSchema`
is the graph itself.

```{note}
The graph is fixed. It is a closed vocabulary of four names, not an extension
point, and a new kind of dataset is expected to reuse it rather than add to it.
That constraint is what keeps evaluators task-generic: {class}`.Balance` does not
know whether it is looking at images or audio clips, only that it has rows at
some level and factors defined on them.
```

## Propagation: values move down, never up

A factor is stored **once**, at its own level, and read from descendant rows by
propagation. A `unit`-level factor is written on the 50 image rows; asking for it
on the 93 detection rows gathers each detection's image value.

Three rules follow, and together they explain most of what you will observe:

1. **Values propagate downwards.** A factor defined at `unit` is readable at
   `instance`, because every instance has a unit ancestor.
2. **Values never propagate upwards, and are never aggregated.** A factor defined
   at `instance` is null on `unit` rows. DataEval will not silently average the
   box areas in an image to invent a unit-level value — if you want that
   summary, you compute it and add it as a factor in its own right.
3. **A row with no ancestor at a factor's level holds null**, even on a branch
   where propagation would otherwise apply.

Propagation is why the replication problem described at the top of this page does
not disappear — it is *managed*. A `unit`-level factor genuinely is replicated
when read from detection rows, and its marginal distribution there genuinely is
weighted by detections-per-image. What levels give you is the ability to *not*
read it that way: {meth}`.Metadata.at` reads the same factor at its own level,
once per image, and the weighting is gone.

### Propagation is what lets levels be compared

Downward propagation is not only a storage economy — it is what makes a factor at
one level comparable against a factor at another, because both are readable from
the finer level's rows.

Background statistics are the clearest case. `compute_stats(per_background=True)`
measures each image's pixels *outside* its bounding boxes: the scene an image was
captured in. That is a property of the image and of nothing inside it, so it is
stored once per image, at `unit`:

```python
stats = compute_stats(
    dataset,
    stats=ImageStats.VISUAL_BRIGHTNESS,
    per_background=True,
    normalize_pixel_values=False,
)
md.add_factors(stats)
md.factor_names  # ['unit_background_brightness', 'unit_brightness', 'instance_brightness', ...]
```

Read at `instance`, `unit_background_brightness` gathers each detection's own
image's background — so a single frame holds each object's brightness beside the
brightness of the scene behind it, and "are these objects only ever annotated
against dark backgrounds?" becomes a question about two columns of one table.
Read at `unit`, via `md.at("unit")`, it is one value per image, unweighted by how
many detections that image happened to contain.

Note what does *not* happen. There is no `instance_background_brightness`: a
detection has no background of its own, and DataEval will not invent one by
copying its image's value into a new instance-level factor. The value stays at
the level it was measured at, and propagation does the rest.

## The tracking diamond

For image-based tasks the graph is a chain, and the rules above are all there is.
Tracking is different, and the difference is worth understanding because it is the
reason the graph is a DAG rather than a tree.

In a tracking dataset, one dataset item is a video — a `sequence`. Inside it,
`unit` means a frame, not a dataset item. And a detection belongs to two things at
once: it was observed **in a frame**, and it is an observation **of a track**.
`unit` and `track` are therefore siblings under `sequence`, and `instance`
descends from both. The graph is a diamond.

Two consequences follow, and neither arises for classification or detection:

**Siblings do not propagate to each other.** A per-frame factor — say, the frame's
motion blur — has no value on a track row, because a track spans many frames and
no single value is the right one. Equally, a per-track factor — the track's total
lifetime — has no value on a frame row. Such a factor stays in
{attr}`.Metadata.dataframe`, where you can always read it, but it is excluded from
factor analysis at a view that cannot see it, rather than being filled with an
invented value.

**A row may have one parent and not the other.** A detection that no tracker
linked — conventionally `track_id == -1` — sits in a frame but belongs to no
track. Per-track factors are null on it. A column that is partly null cannot be
binned, because discretizing means ordering the values and `None` does not order
against a number, so such a factor is left out of factor analysis at the instance
level. Read at its own level, via `md.at("track")`, it is complete and usable.

```{note}
Both consequences are visible rather than silent. A factor excluded from factor
analysis at some view is absent from {attr}`.Metadata.factor_names` and
{attr}`.Metadata.factor_data` at that view, and DataEval logs the reason at debug
level. It is never quietly replaced by a default.
```

## Three jobs a level can do

Three of {class}`.Metadata`'s accessors name a level, and they answer different
questions. Confusing them is the most common source of surprise.

**{attr}`.Metadata.item_level` — where one row is one dataset item.**
`md.rows_at(md.item_level)` is the task-generic spelling of "one row per thing the
dataset yields". For image tasks that is `unit`; for tracking it is `sequence`,
because one dataset item is a whole video. This is a structural fact about the
task, so it is read-only.

**{attr}`.Metadata.label_level` — where the class labels live.**
The rows {attr}`.Metadata.class_labels`, `score` and `box` describe. It is
`instance` on every task. Note that it is always distinct from the item level,
which is deliberate: an item carrying no label at all — an unlabeled image, or an
image with no detections — still has an item row and keeps every factor on it.
Had labels and items shared a level, unlabeled items would have to be dropped.

**{attr}`.Metadata.view` — which rows the array-shaped accessors project.**
This is the movable one. {attr}`.Metadata.factor_data`,
{attr}`.Metadata.factor_names`, {attr}`.Metadata.is_discrete` and
{attr}`.Metadata.shape` — and `len()`, iteration and indexing — all describe the
rows at the view. The dataframe is unaffected; it always holds every level.

The view defaults to `label_level`, which is what keeps a projection aligned with
{attr}`.Metadata.class_labels` row for row. Moving it is how you change the
question you are asking:

```python
md.factor_data.shape[0]  # 93 — one row per detection
md.at("unit").factor_data.shape[0]  # 50 — one row per image
```

Those two arrays answer genuinely different questions about the same data, and
neither is more correct. The first asks "across all detections, how do these
factors co-vary?"; the second asks "across all images". For a `unit`-level factor,
only the second gives it one vote per image.

```{important}
Prefer {meth}`.Metadata.at` over assigning to {attr}`.Metadata.view`. Evaluators
hold a reference to the metadata they were given, so mutating the view in place
changes what an already-constructed evaluator will read. {meth}`.Metadata.at`
returns an independent copy sharing the same structuring and binning work, which
lets two evaluators read two levels of the same dataset at once.
```

Moving the view **up** has one hard limit: {attr}`.Metadata.class_labels` raises
above the label level rather than inventing an answer. An image has several
detections, or none — there is no single label to return, and silently returning
one would hand an evaluator a label array that does not correspond to its factor
rows.

## Binning happens at a factor's own level

{class}`.Metadata` discretizes continuous factors into bins so that categorical
analyses can consume them. **Each factor is binned at its own level** — over the
rows that hold one value per entity — and the resulting bin assignments then
propagate downwards like any other value.

This is not an implementation detail; it is what makes results comparable across
levels. Consider altitude, recorded once per image, on a detection dataset:

- Binned at `unit`, its cut points are computed over 50 altitude readings — the
  real distribution of altitudes across the images.
- Binned at `instance`, they would be computed over 93 readings, each image's
  altitude repeated once per detection it happens to contain. Crowded images would
  pull the cut points toward their own altitudes.

Because binning happens at the factor's own level, an entity's bin number is the
same wherever you read it from. The altitude bin you see on an image row is the
altitude bin you see on every detection inside it. Changing the view changes which
rows are counted; it never changes what any of them says.

The same reasoning governs the binning controls. A bin count passed through
`continuous_factor_bins` applies to the factor's values at its own level, and
`auto_bin_method` reads that same distribution — which matters most for
`"uniform_count"` and `"clusters"`, whose cut points depend on how the values are
distributed rather than only on their range.

See {ref}`binning-levels` for a worked example with real numbers.

## Units, not images

The media-unit level is called `unit` rather than `image`, and this is the one
piece of the vocabulary chosen for what it enables rather than for what it
describes today.

Every level name in the graph describes a *structural role*: `sequence` is "an
ordered run", `instance` is "one labelled thing", `track` is "one identity across
time". `image` was the exception — it named a medium. That was harmless while
every dataset was a computer-vision dataset, and became a problem the moment it
was not. A tabular dataset has exactly the same structure as an image
classification dataset: one row per record, one label per record. It should reuse
the graph unchanged. Being told its records live at the `image` level is, at best,
a lie you learn to ignore.

So the level is `unit`, and the medium's own word for it is carried separately, as
data, by {attr}`.Metadata.unit_type`:

```python
Metadata(image_dataset).unit_type  # 'image'
Metadata(video_dataset).unit_type  # 'frame'
Metadata.from_factors(factors).unit_type  # 'item'
```

{attr}`.Metadata.unit_type` is descriptive only. Nothing in structuring, binning
or projection consults it; it exists so that error messages, reports and your own
code can speak the caller's language. It is deliberately a plain string rather
than part of {data}`.FactorLevel` — a new modality supplies a new value and edits
no type, while the level vocabulary stays closed and task-independent.

The payoff is that generic code stays generic. Every {class}`.Metadata` writes
`"unit"` into its `level` column regardless of modality, so a pipeline that
concatenates metadata from an image dataset and an audio dataset can filter both
the same way, and {class}`.Balance`, {class}`.Diversity` and {class}`.Parity`
need no modality awareness at all.

```{note}
`"image"` is still accepted wherever a level name is taken — `rows_at("image")`,
`at("image")`, `view="image"` — and resolves to `"unit"` with a
`DeprecationWarning`. It is removed in v1.2.0. Note that level names always read
back as `"unit"`, so a comparison like `md.item_level == "image"` is `False`; use
`md.unit_type == "image"` to ask about the medium.
```

## Levels and ontologies are different graphs

DataEval describes your data with two DAGs, and it is worth being explicit about
how they differ, because [Ontology](Ontology.md) is also a page about a directed
acyclic graph.

An {class}`.Ontology` is a graph of **concepts**, related by *is-a*. It says a
sedan is a land vehicle, and that a land vehicle is a vehicle. It describes what
your labels **mean**.

The level graph is a graph of **granularities**, related by *containment*. It says
a detection sits in a frame, and a frame sits in a video. It describes where each
fact **lives**.

They are orthogonal, and the same detection has a position in both: it is an
`instance`-level row (level graph) carrying a class label that resolves to some
concept (ontology). Neither constrains the other. A question like "is this label
valid?" is an ontology question; "at what granularity was this measured?" is a
level question.

## Related concept pages

- [Dataset Bias and Coverage](DatasetBias.md) — the evaluators that consume
  {class}`.Metadata`, and where the choice of view changes what they measure.
- [Ontology](Ontology.md) — the other graph, over label concepts rather than
  granularities.
- [Validation and Trust](ValidationAndTrust.md) — how the level a factor is judged
  at affects the confidence you can place in a result.
- [Acting on Results](ActingOnResults.md) — mapping findings back onto the rows
  they came from.

## See this in practice

### How-to guides

- [How to bin factors by level](../notebooks/h2_bin_factors_by_level.py) — a worked
  example of binning at a factor's own level, with the numbers that show why the
  alternative distorts.
