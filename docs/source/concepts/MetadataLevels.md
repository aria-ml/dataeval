<!-- markdownlint-disable MD051 -->

# Metadata Levels

A dataset's metadata does not fit cleanly in one table. Brightness is a property of an image; a
bounding box is a property of one detection inside that image; a frame rate is a
property of the whole video the image came from. Each of those facts is recorded
once, about a different kind of thing.

Flattening them into a single table forces a choice, and neither option is perfect.
Replicate the coarse values down — copy an image's brightness onto each of its
detections — and an image holding twelve detections now contributes twelve
brightness readings while an image holding one contributes a single reading. Any
statistic computed over that column is silently weighted by detection count.
Aggregate the fine values up instead — average the box areas per image — and the
individual detections stop existing as units of analysis at all, reducing 'resolution'.

{class}`.Metadata` avoids the choice. It keeps every fact at the granularity it
was actually measured, in one dataframe with rows of several kinds, and
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
  `image`; [Level names](#level-names) explains why it is not.
- **`track`** — one identity a tracker assigns, of which each observation is one
  detection.
- **`instance`** — one labelled thing. A detection for object detection and
  tracking; the image itself for whole-image {term}`classification <Classification>`.
  Every supervised task has this level.

Each task keeps the subset of the four levels that it needs, and the graph collapses
edges through whatever it omits — so an image-based task, which has neither
`sequence` nor `track`, correctly reports `unit` as `instance`'s only parent.

| Task                                       | Levels                                  | Item level   | Label level          | `unit_type` |
| ------------------------------------------ | --------------------------------------- | ------------ | -------------------- | ----------- |
| `IC` — image classification                | `unit`, `instance`                      | `unit`       | `instance`           | `image`     |
| `OD` — object detection                    | `unit`, `instance`                      | `unit`       | `instance`           | `image`     |
| `MOT` — multi-object tracking              | `sequence`, `unit`, `track`, `instance` | `sequence`   | `instance`           | `frame`     |
| `factors` — {meth}`.Metadata.from_factors` | as requested, `unit` by default         | as requested | *same as item level* | `item`      |

The `factors` row describes the bare-array form. A `from_factors` call given a
`source_index` that names both items and labels has a shape to copy after all, and
takes object detection's exactly — `unit`, `instance`, labels at `instance` — which is
what lets {func}`.compute_stats` output be reimported without its dataset.

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
captured in. That is a property of the image and of nothing inside it, so
{meth}`.Metadata.add_factors` reads the `source_index` those values arrive with
and stores them once per image, at `unit`, beside the per-detection ones:

```python
md.factor_names
# ['unit_background_brightness', 'unit_brightness', 'instance_brightness', ...]
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
track. Per-track factors are null on it, and that null is not a missing
measurement: it is the statement that this row has no such ancestor. A factor
that cannot describe every row of a view is left out of factor analysis at that
view rather than analysed against rows it says nothing about. Read at its own
level, via `md.at("track")`, it is complete and usable — and it can be rolled up
into `sequence` (below) whatever the current view happens to be, because that is
a question about the data rather than about the view.

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
dataset yields". For image tasks that is `unit` (image); for tracking it is `sequence` (video).
This is a structural fact about the task, so it is read-only.

**{attr}`.Metadata.label_level` — where the class labels live.**

The rows {attr}`.Metadata.class_labels`, `score`, and `box` describe. It is
`instance` on every *dataset* task, and on those it is always distinct from the
item level. That separation is deliberate: an item carrying no label at all — an
unlabeled image, or an image with no detections — still has an item row, but no instance rows.

{meth}`.Metadata.from_factors` is the exception, because there is no dataset to
impose a shape: it places both the item level and the label level at whichever
level the caller asked for, which is `unit` by default. So a single-level
`from_factors` metadata has `item_level == label_level == "unit"`.

Like {attr}`.Metadata.item_level`, {attr}`.Metadata.label_level` is read-only.

**{attr}`.Metadata.view` — which rows the array-shaped accessors project.**

This changes based on the level of data currently being accessed. The view defaults to
`label_level`, which keeps the projection aligned with {attr}`.Metadata.class_labels`
row for row. {attr}`.Metadata.factor_data`, {attr}`.Metadata.factor_names`,
{attr}`.Metadata.is_binned`, {attr}`.Metadata.is_discrete`, and {attr}`.Metadata.shape`
— and `len()`, iteration, and indexing — all describe the rows at the view. The dataframe
is unaffected; it always holds every level.

Changing the view of the data allows DataEval's tools to answer different questions
about the dataset. The first view below directs inquiries towards variations across detections (e.g.
"across all detections, how do these factors co-vary?"). The second invites analysis
"across all images" instead.

```python
md.factor_data.shape[0]  # 93 — one row per detection
md.at("unit").factor_data.shape[0]  # 50 — one row per image
```

```{important}
The two spellings of a view change differ in mutability. Assigning to
{attr}`.Metadata.view` mutates the instance, and evaluators reading it hold a reference to
the metadata they were given rather than a snapshot of it — so a view assigned
after construction changes what an already-built evaluator reads.
{meth}`.Metadata.at` returns an independent copy that shares the structuring and
binning work already done, which is what lets two evaluators read two levels of
one dataset at once.
```

### Choosing which factors to analyze: `inherited`

The view chooses *which rows* are projected. `inherited` chooses *which factors* are analysed on them.

With `inherited=True`, the default, a view analyzes every factor it can read —
its own, plus everything propagated down from more granular levels. With `inherited=False` it
analyzes only the factors defined **at** the view itself.

On an object detection dataset with image metadata plus per-image and
per-detection brightness from {func}`.compute_stats`:

| view                       | `inherited=True` (default)                                       | `inherited=False`     |
| -------------------------- | ---------------------------------------------------------------- | --------------------- |
| `at("unit")` — 50 rows     | `angle`, `location`, `time_of_day`, `unit_brightness`, `weather` | *the same five*       |
| `at("instance")` — 93 rows | all five above, plus `instance_brightness`                       | `instance_brightness` |

Two observations about this example: The row count is set by the view alone — `inherited`
never changes how many rows there are, only how many columns describe them. And
at the coarsest level the two columns agree, because nothing sits above `unit`
for it to inherit.

`inherited=False` is how you ask a question about one level's own measurements,
uncontaminated by replicated context: "do the detections in this dataset vary
among themselves?" rather than "do detections vary, including by which image
they came from?".

## Aggregation: moving values up

Rule 2 above says values never propagate upwards and are never aggregated
silently. {meth}`.Metadata.agg` is the way to do it **loudly** — you say which
rows to roll up, into what, and by what summary, and the result becomes an
ordinary factor at the coarser level:

```python
rolled = md.agg(
    "instance",
    "unit",
    pl.len().alias("n_detections"),
    pl.col("instance_brightness").mean().alias("mean_bright"),
)

rolled.at("unit").factor_names
# ['angle', 'location', 'mean_bright', 'n_detections',
#  'time_of_day', 'unit_brightness', 'weather']
```

`n_detections` is now a per-image fact stored once per image, and every level
rule applies to it unchanged — it bins at `unit`, propagates down to detections,
and is analysed at `unit` with one vote per image.

Two rules govern what `agg` will accept.

**A row with no ancestor at the target level takes no part.** An untracked
detection belongs to no track, so it is neither counted by one nor averaged into
one. Conversely a target row with nothing beneath it answers **null** by default —
nothing was measured there, which is a different statement from measuring zero.
An expression carries no identity element to fall back on, so `agg` cannot know
that `pl.len()` of no rows is zero while `pl.col(x).mean()` of no rows is
undefined; pass `empty=` to say which it is. {meth}`.Metadata.aggregate`, below,
knows it from the reduction's name.

**Aggregating an inherited factor requires `unique_by=`.** This is the
replication problem from the top of this page, arriving from the other direction.
Averaging a *per-image* value over the detections in a track weights each image
by how many detections it happened to contribute — almost never the intended
question. Rather than guess, `agg` refuses an expression over a column defined
above the source level unless you say what to count once:

```python
# Refused: unit_brightness repeats once per detection beneath its image.
md.agg("instance", "track", pl.col("unit_brightness").mean())

# Accepted: each frame contributes one reading to its track.
md.agg("instance", "track", pl.col("unit_brightness").mean(), unique_by="unit")
```

Counting rows never trips this, because `pl.len()` reads no columns at all — so
"how many detections are under this?" is always a question about the source
level itself.

### Naming the summary: `aggregate`

{meth}`.Metadata.agg` takes an expression and asks you to know what it means.
{meth}`.Metadata.aggregate` takes the **name** of a reduction, and a name carries
things an expression cannot:

```python
rolled = md.aggregate("instance_brightness", level="sequence", how="mean")
# adds `instance_brightness_mean`, one value per video
```

The source level is inferred from where each factor is defined, so you say what
you want and where, not where each number happens to live. Naming several
factors at different levels is fine — a `sequence` destination fed from `unit`
and from `track` is genuinely two roll-ups, and is done as two.

A mapping asks for a different reduction per factor, and its keys are the factors
when none are named:

```python
md.aggregate(level="unit", how={"instance_brightness": "mean", "occlusion": "mode"})
```

Naming no factor at all is a **rule** rather than a request: every factor below
the destination that the reduction applies to.

```python
md.aggregate(level="sequence", how="mean")  # every numeric factor below `sequence`
```

The **positional** reductions — the ones that read a bag of values and do not care
what order they arrive in — are `count`, `n_unique`, `sum`, `mean`, `median`,
`std`, `var`, `min`, `max`, `mode`, `first`, `last`, `any` and `all`.

### Reading a level as a series

Four more are **temporal**: they are functions of the ordered series rather than
of the bag, so they need an ordering to exist at all.

- `variability` — mean absolute change per unit of the ordering key. A *rate*, so
  a series with a frame missing from the middle is not twice as jittery for it.
  Distinct from `var`, which is order-invariant: a slow illumination drift and a
  strobing feed can have identical variance and wildly different variability.
- `trend` — least-squares slope of the values against the ordering key, in units
  per key unit.
- `changes` — transitions between consecutive distinct values.
- `longest_run` — longest consecutive stretch of one value, or of values within
  `options={"tolerance": ...}` of the last.

The ordering column is inferred from the source level, preferring a wall-clock
`time_s` over a presentation `pts` over a position, and only a column the level
holds *itself* is eligible. Name one with `order_by=` on an
{class}`.Aggregator`. A level that carries no ordering is **refused** rather than
run against row order, which is an artifact of the walk and not time — so a
named factor at such a level raises, and the rule form passes the level over.

```python
from dataeval.types import Aggregator

md.aggregate(Aggregator("variability", "unit", "sequence", ("brightness",)))
md.aggregate(
    Aggregator("longest_run", "unit", "sequence", ("brightness",), options={"tolerance": ("iqr", (None, 1.5))})
)
```

A tolerance is a {data}`.ThresholdLike`, so a bare number means a multiplier on
the default rather than a distance; say which you mean —
`("constant", (None, 0.1))` for 0.1 in the factor's own units, or
`("iqr", (None, 1.5))` for 1.5 times the spread of the changes actually
observed. A relative one is read off the data, which makes the resolved
aggregator a fit: it comes back carrying the number rather than the recipe, so
replaying it against a second dataset reuses it instead of re-deriving a
different one.

`changes` and `longest_run` count positions, so an uneven ordering key distorts
them — a reading that was never taken is not in the series at all, and a run
reads straight through where it would have been.
{attr}`.AggregationRecord.gaps` counts the steps larger than the tightest one,
and DataEval says so at info level.

**A name states which values it is about.** `mean` over a class label is a
category error that an expression would compute anyway and hand back as a
number. Asked for by name, it is refused before evaluation, and the refusal says
what would have worked:

```python
# `occlusion` is a per-detection string factor
md.aggregate("occlusion", level="unit", how="mean")
# ValueError: 'mean' does not apply to 'occlusion', whose values are String;
# 'mean' takes numeric values. Reductions that apply to 'occlusion' are
# ['count', 'first', 'last', 'max', 'min', 'mode', 'n_unique'].
```

A factor you *name* is a request, so a mismatch is refused. A factor selected by
the rule form is filtered out instead, with a line at info level saying what was
passed over — selecting is what the rule is for.

**A name is asked for without inspecting its inputs, so it is strict by default.**
`aggregate` applies `min_coverage=1.0`: a destination whose rows did not *all*
record a value answers null rather than summarizing the rest. `agg` defaults the
opposite way, because an expression is written by someone who has looked. Lower
the threshold with an {class}`.Aggregator`, and read
{attr}`.Metadata.last_aggregation` to see the coverage that would have been
answerable.

**A name states what an empty group answers.** `count`, `sum` and `n_unique`
answer `0`; `any` answers `False` and `all` answers `True`; `mean`, `median`,
`min`, `max`, `std`, `var`, `mode`, `first` and `last` answer null. A frame
holding no detections genuinely has zero of them, and just as genuinely has no
mean of anything — only the reduction knows which.

**A name is the durable record of the operation.** A rolled-up factor is called
`f"{factor}_{how}"`. {attr}`.FactorInfo.aggregated_from` records the level a
factor was rolled up *from* and deliberately not what was done to it, so the name
is what tells `brightness_mean` from `brightness_median`.

Every *factor* can be rolled up, not only the ones the current view admits into
factor analysis. Whether a per-track factor is readable on detection rows is a
question about the view; whether it can be averaged over a sequence is a question
about the data. Reserved columns such as `class_label` are not factors and are
not reachable by name here — a coarse class factor comes from {meth}`.Metadata.agg`,
which reads the store's columns directly.

### Choosing a route through the diamond

For a chain there is one way up and nothing to choose. On the tracking diamond a
detection reaches its sequence two ways — through its frame, and through its
track — and the two do not agree about untracked detections, because one branch
has no answer for them.

By default a roll-up takes **every** route, so it reaches a row wherever any
branch does. `via=` narrows it to one branch:

```python
from dataeval.types import Aggregator

md.aggregate(Aggregator("mean", "instance", "sequence", ("box_area",)))
# every detection contributes, through its frame

md.aggregate(Aggregator("mean", "instance", "sequence", ("box_area",), via="track"))
# only the detections a tracker linked; lands as `box_area_mean_via_track`
```

These are **different questions, not different spellings of one**. The first asks
"how much of this, per video"; the second asks "how much of this per video, among
things a tracker committed to". They differ by exactly the untracked detections,
and going up in two hops — `instance` to `track`, then `track` to `sequence` —
gives the second answer, not the first. Roll-up is not associative across a
branch that stops short.

The route appears in the output name whenever it is not the default, so the two
cannot be confused once computed. DataEval also logs, at info level, how many
rows took no part for want of an ancestor — zero for every complete route, and
exactly the untracked count for a partial one.

```{note}
Where both branches *do* have an answer they must give the same one. A detection
whose track sits in one video while its frame sits in another is a contradiction
rather than a preference, and DataEval raises rather than picking a winner.
```

### Declaring a roll-up apart from running it

{class}`.Aggregator` is the roll-up as a value: the reduction, the levels it runs
between, the factors, and the modifiers `aggregate`'s keywords do not carry —
`via`, `unique_by`, an output suffix. Everything except the factor set is
checkable against a level graph with no dataset in hand, so a roll-up can be
declared next to the thing that produces the numbers and be wrong at import time
rather than at analysis time:

```python
from dataeval.types import Aggregator, FactorLevelSchema

schema = FactorLevelSchema.of("sequence", "unit", "track", "instance")
Aggregator("mean", "unit", "sequence").validate(schema)  # no dataset in hand
Aggregator("mean", "unit", "track").validate(schema)
# ValueError: ... 'track' does not sit above 'unit' in this level graph.
```

Leaving `source` as `None` means "infer it per factor", which is what
`aggregate`'s keyword form does. Resolving that against a dataset reads the
answer off the data, so the resolved aggregator records itself as `derived` — the
same distinction {class}`.BinSpec` draws between edges a caller declared and
edges fitted to a draw.

## Narrowing the population: `where` and `having`

{meth}`.Metadata.where` and {meth}`.Metadata.having` both take a predicate at
some level and return a new metadata over fewer rows. They differ in **which
direction the cut travels**, and the difference is easiest to see by giving both
the same predicate:

```python
md.level_counts  # {'unit': 50, 'instance': 93}
md.where(pl.col("class_label") == 0, "instance").level_counts  # {'unit': 50, 'instance': 20}
md.having(pl.col("class_label") == 0, "instance").level_counts  # {'unit': 17, 'instance': 38}
```

`where` keeps **the matching rows**: the 20 detections of class 0, and every
image, because `where` never filters upwards — an image is still an image
whatever its detections turn out to be.

`having` keeps **the entities that have a match**: the 17 images holding at
least one class-0 detection, and then all 38 detections in those images, not
just the class-0 ones. It is the "images containing a person" filter, and it
travels up first and then back down.

One rule decides everything that follows in both cases:

```{important}
**A row survives only if every ancestor it actually has survives.**

Applied downwards from wherever the filter seeds it. A row with *no* ancestor at
some level is not failing this test — it has nothing there to lose — which is why
an untracked detection is not swept away by a filter over tracks.
```

Two consequences worth expecting. A filter can only ever **add** factors to the
analysis, never remove one: dropping the rows that lacked an ancestor can turn a
partly-null column into a total one, and a total column can be binned. And
`where` at one level leaves that level's **siblings** untouched — filtering
frames does not remove tracks, because a track is not beneath a frame.

### A filtered metadata still holds its whole dataset

This is the one thing about filtering that will bite silently if it is not
stated. Filtering removes *rows*; it does not remove *items* from the dataset the
metadata is bound to. So anything computed from that dataset — embeddings above
all — still describes the original population, and pairing the two row-for-row is
a misalignment that raises nothing.

{attr}`.Metadata.is_filtered` records that this happened, and the evaluators that
take both a metadata and something dataset-shaped refuse a filtered one outright
rather than silently mispairing it.

{meth}`.Metadata.selected_items` is what brings the dataset side back into
correspondence: it returns the surviving item indices, which a dataset view can
be built from.

```python
narrowed = md.having(pl.col("class_label") == 0, "instance")
narrowed.selected_items()  # 17 item indices
```

It succeeds only when the filter cut along item boundaries. The `where` above
kept 20 of 93 detections while keeping all 50 images, so it has no item-level
answer to give — a dataset can hand back an image, not three of its detections —
and `selected_items` says so rather than returning a subset that does not
correspond.

## Factors only some rows declare

A dataset does not always record everything for everything. One image's metadata
omits a key the others carry; one frame of a video declares no timestamp. DataEval's
answer, by default, is **all-or-nothing**: such a factor is dropped for every row,
and {attr}`.Metadata.dropped_factors` says so.

That is the conservative reading. A factor present for part of a dataset can
mislead an analysis that does not know it is part absent — a balance result over
"the rows that happened to have a value" is a statement about a population nobody
chose. But it also throws away the part that *was* recorded, which for a large
dataset with one incomplete entry is the whole factor.

`partial_factors=True` asks for the other reading:

```python
md = Metadata(dataset, partial_factors=True)
```

The rows that recorded nothing get a **missing value**, which binning places in a
bin of its own — {attr}`.BinSpec.missing_code` for a cut,
{attr}`.LevelSpec.missing_code` for a vocabulary. "Not recorded" never becomes one
of the factor's values, so it cannot be mistaken for a category the data contains,
and the code space is fixed by the vocabulary rather than by how much happened to
be missing in this draw.

It is one policy, read everywhere structuring meets an incompletely declared
value — a metadata key some items omit, and a per-frame timing some frames omit.
Two opposite answers to that question in one pass would be the harder thing to
explain.

```{note}
A factor **no** row declares is dropped either way. That is a factor the dataset
does not carry, rather than one it carries incompletely, and an all-null column
says nothing that its absence does not. The same applies to a key inconsistent
among the *targets within one item*: by the time the mismatch is visible, which
targets it came from is no longer recoverable.
```

Aggregation is where the difference pays off. A sequence-level mean over the 999
frames that did declare a timestamp is a perfectly good number, and under the
default the factor was gone before anything could ask for it. See
{meth}`.Metadata.aggregate`'s `min_coverage`, which decides how much of a
destination must be recorded for it to answer.

## Binning happens at a factor's own level

{class}`.Metadata` discretizes continuous factors into bins so that categorical
analyses can consume them. **Each factor is binned at its own level** — over the
rows that hold one value per entity — and the resulting bin assignments then
propagate downwards like any other value.

Consider altitude, recorded once per image, on a detection dataset:

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

## Choosing a level for a video measurement

Video datasets force a choice between four structural levels: a sequence, its
frames, its tracks, and its detections.

The core rule is: **Store a value at the level where it was measured.** For
example, store a per-frame metric at the `unit` level, even if you eventually
want a sequence-level summary. Use {meth}`.Metadata.agg` to compute
coarser-grained metrics from finer ones. This keeps both the raw measurements
and the summaries available.

| Measured Entity | Level | Which column names a row |
| --- | --- | --- |
| Property of the whole video (e.g., codec, resolution, platform) | `sequence` | — the item names it |
| Property of one frame (e.g., brightness, blur, scene cuts) | `unit` | `unit_index` |
| Property of one track (e.g., mean speed, total displacement) | `track` | `track_id` |
| Property of one detection (e.g., box area, confidence, IoU) | `instance` | `target_index` |

Every one of those names a row **within its item**, because `unit_index`,
`track_id` and `target_index` all restart in every sequence. DataEval matches on
`(item_index, key)` for that reason. A sequence needs no key of its own: there is
one per item, so the item names it outright.

The `*_index` columns are DataEval's own *positional* identity: dense, zero-based,
rebuildable from order alone. They are distinct from the dataset's own identifiers,
which are **reserved** and never treated as factors. `item_id` carries the datum's
own `id` — the value MAITE requires on every item — on every row, so any row can be
traced back to its source item; `track_id` does the same for a tracking observation.
It is a lookup, not a key: a view that draws one item more than once
(`ClassBalance` oversampling, `Indices([0, 0, 1])`) puts one id on several rows, and
that is the truth about where those rows came from. A `Metadata` built from raw
factors has no datum to ask, so its `item_id` is the positional index.

Neither identifier is binned, correlated, or reported on: an identifier names a
row, it does not measure the data. A metadata key spelled `id` is carried onto
`item_id` rather than read as a factor, and a factor that would shadow a reserved
id name is renamed out of it.

```{warning}
The key for a detection is `target_index`, **not** `instance_index`.
`instance_index` numbers detections within a *frame*, so it repeats across the
frames of one sequence — it names one row on an image dataset and several on a
video, which is the worst way for a key to be wrong. `target_index` is dense
within the item on every task.
```

### Naming a row: keys or addresses

There are two spellings of that same table, and they agree by construction.

**A column at a time** — the bulk form, and what a producer emitting whole arrays
uses. Pass the key column alongside the values:

```python
md.add_factors(
    {"item_index": [...], "track_id": [...], "speed": [...]},
    level="track",
    key="track_id",
)
```

**One value at a time** — the scalar form, and what an evaluator's findings come
back as. A {class}`.SourceIndex` is `(item, key, level)`, an address naming one
row:

```python
md.add_factors({"speed": [...]}, source_index=[SourceIndex(0, 5, "track"), ...])
```

An address that states no level is the **task-generic** reading: the item level
with no key, the label level under one. `SourceIndex(3)` names image 3 on an
image dataset and video 3 on a tracking one, and `SourceIndex(3, 7)` names
detection 7 of item 3 on either. That is what every producer emits, and it is why
{func}`.compute_stats` output places correctly without knowing which kind of
dataset it measured.

State a level only where an unstated one would resolve to a different one — for a
frame or a track. Two spellings of one address are not equal to each other, so a
result keyed on addresses would hold both.

An address names a row **without saying what it sits inside**, which is what lets
one tuple reach every level of a diamond. The consequence is that addresses can
place values into rows that already exist but cannot *build* the rows: nothing in
a source index says which frame a detection was seen in, so
{meth}`.Metadata.from_factors` given addresses builds the two-level shape only.
Construct from the dataset, then place.

### The impact of misaligned storage levels

Storing a measurement away from its native level changes how the data behaves
during analysis.

- **Storing coarser than measured (Aggregation):** If you store frame-level
  brightness directly at the `sequence` level, you lose the individual frame
  variations. Always store readings at their native level first, and use
  {meth}`.Metadata.agg` to generate summaries (e.g., `.agg("unit", "sequence",
  pl.col("blur").mean())`). This preserves both individual frame metrics and
  sequence-level averages for binning.
- **Storing finer than measured (Replication):** Never replicate a coarser
  value (like a `sequence` property) manually across finer rows (like `unit` or
  `instance`). Since DataEval automatically propagates values down the
  hierarchy, manually replicating them adds redundant data. This distorts
  analysis and binning by over-representing replicated values.
- **Storing on sibling branches:** The `unit` (frames) and `track` levels are
  siblings. You cannot store a per-frame value directly at the `track` level
  because a track spans multiple frames. Instead, aggregate from `instance` to
  `track` using {meth}`.Metadata.agg` with the `unique_by="unit"` constraint.
  This ensures frames with multiple detections do not over-weight the track's
  average.

### Handling empty frames and untracked detections

Video datasets contain structures that standard image datasets do not:

- **Empty frames:** A frame with no detections still has a `unit` row, but has
  no `instance` rows. Storing frame-level measurements at the `instance` level
  silently discards data for all empty frames. Storing them at `unit` keeps the
  data complete.
- **Untracked detections:** A detection with no active track (e.g.,
  `track_id == -1`) belongs to a frame (`unit`) but not to any `track`.
  Track-level factors will be null on it. If you need to analyze untracked
  detections, measure and store those factors at the `instance` level.

### Built-in track statistics

DataEval automatically computes two per-track metrics during structuring:

| Column | Meaning | Equal to |
| --- | --- | --- |
| `track_length` | Number of frames the track was observed in | `n_appearances` in {func}`.track_stats` |
| `frame_span` | Inclusive frame span from first to last | `track_duration` in {func}`.track_stats` |

These metrics differ when a track has temporal gaps (e.g., a track visible only
in frames 0 and 2 has `track_length == 2` and `frame_span == 3`).

Because DataEval computes these automatically, you do not need to recalculate
them via {func}`.track_stats` or manual aggregation (such as running `pl.len()`
at the `instance` level).

## Saving and reading Metadata

Building metadata reads every item of the dataset — decoding images, unpacking
targets, accumulating tracks. Everything after that is arithmetic over the rows
that walk produced, and those rows are small next to the dataset.
{meth}`.Metadata.save` writes them so the walk is paid for once:

```python
Metadata(dataset).save("metadata.dem")
md = Metadata.load("metadata.dem", dataset)
```

This file holds the rows at each level, the
links between them, and which factor is defined where. It does **not** hold the three following things:

- **The dataset.** It cannot be serialized, so {meth}`.Metadata.load` takes a
  live one and binds it — which is also what lets the item counts be checked
  against each other rather than diverging silently.
- **Binning.** Bin assignments are derived from the values, so the file keeps the
  values and the binning configuration is supplied at load. One file therefore
  serves any `continuous_factor_bins` you later want from it. For the same
  reason `exclude`, `include`, and `view` are not stored: they are how a reader
  asks its question, not what the rows are.
- **The per-item metadata dicts** behind {attr}`.Metadata.raw`. They hold
  whatever the dataset put there, of unbounded size, so {attr}`.Metadata.raw`
  raises on a loaded instance rather than answering as though the dataset had
  carried none.

A filtered metadata saves and reloads as filtered, so
{attr}`.Metadata.is_filtered` still reports `True` and the evaluators that refuse
one still refuse it.

```{warning}
This is a **cache, not an interchange format.** The file holds DataEval's
internal per-level layout, and that layout may change in any release. Each file
records the format version and the level graph it was written against, and
{meth}`.Metadata.load` refuses anything it does not recognize rather than
restoring old rows against a new graph — which would raise nothing and answer
wrongly. To keep metadata for anything other than a cache — a record, or another
tool — write {attr}`.Metadata.dataframe` to a parquet file instead.
```

So {class}`.MetadataFormatError` is the *designed* outcome for a stale file
rather than a bug to work around. It carries exactly one piece of information:
that these rows cannot be trusted against this version, and the dataset walk
they stood in for has to be paid again.

## Level names

The media-unit level is called `unit` rather than `image`, and this is the one
piece of the vocabulary chosen for what it enables rather than for what it
describes today.

Every level name in the graph describes a *structural role*: `sequence` is "an
ordered run" of `units`, `instance` is "one labelled thing", and `track` is "one identity across
time", or "an ordered run" of `instances`.

Each medium will have a familiar/colloquial term for the thing at each level.
This is carried separately, and can be referenced by {attr}`.Metadata.unit_type`:

```python
Metadata(image_dataset).unit_type  # 'image'
Metadata(video_dataset).unit_type  # 'frame'
Metadata.from_factors(factors).unit_type  # 'item'
```

{attr}`.Metadata.unit_type` is descriptive only. Nothing in structuring, binning
or projection consults it; it exists so that error messages, reports, and your own
code can speak the caller's language. It is deliberately a plain string rather
than part of {data}`.FactorLevel` — a new modality supplies a new value and edits
no type, while the level vocabulary stays closed and task-independent.

The payoff is that generic code stays generic. Every {class}`.Metadata` writes
`"unit"` into its `level` column regardless of modality, so a pipeline that
concatenates metadata from an image dataset and an audio dataset can filter both
the same way, and {class}`.Balance`, {class}`.Diversity` and {class}`.Parity`
need no modality awareness at all.

```{note}
Previous versions were image-dataset-centric, and used the terms "image" and "target".
Those spellings stopped resolving in v1.2.0: `rows_at`, `at` and `view` take a level
name only, and the retired names raise rather than warn. Level names always read back
as the level vocabulary — `"unit"` for the media-unit level — so
`md.item_level == "image"` is `False` even on an image dataset. The level names
answer where a fact lives, and {attr}`.Metadata.unit_type` answers what the medium
is called.
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

- [Binning](Binning.md) — what the discretization on this page's factors actually
  does to them, and why the cut points are a choice rather than a detail.
- [Image Statistics](ImageStatistics.md) - more factors that can be calculated from raw data
  and added to Metadata on demand.
- [Dataset Bias and Coverage](DatasetBias.md) — the evaluators that consume
  {class}`.Metadata`, and where the choice of view changes what they measure.
- [Ontology](Ontology.md) — the other graph, over label concepts rather than
  granularities.
- [Validation and Trust](ValidationAndTrust.md) — how the level a factor is judged
  at affects the confidence you can place in a result.
- [Acting on Results](ActingOnResults.md) — mapping findings back onto the rows
  they came from.

## See this in practice

### Tutorials

- [Analyze a dataset across its levels](../notebooks/tt_analyze_across_levels.py) —
  everything on this page worked end to end on an aerial detection dataset: the same
  factor reporting two distributions, `at` and `inherited`, `agg`, `where` versus
  `having`, and the bias question that can only be asked once a level is chosen.

### How-to guides

- [How to bin factors by level](../notebooks/h2_bin_factors_by_level.py) — a worked
  example of binning at a factor's own level, with the numbers that show why the
  alternative distorts.
