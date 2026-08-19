<!-- markdownlint-disable MD051 -->

# Binning

Shannon entropy is not defined over altitude in meters, and neither is a
{term}`chi-square test <Chi-Square Test of Independence>`. Both are defined over
a **finite alphabet** — a fixed set of categories a value can take — and a
continuous factor does not have one. So before either can run, the factor has to
be cut into pieces.

That cutting is **binning**, and DataEval does it for you by default.
{class}`.Metadata` {term}`discretizes <Discretization>` every factor before any
evaluator sees it: by the time one reads a factor, it is looking at integer
codes, not at altitudes. The bias and diversity results you get back describe
your factors *as binned*.

This page follows that in four movements: **what** binning is and how it differs
from the relabeling that discrete and categorical factors get, **why** it
matters and which evaluators depend on it, **how** DataEval places the edges,
and how the cut shows up in results. It is the metadata-side counterpart to
[Embeddings](Embeddings.md): the choice of feature extractor shapes every
geometric result, and the choice of bins shapes every metadata one.

## What binning is

{term}`Binning` cuts a continuous factor into intervals and hands each interval
an integer. That is the whole operation. What makes it worth a page is that it
sits beside a second operation that also produces integers but keeps every
distinction, and that DataEval picks between the two per factor, based on what
each factor is.

### Three questions about every factor

Three separate questions get asked of every factor, and they are usually
collapsed into one. Separating them is what makes `factor_info` readable.

```{graphviz}

   digraph factor_kinds {
      node [shape=plaintext]
      rankdir=TB

      // Renders as one three-row table. Each row is a band (a question), each column is
      // a kind of factor, so reading a column downward traces that factor's whole path:
      //
      //   ┌──────────────┬─────────────┬──────────────┐
      //   │  CONTINUOUS  │  DISCRETE   │ CATEGORICAL  │   what is the variable?
      //   ├──────────────┴─────────────┼──────────────┤
      //   │         NUMERICAL          │ NON-NUMERICAL│   how is it stored?
      //   ├──────────────┬─────────────┴──────────────┤
      //   │    BINNED    │         DIGITIZED          │   what did DataEval produce?
      //   └──────────────┴────────────────────────────┘
      //
      // The COLSPANs are what make the columns line up, so changing a WIDTH or a
      // COLSPAN breaks the reading rather than just the appearance.

      // The cell fills are fixed in both themes, so each label carries its own dark
      // FONT COLOR. Left to the theme's foreground variable it would turn white in
      // dark mode and vanish against the pale cells.
      taxonomy [label=<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10">
          <TR>
            <TD BGCOLOR="#e2efda" WIDTH="200"><FONT COLOR="#1f2328">CONTINUOUS</FONT></TD>
            <TD BGCOLOR="#d6dce5" WIDTH="180"><FONT COLOR="#1f2328">DISCRETE</FONT></TD>
            <TD BGCOLOR="#ffd966" WIDTH="160"><FONT COLOR="#1f2328">CATEGORICAL</FONT></TD>
          </TR>
          <TR>
            <TD BGCOLOR="#c6e0b4" COLSPAN="2"><FONT COLOR="#1f2328">NUMERICAL</FONT></TD>
            <TD BGCOLOR="#ffd966"><FONT COLOR="#1f2328">NON-NUMERICAL</FONT></TD>
          </TR>
          <TR>
            <TD BGCOLOR="#b4c7e7"><FONT COLOR="#1f2328">BINNED</FONT></TD>
            <TD BGCOLOR="#dae3f3" COLSPAN="2"><FONT COLOR="#1f2328">DIGITIZED</FONT></TD>
          </TR>
        </TABLE>
      >]
   }
```

Read it **down a column**: each band is one of the questions, so a column traces
a single factor's path from what it is to what the evaluators receive.

1. **What is the variable?** {term}`Continuous <Continuous Variable>` values can
   fall anywhere in a range — brightness, altitude, box area.
   {term}`Discrete <Discrete Variable>` values land on a countable set — a count
   of people, an ISO setting, a year. {term}`Categorical <Categorical Variable>`
   values name a group with no arithmetic between them — `"rainy"`, `"urban"`, a
   sensor ID. A fact about the world, unchanged by how you process the data.

2. **How is it stored?** Numerical or not — a fact about the array's dtype, not
   about the variable, and the two need not agree: sensor IDs are categorical
   but stored as numbers. Dtype is read first and decides the label, so
   `continuous` and `discrete` are always numeric and `categorical` never is.

3. **What did DataEval produce?** Non-negative integers either way, by one of two
   operations that are not equivalent. This band is the only one that describes
   a choice rather than an observation, and it is the one this page is about.

The middle column is the one to hold on to: a discrete numeric factor is
*numerical* and *not continuous* at once, needing no cutting because it already
has a finite alphabet. It is why the two words are not interchangeable.

### Binned is not digitized

Digitizing and binning both hand the evaluators integers. They differ in what
they keep:

:::{list-table}
:widths: 16 42 42
:header-rows: 1

- - Property
  - {term}`Digitizing <Digitization>`
  - {term}`Binning`
- - What it does
  - Assigns one integer per distinct value
  - Assigns one integer per interval of values
- - Information
  - **Lossless.** The map is one-to-one and invertible
  - **Lossy by design.** Many values share one code, which is what gives the
    factor a finite alphabet
- - Choice involved
  - None. The distinct values are whatever they are
  - **The bin edges.** Which ones are right depends on the question being asked
- - Applies to
  - Categorical factors, and discrete factors the sample can afford
  - Continuous factors, and discrete numeric factors carrying more levels than the
    sample supports
:::

Digitizing is bookkeeping — a relabeling that any downstream count would have
produced anyway. Binning is a summary: it trades resolution for the finite
alphabet the evaluators need, and how much resolution it trades depends on where
the edges fall. **Everything on this page that involves a decision involves
binning, not digitizing.**

### Reading `factor_info`

The diagram's three kinds map exactly onto what {attr}`.Metadata.factor_info`
reports, so it doubles as a legend for that output:

:::{list-table}
:widths: 26 24 22 28
:header-rows: 1

- - Factor
  - `factor_type`
  - Companion
  - Produced by
- - Continuous numeric
  - `"continuous"`
  - `is_binned=True`
  - Interval cuts placed by `auto_bin_method`
- - Discrete numeric, within the [level budget](#a-discrete-factor-can-still-be-binned)
  - `"discrete"`
  - `is_digitized=True`
  - Ordinal encoding of the distinct values
- - Discrete numeric, over that budget
  - `"discrete"`
  - `is_binned=True`
  - Interval cuts placed by `auto_bin_method`
- - Non-numeric
  - `"categorical"`
  - `is_digitized=True`
  - Ordinal encoding of the distinct values
- - Non-numeric, near-unique
  - — dropped —
  - — none —
  - `"cardinality_over_budget"` in {attr}`.Metadata.dropped_factors`
:::

Each surviving factor also carries the map itself in
{attr}`.FactorInfo.encoding`: a `BinSpec` holding the cut points where the
companion says `is_binned`, a `LevelSpec` holding the value per code where it
says `is_digitized`. That is what lets a code be read back as `[0, 12.4)` or
`rain` rather than as `3`.

Four factors as `factor_info` reports them:

```text
altitude_m   continuous   binned=True    ← cut into intervals, lossy
epoch_s      discrete     binned=True    ← too many levels to afford, cut anyway
n_people     discrete     binned=False   ← relabeled, lossless
weather      categorical  binned=False   ← relabeled, lossless
```

`factor_type` and `is_binned` are therefore independent: the type records what the
factor is, the companion flag records what was done to it, and `discrete` appears
against both. Selecting the factors that were summarized rather than relabeled
means reading `is_binned`, not `factor_type`.

The discretized values live in a **companion column** rather than replacing the
factor: `altitude_m↕` holds the bin indices, `weather#` the ordinal codes.
{attr}`.Metadata.factor_data` is the view built from those companions and is
what the bias evaluators read; {meth}`.Metadata.rows_at` carries the native
values, so the dataframe keeps the measurements at full resolution. The summary
sits alongside them, and is what the evaluators read.

Because dtype decides the label, a **categorical factor stored as integers** —
sensor IDs, encoded weather codes — is reported as `discrete`. The evaluators
are unaffected: both digitize identically and both reach them as
`is_binned=False`. Your own code is not, so anything selecting factors by
`factor_type` will miss it. Dtype is the only input to that label, so the same
factor stored as strings reports as `categorical`.

There is a fourth path. Naming a factor in `continuous_factor_bins` sends it
down the first row **regardless of what its numeric values mean**, marking it
continuous and binned even if they are categorical codes. A non-numeric column
named there raises `TypeError` rather than being coerced. That is the intended
escape hatch for a numeric factor you want cut on edges of your own, so it is
worth naming only factors you mean to treat as continuous — the label holds for
the life of that {class}`.Metadata`.

### Binning moves a factor one way

Binning does not just relabel a continuous factor. It moves it along the first
band, toward the categorical end:

```{graphviz}

   digraph ladder {
      rankdir=LR
      node [shape=box, style="rounded,filled", fontsize=11, height=0.5]
      edge [fontsize=10]

      // Renders as three rounded boxes left to right, the second arrow dashed:
      //
      //   ( CONTINUOUS ) ──auto_bin_method──> ( BINNED ) ┈┈what it now means┈> ( CATEGORICAL )
      //
      // Fills match the first diagram's bands so the two read as one picture.

      // Node text is pinned to its fill; edge labels sit on the page background and
      // are left to the theme so they follow light and dark mode.
      cont [label="CONTINUOUS", fillcolor="#e2efda", fontcolor="#1f2328"]
      binned [label="BINNED", fillcolor="#b4c7e7", fontcolor="#1f2328"]
      cat [label="CATEGORICAL", fillcolor="#ffd966", fontcolor="#1f2328"]

      cont -> binned [label="  auto_bin_method"]
      binned -> cat [label="  what it now means", style=dashed]
   }
```

A binned factor lands in between: **stored as a number, ordered like a number,
and meaning a category.** Bin `3` is not three of anything, and bin `1` to bin
`3` is not twice bin `1` to bin `2` unless the edges happen to be equally
spaced.

Downstream code reads the number, because the number is nearly all it gets.
Evaluators reach factors through {class}`.MetadataLike`, whose four required
members are `factor_names`, `factor_data`, `class_labels` and `is_binned` —
`factor_info` is not one of them. Everything a factor's treatment contributes is
that one bool, which is `True` for a binned continuous factor:

- {func}`.split_dataset` groups on bin index, so two adjacent bins are as
  different as two distant ones.
- A missing value's bin (see [below](#missing-values-get-a-bin-with-a-position-but-no-meaning))
  sits *above* the highest observed bin, so anything reading position as
  magnitude reads "missing" as "large".

The arrow only runs left to right: a bin index does not carry altitude back.
That is why the edges are worth knowing — they set the vocabulary every metadata
result is expressed in.

## Why it matters

**Binning is what makes the measurement possible.** Entropy, a contingency
table, a group identity — each needs a finite set of categories to count into,
and a continuous factor supplies none. Cutting one into intervals is what puts
altitude, brightness and box area on the same footing as weather and city, so a
single set of evaluators can read all of them.

**The edges decide what the measurement can see.** Two values in the same bin
are one value to everything downstream, and two in different bins are simply
different — no closer for being adjacent. That is the summary doing its job, and
it is also why edges placed at the boundaries your question cares about give
sharper answers than edges placed anywhere else.

How much each point matters depends on the consumer.

### Which consumers require it

Not every consumer needs binning, and the ones that do need it for different
reasons — most of them mathematically, one as a consequence of what it is
handed.

:::{list-table}
:widths: 26 18 56
:header-rows: 1

- - Consumer
  - Binning is
  - Why
- - {class}`.Diversity`
  - **Required**
  - Shannon and Simpson indices are sums over the probabilities of a finite set
    of categories. A continuous factor has no such set.
- - {func}`.parity` ⚠️, {class}`.Parity` ⚠️
  - **Required**
  - A contingency table needs cells to count into. Coarser bins avoid the sparse
    cells reported in `insufficient_data`, at the cost of resolution.
- - {func}`.split_dataset` with `split_on`
  - **Required**
  - Grouping needs discrete group identities. Note that `split_on` is ignored
    outright on object detection datasets, with a log message and no error.
- - {class}`.Balance`
  - **Required in practice**
  - It reads `factor_data`, which is bin and category indices throughout, so the
    contingency-table estimator runs on every factor, at whatever resolution the
    binning upstream produced. What `Balance` forwards is not *which estimator*
    to use — that is read from the column's values — but whether each factor's
    set of values is its own or an artifact of the cuts, which decides how the
    `factors` DataFrame is normalized. See [below](#binning-reaches-the-three-outputs-differently).
- - {func}`.factor_deviation`, {func}`.factor_predictors`
  - **Not applied**
  - Both take a plain mapping of factor name to raw array and never touch
    `factor_data`, so they see the values you measured.
:::

The `Balance` row is a property of what it passes, not of the estimator it calls.
{func}`.mutual_info` given raw values still routes the measured columns to the
neighbor-based estimator; the resolution was set upstream in `factor_data`. One
visible consequence: `Balance.num_neighbors` tunes a neighborhood only the
measured path consults, which `Balance` does not produce — it is deprecated for
that reason and warns when set.

## How DataEval bins

With `continuous_factor_bins=None` — the default — {class}`.Metadata` classifies
each factor on its own. Non-numeric columns are ordinal-encoded
immediately, unless they carry more distinct values than [the level
budget](#a-discrete-factor-can-still-be-binned) allows, in which case they are
dropped. Numeric columns go to {func}`.is_continuous`, and its verdict picks the
treatment.

### The continuous/discrete heuristic

{func}`.is_continuous` is a heuristic, not a hypothesis test: it reports no
p-value and has no stated error rate. It combines three signals:

1. **Near-neighbor uniformity.** Under a continuous distribution, a point is
   equally likely to lie anywhere between its two neighbors. Normalizing those
   positions to [0, 1] gives a distribution that is near-uniform for continuous
   data and lumpy for discrete data; the Wasserstein distance from uniform
   measures the gap, against a threshold of `0.5 / sqrt(n)`.
2. **Duplicate fraction.** Genuinely continuous values drawn into floating point
   collide with probability zero, so exact duplicates are evidence of discrete
   support.
3. **Lattice (GCD) test.** Discrete values on a regular grid have gaps that are
   near-integer multiples of a base unit. This catches integer-valued
   distributions before enough collisions accumulate for signal 2 to fire.

A sample is called continuous when signal 1 clears its threshold and neither
secondary signal fires. If exactly one secondary fires, the stricter
`0.3 / sqrt(n)` threshold decides. If both fire, the sample is discrete
regardless. **All five constants are tuned empirically rather than derived** —
the two Wasserstein thresholds, a `0.005` duplicate tolerance, a `0.85` lattice
cutoff, and a `0.05` near-integer tolerance inside the GCD test — so on data
whose support you already know, the verdict is quick to confirm.
{func}`.is_continuous` is public, and its verdict on an array is the same one
{class}`.Metadata` acts on. Three samples of 200 values:

:::{list-table}
:widths: 36 16 48
:header-rows: 1

- - Sample
  - Verdict
  - Which signals fired
- - Draws from a standard normal
  - continuous
  - None — near-uniform neighbor positions, no collisions, no lattice
- - Integers drawn from 0-99
  - discrete
  - All three: lumpy neighbor positions, duplicates, and a lattice
- - The same normal draws rounded to one decimal
  - discrete
  - All three, created by the rounding alone
:::

That last case is the common one. **Rounding a continuous quantity for storage
makes it discrete**, so a brightness value written to one decimal place will be
digitized into one category per distinct value rather than binned.

### Floors that override everything

- **Fewer than 20 observations → discrete, unconditionally.** On a small
  dataset every distinct float becomes its own category, and the factor ends up
  nearly one-to-one with the sample index.
- **Fewer than 3 distinct values → discrete.** The near-neighbor construction
  needs interior points.

Both counts are taken *after* dropping non-finite values, so a factor that is
mostly missing falls under the floor and is called discrete. Infinities are
dropped alongside `NaN`: a missing value has no position on the line and an
infinity no finite distance to its neighbors, so neither carries the spacing
information the test reads. These are the same values the bin edges are placed
on, so the verdict and the edges are read off one set of numbers.

### A discrete factor can still be binned

A discrete verdict says the factor's support is countable. It does not promise
the support is *small*, and an integer factor can be discrete while taking a
value per entity — pixel areas, epoch seconds, identifiers. Scoring one of those
a value at a time is what makes it report a correlation with anything it is
measured against, because a contingency table with more cells than observations
records which cells were hit rather than anything about the factor.

So no numeric factor carries more levels than the sample's level budget,
`max(20, sqrt(n))`, with `n` counted at the factor's own level.

`sqrt(n)` is the square-root rule for histogram bin counts; the floor of 20 keeps
an ordinary categorical factor intact on a small sample. Both are standard rules
of thumb rather than derived quantities. A discrete factor over that budget is
binned rather than ordinal-encoded — reported as `factor_type="discrete"` with
`is_binned=True` — and the same budget caps the bin count on the continuous
path.

The budget is the point past which a contingency table stops describing the data,
and nothing more than that. It is **not** the count that reads best, which sits
well under it and moves with the sample size —
[measured here](#a-fine-cut-costs-the-same-correlation-from-the-other-side). A
factor sitting just inside the budget clears what the budget guards against and
is still cut more finely than the sample rewards.

A different line binds non-numeric columns, and it is not the budget. The budget
answers *how many cells can this sample fill*, which is the right question for
choosing a bin count and the wrong one for deciding whether a column is a factor
at all — twenty-five cities over a hundred images overruns it and is a perfectly
good factor. What is asked instead is whether the column **names its rows rather
than grouping them**: a value per row is not a category, every contingency table
over it is a table of ones, and there is no order along a set of strings to cut it
into fewer. Such a column is **dropped**, leaving {attr}`.Metadata.factor_names`
and gaining a `"cardinality_over_budget"` entry in
{attr}`.Metadata.dropped_factors`.

A numeric column at the same cardinality is binned and kept — that asymmetry is
the point, and it extends to timestamps, which are ordered and so cut into
intervals like any other measured quantity. Map the values onto a smaller
vocabulary — a coarser taxonomy, a prefix, a lookup — to keep a column that is
meant to be categorical.

A merely *wide* vocabulary is not dropped. A factor whose levels are thin for the
sample is what {attr}`.ParityOutput.insufficient_data` exists to report, and
removing the factor would foreclose the mechanism that says so.

That report names its levels rather than numbering them, and
{meth}`.Metadata.code_names` is the same lookup asked directly — what code 3 of
`illum_lux` means, without going through an evaluator to find out. Names come from
the record, so a declared cutoff reads as `< 0` rather than as whatever the coldest
row happened to be. A factor cut at `[-np.inf, 0.0, 10.0, np.inf]` reads back as
`< 0`, `[0, 10)` and `>= 10`, and every bin the cut declares is named whether or
not this sample reached it. Anything rendering a factor's codes itself — a report,
an export, a plot axis — gets the strings the evaluators use rather than
approximating them.

Both the automatic binning and the drop raise a `UserWarning`, naming every factor
affected, in addition to the per-factor detail on the `dataeval.metadata` logger.

### Where the decision is made matters

On {term}`object detection <Object Detection>` metadata, factors live at
different [levels](MetadataLevels.md). A factor recorded at the **unit** level
is classified and binned on its per-image values, not on the copy replicated
onto each detection — otherwise the replication would look like duplicates and
signals 2 and 3 would call a continuous factor discrete. A factor recorded at
the **instance** level is scored on every detection, because there the repeats
are genuine observations.

The 20-sample floor applies to whichever set the factor is judged on, so a
unit-level factor on fewer than 20 images is always called discrete no matter
how many detections those images carry. {ref}`binning-levels` works through the
consequences.

### Choosing a method

`auto_bin_method` selects how edges are placed for factors judged continuous —
and for [discrete factors that overrun the level budget](#a-discrete-factor-can-still-be-binned).
All three place edges from the finite values only, then set the outermost edges
to ±∞ so that finite values beyond the observed range would still land in an end
bin rather than forming their own. That absorbs `-inf`; `+inf` escapes it, for
the reason given [below](#missing-values-get-a-bin-with-a-position-but-no-meaning).

:::{list-table}
:widths: 20 40 40
:header-rows: 1

- - Method
  - Preserves
  - Costs
- - `"uniform_width"` *(default)*
  - The **shape** of the distribution. Equal-width bins mean bar height is
    proportional to density, so a histogram of the result looks like the data.
  - Statistical power is unequal across bins — tail bins may hold a handful of
    samples while the mode holds hundreds.
- - `"uniform_count"`
  - **Power per bin.** Quantile edges give every bin roughly equal occupancy,
    which is what contingency tests want.
  - Shape. Bin width now varies inversely with density, so equal bins do not
    mean equal ranges and the result cannot be read as a histogram.
- - `"clusters"`
  - **Natural gaps.** Edges are derived from DataEval's HDBSCAN port, so
    genuinely multimodal factors get cuts between the modes rather than through
    them.
  - Stability. It inherits the clusterer's sensitivity, and there is no
    guarantee the number of modes is stable across datasets.
:::

Those trade-offs line up with particular consumers. `"uniform_count"` is the one
that matches what {func}`.parity` wants, since equal occupancy is what keeps
cells out of `insufficient_data`. `"clusters"` is the only one that will not cut
through a mode on a visibly multimodal factor. The default trades both for a
result that reads back as a histogram of the data.

All three place edges by reading the values, so where a factor has a threshold
that comes from the world rather than the data, the choice of method is not the
lever that reaches it. `continuous_factor_bins` is.

## Matching the cut to the question

Automatic binning reads the values and nothing else, which is the right default:
most factors have no privileged cut point, and edges derived from the
distribution describe them well. Some factors do have one — a value at which the
meaning of the factor changes — and that value is not something the distribution
can reveal. Declaring it is what this section works through.

### A worked example: temperature

Take 400 road-scene images with the air temperature recorded for each, ranging
from -18.6 °C to 22.4 °C. A quarter of them were shot below freezing. That
boundary is the one that matters: below 0 °C the road may be iced, above it the
same road is merely wet, and a model that has never seen ice will fail on it. So
the question you want to ask the dataset is *"are pedestrians under-represented
in freezing conditions?"* — a {class}`.Parity` ⚠️ test between the class labels
and a temperature factor.

That test needs categories, so the temperature has to be cut. Here is what the
default does with it:

```text
uniform_width — 3 bins
  bin 1   n= 17   [-18.6,  -5.1]   100% freezing
  bin 2   n=299   [ -4.9,   8.7]    29% freezing   ←
  bin 3   n= 84   [  8.7,  22.4]     0% freezing
```

Three quarters of the dataset landed in one bin that is 29% icy and 71% wet.
Every evaluator downstream now sees "temperature = 2" for both an iced road and
a mild wet one, so no result computed from this factor can distinguish them. The
`Parity` test runs and returns a number, computed over a category that mixes
iced and wet roads. It is a sound answer about that category, and not an answer
about freezing.

Switching method moves the edge but not the mixing:

```text
uniform_count — 3 bins
  bin 1   n=134   [-18.6,   1.1]    77% freezing   ←
  bin 2   n=133   [  1.1,   6.2]     0% freezing
  bin 3   n=133   [  6.2,  22.4]     0% freezing
```

The edge moved, but it still straddles zero. It always will, because **0 °C is
not a feature of the distribution — it is a feature of water.** Every automatic
method places edges by reading the values: equal spans, equal counts, or gaps
between modes. None of them has any way to know that one particular value on
that axis is a phase change. The information needed to cut this factor for
*this* question is not in the factor.

It has to arrive from outside the factor, through `continuous_factor_bins` —
declaring the edges `[-np.inf, 0.0, np.inf]` for `temp_c` puts the one cut where
the meaning is instead of where the values happen to fall:

```text
declared edges — 2 bins
  bin 1   n=103   [-18.6,   0.0)   100% freezing
  bin 2   n=297   [  0.0,  22.4]     0% freezing
```

Two bins, cut in the one place that carries meaning, and the `Parity` test now
answers the question that was asked. A longer edge list adds resolution —
`[-np.inf, -5, 0, 5, np.inf]` separates hard freeze from marginal freeze — but
the edge at zero is the one doing the work.

### A second collection

Now shoot the same road again in early spring. Same camera, same `temp_c`
factor, same call to {class}`.Metadata` — and only 2% of the images below
freezing this time:

```text
uniform_width — 4 bins
  bin 1   n= 37   [ -4.6,   2.7]    24% freezing
  bin 2   n=211   [  2.7,  10.0]     0% freezing
  bin 3   n=139   [ 10.0,  17.1]     0% freezing
  bin 4   n= 13   [ 17.8,  24.5]     0% freezing
```

Nothing was configured differently. The winter collection came out in three
bins and this one in four, because the count is derived from the values rather
than set by you. Two things follow:

- **Contextually**, the freezing signal has all but vanished — nine images out
  of 400 — and it is *still* not isolated. Bin 1 mixes freezing with mild, the
  same mixing winter's bin 2 had, at a different place on the axis.
- **Comparatively**, the two collections no longer share a vocabulary.
  `temp_c = 1` means "hard freeze, -18.6 to -5.1" in the winter data and "-4.6
  to 2.7, mostly above freezing" in the spring data. The integer is the same
  and the meaning is not.

The second is the one that spreads. A {class}`.Diversity` index over a
three-letter alphabet is not on the same scale as one over a four-letter
alphabet, and a {class}`.Balance` score that moved between the two collections
may be reporting a change in the road or only a change in where the cuts fell —
with nothing in either output to tell you which. Anything that reads
`factor_data` inherits this; {func}`.factor_deviation` and
{func}`.factor_predictors` escape it precisely because they read the raw values
instead.

Explicit edges settle both at once. `[-np.inf, 0.0, np.inf]` gives the spring
data bin 1 = 9 images below freezing, bin 2 = 391 above — the same two
categories, meaning the same two things, as the winter data. The factor becomes
comparable across collections because its vocabulary is now fixed by you rather
than derived per dataset. [The bin count is a function of the data,
not a setting](#the-bin-count-is-a-function-of-the-data-not-a-setting) has the
measurements behind this.

### Other thresholds that come from the world

The same shape recurs whenever a factor has a threshold set outside the data:
dawn and dusk in a time-of-day factor, a sensor's rated range in a distance
factor, the resolution below which an object is unrecognizable in a box-area
factor. **In each case, a cut derived from the values is a good summary of the
distribution and not a summary of the boundary you care about** — so that
boundary is worth declaring. These are the factors `continuous_factor_bins`
exists for; the rest are well served by the default.

## Pitfalls to check for

The automatic path derives its cut from the values, which has consequences worth
knowing so you can recognize the factors that would be better off declared. Each
of these is visible before you read anything computed from the factor.

### The bin count is a function of the data, not a setting

`"uniform_width"` does not take a bin count. It starts from NumPy's
`histogram(bins="auto")`, then *reduces* the count — at most 20 times — while
any non-empty bin holds fewer than 10 samples. The 10-sample floor keeps every
bin populated enough to count into, and the tails trip it readily, so the count
that comes out is a property of the particular draw:

```text
500 draws from a standard normal, 200 different seeds

bins     3    4    5    6    7    8   10
seeds   12   63   59   39   20    6    1
```

Between **3 and 10 bins** for the same distribution at the same sample size. The
consequence is not that any one of those is wrong — it is that **the same factor
measured on two datasets can be cut into different numbers of bins**, so binned
factor values, and every score computed from them, are not comparable across
runs. `"uniform_count"` inherits this count and moves only the edges.

The [level budget](#a-discrete-factor-can-still-be-binned) caps the count from
above and the reduction floors it at two, so the result is bounded — but a bound
is not a setting, and anything between those two ends is still chosen by the
draw.

### A factor can reduce to two bins

The reduction stops at two bins, so a factor never comes back constant. Two is
the floor, and a small sample with a far-off tail reaches it every time: the
outlier stretches the range, the interior bins come out sparse, and the
reduction runs all the way down — 20 to 80 draws from a standard normal plus
three outliers near 50 came back as two bins on every one of 2,000 seeds.

A continuous factor reduced to a **binary split** still answers every question
put to it — a {class}`.Diversity` index over two categories, a {class}`.Parity`
table with two columns, two groups under {func}`.split_dataset` — at the
resolution two categories allow. On this shape the cut lands between the mode
and the outliers, which is where the sample supports one, and a binary answer to
a question about a spread of values is usually not the one you wanted.

The level count is what tells you. `factor_data` columns follow `factor_info`
order, so the level count of a binned factor is worth checking before reading
anything computed from it. A column carrying two distinct codes is a factor that
arrived as a binary split, whatever resolution the values behind it had.

### Explicit edge lists produce more bins than edges

`continuous_factor_bins` accepts either an integer count or a sequence of edges,
and they behave differently at the boundaries. An **integer** gets the ±∞
treatment described above. An explicit **sequence** is used verbatim, so values
outside its range fall into open-ended bins on either side. For a factor holding
`-5.0`, `5.0`, `15.0` and `25.0`:

:::{list-table}
:widths: 34 22 44
:header-rows: 1

- - `continuous_factor_bins` value
  - Codes produced
  - What the outermost edges are
- - The edge sequence `[0, 10, 20]`
  - `0, 1, 2, 3`
  - `0` and `20`, so anything beyond them gets a bin of its own
- - The count `3`
  - `1, 2, 3, 3`
  - ±∞, so the end bins absorb everything past the observed range
:::

Three edges describe two intervals, but four bins come back: one below `0` and
one above `20`. ±∞ present in the sequence is what makes the given edges the
outer limits — the treatment an integer count receives automatically and a
sequence does not.

### Missing values get a bin with a position but no meaning

A `NaN` is not a small value, a large value, or a value between two edges, so it
is given a bin of its own above the bins holding observed values — automatic or
explicit, on binned and digitized factors alike.

- **Edges are placed on the finite values alone**, so a missing value does not
  shift where the cuts fall.
- **`+inf` is not absorbed.** `-inf` lands in the first bin, but because the top
  edge is `+inf` and digitizing is right-open, `+inf` falls past it into a bin of
  its own, one above the highest finite bin. The missing bin then sits one above
  *that*, so a factor carrying both ends up with two categories that are not
  values.
- **The position is an artifact.** The missing bin sits at the top because the
  codes have to be contiguous — a gap would show up downstream as an empty
  category — not because missing is large. Anything treating the factor as
  ordinal reads that position as a value.

So where the missing rate is material, part of any finding about that factor is
a finding about its absence, and the output does not separate the two.

### A nanosecond timestamp is read to 256 ns

A capture time is totally ordered, so it cuts into intervals exactly like a
number — DataEval reads one as a float so that `NaT` can travel as `NaN` and be
given the missing bin like any other gap. A float64 carries 53 bits of mantissa,
and near the current epoch (~1.8e18 nanoseconds) consecutive representable values
are **256 ns** apart, so two capture times closer together than that become one
number.

`datetime64[us]` and `datetime64[ms]` are exact. Only `[ns]` — polars' and pandas'
default unit — is affected, and only below a resolution no bin edge on a capture
time is placed at. What it does change is the *distinct count* of such a column,
which feeds the continuous/discrete verdict and the near-uniqueness test that
drops identifiers. If a sub-microsecond difference is one you need kept, cast the
column to `datetime64[us]` before handing it to {meth}`.Metadata.add_factors`.

### The automatic path announces itself

Structuring raises a `UserWarning` naming every factor it binned on the caller's
behalf, because the per-factor `WARNING` it also writes to the `dataeval.metadata`
logger reaches nobody by default — DataEval attaches a `NullHandler` there, which
suppresses Python's last-resort stderr handler. One aggregated warning rather than
one per factor, so it stays readable on a dataset carrying many of them.

The record survives the warning. Each factor's {attr}`.FactorInfo.encoding` says
where its cuts fell and who chose them — `provenance="derived"` where DataEval
chose both the count and the placement, against `"count"`, `"edges"` or
`"declared"` where the caller did — so a run that binned automatically is
distinguishable from one handed explicit edges long after the warning scrolled
past. `repr(md)` reports the count as `auto_encoded=N`.

## How the cut shows up in a score

The sections above are about where the edges land. This one is about the
arithmetic that follows from having edges at all: how much association a cut
carries through, and how the number is normalized so that two different cuts can
be read on one scale.

### Binning reaches the three outputs differently

{class}`.Balance` returns three DataFrames, and binning does not reach them
equally. The difference is what each one divides by:

- **`balance` and `classwise`** divide by the entropy of the **class label**,
  which is never binned. Cutting a factor more finely moves the numerator toward
  the dependence the unbinned values carry and leaves the denominator alone, so
  the score converges. On a factor with a true dependence of 0.41, cutting it into
  2, 8, 32 and 256 bins gives 0.27, 0.39, 0.40 and 0.41 — settled by about 16 bins.
- **`factors`** compares two metadata factors, and *both* may be binned. There is
  no fixed reference, so a factor whose alphabet came out of the cuts contributes
  no entropy ceiling: such a pair is scored by the Linfoot transformation instead.
  Scoring it against a binned factor's entropy would make the reported association
  shrink as the same data is cut more finely — 0.40 at four bins down to 0.14 at
  128, on identical data.

A factor whose values *are* its own alphabet — a category, a count, a rating —
keeps the entropy ceiling in `factors`, so a duplicated categorical factor still
reads 1.0. The two conventions are each correct in one regime, which is why the
choice is made per factor rather than globally.

Normalization does not add back resolution. A factor cut into four bins is
scored on four bins' worth of information whichever denominator is used. What the
Linfoot branch buys is dropping the entropy-ceiling artifact that *grows* with bin
count — the reported score no longer inflates for a bin count
[the data derived rather than the caller](#the-bin-count-is-a-function-of-the-data-not-a-setting)
settled on, though the mutual information underneath still reflects what those
bins retained.

### The Linfoot branch has a ceiling of its own, and it is divided out

Dropping the entropy denominator does not by itself leave the Linfoot branch
scale-free. Mutual information between two factors cannot exceed the smaller of
their entropies whatever produced the codes, so the largest value the
transformation can return is bounded by their alphabets too — just in the
opposite direction from the entropy ceiling. Left alone, a factor scored against
an identical copy of itself would not read 1.0 unless it had enough levels:

:::{list-table}
:widths: 30 20 50
:header-rows: 1

- - Factor
  - Transformation alone would give
  - Ceiling, `1 - exp(-2·min(H₁, H₂))`
- - 2 bins, equal occupancy
  - 0.750
  - 0.750
- - 3 bins, equal occupancy
  - 0.889
  - 0.889
- - 4 bins, equal occupancy
  - 0.937
  - 0.938
- - 8 bins, equal occupancy
  - 0.984
  - 0.984
- - 16 bins, equal occupancy
  - 0.996
  - 0.996
- - 2 bins, 90/10 split
  - 0.473
  - 0.473
:::

The ceiling is exact, not approximate, and it depends on how full the bins are
rather than only on how many there are — a lopsided binary split would be capped
near 0.47.

That ceiling is the mirror image of the artifact the Linfoot branch exists to
remove. The entropy denominator deflates a score as bins are *added*; this one
would deflate it as bins are *removed*. Both are properties of the cut rather
than of the data, so **`factors` divides by the reachable maximum**, exactly as
the entropy branch divides by the entropy. A duplicated factor reads 1.0 on
either branch, at any cut, however unevenly its bins are filled.

What this does *not* do is add resolution the cut did not keep — see below.

### What a coarse cut costs a correlation

The practical consequence is that `factor_correlation_threshold` — 0.5 by default
— is crossed at very different true dependences depending on how finely each side
was cut. Measured on bivariate normal pairs where the true dependence on the
Linfoot scale is exactly ρ², at n = 5,000:

:::{list-table}
:widths: 30 35 35
:header-rows: 1

- - Both factors cut into
  - `mi_value` reaches 0.5 at a true dependence of
  - Reported at true dependence 0.64
- - 2 bins
  - 0.73
  - 0.42
- - 3 bins
  - 0.64
  - 0.50
- - 5 bins
  - 0.57
  - 0.57
- - 16 bins
  - 0.51
  - 0.62
- - unbinned values, for reference
  - 0.49
  - 0.65
:::

A pair with a true dependence of 0.64 — a strong relationship by any standard —
is flagged at 16 bins and is not flagged at 2. **The coarser the cut, the more
dependence a pair must carry before `is_correlated` fires.**

What remains after the ceiling is divided out is the cut doing its job: two
binary variables cut from a strongly correlated pair genuinely are less dependent
than the values they came from, and reporting less is correct. It does mean a
declared cutoff — which is usually coarse, two or three bins — buys its meaning
at the price of sensitivity, and that a pair sitting just under the threshold is
worth re-reading at a finer cut before concluding anything. The level counts are
worth reading beside the scores, and a factor's level count is the number of
distinct codes its `factor_data` column carries.

### A fine cut costs the same correlation, from the other side

Cutting finer does not recover it either, because both tails lose the signal. A
cut so fine that the contingency table has more cells than the sample has rows
records which cells were hit rather than anything about the factor, and the
chance correction then subtracts nearly everything the counts found. Recovered
dependence for the same ρ = 0.9 pair — a true value of 0.810 — cut into quantiles
at each count, 8 seeds:

:::{list-table}
:widths: 12 11 11 11 11 11 11 11 11
:header-rows: 1

- - Samples
  - k = 2
  - k = 4
  - k = 8
  - k = 16
  - k = 32
  - k = 64
  - k = 128
  - Level budget
- - 200
  - 0.582
  - 0.710
  - **0.729**
  - 0.606
  - 0.302
  - 0.072
  - 0.011
  - 20
- - 1,000
  - 0.576
  - 0.705
  - 0.761
  - **0.770**
  - 0.667
  - 0.376
  - 0.135
  - 31
- - 5,000
  - 0.582
  - 0.705
  - 0.766
  - 0.788
  - **0.789**
  - 0.713
  - 0.458
  - 70
:::

**The count that reads best moves with the sample size**, from about 8 at n = 200
to about 32 at n = 5,000, so one setting is unlikely to suit two datasets of
different sizes. This is the same finding as
[the bin count being a function of the data](#the-bin-count-is-a-function-of-the-data-not-a-setting),
arriving from the other direction: there it is the automatic path that cannot be
pinned, here it is the declared one that cannot be pinned *once*.

**The level budget is a ceiling, not a target.** It is the last column above, and
in every row it sits past the peak — cutting at exactly the budget reports 0.507
where 8 bins report 0.729 at n = 200, and 0.691 against 0.789 at n = 5,000.
Staying under the budget is what keeps a factor's table readable; it is not what
makes the cut a good one. DataEval warns when a **declared** encoding overruns
the budget and deliberately names no count in its place, because the number it
would name is this one.

A cut cannot avoid this trade: the sensitivity a fine cut buys is given back
through the sparsity it creates. The one read with no `k` to choose is the
unbinned one, which is what {attr}`~dataeval.bias.Balance.factor_source` selects
and why it prefers native values wherever nobody declared a cut.

For calibration at the other end, the largest score two **independent** factors
produced over 40 seeds, both carrying 16 levels — the level at which a reported
association means nothing. One seed set, all three regimes, so the columns are
comparable to each other:

:::{list-table}
:widths: 20 27 27 26
:header-rows: 1

- - Samples
  - Binned pair (Linfoot)
  - Own-alphabet pair (entropy)
  - Native values (estimator)
- - 200
  - 0.137
  - 0.034
  - 0.082
- - 1,000
  - 0.058
  - 0.011
  - 0.020
- - 5,000
  - 0.008
  - 0.001
  - 0.003
:::

Every floor sits far below the 0.5 default, so the threshold is unlikely to fire
on noise — but at n = 200 a spurious 0.137 is within reach, and that is a
sample-size limit rather than a binning one.

Reading the two tables together answers the question
{attr}`~dataeval.bias.Balance.factor_source` turns on. A pair read as **native
values** reports 0.5 at a true dependence of 0.49 and carries the *lowest* null
floor of the three — the closest to calibrated, and the one that discards nothing.
Every cut under-reports, and the coarser the cut the further it under-reports:
0.73 at two bins against 0.51 at sixteen. So the span the single threshold has to
cover is set by **how coarsely factors were cut**, not by which channel read
them; reading unbinned values moves the near edge of that span from 0.51 to 0.49,
on a range already 0.24 wide.

That is why `factor_source="auto"` reads native values only where nobody declared
anything. A declared cutoff is a claim, and the under-reporting is that claim
being honored — you asked about *freezing*, not about temperature. Where no one
claimed anything, there is nothing to honour, and the calibrated read is the
better one.

## What the choice determines

Binning is the one band of the taxonomy that records a decision rather than an
observation, and three properties of that decision are what the rest of this
page has been describing.

**The classification is visible in exactly one place.**
{attr}`.Metadata.factor_info` is where a factor's kind and treatment are
reported, and nothing downstream carries either, because {class}`.MetadataLike`
does not expose it. Both ways a factor can land in a row you did not expect — an
integer-coded category read as `discrete`, a rounded continuous quantity read
the same way — are visible nowhere else. Two inputs decide which row a factor
lands in: its dtype, and whether it is named in `continuous_factor_bins`.

**A threshold that comes from the world has to be declared.** Where a factor has
a value at which the world changes — freezing, sunset, a detection-size floor —
no method that places edges by reading values can find it, so the factor gets a
cut describing its distribution instead. `continuous_factor_bins`,
`factor_levels` and `encoding` are what carry such a threshold in, and the only
inputs that fix a factor's vocabulary across datasets rather than letting each
draw derive its own count and edges.

**Sensitivity to the cut is something you measure.** A result says which cut
produced it — {attr}`~dataeval.Metadata.encoding_digest` on every bias result,
and `scored_as` on {attr}`.BalanceOutput.factors` — so two scores can be told
apart. Comparing them is the step that shows whether a conclusion is about the
data or about the cut, and it is one you run yourself.

For the provenance side — what to record so that two runs stay comparable, and
which caveats attach to each evaluator — see
[Validation and Trust](ValidationAndTrust.md#metadata-binning-a-policy-applied-to-every-factor).

## Related concept pages

- [Metadata Levels](MetadataLevels.md) — the level a factor lives at is the level
  its bins are cut at, and why that keeps a bin number the same wherever it is read.
- [Dataset Bias and Coverage](DatasetBias.md) — the evaluators that consume binned
  factors, and what each of them does with the integers.
- [Validation and Trust](ValidationAndTrust.md#metadata-binning-a-policy-applied-to-every-factor)
  — binning as one entry in the per-evaluator caveat inventory, alongside the
  extractor choice it mirrors.
- [Embeddings](Embeddings.md) — the same question on the geometric side: a
  transformation applied before measurement, chosen for the same kind of reason.

## See this in practice

### How-to guides

- [How to control and reuse a factor's binning](../notebooks/h2_control_factor_binning.py)
  — declaring the cuts you mean, ratifying the ones you do not, and carrying the
  whole map to the next collection so two results are comparable.
- [How to bin factors by level](../notebooks/h2_bin_factors_by_level.py) — a worked
  example of binning at a factor's own level, with the numbers that show why the
  alternative distorts.
- [How to configure logging](../notebooks/h2_configure_logging.py) — surfacing the
  auto-binning warnings the default silences.
