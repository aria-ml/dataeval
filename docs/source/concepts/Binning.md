<!-- markdownlint-disable MD051 -->

# Binning

**Binning** converts a continuous factor into discrete categories, which analyses that need a finite set of categories
require. Shannon entropy and a {term}`chi-square test <Chi-Square Test of Independence>` are defined over a
**finite alphabet** (a fixed set of categories a value can take), and a continuous factor does not have one. The choice
of bin boundaries affects the resulting distributions and can influence every downstream statistic computed from the
factor.

DataEval bins by default. {class}`.Metadata` {term}`discretizes <Discretization>` every factor before any evaluator
reads it, so an evaluator receives integer codes and not altitudes, and the bias and diversity results describe the
factors as binned.

This page covers what binning is and how it differs from the relabeling that discrete and categorical factors receive,
why it matters and which evaluators depend on it, how DataEval places the edges, and how the cut shows up in results.
It is the metadata-side counterpart to [Embeddings](Embeddings.md): the feature extractor shapes every geometric
result, and the bins shape every metadata result.

## What binning is

{term}`Binning` cuts a continuous factor into intervals and assigns each interval an integer. A second operation,
digitizing, also produces integers but keeps every distinction, and DataEval chooses between the two for each factor
according to what the factor is.

### Three questions about every factor

Three separate questions are asked of every factor. They are often treated as one, and separating them makes
`factor_info` readable.

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

Read it down a column: each band is one of the questions, so a column traces a
single factor's path from what it is to what the evaluators receive.

1. What is the variable? {term}`Continuous <Continuous Variable>` values can
   fall anywhere in a range, such as brightness, altitude or box area.
   {term}`Discrete <Discrete Variable>` values land on a countable set, such as
   a count of people, an ISO setting or a year.
   {term}`Categorical <Categorical Variable>` values name a group with no
   arithmetic between them, such as `"rainy"`, `"urban"` or a sensor ID. This is a fact
   about the world and does not change with how the data is processed.

2. How is it stored? Numerical or not. It is a fact about the array's dtype, not
   about the variable, and the two need not agree: sensor IDs are categorical
   but stored as numbers. Dtype is read first and decides the label, so
   `continuous` and `discrete` are always numeric and `categorical` never is.

3. What did DataEval produce? Non-negative integers either way, by one of two
   operations that are not equivalent. This band is the only one that describes
   a choice rather than an observation, and it is the one this page is about.

The middle column shows why the words are not interchangeable: a discrete numeric factor is numerical and not
continuous, and it needs no cutting because it already has a finite alphabet.

### Binning compared with digitizing

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
  - Lossless. The map is one-to-one and invertible
  - Lossy by design. Many values share one code, which is what gives the factor
    a finite alphabet
- - Choice involved
  - None. The distinct values are whatever they are
  - The bin edges. Which ones are right depends on the question being asked
- - Applies to
  - Categorical factors, and discrete factors the sample can afford
  - Continuous factors, and discrete numeric factors carrying more levels than the
    sample supports
:::

Digitizing is a relabeling that any downstream count would have produced anyway. Binning is a summary: it trades
resolution for the finite alphabet the evaluators need, and how much resolution it trades depends on where the edges
fall. Every decision discussed on this page concerns binning and not digitizing.

### Reading `factor_info`

The diagram's three kinds map one-to-one onto what {attr}`.Metadata.factor_info`
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
  - dropped
  - none
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

Because dtype decides the label, a categorical factor stored as integers, such
as sensor IDs or encoded weather codes, is reported as `discrete`. The
evaluators are unaffected: both digitize identically and both reach them as
`is_binned=False`. Your own code is not, so anything selecting factors by
`factor_type` will miss it. Dtype is the only input to that label, so the same
factor stored as strings reports as `categorical`.

There is a fourth path. Naming a factor in `continuous_factor_bins` sends it
down the first row regardless of what its numeric values mean, marking it
continuous and binned even if they are categorical codes. A non-numeric column
named there raises `TypeError` rather than being coerced. That is the intended
mechanism for a numeric factor you want cut on edges of your own, so it is
worth naming only factors you mean to treat as continuous: the label holds for
the life of that {class}`.Metadata`.

### What a binned factor represents

Binning does more than relabel a continuous factor. It moves the factor along the first band, toward the categorical
end:

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

A binned factor lands in between: stored as a number, ordered like a number, and
meaning a category. Bin `3` is not three of anything, and bin `1` to bin `3` is
not twice bin `1` to bin `2` unless the edges happen to be equally spaced.

Downstream code reads the number, because the number is nearly all it gets.
Evaluators reach factors through {class}`.MetadataLike`, whose four required
members are `factor_names`, `factor_data`, `class_labels` and `is_binned`.
`factor_info` is not one of them. Everything a factor's treatment contributes is
that one bool, which is `True` for a binned continuous factor:

- {func}`.split_dataset` groups on bin index, so two adjacent bins are as
  different as two distant ones.
- A missing value's bin (see
  [below](#missing-values-get-a-bin-with-a-position-but-no-meaning)) sits above
  the highest observed bin, so anything reading position as magnitude reads
  "missing" as "large".

The mapping is one-way: a bin index does not carry the altitude back. The edges therefore set the vocabulary in
which every metadata result is expressed.

## Why it matters

Binning is what makes the measurement possible. Entropy, a contingency table, a
group identity: each needs a finite set of categories to count into, and a
continuous factor supplies none. Cutting one into intervals is what puts
altitude, brightness and box area on the same footing as weather and city, so a
single set of evaluators can read all of them.

The edges decide what the measurement can see. Two values in the same bin are
one value to everything downstream, and two in different bins are different, no
closer for being adjacent. This is the intended effect of summarizing, and it is also why
edges placed at the boundaries relevant to the question give more specific answers
than edges placed elsewhere.

How much each point matters depends on the consumer.

### Which consumers require it

Not every consumer needs binning, and the ones that do need it for different
reasons: some mathematically, and one only where a cut was chosen.

:::{list-table}
:widths: 26 18 56
:header-rows: 1

- - Consumer
  - Binning is
  - Why
- - {class}`.Diversity`
  - Required
  - Shannon and Simpson indices are sums over the probabilities of a finite set
    of categories. A continuous factor has no such set.
- - {func}`.parity` ⚠️, {class}`.Parity` ⚠️
  - Required
  - A contingency table needs cells to count into. Coarser bins avoid the sparse
    cells reported as `too_thin` in {attr}`.ParityOutput.cells`, at the cost of
    resolution.
- - {func}`.split_dataset` with `split_on`
  - Floats and declared cuts only
  - Grouping needs discrete group identities, and a factor's recorded values are
    already that, so a video or a site is one group however many there are. A
    float factor, whose every value would be a group of its own, and a factor
    with a declared cut are grouped by their bins. On object detection datasets
    each group is split as one unit, so the split fractions count groups, not
    images.
- - {class}`.Balance`
  - Only for a cut somebody chose
  - It reads each factor either as codes (bin and category indices) or as the
    measured values, as `factor_source` selects, and `"auto"` is the default.
    A cut that was declared, requested by count or accepted is read as codes,
    and a contingency table runs over those bins. A factor whose cut nobody
    chose is read from its measured values, which go to a neighbor-based
    estimator that has no bins and is not corrected for chance. A category is
    always read as codes. See
    [below](BinEdges.md#binning-reaches-the-three-outputs-differently).
- - {func}`.factor_deviation`, {func}`.factor_predictors`
  - Not applied
  - Both take a plain mapping of factor name to raw array and never touch
    `factor_data`, so they see the values you measured.
- - {class}`.Outliers` over metadata factors (in development)
  - Never applied
  - A threshold over bin codes measures the bin edges, not the values. It
    reads raw values only, and the continuous/discrete heuristic below does
    not decide which factors it tests. Re-binning a metadata does not change
    an outlier result. See
    [Outlier detection over annotation and metadata factors](DataIntegrity.md#outlier-detection-over-annotation-and-metadata-factors).
:::

The estimator that handles a column depends on the column's own values: integral
values are codes and are tabulated, and anything with a fractional part was
measured and goes to the neighbor-based estimator. `Balance.num_neighbors` tunes
that estimator, so it has an effect only on the measured columns. With
`factor_source="coded"`, which reproduces the behavior of releases before 1.1, no
column reaches the estimator and `num_neighbors` has no effect. With `"values"`
every factor is read from its measured values and every declared cut is ignored.

## How DataEval bins

With `continuous_factor_bins=None`, the default, {class}`.Metadata` classifies
each factor on its own. Non-numeric columns are ordinal-encoded immediately,
unless they carry more distinct values than
[the level budget](#a-discrete-factor-can-still-be-binned) allows, in which case
they are dropped. Numeric columns go to {func}`.is_continuous`, and its verdict
picks the treatment.

### The continuous/discrete heuristic

{func}`.is_continuous` is a heuristic, not a hypothesis test: it reports no
p-value and has no stated error rate. It combines three signals:

1. Near-neighbor uniformity: under a continuous distribution, a point is equally
   likely to lie anywhere between its two neighbors. Normalizing those positions
   to [0, 1] gives a distribution that is near-uniform for continuous data and
   lumpy for discrete data; the Wasserstein distance from uniform measures the
   gap, against a threshold of `0.5 / sqrt(n)`.
2. Duplicate fraction: continuous values drawn into floating point collide with
   probability zero, so exact duplicates are evidence of discrete support.
3. Lattice (GCD) test: discrete values on a regular grid have gaps that are
   near-integer multiples of a base unit. This catches integer-valued
   distributions before enough collisions accumulate for signal 2 to fire.

A sample is called continuous when signal 1 clears its threshold and neither
secondary signal fires. If a single secondary signal fires, the stricter
`0.3 / sqrt(n)` threshold decides. If both fire, the sample is discrete
regardless. All five constants are tuned empirically rather than derived: the
two Wasserstein thresholds, a `0.005` duplicate tolerance, a `0.85` lattice
cutoff, and a `0.05` near-integer tolerance inside the GCD test. So on data
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
  - None: near-uniform neighbor positions, no collisions, no lattice
- - Integers drawn from 0-99
  - discrete
  - All three: lumpy neighbor positions, duplicates, and a lattice
- - The same normal draws rounded to one decimal
  - discrete
  - All three, created by the rounding alone
:::

That last case is common. Rounding a continuous quantity for storage makes it discrete, so a brightness value
written to one decimal place is read as discrete and, within the level budget, digitized into one category per
distinct value rather than binned.

### Floors on the continuous/discrete verdict

- Fewer than 20 observations → discrete, unconditionally. On a small dataset
  every distinct float becomes its own category, and the factor ends up nearly
  one-to-one with the sample index.
- Fewer than 3 distinct values → discrete. The near-neighbor construction needs
  interior points.

Both counts are taken after dropping non-finite values, so a factor that is
mostly missing falls under the floor and is called discrete. Infinities are
dropped alongside `NaN`: a missing value has no position on the line and an
infinity no finite distance to its neighbors, so neither carries the spacing
information the test reads. These are the same values the bin edges are placed
on, so the verdict and the edges are read off one set of numbers.

### A discrete factor can still be binned

A discrete verdict says the factor's support is countable. It does not imply
the support is small, and an integer factor can be discrete while taking a value
per entity, such as pixel areas, epoch seconds, or a non-reserved
identifier-like column. Scoring one of those a value at a time can make it
report a correlation with anything it is measured against, because a contingency
table with more cells than observations records which cells were hit rather than
anything about the factor. The datum's own `id` is never a factor to begin with:
it is reserved and carried on the `item_id` column, so this question does not
arise for it.

So no numeric factor carries more levels than the sample's level budget,
`max(20, sqrt(n))`, with `n` counted at the factor's own level.

`sqrt(n)` is the square-root rule for histogram bin counts; the floor of 20
keeps an ordinary categorical factor intact on a small sample. Both are standard
rules of thumb rather than derived quantities. A discrete factor over that
budget is binned rather than ordinal-encoded (reported as
`factor_type="discrete"` with `is_binned=True`), and the same budget caps the
bin count on the continuous path.

The budget marks the point past which a contingency table stops describing the
data. It is not the count that reads best, which
sits well under it and moves with the sample size
([measured here](BinEdges.md#effect-of-fine-binning-on-the-correlation-score)). A
factor sitting just inside the budget clears what the budget guards against and
is still cut more finely than the sample rewards.

A different line binds non-numeric columns, and it is not the budget. The budget
answers "how many cells can this sample fill", which is the right question for
choosing a bin count and the wrong one for deciding whether a column is a factor
at all: twenty-five cities over a hundred images overruns it and is a perfectly
good factor. What is asked instead is whether the column names its rows rather
than grouping them: a value per row is not a category, every contingency table
over it is a table of ones, and there is no order along a set of strings to cut
it into fewer. Such a column is dropped, leaving {attr}`.Metadata.factor_names`
and gaining a `"cardinality_over_budget"` entry in
{attr}`.Metadata.dropped_factors`.

A numeric column at the same cardinality is binned and kept: that asymmetry is
intended, and it extends to timestamps, which are ordered and so cut into
intervals like any other measured quantity. Map the values onto a smaller
vocabulary, such as a coarser taxonomy, a prefix or a lookup, to keep a column
that is meant to be categorical.

A merely wide vocabulary is not dropped. A factor whose levels are thin for the
sample is what {attr}`.ParityOutput.cells` exists to report, and removing the
factor would foreclose the mechanism that says so.

That report names its levels rather than numbering them, and
{meth}`.Metadata.code_names` is the same lookup asked directly: what code 3 of
`illum_lux` means, without going through an evaluator to find out. Names come
from the record, so a declared cutoff reads as `< 0` rather than as whatever the
coldest row happened to be. A factor cut at `[-np.inf, 0.0, 10.0, np.inf]` reads
back as `< 0`, `[0, 10)` and `>= 10`, and every bin the cut declares is named
whether or not this sample reached it. Anything rendering a factor's codes
itself, such as a report, an export or a plot axis, gets the strings the
evaluators use rather than approximating them.

Both the automatic binning and the drop raise a `UserWarning`, naming every factor
affected, in addition to the per-factor detail on the `dataeval.metadata` logger.

### Level at which a factor is classified

On {term}`object detection <Object Detection>` metadata, factors live at
different [levels](MetadataLevels.md). A factor recorded at the unit level is
classified and binned on its per-image values, not on the copy replicated onto
each detection. Otherwise the replication would look like duplicates and signals
2 and 3 would call a continuous factor discrete. A factor recorded at the
instance level is scored on every detection, because there the repeats are
genuine observations.

The 20-sample floor applies to whichever set the factor is judged on, so a
unit-level factor on fewer than 20 images is always called discrete no matter
how many detections those images carry. {ref}`binning-levels` works through the
consequences.

### Choosing a method

`auto_bin_method` selects how edges are placed for factors judged continuous and
for
[discrete factors that overrun the level budget](#a-discrete-factor-can-still-be-binned).
All three place edges from the finite values only, then set the outermost edges
to ±∞ so that finite values beyond the observed range would still land in an end
bin rather than forming their own. That absorbs `-inf`; `+inf` escapes it, for
the reason given
[below](#missing-values-get-a-bin-with-a-position-but-no-meaning).

:::{list-table}
:widths: 20 40 40
:header-rows: 1

- - Method
  - Preserves
  - Costs
- - `"uniform_width"` (default)
  - The shape of the distribution. Equal-width bins mean bar height is
    proportional to density, so a histogram of the result looks like the data.
  - Statistical power is unequal across bins: tail bins may hold a handful of
    samples while the mode holds hundreds.
- - `"uniform_count"`
  - Power per bin. Quantile edges give every bin roughly equal occupancy, which
    is what contingency tests want.
  - Shape. Bin width now varies inversely with density, so equal bins do not
    mean equal ranges and the result cannot be read as a histogram.
- - `"clusters"`
  - Natural gaps. Edges are derived from DataEval's HDBSCAN port, so multimodal
    factors get cuts between the modes rather than through them.
  - Stability. It inherits the clusterer's sensitivity, and there is no
    guarantee the number of modes is stable across datasets.
:::

Those trade-offs line up with particular consumers. `"uniform_count"` is the one
that matches what {func}`.parity` wants, since equal occupancy is what keeps
cells from being too sparse to test. `"clusters"` is the only one that will not cut
through a mode on a visibly multimodal factor. The default trades both for a
result that reads back as a histogram of the data.

All three place edges by reading the values, so where a factor has a threshold
that comes from the world rather than the data, the choice of method is not the
lever that reaches it. `continuous_factor_bins` is.

## Choosing bin edges

Automatic binning reads the values and nothing else, which is a reasonable default: many factors have no privileged cut
point. Some factors do have one, a value at which the meaning of the factor changes, and the distribution cannot reveal
it, so it has to be declared. A declared cut also changes how much association a score reports. [Choosing bin
edges](BinEdges.md) works through a temperature example, shows how the cut reaches each output of {class}`.Balance`, and
measures what coarse and fine cuts cost a correlation.

## Limitations of automatic binning

The automatic path derives its cut from the values. The limitations below describe the consequences, so that factors
that are better declared can be recognized. Each is visible before any score computed from the factor is read.

### The bin count is a function of the data, not a setting

`"uniform_width"` does not take a bin count. It starts from NumPy's
`histogram(bins="auto")`, then reduces the count, at most 20 times, while any
non-empty bin holds fewer than 10 samples. The 10-sample floor keeps every bin
populated enough to count into, and the tails trip it readily, so the count that
comes out is a property of the particular draw:

```text
500 draws from a standard normal, 200 different seeds

bins     3    4    5    6    7    8   10
seeds   12   63   59   39   20    6    1
```

Between 3 and 10 bins for the same distribution at the same sample size. The
consequence is not that any one of those is wrong. It is that the same factor
measured on two datasets can be cut into different numbers of bins, so binned
factor values, and every score computed from them, are not comparable across
runs. `"uniform_count"` inherits this count and moves only the edges.

The [level budget](#a-discrete-factor-can-still-be-binned) caps the count from
above and the reduction floors it at two, so the result is bounded, but a bound
is not a setting, and anything between those two ends is still chosen by the
draw.

### A factor can reduce to two bins

The reduction stops at two bins, so a factor never comes back constant. Two is
the floor, and a small sample with a far-off tail can reach it: the
outlier stretches the range, the interior bins come out sparse, and the
reduction runs all the way down: 20 to 80 draws from a standard normal plus
three outliers near 50 came back as two bins on every one of 2,000 seeds.

A continuous factor reduced to a binary split still answers every question put
to it (a {class}`.Diversity` index over two categories, a {class}`.Parity` table
with two columns, two groups under {func}`.split_dataset`) at the resolution two
categories allow. On this shape the cut lands between the mode and the outliers,
which is where the sample supports one, and a binary answer to a question about
a spread of values is usually not the one you wanted.

The level count is what tells you. `factor_data` columns follow `factor_info`
order, so the level count of a binned factor is worth checking before reading
anything computed from it. A column carrying two distinct codes is a factor that
arrived as a binary split, whatever resolution the values behind it had.

### Explicit edge lists produce more bins than edges

`continuous_factor_bins` accepts either an integer count or a sequence of edges,
and they behave differently at the boundaries. An integer gets the ±∞ treatment
described above. An explicit sequence is used verbatim, so values outside its
range fall into open-ended bins on either side. For a factor holding `-5.0`,
`5.0`, `15.0` and `25.0`:

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
outer limits, which is the treatment an integer count receives automatically and
a sequence does not.

### Missing values get a bin with a position but no meaning

A `NaN` is not a small value, a large value, or a value between two edges, so it
is given a bin of its own above the bins holding observed values (automatic or
explicit, on binned and digitized factors alike).

- Edges are placed on the finite values alone, so a missing value does not shift
  where the cuts fall.
- `+inf` is not absorbed. `-inf` lands in the first bin, but because the top
  edge is `+inf` and digitizing is right-open, `+inf` falls past it into a bin
  of its own, one above the highest finite bin. The missing bin then sits one
  above that, so a factor carrying both ends up with two categories that are not
  values.
- The position is an artifact. The missing bin sits at the top because the codes
  have to be contiguous (a gap would show up downstream as an empty category),
  not because missing is large. Anything treating the factor as ordinal reads
  that position as a value.

So where the missing rate is material, part of any finding about that factor is
a finding about its absence, and the output does not separate the two.

### A nanosecond timestamp is read to 256 ns

A capture time is totally ordered, so it cuts into intervals like a number.
DataEval reads one as a float so that `NaT` can travel as `NaN` and be given the
missing bin like any other gap. A float64 carries 53 bits of mantissa, and near
the current epoch (~1.8e18 nanoseconds) consecutive representable values are 256
ns apart, so two capture times closer together than that become one number.

`datetime64[us]` and `datetime64[ms]` are exact. Only `[ns]` (polars' and
pandas' default unit) is affected, and only below a resolution no bin edge on a
capture time is placed at. What it does change is the distinct count of such a
column, which feeds the continuous/discrete verdict and the near-uniqueness test
that drops a near-unique non-reserved column. If a sub-microsecond difference is
one you need kept, cast the column to `datetime64[us]` before handing it to
{meth}`.Metadata.add_factors`.

### Warnings from the automatic path

Structuring raises a `UserWarning` naming every factor it binned on the caller's
behalf, because the per-factor `WARNING` it also writes to the
`dataeval.metadata` logger reaches nobody by default: DataEval attaches a
`NullHandler` there, which suppresses Python's last-resort stderr handler. One
aggregated warning rather than one per factor, so it stays readable on a dataset
carrying many of them.

The record survives the warning. Each factor's {attr}`.FactorInfo.encoding` says
where its cuts fell and who chose them: `provenance="derived"` where DataEval
chose both the count and the placement, against `"count"`, `"edges"` or
`"declared"` where the caller did. So a run that binned automatically is
distinguishable from one handed explicit edges long after the warning scrolled
past. `repr(md)` reports the count as `auto_encoded=N`.

## What the choice determines

Binning is the one band of the taxonomy that records a decision rather than an
observation, and three properties of that decision are what this page and
[Choosing bin edges](BinEdges.md) describe.

The classification is visible in a single place. {attr}`.Metadata.factor_info`
is where a factor's kind and treatment are reported, and nothing downstream
carries either, because {class}`.MetadataLike` does not expose it. Both ways a
factor can land in a row you did not expect (an integer-coded category read as
`discrete`, a rounded continuous quantity read the same way) are visible nowhere
else. Two inputs decide which row a factor lands in: its dtype, and whether it
is named in `continuous_factor_bins`.

A threshold that comes from the world has to be declared. Where a factor has a
value at which the world changes (freezing, sunset, a detection-size floor), no
method that places edges by reading values can find it, so the factor gets a cut
describing its distribution instead. `continuous_factor_bins`, `factor_levels`
and `encoding` are what carry such a threshold in, and the only inputs that fix
a factor's vocabulary across datasets rather than letting each draw derive its
own count and edges.

Sensitivity to the cut is something you measure. A result says which cut
produced it: {attr}`~dataeval.Metadata.encoding_digest` on every bias result,
and `scored_as` on {attr}`.BalanceOutput.factors`. So two scores can be told
apart. Comparing them is the step that shows whether a conclusion is about the
data or about the cut, and it is one you run yourself.

For the provenance side (what to record so that two runs stay comparable, and
which caveats attach to each evaluator), see
[Validation and trust](ValidationAndTrust.md#metadata-binning-a-policy-applied-to-every-factor).

## Related concept pages

- [Metadata levels](MetadataLevels.md): the level a factor lives at is the level
  its bins are cut at, and why that keeps a bin number the same wherever it is read
- [Dataset bias and coverage](DatasetBias.md): the evaluators that consume binned
  factors, and what each of them does with the integers
- [Validation and trust](ValidationAndTrust.md#metadata-binning-a-policy-applied-to-every-factor):
  binning as one entry in the per-evaluator caveat inventory, alongside the
  extractor choice it mirrors
- [Choosing bin edges](BinEdges.md): declaring edges for a question, and what a cut costs in a score
- [Embeddings](Embeddings.md): the same question on the geometric side, a
  transformation applied before measurement and chosen for the same kind of reason

## See this in practice

### How-to guides

- [Control and reuse a factor's binning](../notebooks/h2_control_factor_binning.py):
  declaring the cuts you mean, ratifying the ones you do not, and carrying the
  whole map to the next collection so two results are comparable
- [Configure logging with DataEval](../notebooks/h2_configure_logging.py): surfacing the
  auto-binning warnings the default silences

### Tutorials

- [Analyze a dataset across its levels](../notebooks/tt_analyze_across_levels.py#binning-follows-the-level):
  binning at a factor's own level, checked on an aerial detection dataset
