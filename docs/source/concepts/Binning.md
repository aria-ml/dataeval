<!-- markdownlint-disable MD051 -->

# Binning

Shannon entropy is not defined over altitude in meters, and neither is a
{term}`chi-square test <Chi-Square Test of Independence>`. Both are defined over
a **finite alphabet** — a fixed set of categories a value can take — and a
continuous factor does not have one. So before either can run, the factor has to
be cut into pieces.

That cutting is **binning**, and DataEval applies it without being asked.
{class}`.Metadata` {term}`discretizes <Discretization>` every factor before any
evaluator sees it: by the time one reads a factor, it is looking at integer
codes, not at altitudes. The bias and diversity results you get back describe
your factors *as binned*.

This page explains what that means — what kinds of factor exist, which of them
survive discretization intact and which do not, how DataEval decides, and where
the choice leaks into results. It is the metadata-side counterpart to
[Embeddings](Embeddings.md): where the choice of feature extractor is the
largest uncontrolled variable in the geometric half of the library, binning is
the largest one in the metadata half.

## A worked example: temperature

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
`Parity` test still runs and still returns a number. What it compares is a
category that mixes iced and wet roads, so the number is not an answer to the
question you asked, and nothing in its output says so.

Switching method does not rescue it:

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
that axis is a phase change. The information needed to bin this factor correctly
is not in the factor.

It has to arrive from outside the factor, through `continuous_factor_bins`:

```python
md = Metadata(dataset, continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})

# bin 1   n=103   [-18.6,   0.0)   100% freezing
# bin 2   n=297   [  0.0,  22.4]     0% freezing
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
than set by you. Two separate things broke:

- **Contextually**, the freezing signal has all but vanished — nine images out
  of 400 — and it is *still* not isolated. Bin 1 mixes freezing with mild, the
  same failure winter's bin 2 had, at a different place on the axis.
- **Comparatively**, the two collections no longer share a vocabulary.
  `temp_c = 1` means "hard freeze, -18.6 to -5.1" in the winter data and "-4.6
  to 2.7, mostly above freezing" in the spring data. The integer is the same
  and the meaning is not.

That second failure is the one that spreads. A {class}`.Diversity` index over a
three-letter alphabet is not on the same scale as one over a four-letter
alphabet, and a {class}`.Balance` score that moved between the two collections
may be reporting a change in the road or only a change in where the cuts fell —
with nothing in either output to tell you which. Anything that reads
`factor_data` inherits this; {func}`.factor_deviation` and
{func}`.factor_predictors` escape it precisely because they read the raw values
instead.

Explicit edges fix both failures at once. `[-np.inf, 0.0, np.inf]` gives the
spring data bin 1 = 9 images below freezing, bin 2 = 391 above — the same two
categories, meaning the same two things, as the winter data. The factor becomes
comparable across collections because you fixed its vocabulary instead of
letting each dataset invent its own. [The bin count is a function of the data,
not a setting](#the-bin-count-is-a-function-of-the-data-not-a-setting) has the
measurements behind this.

### The same problem elsewhere

The same shape of problem recurs whenever a factor has a threshold that comes
from the world rather than from the data: dawn and dusk in a time-of-day factor,
a sensor's rated range in a distance factor, the resolution below which an object
is unrecognizable in a box-area factor. **In each case, the default will bin the
factor successfully and silently answer a different question than the one you
asked.** The rest of this page is about recognizing when that has happened.

## Three questions about every factor

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

Digitizing and binning both hand the evaluators integers. Only one of them loses
anything:

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
  - **Lossy.** Many values collapse to one code, irreversibly
- - Choice involved
  - None. The distinct values are whatever they are
  - **The bin edges**, and there is no correct answer independent of the
    question being asked
- - Applies to
  - Categorical factors, and discrete factors the sample can afford
  - Continuous factors, and discrete numeric factors carrying more levels than the
    sample supports
:::

Digitizing is bookkeeping — a relabeling that any downstream count would have
produced anyway. Binning destroys information that cannot be recovered
downstream, and how much it destroys depends on a decision made for you.
**Everything difficult in what follows is a consequence of binning, not
digitizing.**

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
:::

```python
for name, info in md.factor_info.items():
    print(f"{name:12s} {info.factor_type:11s} binned={info.is_binned}")

# altitude_m   continuous  binned=True     ← cut into intervals, lossy
# epoch_s      discrete    binned=True     ← too many levels to afford, cut anyway
# n_people     discrete    binned=False    ← relabeled, lossless
# weather      categorical binned=False    ← relabeled, lossless
```

`factor_type` and `is_binned` are therefore independent: the type records what the
factor is, the companion flag records what was done to it, and `discrete` appears
against both. Selecting the factors that lost information means reading
`is_binned`, not `factor_type`.

The discretized values live in a **companion column** rather than replacing the
factor: `altitude_m↕` holds the bin indices, `weather#` the ordinal codes.
{attr}`.Metadata.factor_data` is the view built from those companions and is
what the bias evaluators read; {meth}`.Metadata.rows_at` carries the native
values, so nothing is destroyed in the dataframe — only in what the evaluators
see.

Because dtype decides the label, a **categorical factor stored as integers** —
sensor IDs, encoded weather codes — is reported as `discrete`. The evaluators
are unaffected: both digitize identically and both reach them as
`is_binned=False`. Your own code is not, so anything selecting factors by
`factor_type` will miss it. Dtype is the only input to that label, so the same
factor stored as strings reports as `categorical`.

There is a fourth path. Naming a factor in `continuous_factor_bins` sends it
down the first row **regardless of what its numeric values mean**, marking it
continuous and binned even if they are categorical codes. A non-numeric column
named there raises `TypeError` rather than being coerced. That is the intended escape
hatch, and also a way to mislabel a factor for the life of that
{class}`.Metadata`.

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

The arrow only runs left to right. You cannot recover altitude from bin index,
which is why a binning you did not choose is not a detail — it is a
lossy transform applied to your data before you saw the results.

## When binning is required, and when it is only a convention

Not every consumer needs it. One of the most-used evaluators bins for no
mathematical reason at all.

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
    contingency-table estimator runs on every factor and the binning upstream is
    the whole of the loss. What `Balance` forwards is not *which estimator* to use
    — that is read from the column's values — but whether each factor's set of
    values is its own or an artifact of the cuts, which decides how the `factors`
    DataFrame is normalized. See [below](#binning-reaches-the-three-outputs-differently).
- - {func}`.factor_deviation`, {func}`.factor_predictors`
  - **Not applied**
  - Both take a plain mapping of factor name to raw array and never touch
    `factor_data`, so they see the values you measured.
:::

The `Balance` row is a property of what it passes, not of the estimator it calls.
{func}`.mutual_info` given raw values still routes the measured columns to the
neighbor-based estimator; the loss happened upstream in `factor_data`. One visible
consequence: `Balance.num_neighbors` tunes a neighborhood that only the measured
path consults, which `Balance` cannot produce — which is why it is deprecated and
warns when set.

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

None of this recovers what binning destroyed. A factor cut into four bins is
scored on four bins' worth of information whichever denominator is used. What the
Linfoot branch buys is dropping the entropy-ceiling artifact that *grows* with bin
count — the reported score no longer inflates for a bin count
[the caller did not choose](#the-bin-count-is-a-function-of-the-data-not-a-setting),
though the mutual information underneath still reflects what those bins retained.

### The Linfoot branch has a ceiling of its own

Dropping the entropy ceiling does not leave the Linfoot branch scale-free. Mutual
information between two factors cannot exceed the smaller of their entropies, so
the largest Linfoot value a pair can reach is bounded by their alphabets too —
just in the opposite direction. A factor scored against an identical copy of
itself does not read 1.0 unless it has enough levels:

:::{list-table}
:widths: 30 20 50
:header-rows: 1

- - Factor
  - Duplicate reads
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
rather than only on how many there are — a lopsided binary split is capped near
0.47. Because
[the automatic path lands between 3 and 10 bins](#the-bin-count-is-a-function-of-the-data-not-a-setting),
an auto-binned pair is typically capped somewhere between 0.89 and 0.99, and a
factor that
[collapsed to a binary split](#a-factor-can-collapse-to-two-bins) is capped at
0.75 — so two identical factors can never report as identical.

This is the mirror image of the artifact the Linfoot branch exists to remove. The
entropy denominator deflates a score as bins are *added*; the Linfoot ceiling
deflates it as bins are *removed*. Both are properties of the cut rather than of
the data.

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
  - 0.88
  - 0.32
- - 3 bins
  - 0.71
  - 0.44
- - 5 bins
  - 0.59
  - 0.54
- - 16 bins
  - 0.52
  - 0.62
- - unbinned values, for reference
  - 0.49
  - 0.65
:::

A pair with a true dependence of 0.64 — a strong relationship by any standard —
is flagged at 16 bins and is not flagged at 2 or 3. **The coarser the cut, the
more dependence a pair must carry before `is_correlated` fires.**

Most of that is the cut doing its job: two binary variables cut from a strongly
correlated pair genuinely are less dependent than the values they came from, and
reporting less is correct. Part of it is the ceiling above, which is an artifact.
Either way, the number to compare a `factors` score against is not 1.0 but the
ceiling for that pair, and the level counts are worth reading beside the scores:

```python
[len(np.unique(md.factor_data[:, i])) for i in range(md.factor_data.shape[1])]
```

For calibration at the other end, the largest score two **independent** factors
produced over 40 seeds, both carrying 16 levels — the level at which a reported
association means nothing:

:::{list-table}
:widths: 25 37 38
:header-rows: 1

- - Samples
  - Binned pair (Linfoot)
  - Own-alphabet pair (entropy)
- - 200
  - 0.135
  - 0.034
- - 1,000
  - 0.044
  - 0.009
- - 5,000
  - 0.009
  - 0.002
:::

Both floors sit far below the 0.5 default, so the threshold is not at risk of
firing on noise — but at n = 200 a spurious 0.135 is within reach, and that is a
sample-size limit rather than a binning one.

## How DataEval decides

With `continuous_factor_bins=None` — the default — {class}`.Metadata` classifies
each factor without being asked. Non-numeric columns are ordinal-encoded
immediately. Numeric columns go to {func}`.is_continuous`, and its verdict picks
the treatment.

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
regardless. **All five constants are tuned values carrying no derivation** — the
two Wasserstein thresholds, a `0.005` duplicate tolerance, a `0.85` lattice
cutoff, and a `0.05` near-integer tolerance inside the GCD test — so the verdict
is worth checking on data whose support you already know.
{func}`.is_continuous` is public, and its verdict on an array is the same one
{class}`.Metadata` acts on:

```python
from dataeval.core import is_continuous

is_continuous(rng.normal(size=200))  # True
is_continuous(rng.integers(0, 100, size=200))  # False — lumpy NNN, duplicates, lattice
is_continuous(np.round(rng.normal(size=200), 1))  # False — rounding creates all three
```

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

So no numeric factor carries more levels than the sample's level budget:

```python
max(20, sqrt(n))  # n at the factor's own level
```

`sqrt(n)` is the square-root rule for histogram bin counts; the floor of 20 keeps
an ordinary categorical factor intact on a small sample. Both are rules of thumb,
not derived quantities. A discrete factor over that budget is binned rather than
ordinal-encoded — reported as `factor_type="discrete"` with `is_binned=True` — and
the same budget caps the bin count on the continuous path. The cap applies to
numeric columns only: a non-numeric column keeps one category per distinct value
however many there are, since there is no axis along which to merge them.

Like automatic binning generally, this announces itself only as a `WARNING` on the
`dataeval.metadata` logger.

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

## Choosing a method

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
through a mode on a visibly multimodal factor. The default gives up both to keep
the result readable as a histogram.

All three place edges by reading the values, so where a factor has a threshold
that comes from the world rather than the data, the choice of method is not the
decision that matters. Setting `continuous_factor_bins` is.

## Pitfalls of automatic binning

### The bin count is a function of the data, not a setting

`"uniform_width"` does not take a bin count. It starts from NumPy's
`histogram(bins="auto")`, then *reduces* the count — at most 20 times — while
any non-empty bin holds fewer than 10 samples. The 10-sample floor is aggressive
and the tails keep tripping it, so the count that comes out is a property of the
particular draw:

```python
# 500 draws from a standard normal, 200 different seeds
Counter({4: 63, 5: 59, 6: 39, 7: 20, 3: 12, 8: 6, 10: 1})
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

### A factor can collapse to two bins

The reduction stops at two bins, so a factor cannot come back constant and
information-free. Two is the floor, not a reassuring one: a small sample with a
far-off tail hits it every time, and the reduction has nowhere left to go.

```python
# 20-80 draws from a standard normal plus 3 outliers near 50, 2000 seeds
Counter({2: 2000})
```

A continuous factor reduced to a **binary split** still answers every question
put to it — a {class}`.Diversity` index over two categories, a {class}`.Parity`
table with two columns, two groups under {func}`.split_dataset` — and the
resolution it had is gone. The cut sits wherever the reduction left it, which on
this shape is between the mode and the outliers rather than anywhere meaningful.
Nothing in the output says the factor arrived this coarse.

It is visible as a shape. `factor_data` columns follow `factor_info` order, so
the level count of a binned factor is worth checking before reading anything
computed from it:

```python
len(np.unique(md.factor_data[:, i]))  # 2 — the i-th factor is a binary split
```

### Explicit edge lists get bins you did not ask for

`continuous_factor_bins` accepts either an integer count or a sequence of edges,
and they behave differently at the boundaries. An **integer** gets the ±∞
treatment described above. An explicit **sequence** is used verbatim, so values
outside its range fall into open-ended bins on either side:

```python
# a factor holding [-5.0, 5.0, 15.0, 25.0]
Metadata(ds, continuous_factor_bins={"f": [0, 10, 20]})  # → bins [0, 1, 2, 3]
Metadata(ds, continuous_factor_bins={"f": 3})  # → bins [1, 2, 3, 3]
```

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

### The automatic path announces itself only in logs

Each auto-binned factor emits a `WARNING` on the `dataeval.metadata` logger —
the only record that its edges were derived rather than given. DataEval attaches
a `NullHandler` to that logger, so the default is silence, and a run that binned
every factor automatically is indistinguishable from one handed explicit edges
for all of them.

## What the choice determines

Binning is the one band of the taxonomy that records a decision rather than an
observation, and three properties of that decision are what the rest of this
page has been describing.

**The classification is visible in exactly one place.**
{attr}`.Metadata.factor_info` is where a factor's kind and treatment are
reported, and nothing downstream carries either, because {class}`.MetadataLike`
does not expose it. Both ways of landing in the wrong row — an integer-coded
category read as `discrete`, a rounded continuous quantity read the same way —
are silent everywhere else. Two inputs decide which row a factor lands in: its
dtype, and whether it is named in `continuous_factor_bins`.

**The thresholds that matter are not in the data.** Where a factor has a value
at which the world changes — freezing, sunset, a detection-size floor — no
method that places edges by reading values can find it, and the factor is binned
successfully anyway. `continuous_factor_bins` is the only input that carries
such a threshold in, and the only one that fixes a factor's vocabulary across
datasets rather than letting each draw derive its own count and edges.

**Nothing marks a result as binning-sensitive.** A score computed at one setting
and a score computed at another are both returned without reference to the cut
behind them. The difference between the two is the only available evidence that
a conclusion is about the binning rather than about the data, and DataEval
neither computes it nor warns that it might matter.

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
  extractor problem it mirrors.
- [Embeddings](Embeddings.md) — the same question on the geometric side: a
  transformation applied before measurement that no result object records.

## See this in practice

### How-to guides

- [How to bin factors by level](../notebooks/h2_bin_factors_by_level.py) — a worked
  example of binning at a factor's own level, with the numbers that show why the
  alternative distorts.
- [How to configure logging](../notebooks/h2_configure_logging.py) — surfacing the
  auto-binning warnings the default silences.
