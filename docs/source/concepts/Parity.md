<!-- markdownlint-disable MD051 -->

# Parity

{class}`.Parity` ⚠️ tests whether metadata factors are associated with the class
labels. It reports a test for each factor as a whole, and a test for each
*cell*, one combination of a factor level and a class, of whether the cell
holds more or fewer rows than it would if the factor and the class were
independent. The evaluator calls the core function {func}`.parity` ⚠️. Both are
experimental, and their API may change or be removed in a later release.

{class}`.Balance` measures the same factor–class dependence as a normalized
mutual information score (see
[Dataset Bias and Coverage](DatasetBias.md#measuring-bias-mutual-information-and-evenness)).
Parity adds a hypothesis test, results for individual cells, and p-values that
account for rows grouped into images or sequences.

## What it answers

### The question

[Binning](Binning.md#a-worked-example-temperature) cuts a temperature factor to
answer one question: are pedestrians under-represented in freezing conditions?
Parity's factor test asks whether the class mix below 0 °C differs from the
class mix above it. Its cell test, which answers the question as asked, checks
whether the pedestrians-below-freezing cell holds fewer images than it would if
pedestrians were as common in freezing conditions as in the dataset as a whole.

Parity answers three questions:

- Is a factor associated with the class?
- Which cells depart from independence?
- Which cells are too thin to judge?

### What it does not measure

Parity does not measure coverage. A dataset that is 99% day and 1% night for
every class has no association:
Cramér's V is 0 and no cell departs from independence, because every class has
the same small share of night images. For a night mission the dataset may still
hold too few night images.

DataEval has no metric yet for shortfall against a target distribution or
against a minimum count per condition. To check counts against a mission's
targets, compare them directly, using the `observed` column of
{attr}`.ParityOutput.cells` or the factor in {class}`.Metadata`.
{class}`.Diversity` shows how unevenly a factor's values are sampled. Parity's
`deficit` is the shortfall against independence.

### The table: observed, expected, lift and deficit

For one factor with $r$ observed levels, a class axis with $K$ observed classes
and $n$ rows, $O_{ij}$ counts the rows at level $i$ with class $j$. If the
class mix were the same at every level, the expected count would be

$$E_{ij} = \frac{n_{i\cdot}\, n_{\cdot j}}{n}$$

the level's row count times the class's share of all rows. Parity reports two
measures of how far a cell departs from $E_{ij}$:

- **lift** $= O_{ij} / E_{ij}$: 1 under independence, 0 for an empty cell, and
  2 for a cell holding twice its expected count.
- **deficit** $= E_{ij} - O_{ij}$: the number of rows the cell is short of
  independence, negative for a surplus. It counts rows, and on detection data
  the rows are detections.

Suppose the 400 road-scene images of the Binning example show 120 pedestrians,
220 vehicles and 60 cyclists, one class per image, and the 103 freezing images
show 12 pedestrians, 80 vehicles and 11 cyclists. With temperature cut at 0 °C
and the default thresholds, Parity reports these cells (some columns omitted):

:::{list-table}
:widths: 10 16 12 12 10 12 14 14
:header-rows: 1

- - `level`
  - `class_name`
  - `observed`
  - `expected`
  - `lift`
  - `deficit`
  - `p_value_adjusted`
  - `status`
- - `< 0`
  - pedestrian
  - 12
  - 30.9
  - 0.39
  - 18.9
  - 1.7e-6
  - `underrepresented`
- - `< 0`
  - vehicle
  - 80
  - 56.65
  - 1.41
  - -23.35
  - 2.2e-7
  - `overrepresented`
- - `< 0`
  - cyclist
  - 11
  - 15.45
  - 0.71
  - 4.45
  - 0.20
  - `no_finding`
- - `>= 0`
  - pedestrian
  - 108
  - 89.1
  - 1.21
  - -18.9
  - 1.7e-6
  - `no_finding`
- - `>= 0`
  - vehicle
  - 140
  - 163.35
  - 0.86
  - 23.35
  - 2.2e-7
  - `no_finding`
- - `>= 0`
  - cyclist
  - 49
  - 44.55
  - 1.10
  - -4.45
  - 0.20
  - `no_finding`

:::

The first row answers the question. The freezing images hold 12 pedestrians
where independence predicts 30.9, a deficit of 18.9, with an adjusted p-value
of 1.7 × 10⁻⁶. In the rest of the table:

- The factor's V is 0.27, below the default `score_threshold` of 0.3, so the
  factor as a whole is `weak`. The pedestrian cell is still `underrepresented`,
  because cell statuses do not depend on the factor's V.
- The cyclist cell is 29% short (lift 0.71), but with 60 cyclists in the
  dataset the shortfall cannot be distinguished from chance (adjusted
  p = 0.20).
- Above freezing, the pedestrian and vehicle cells also depart significantly,
  with lifts of 1.21 and 0.86. Both lie inside the default lift band of 1.25×.
  Each row and column of the table sums to a fixed total, so a deficit in one
  cell is offset by surpluses in others, and the band keeps such small
  offsetting departures from being reported as findings.

```{figure} ../_static/images/concepts/parity/lift_band.svg
---
alt: >-
  Dot plot of the six cells' lift on a log axis, with a shaded band from 0.80 to 1.25. Below 0 °C,
  pedestrian is at 0.39, filled, underrepresented; vehicle at 1.41, filled, overrepresented; cyclist
  at 0.71, hollow, no_finding. At or above 0 °C, pedestrian at 1.21 and vehicle at 0.86 are filled
  but inside the band, no_finding; cyclist at 1.10 is hollow, no_finding.
---

The cells of the temperature example. Filled markers are significant after the
adjustment. No cell in this table is sparse, so a cell is a finding when it is
significant and its lift lies outside the shaded band. The two significant cells
above freezing are `no_finding`.
```

## Why rows are not independent

Pearson's chi-squared test treats every row as an independent draw. In computer
vision datasets the rows are often dependent:

- Frames of one video sequence share its scene and weather, and usually the
  class of the tracked object.
- Detections in one image share the image's conditions, so a per-image factor
  such as time of day or sensor has one value across all of them.
- A crowded scene puts many detections at one level of a factor, so a few
  images can supply most of a level's rows.

The same holds when the rows come from a derived view of a dataset. The frames
of a `SequenceFrames` view are not independent samples of their video, and the
crops of a `DetectionCrops` view are not independent samples of their image.
Parity groups such rows by the video or image they came from, however the view
was thinned or reordered, and reports `group_source="source"`. A video
contributes one unit whether the view keeps all of its frames or every other
one.

Dependent rows carry less information than the same number of independent rows,
and a test that treats them as independent overstates the evidence. The table
below gives false-positive rates at α = 0.05 of a row-level chi-squared test and
of Parity's grouped test (see [Theory](#theory)), run on the same simulated
datasets: 2,000 per design, and 4,000 for the last. Each factor was generated
independently of the class, so a calibrated test rejects 5% of the time.

:::{list-table}
:widths: 60 20 20
:header-rows: 1

- - Design (factor independent of the class)
  - Row-level test
  - Parity's grouped test
- - Sequence-level factor against detection labels, 40 sequences of 50 frames
  - 99.4%
  - 1.7%
- - Detection-level factor whose values cluster by image (object detection)
  - 51.0%
  - 4.2%
- - Frame-level factor in multi-object tracking
  - 99.6%
  - 2.2%
- - 20 images of 20 detections at one level, 380 single-detection images at
    the other
  - 54.0%
  - 5.3%

:::

```{figure} ../_static/images/concepts/parity/row_level_fpr.svg
---
alt: >-
  Horizontal bar chart of the table's false-positive rates, with a dashed line at the nominal 5%.
  The row-level test's bars reach 99.4%, 51.0%, 99.6% and 54.0%. Parity's grouped test's bars reach
  1.7%, 4.2%, 2.2% and 5.3%.
---

The rates in the table above. The dashed line marks the nominal 5%.
```

Row-level dependence also inflates the effect size. For a sequence-level
factor generated independently of the class over 7 sequences of 300 frames,
with a class mix that varied strongly between sequences, the row-level
bias-corrected V averaged 0.358, above the 0.3 threshold. Over 40 such
sequences it averaged 0.150. The same problem affects comparisons between
datasets. A row-level label-shift test declared two detection datasets
different in 84–91% of trials, although they had the same class distribution;
they differed in image counts and detections per image (see
[Label parity](DistributionShift.md#label-parity)).

## Theory

### The null hypothesis

The null hypothesis is label-proportion homogeneity: the class distribution of
rows is the same at every level of the factor. Groups, such as images or
sequences, are assumed independent of each other. Rows inside a group may
depend on each other in any way.

This null is weaker than the permutation null, under which whole groups,
including their sizes, are exchangeable across levels. Group sizes may
therefore differ by level, as when one site has crowded scenes. On the last
design in the table above, 20 images of 20 detections at one level against 380
single-detection images at the other (the 20×20 versus 380×1 design), the
grouped test described below rejected 5.3% ± 0.4% of 4,000 simulated null
datasets at α = 0.05. A permutation-based covariance, which assumes
exchangeable groups, rejected 38.9%.

### Groups, and how they are found

Parity resolves one grouping, the independent sampling unit, for the whole
call, from the structure of its input. It records where the grouping came from
in {attr}`~.ParityOutput.group_source` and the level it grouped by in
{attr}`~.ParityOutput.group_level`:

:::{list-table}
:widths: 38 38 24
:header-rows: 1

- - Input
  - Grouped by
  - `group_source`
- - A {class}`.Metadata` over a derived view: a `SequenceFrames`, a
    `DetectionCrops`, a `View` or a {func}`.merge_datasets` of either, or a
    `View` or merge that repeats items of a plain dataset
  - The root source item behind each row: the video for frames, the image for
    crops, the dataset item for a `View` or merge that repeats items
  - `"source"`, with no `group_level`, when some source item stands behind more
    than one item; otherwise the rows group as any other {class}`.Metadata`
- - Any other {class}`.Metadata` whose `item_indices` map several rows to one
    item, at any view: detections, frames, tracks, or item ids repeated at the
    item level
  - The dataset item each row belongs to: the image for object detection, the
    sequence for tracking and video
  - `"level"`, with `group_level` naming the item level
- - A custom container whose `item_indices` (label → item) map several rows to
    one item
  - Those item indices
  - `"item"`
- - A {class}`.Metadata` or custom container whose `item_indices` are
    one-to-one with the rows (image classification, a detection dataset viewed
    at its images, or object detection with one detection per image)
  - Nothing: each row is its own item, and the ungrouped test runs
  - `"rows"`
- - A custom container that records neither, with `assume_independent=True`
  - Nothing: the caller declares each row independent
  - `"declared"`
- - A custom container that records neither, without the declaration
  - —
  - raises `ValueError`

:::

Grouping by the dataset item is valid at any level a factor was measured at,
because every row belongs to exactly one item. A per-image factor and a
per-detection factor are tested over the same groups. A factor whose level is
unknown is also tested over the item groups and does not change how the other
factors are grouped. Parity raises an error for a {class}`.Metadata` with a
level above its item level, which no dataset structure produces today, instead
of guessing which level is independent. It also raises when `item_indices` does
not hold one entry per row.

If a custom {class}`.MetadataLike` records neither {class}`.Metadata` levels
nor `item_indices`, Parity raises `ValueError` before computing anything:

```text
Cannot establish independent sampling units for this metadata. Provide
`item_indices` (label -> item) or use `Metadata` levels, or pass
`assume_independent=True` if each row is its own independent item. Row-level
inference gives invalid p-values when several rows belong to one image or
sequence.
```

The {class}`.MetadataLike` protocol documents a missing `item_indices` as one
label per item. Parity does not apply that default, because a missing
`item_indices` gives no evidence that the rows are independent. A
{class}`.Metadata` built from a dataset always records its structure and never
raises this error. The core function {func}`.parity` takes the grouping as an
argument: pass either `groups`, one group id per row, or
`assume_independent=True`. It raises when given neither or both.

The grouping records which rows belong together. It cannot show whether the
groups are independent of each other. Consecutive sequences from one flight,
images from one collection pass over the same scene, and tiles cut from one
frame are dependent groups, and the metadata does not reveal it. Parity does not
correct for dependence between groups, and such dependence makes its p-values
too small. The caller chooses the sampling unit: when several items are known
to depend on each other, pass one id per independent unit, as `groups` to
{func}`.parity` or as `item_indices` on a custom container.

### The factor test

#### Design effect

With groups, each group $s$ contributes a score

$$U_s = \sum_{\text{rows in } s} (x - \hat p_f)(y - \hat p_y)^\top$$

where $x$ and $y$ are a row's level and class as one-hot vectors, and
$\hat p_f$ and $\hat p_y$ are the level and class proportions over all rows.
The scores sum to $O - E$. Over $S$ groups, their cluster sandwich covariance

$$V = \frac{S}{S-1} \sum_s \operatorname{vec}(U_s) \operatorname{vec}(U_s)^\top$$

estimates the variance of $O - E$ without any assumption about how the rows
inside a group relate. With $G = \operatorname{diag}(1/E)$, Pearson's statistic
is $X^2 = \operatorname{vec}(O - E)^\top G \operatorname{vec}(O - E)$ with
$\mathrm{df} = (r-1)(K-1)$, and the mean generalized design effect of Rao and
Scott is

$$\bar\delta = \frac{\operatorname{tr}(GV)}{\mathrm{df}}$$

$\bar\delta$ is the factor by which grouping inflates the table's variance
relative to independent rows. It is 1 for independent rows, and 25 when the
table carries about as much information as $n/25$ independent rows. Parity
uses $\bar\delta$ as computed and does not floor it at 1. A value below 1
indicates negative dependence inside groups, for example pairs of detections
that usually hold one of each class. In a simulation of that case (200 pairs,
$\bar\delta$ near 0.10, 2,000 null datasets) the test rejected 4.4% at
α = 0.05. When $\bar\delta < 1$, $n/\bar\delta$ exceeds $n$, so it is a
computational analog of an effective sample size and cannot be read as a count
of independent observations. Parity reports $\bar\delta$ as
`design_effect`.

#### Effective groups

A level that occurs in only a few groups rests on only a few independent
observations, however many rows it holds. For each level $i$, with $X_{si}$
rows of group $s$ at that level, the [Kish (1965)](#ref5) effective number of
groups is

$$k_i = \frac{\left(\sum_s X_{si}\right)^2}{\sum_s X_{si}^2}$$

It equals the number of groups when they contribute equally and falls toward 1
as one group dominates.

```{figure} ../_static/images/concepts/parity/effective_groups.svg
---
alt: >-
  Two horizontal bars of 400 rows each, divided into one segment per image. Level A has 100 equal
  segments of 4 rows, labeled k = 100. Level B has one segment of 200 rows followed by 50 segments
  of 4 rows, labeled k = 3.9.
---

Two levels of 400 rows each. Level A spreads its rows over 100 images of 4
rows, so $k = 100$. Level B puts 200 rows in one image and the rest in 50
images of 4, so $k = 3.9$. A factor with these two levels has $d = 2.9$, so all
of its testable cells are `too_thin` (see [The cell tests](#the-cell-tests)).
```

The design degrees of freedom are $d = \min_i k_i - 1$,
set by the level with the fewest effective groups. When the class is constant
within every group, for instance one class per sequence, the classes' effective
group counts also enter the minimum. Classes that vary within groups are left
out of the minimum, because including them would let one rare class make every
factor untestable. With them left out, the test held its level when a rare
class was concentrated in a few bursty groups, groups in which most rows are
that class (0–2% false positives). Parity reports the minimum, $\min k$, as
`effective_groups`.

#### The Rao–Scott F test

The second-order correction also accounts for how unevenly the design effect is
spread across the table:

$$1 + a^2 = \frac{\operatorname{tr}\left((GV)^2\right)}{\mathrm{df}\cdot\bar\delta^2}$$

$$\nu = \frac{\mathrm{df}}{1 + a^2}$$

The factor's p-value is the upper tail of $F(\nu, \nu d)$ at
$X^2 / (\bar\delta\,\mathrm{df})$. This is the second-order Rao–Scott F test
with design degrees of freedom ([Rao & Scott, 1984](#ref8);
[Thomas & Rao, 1987](#ref9); the `statistic = "F"` option of R's
`survey::svychisq`), using Kish effective groups in place of the number of
primary sampling units. Parity reports $\nu$ as `df_num`, $\nu d$ as `df_den`
and `test="rao_scott_f"`. The core function {func}`.parity` also reports
$X^2/\bar\delta$ as `chi_squared`.

#### Ungrouped data

When every row is its own item (`group_source` is `"rows"` or `"declared"`, or
all of the core function's group ids are distinct), Parity runs Pearson's
chi-squared test without continuity correction, which gives the same result as
`scipy.stats.chi2_contingency(table, correction=False)`. In that case
$\bar\delta = 1$, the reference distribution is $\chi^2$ with $(r-1)(K-1)$
degrees of freedom (`df_den` is infinite), `effective_groups` is the row count,
and `test` is `"pearson"`.

### The effect size: V

The effect size, reported as `score`, is Cramér's V with the
[Bergsma (2013)](#ref3) bias correction, on a scale from 0 (no association)
to 1. Bergsma subtracts the mean of $\phi^2 = X^2/n$ under independence,
$\mathrm{df}/(n-1)$. Under grouping the null mean of $X^2$ is
$\bar\delta\cdot\mathrm{df}$, so Parity subtracts $\bar\delta$ times as much:

$$\tilde\phi^2 = \max\!\left(0,\ \frac{X^2}{n} - \frac{\bar\delta\,\mathrm{df}}{n-1}\right)$$

$$V = \sqrt{\frac{\tilde\phi^2}{\min(\tilde r - 1,\ \tilde k - 1)}}$$

where $\tilde r$ and $\tilde k$ are Bergsma's corrected dimensions. In a
simulated sequence-level factor with a milder sequence effect than the one in
[Why rows are not independent](#why-rows-are-not-independent), the correction
reduced the average null V from 0.212 to 0.080 over 7 sequences, and from 0.087
to 0.028 over 40. The bias that remains comes from Bergsma's truncation at 0,
which leaves a null mean of 0.074 even on 40 independent rows. With a true V
near 0.11, the corrected estimate averaged 0.108, 0.096 and 0.108 over 7, 40
and 200 sequences, and the row-level V averaged 0.233, 0.136 and 0.116.

### The cell tests

Each cell gets an exact two-sided test of whether it holds more or fewer rows
than homogeneity predicts:

- Without groups, Parity uses the hypergeometric test of the cell against the
  rest of its table (the cell-versus-rest 2×2 collapse), conditional on the
  margins. It is exact at any expected count. The two-sided p-value is twice
  the smaller tail, capped at 1.
- With groups, Parity uses the effective-count exact test of
  [Korn & Graubard (1998)](#ref6). The cell's design effect, its variance
  (described below) divided by the variance independent rows would give, is
  floored at 1, so a cell never claims more information than its rows carry.
  The design effect is then multiplied by $(t_{d_{ij},\,0.975} / z_{0.975})^2$,
  where $d_{ij} = \min(d, k_j - 1)$ and $k_j$ is the class's effective number
  of groups. The cell is tested as an exact binomial of $O/\text{deff}$
  successes in $n_i/\text{deff}$ trials at the class's proportion.

With groups, every cell of a factor uses the factor's $d$, set by the level
with the fewest effective groups, capped by the class's own effective groups. A cell at a
well-populated level uses the same $d$ because the columns of $O - E$ sum to
zero: a common level's deviation is tied to the rare levels' deviations and is
no more reliable than theirs. A level with at least two but fewer than five
effective groups therefore makes all of its factor's testable cells
`too_thin`, and with fewer than two effective groups the factor is untestable.

Parity does not use the normal approximation to the standardized residual. At
an expected count of 5, that approximation understates the evidence for an
empty cell by a factor of 1.8 and overstates the evidence for a cell holding 15
by a factor of 63.

#### The grouped cell variance floor

A grouped cell's sandwich variance is estimated from the groups that fill it.
When a class comes in bursts, with most of an image or sequence showing that
one class, a cell's count comes from a few bursts. A cell that by chance
receives fewer bursts than its share is then both short and low in estimated
variance, and its test reports a chance shortfall as a precise deficit. With
the sandwich variance alone, in simulated null tables with a per-image factor
and classes that came in per-image bursts, at least one cell was flagged after
the BH adjustment in about 95% of 20×20 tables over 2,000 images and in every
40×40 table over 4,000 images. A 100×100 table over 100,000 images had 84 cells
reported `underrepresented`. Parity therefore takes each grouped cell's variance
as the larger of two estimates:

- The cell's sandwich variance. It makes no assumption about how rows inside a
  group relate, and it is estimated from the cell's own groups, so it shrinks
  when the cell happens to be short.
- A cluster-model variance. Between groups, a cell's deviation is a sum over
  groups of each group's share of rows at the level times its count of the
  class in excess of the class's overall proportion. The model replaces each
  group's squared excess count with its expectation under the
  [Kish (1965)](#ref5) cluster model, $m\,q\,(1 + (m - 1)\rho)$, where $m$ is
  the group's size, $q = p(1 - p)$ is the class's binomial variance and $\rho$
  is the class's within-group correlation. The expectation uses only the
  group's size; the group's observed count of the class does not enter it. For
  a group at a given level, $\rho$ is the largest of four moment estimates:
  - the class's correlation pooled over the groups at the other levels;
  - the largest of the class's correlations at each other level separately,
    which keeps the variance of a class that bursts at some levels only, as
    scene context can make it;
  - the level's correlation pooled over the other classes, which keeps the
    variance of a level whose groups burst;
  - the largest of the level's correlations in each other class separately,
    which keeps the variance of a level that bursts in a few classes, for every
    class at that level.

For a group at the cell's level, none of the four estimates uses the cell: the
first two are computed at other levels, and the last two from other classes at
that level. With few
classes, the other classes' counts at a level mirror the class's own (exactly,
with two classes), so the last two can raise the floor but do not protect a
cell that falls short; the first two do. A level with larger groups keeps its
larger variance through $m$. Groups at other levels also contribute to a cell's
variance, weighted by the square of the level's share, and the correlation
estimates for those groups include the cell's own groups.

Taking the larger of the two variances can only widen a cell's reference
distribution. The factor test uses neither. The model's assumptions, the case
it cannot handle, and its calibration and cost in power are in
[A4](#a4-the-grouped-cell-variance-model) and
[Calibration and power](#calibration-and-power).

### Untestable factors and cells

An untestable result is NaN, is excluded from the multiplicity adjustment, and
has status `untestable`. It does not indicate independence. A factor is
untestable when:

- it has fewer than two observed levels, or the class axis has fewer than two
  observed classes;
- with groups, there are fewer than two groups, fewer than two effective groups
  ($d < 1$), $\operatorname{tr}(GV)$ is zero (at or below $10^{-12}$), or the
  table has more than 10,000 cells, the size limit of the grouped test (a
  warning is logged).

With one sequence per level, a Rao–Scott test returned p ≈ 0.41 for both null
data and a perfect shortcut, which is why $d < 1$ is treated as untestable.

A cell is untestable when its factor is, when its estimated variance is
degenerate (at or below $10^{-12}$ of the largest cell variance in its table),
or when its class occurs in a single group. A class seen in one image or
sequence leaves the cell no design degrees of freedom ($d_{ij} = 0$). A class
seen in slightly more than one effective group is testable, but with fewer than
4 design degrees of freedom its cells are `too_thin`.

### Multiplicity

`p_value_adjusted` is [Benjamini–Hochberg](#ref1) adjusted across the testable
factors of one call, and separately across its testable cells. Every flag and
status uses the adjusted value. Cells in one table share its margins, so their
tests are dependent, and they need not satisfy the positive-dependence
condition under which Benjamini–Hochberg controls the false discovery rate
([Benjamini & Yekutieli, 2001](#ref2)). The adjusted p-values therefore carry
no unconditional false-discovery-rate guarantee.

In a table with any association, almost no cell's true lift is exactly 1, so
the error that matters is a false actionable finding: a cell reported as under-
or overrepresented whose true lift lies inside the lift band. In 1,000
simulated 4×5 tables of 4,000 rows, each with two planted deviations among
dependent cells, 1.8% of the actionable cells were false in this sense. Flagging
on significance alone, without the band, 12.7% were.

## Reading the output

### Factor status

`has_insufficient_data` is a screen on the reference distribution. It is True
when the expected counts divided by $\bar\delta$ fail [Cochran's](#ref4)
criterion (an expected count below 1, or more than 20% of them below 5), or
when $d < 4$. The screen does not guarantee validity; the calibration evidence
is for the F test itself (see [Calibration and power](#calibration-and-power)).
The factor `status` combines the screen with the test, in this order of
precedence:

:::{list-table}
:widths: 20 80
:header-rows: 1

- - `status`
  - Meaning
- - `untestable`
  - NaN results: the design cannot answer the question.
- - `associated`
  - `is_associated`: adjusted p at or below `p_value_threshold` (0.05) and V at
    or above `score_threshold` (0.3).
- - `weak`
  - Significant, with V below `score_threshold`.
- - `inconclusive`
  - Not significant, and `has_insufficient_data`: too little evidence to treat
    the result as a negative.
- - `not_detected`
  - Not significant, with sufficient data.

:::

```{graphviz}
---
alt: >-
  Decision flow for a factor's status. Not testable leads to untestable. A testable factor whose
  adjusted p is at or below p_value_threshold is associated when V is at or above score_threshold
  and weak otherwise. A factor that is not significant is inconclusive when has_insufficient_data is
  true and not_detected otherwise.
---

   digraph factor_status {
      rankdir=TB
      nodesep=0.3
      ranksep=0.35
      node [shape=box, style="rounded,filled", fontsize=11, height=0.4]
      edge [fontsize=10, arrowsize=0.7]

      // Read top to bottom: a question is asked only of a factor that reached it, so the
      // first status reached is the factor's. Questions are gray and statuses pale blue.
      // Node text is pinned to its fill; edge labels sit on the page background and are
      // left to the theme so they follow light and dark mode.
      testable [label="testable?", fillcolor="#eaeef2", fontcolor="#1f2328"]
      significant [label="adjusted p at or below\np_value_threshold?", fillcolor="#eaeef2", fontcolor="#1f2328"]
      strong [label="V at or above\nscore_threshold?", fillcolor="#eaeef2", fontcolor="#1f2328"]
      enough [label="has_insufficient_data?", fillcolor="#eaeef2", fontcolor="#1f2328"]

      untestable [label="untestable", fillcolor="#ffffff", fontcolor="#1f2328", style="rounded,filled,dashed"]
      associated [label="associated", fillcolor="#dae3f3", fontcolor="#1f2328"]
      weak [label="weak", fillcolor="#dae3f3", fontcolor="#1f2328"]
      inconclusive [label="inconclusive", fillcolor="#dae3f3", fontcolor="#1f2328"]
      not_detected [label="not_detected", fillcolor="#dae3f3", fontcolor="#1f2328"]

      {rank=same; testable; untestable}
      {rank=same; strong; enough}

      testable -> untestable [label=" no"]
      testable -> significant [label=" yes"]
      significant -> strong [label=" yes"]
      significant -> enough [label=" no"]
      strong -> associated [label=" yes"]
      strong -> weak [label=" no"]
      enough -> inconclusive [label=" yes"]
      enough -> not_detected [label=" no"]
   }
```

### Cell status

A cell carries three judgments in separate columns:

- `is_deviant`: the cell is testable and its adjusted p-value is at or below
  `p_value_threshold`. It ignores lift and sparsity, so every significant cell
  is visible in this column.
- `is_sparse`: the cell's effective expected count, $E/\text{deff}$, is below 5
  ($E$ below 5 without groups), or the cell has fewer than 4 design degrees of
  freedom.
- `status`: the reading to act on, assigned in this order of precedence:

:::{list-table}
:widths: 20 44 36
:header-rows: 1

- - `status`
  - Condition
  - Reading
- - `untestable`
  - Not testable
  - The design cannot answer this.
- - `too_thin`
  - `is_sparse`
  - Collect more data at this level to find out. Never reported as a finding,
    whatever its p-value.
- - `underrepresented`
  - `is_deviant`, and lift at or below `1/lift_threshold` (0.80)
  - A collection candidate.
- - `overrepresented`
  - `is_deviant`, and lift at or above `lift_threshold` (1.25)
  - Evidence of a possible shortcut.
- - `no_finding`
  - Otherwise, including significant cells inside the band
  - Nothing to act on at this band.

:::

```{graphviz}
---
alt: >-
  Decision flow for a cell's status. Not testable leads to untestable. A testable, sparse cell is
  too_thin. A cell that is not sparse and not deviant is no_finding. A deviant cell is
  underrepresented when its lift is at or below 0.80, overrepresented when it is at or above 1.25,
  and no_finding when its lift is inside the band.
---

   digraph cell_status {
      rankdir=TB
      nodesep=0.3
      ranksep=0.35
      node [shape=box, style="rounded,filled", fontsize=11, height=0.4]
      edge [fontsize=10, arrowsize=0.7]

      // Read top to bottom: a question is asked only of a cell that reached it, so the first
      // status reached is the cell's. Questions are gray; statuses take the colors of the
      // figures on this page. The lift edges show the default lift_threshold of 1.25. Node
      // text is pinned to its fill; edge labels sit on the page background and are left to
      // the theme so they follow light and dark mode.
      testable [label="testable?", fillcolor="#eaeef2", fontcolor="#1f2328"]
      sparse [label="sparse? E/deff below 5,\nor fewer than 4 design df", fillcolor="#eaeef2", fontcolor="#1f2328"]
      deviant [label="deviant? adjusted p at or\nbelow p_value_threshold", fillcolor="#eaeef2", fontcolor="#1f2328"]
      band [label="lift outside the band?", fillcolor="#eaeef2", fontcolor="#1f2328"]

      untestable [label="untestable", fillcolor="#ffffff", fontcolor="#1f2328", style="rounded,filled,dashed"]
      too_thin [label="too_thin", fillcolor="#fff1c2", fontcolor="#1f2328"]
      no_finding [label="no_finding", fillcolor="#d0d7de", fontcolor="#1f2328"]
      under [label="underrepresented", fillcolor="#f6c6a4", fontcolor="#1f2328"]
      over [label="overrepresented", fillcolor="#c6dbef", fontcolor="#1f2328"]

      {rank=same; testable; untestable}
      {rank=same; sparse; too_thin}
      {rank=same; band; no_finding}

      testable -> untestable [label=" no"]
      testable -> sparse [label=" yes"]
      sparse -> too_thin [label=" yes"]
      sparse -> deviant [label=" no"]
      deviant -> band [label=" yes"]
      deviant -> no_finding [label=" no"]
      band -> no_finding [label=" inside"]
      band -> under [label=" lift ≤ 0.80"]
      band -> over [label=" lift ≥ 1.25"]
   }
```

`is_actionable` is True for `underrepresented` and `overrepresented` cells. On a
large table even small lifts are significant, so these statuses also require
the lift to fall outside the band set by `lift_threshold`. The default of 1.25
is an interpretable practical-significance choice with no statistical
derivation. Setting it to 1 disables the band.

Sparsity takes precedence over deviance because a handful of rows can produce a
significant p-value. In a 40-row table where a rare level holds 4 rows, all of
class A, and the other 36 rows are all class B, the three cells expected to
hold 0.4, 3.6 and 3.6 rows have an adjusted p-value of 2.2 × 10⁻⁵. They are
`is_deviant` and are reported `too_thin`. Not being sparse is necessary for a
reliable cell but not sufficient: reliability also depends on the grouped test
having been validated for the data's structure (see
[Assumptions and their limits](#assumptions-and-their-limits)).

A cell's status does not depend on the factor's V, so a large, reliable gap is
reported even when the factor's overall association is small. In a 3,000-row
table where a level of 60 rows holds none of a class that independence expects
30 of, V is 0.142 and the factor is `weak`, while the empty cell is
`underrepresented` with an adjusted p-value of 9.5 × 10⁻¹⁹.

### What the output holds

{attr}`.ParityOutput.factors` has one row per factor, in the metadata's factor
order, with the columns `status`, `score` (V), `p_value`, `p_value_adjusted`,
`is_associated`, `has_insufficient_data`, `design_effect`, `effective_groups`,
`df_num` and `df_den`.

{attr}`.ParityOutput.cells` has one row per factor, observed level and observed
class, including cells with an observed count of zero, with levels and classes
decoded to names: $r \cdot K$ rows for a factor with $r$ levels. A level or
class that never occurs has no row. Parity tests combinations of the levels and
classes that are present, and a condition missing from the dataset entirely is a
coverage question (see [What it does not measure](#what-it-does-not-measure)).

Cells are sorted by factor name, then by status in the order
`underrepresented`, `overrepresented`, `too_thin`, `no_finding`, `untestable`,
then by deficit, largest first. The order is for reading convenience. It does
not rank collection priority, since neither deficit nor p-value measures
collection cost or mission value.

The output also records `test`, `group_source`, `group_level`, `n_rows` and
`n_groups`.

## When to use it

Use Parity to find where a factor and the class depart from independence, cell
by cell, with p-values that account for rows grouped into images or sequences.
Its `underrepresented` cells are collection candidates, and its
`overrepresented` cells point to possible shortcuts.

Parity reads categorical factors, so continuous factors must be binned first
(see [Binning](Binning.md)). On grouped data it needs enough independent images
or sequences at each level: check `effective_groups` and
`has_insufficient_data` before reading a non-significant result as a negative.
To compare counts with a mission's targets, see
[What it does not measure](#what-it-does-not-measure). To test whether two
datasets share a class distribution, use {func}`.label_parity` (see
[Label parity](DistributionShift.md#label-parity)).

## Assumptions and their limits

### A1. Factors are categorical codes

A factor's categories come from binning (see [Binning](Binning.md)). The
missing-value bin is an ordinary category, so if a factor's missingness is
associated with the class, the missing-value level shows up as associated.
Where the missing rate is material, check whether a finding is about the
factor's values or about their absence.

### A2. Groups are independent of each other

Nothing is assumed about rows inside a group. Dependence between groups is not
corrected and makes the p-values anti-conservative (see
[Groups, and how they are found](#groups-and-how-they-are-found)). No check on
the metadata can detect it. With `group_source="declared"`, the p-values are
valid only if the caller's declaration that every row is independent is true.

### A3. The reference distributions are asymptotic

The factor test's F reference and the grouped cell tests' effective-count
reference are asymptotic in the number of effective groups at each level. Their
calibration was measured in simulation (see
[Calibration and power](#calibration-and-power)), and the results apply to the
designs tested.

### A4. The grouped cell variance model

The cluster-model floor on a grouped cell's variance (see
[The grouped cell variance floor](#the-grouped-cell-variance-floor)) rests on
three assumptions.

**Correlation does not depend on group size.** Size enters only through
$1 + (m - 1)\rho$. In checks with sticky sequences of 4 to 100 frames, and
log-normal image sizes with $\sigma$ up to 2, at most 5% of null tables had a
flagged cell, against up to 91% with the sandwich variance alone.

**For each level and class, at least one of the four estimates is at least the
cell's own correlation.** This fails when a class comes in bursts at one level
of the factor only and that level's other classes do not burst. With two
classes there are no other classes. The cell's own groups are then the only
evidence of its variance. A cell that by chance holds none of the class's bursts
looks the same as a cell whose class is truly scarce at that level, and no
variance estimated from the data can distinguish them.

**Only the between-group part is modeled.** When a factor's value varies inside
a group, the within-group part of a cell's deviation is covered by the sandwich
variance alone.

#### Contraindication: a class that bursts at one level only

The second assumption fails in this case. Parity does not detect it, so check
your data for:

- a rare class seen in one or a few crowded images, all at one level;
- a class axis of two classes, one of them bursty at a single level;
- a factor whose levels each have their own bursty class.

```{figure} ../_static/images/concepts/parity/burst_contraindication.svg
---
alt: >-
  Two draws side by side, each showing level 0 and level 1 as grids of image tiles shaded by
  rare-class rows. In draw A, two tiles at level 0 are bursts, and the cell is too_thin with 17 observed
  against 16.9 expected. In draw B, no tile at level 0 holds the rare class, and the cell is
  underrepresented with 0 observed against 17.3 expected, adjusted p 0.0033. At level 1 in both
  draws the rare class appears as scattered tiles of 1 to 4 rows.
---

Two datasets drawn from the two-class null design that CI runs for this case:
1,000 images of 10 rows at 10 levels. The rare class has the same expected share
at every level. At level 0 it arrives only in bursts, about two bursty images
per dataset, and at the other levels, shown here by level 1, it is spread
thinly. In draw A, level 0 holds two bursts, the cell's variance reflects them,
and its effective expected count falls below 5, so the cell is `too_thin`. In
draw B, level 0 holds no burst. Neither the cell's sandwich variance nor any of
the four correlation estimates reflects the class's bursts, and Parity reports
the empty cell `underrepresented`. A dataset in which the rare class is truly
absent at level 0 looks the same as draw B.
```

In simulated null data, the share of tables with a falsely flagged cell,
against a nominal 5%, was:

:::{list-table}
:widths: 64 18 18
:header-rows: 1

- - Null design
  - Tables per run
  - Flagged
- - Two classes; the rare one bursts at one of 3 to 10 levels, about two
    bursty images there
  - 200–400
  - 11–17%
- - The same with about one bursty image there
  - 200–400
  - 2–3%
- - One such cell in a 2×2 to 3×4 table, 2 to 3 bursty images
  - 200–400
  - 3–10%
- - Each of 10 / 20 / 40 levels bursting in a class of its own
  - 200–500
  - 7–8% / 26–33% / 49–54%

:::

For such a class at such a level, treat an `underrepresented` cell as a reason
to inspect its images, and do not report it as a finding.

### A5. V describes the row-level association

Its null bias is corrected by matching the first moment (see
[The effect size: V](#the-effect-size-v)). Its sampling uncertainty is not
reported. A fixed `score_threshold` corresponds to different effect sizes
(Cohen's w) at different factor cardinalities, and Parity does not adjust for
this.

### A6. Multiplicity

The p-values are BH-adjusted, which gives no unconditional false-discovery-rate
guarantee on dependent cells (see [Multiplicity](#multiplicity)).

### A7. The test is nominal

Ordered bins are treated as unordered. The test never reads a code as a
magnitude, and an ordinal trend gains no power from its order.

## Calibration and power

All rates in this section were measured by simulation. They apply to the
designs tested and do not establish calibration for other designs.

### Factor test

Calibration was measured by simulating null data in the scenarios below. At
α = 0.05, each false-positive rate is at or below the nominal rate plus Monte
Carlo error:

:::{list-table}
:widths: 62 18 20
:header-rows: 1

- - Null scenario
  - Simulated datasets
  - False-positive rate
- - A rare level on 5% of 300 images (about 15), with heavy-tailed detection
    counts (log-normal(3, 1.5) per image)
  - 500
  - 3.0%
- - A rare level on 2% of 1,000 images, with the same counts
  - 500
  - 1.0%
- - A rare level (20% of 300 images) holding the large images, about 20
    detections each against about 2
  - 500
  - 3.6%
- - A sequence-level factor, 40 sequences of 50 frames
  - 2,000
  - 1.7%
- - Balanced object detection, 300 images
  - 500
  - 3.6%
- - A rare class in 2 to 10 bursty groups, among 60 or 200 sequences
  - 400 per design
  - 0–2%
- - 20 images of 20 detections at one level, 380 single-detection images at
    the other
  - 4,000
  - 5.3% ± 0.4%
- - Negative dependence inside groups, design effect near 0.10
  - 2,000
  - 4.4%

:::

With few effective groups the test was conservative in every scenario
evaluated. The grouped test's rates on the four designs of
[Why rows are not independent](#why-rows-are-not-independent) are in that
section's table.

### Cell tests

At α = 0.05, the grouped cell tests had false-positive rates of 0–0.2% in six
simulated small-table scenarios with four classes and rare levels, on
heavy-tailed images or sequence-level factors.

Excluding the contraindicated case in [A4](#a4-the-grouped-cell-variance-model),
72 null designs with bursty classes, 500 tables each (100 for the largest),
stayed within the Monte Carlo band. In 69 of them no table had a flagged cell
after the adjustment, and in none did more than 1% of tables. Cell
false-positive rates were 0–0.9%, and 3.3% in a 45×45 control whose factor
varies by row (nominal 5%). The designs covered:

- per-image factors at 20×20 and 40×40, including levels whose groups differ in
  size;
- two-level factors: rare, clustered within images, holding groups ten times
  larger, or alone bursty, against up to 100 classes;
- two to six classes, with a rare class held in bursts by 3–6% of images at
  every level, at 2 to 100 levels;
- burstiness that differs by level whenever another level shares it: half the
  levels bursty, two classes bursty at half the levels;
- a class constant within groups with the factor bursty, and both bursty;
- classes that burst at some levels only, each shared with another level or
  class:
  - blocks of 2 to 10 levels bursting in their own 2 to 10 classes, at 10×10 to
    40×40;
  - one level bursting in half the classes and the others in the other half;
  - bursts wherever the level's and the class's parities match;
  - a single level bursting in three of 20 classes;
- sticky sequences and log-normal image sizes, as in
  [A4](#a4-the-grouped-cell-variance-model).

A 100×100 table over 100,000 images had no flagged cell.

### Checks in CI

CI re-runs eight null designs of the kind in the factor-test calibration table,
including
10 sequences, a frame-level factor in multi-object tracking and the 20×20 versus
380×1 design, with 300 seeded datasets each. It asserts that the factor and
cell false-positive rates stay at or below α plus three Monte Carlo standard
errors.

The core tests also re-run eleven null designs with bursty classes, with 40 to
200 seeded datasets each:

- the 20×20, 40×40 and unequal-size 40×40 tables;
- the rare clustered two-level factor;
- the large-group two-level factor, with bursts of 90% and 50% of an image;
- a rare bursty class among two and among three classes;
- one level bursting in half the classes and the others in the other half;
- blocks of two levels bursting in their own two classes, at 20×20 over 2,000
  images and at 40×40 over 4,000, where each cell holds about 2.5 bursts.

They also run the two-class design with a rare class bursting at one level only
(the contraindication in [A4](#a4-the-grouped-cell-variance-model)) and assert
that its false-alarm rate stays below 25%. The measured rate is 11–17%, and the
sandwich variance alone gives about 18%. The 25% ceiling is a regression guard:
it fails only if a change makes this case worse than having no floor at all,
and it makes no calibration claim.

The 20×20 and 40×40 tables also run through the evaluator, 100 and 30 tables
respectively. The evaluator tests assert that the share of tables with any
flagged or actionable cell after the adjustment, and the cell false-positive
rate, stay within the same bound.

### Power

Power depends on the number of independent groups, which is the cost of
p-values that remain valid under grouping. A dataset of few sequences has
little power however many frames it holds. Factor-test power at α = 0.05:

:::{list-table}
:widths: 52 28 20
:header-rows: 1

- - Planted gap
  - Size
  - Power
- - Object detection: one class at a tenth of its share at a level holding 20%
    of images, about 5 detections per image
  - 100 / 300 images
  - 0.80 / 1.00
- - Sequences: one class at 0.15 of its share at one of three levels, 50 frames
    per sequence
  - 20 / 60 / 200 sequences
  - 0.02 / 0.83 / 1.00

:::

At 20 sequences the test almost never detects the gap. Read `effective_groups`
alongside any non-significant result.

In these designs, the gap cell's own test has less power than the factor test
at small sizes. The variance floor costs power for deficits held by few groups, for two reasons: the
cluster-model variance does not shrink when a cell is short, and when each level
holds few bursts, the largest of several noisy per-level and per-class
correlation estimates widens every cell's reference distribution. Power to find
the planted gap in its cell (adjusted p ≤ 0.05), with and without the floor:

:::{list-table}
:widths: 52 24 24
:header-rows: 1

- - Planted gap
  - Power
  - Sandwich variance only
- - Object-detection gap of the factor-test power table above, 100 / 300
    images
  - 0.20 / 1.00
  - 0.85 / 1.00
- - Sequence gap of the factor-test power table above, 20 / 60 / 200
    sequences
  - 0.00 / 0.08 / 1.00
  - 0.00 / 0.71 / 1.00
- - A class at 30% of its share at a small-image level, five levels over 600
    images, one level holding the large images
  - 0.14
  - 0.57
- - One class at a tenth of its share at one level, 20 levels and 20 bursty
    classes, 2,000 images
  - 0.00
  - not calibrated
- - The same class at 30% of its share, 20,000 images
  - 1.00
  - 1.00

:::

In the 20×20 table over 2,000 images each cell holds about five bursts, and even
a 90% deficit went undetected. With ten times as many images, or a 10×10 table
over 2,000 images, the planted deficit was always found.

## Acting on a result

[Acting on Results](ActingOnResults.md#parity) covers each finding in detail.
In brief:

- `underrepresented` cells are collection candidates, ranked by `deficit`:

  ```python
  result = Parity().evaluate(dataset)
  gaps = result.cells.filter(pl.col("status") == "underrepresented").sort("deficit", descending=True)
  ```

  The deficit is measured against independence. On grouped data it counts rows,
  so a deficit of 30 detections may correspond to a few crowded images.
- `too_thin` cells need more data at that level before they can be judged in
  either direction.
- `inconclusive` and `untestable` factors do not show that the factor is
  independent of the class.
- With `group_source="declared"`, the p-values are valid only if every row is
  independent; confirm this before citing them.

## Related concept pages

- [Dataset Bias and Coverage](DatasetBias.md) — Balance and Diversity, which
  score factor–class dependence and sampling evenness, and coverage in
  embedding space
- [Binning](Binning.md) — how continuous factors are cut into the levels
  Parity tests
- [Metadata Levels](MetadataLevels.md) — the item levels Parity groups rows by
- [Distribution Shift](DistributionShift.md#label-parity) — `label_parity`,
  which compares the class distributions of two datasets with the same grouped
  test
- [Acting on Results](ActingOnResults.md#parity) — what to do with each
  finding
- [Validation and Trust](ValidationAndTrust.md) — Parity's evidence and
  stability status

## See this in practice

### Tutorials

- [Analyze bias against a custom class axis](../notebooks/tt_analyze_by_custom_class_axis.py)

### How-to guides

- [How to measure label independence](../notebooks/h2_measure_label_independence.py)

## References

The method rests on [Rao & Scott (1981)](#ref7) and [Rao & Scott (1984)](#ref8)
for the design-effect correction, [Thomas & Rao (1987)](#ref9) for the F
reference, [Korn & Graubard (1998)](#ref6) for the grouped cell test,
[Kish (1965)](#ref5) for effective groups, [Bergsma (2013)](#ref3) for V,
[Cochran (1954)](#ref4) for the sufficiency screen, and
[Benjamini & Hochberg (1995)](#ref1) and [Benjamini & Yekutieli (2001)](#ref2)
for the multiplicity adjustment and its limits.

1. [Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
   a practical and powerful approach to multiple testing. *Journal of the Royal
   Statistical Society: Series B*, 57(1), 289–300.
   [paper](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x)]{#ref1}

2. [Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery
   rate in multiple testing under dependency. *Annals of Statistics*, 29(4),
   1165–1188.]{#ref2}

3. [Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T.
   *Journal of the Korean Statistical Society*, 42(3), 323–328.
   [paper](https://www.sciencedirect.com/science/article/abs/pii/S1226319212001032)]{#ref3}

4. [Cochran, W. G. (1954). Some methods for strengthening the common
   chi-square tests. *Biometrics*, 10(4), 417–451.]{#ref4}

5. [Kish, L. (1965). *Survey Sampling*. Wiley.]{#ref5}

6. [Korn, E. L., & Graubard, B. I. (1998). Confidence intervals for proportions
   with small expected number of positive counts estimated from survey data.
   *Survey Methodology*, 24(2), 193–201.]{#ref6}

7. [Rao, J. N. K., & Scott, A. J. (1981). The analysis of categorical data from
   complex sample surveys: Chi-squared tests for goodness of fit and
   independence in two-way tables. *Journal of the American Statistical
   Association*, 76(374), 221–230.]{#ref7}

8. [Rao, J. N. K., & Scott, A. J. (1984). On chi-squared tests for multiway
   contingency tables with cell proportions estimated from survey data.
   *Annals of Statistics*, 12(1), 46–60.]{#ref8}

9. [Thomas, D. R., & Rao, J. N. K. (1987). Small-sample comparisons of level
   and power for simple goodness-of-fit statistics under cluster sampling.
   *Journal of the American Statistical Association*, 82(398),
   630–636.]{#ref9}
