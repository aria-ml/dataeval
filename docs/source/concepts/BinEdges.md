# Choosing bin edges

[Binning](Binning.md) cuts a continuous factor into intervals before any evaluator reads it, and the automatic path
derives the edges from the values. This page covers the case where the edges should come from the question instead, and
what a cut costs in the scores computed from it. It assumes the vocabulary of [Binning](Binning.md): bins, digitizing,
the level budget and `continuous_factor_bins`.

## Matching the cut to the question

Automatic binning reads the values and nothing else, which is a reasonable default:
many factors have no privileged cut point, and edges derived from the
distribution describe them well. Some factors do have one, a value at which the
meaning of the factor changes, and that value is not something the distribution
can reveal. Declaring it is what this section works through.

### A worked example: temperature

Take 400 road-scene images (an illustrative example) with the air temperature recorded for each, ranging
from -18.6 °C to 22.4 °C. A quarter of them were shot below freezing. That
boundary is the one that matters: below 0 °C the road may be iced, above it the
same road is merely wet, and a model that has never seen ice may fail on it. So
the question you want to ask the dataset is "are pedestrians under-represented
in freezing conditions?", which is a {class}`.Parity` ⚠️ test between the class
labels and a temperature factor.

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

The edge moved, but it still straddles zero, because 0 °C is a property of water and not of the distribution of
values. Every automatic method places edges by reading the values: equal spans, equal counts, or gaps between modes.
None of them can identify a particular value on that axis as a phase change. The information needed to cut this factor
for this question is not in the factor.

It has to arrive from outside the factor, through `continuous_factor_bins`.
Declaring the edges `[-np.inf, 0.0, np.inf]` for `temp_c` puts the one cut where
the meaning is instead of where the values happen to fall:

```text
declared edges — 2 bins
  bin 1   n=103   [-18.6,   0.0)   100% freezing
  bin 2   n=297   [  0.0,  22.4]     0% freezing
```

With two bins cut at the one meaningful value, the `Parity` test answers the question
that was asked. A longer edge list adds resolution
(`[-np.inf, -5, 0, 5, np.inf]` separates hard freeze from marginal freeze), but
the edge at zero is the one doing the work.

The answer is read from a cell rather than from the factor. The
pedestrians-below-freezing row of {attr}`.ParityOutput.cells` gives the
pedestrian images observed below 0 °C, the number expected if pedestrians were
as common there as in the dataset as a whole, the `deficit` between the two,
and a `status`: `underrepresented` when the shortfall is significant and at
least the lift band (lift 0.80 or less by default), `too_thin` when there are
too few freezing images to tell. The factor's overall Cramér's V does not enter
that judgment, so a clear shortfall in one cell is reported even when the
factor as a whole is only weakly associated. The
[Parity page](Parity.md#the-table-observed-expected-lift-and-deficit)
works this table through.

### A second collection

Now shoot the same road again in early spring. Same camera, same `temp_c`
factor, same call to {class}`.Metadata`, and only 2% of the images below
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

- Contextually, the freezing signal has all but vanished (nine images out of 400)
  and it is still not isolated. Bin 1 mixes freezing with mild, the same mixing
  winter's bin 2 had, at a different place on the axis.
- Comparatively, the two collections no longer share a vocabulary. `temp_c = 1`
  means "hard freeze, -18.6 to -5.1" in the winter data and "-4.6 to 2.7, mostly
  above freezing" in the spring data. The integer is the same and the meaning is
  not.

The second effect propagates to every downstream score. A {class}`.Diversity` index over a
three-letter alphabet is not on the same scale as one over a four-letter
alphabet, and a {class}`.Balance` score that moved between the two collections
may be reporting a change in the road or only a change in where the cuts fell,
with nothing in either output to tell you which. Anything that reads
`factor_data` inherits this; {func}`.factor_deviation` and
{func}`.factor_predictors` escape it precisely because they read the raw values
instead.

Explicit edges address both. `[-np.inf, 0.0, np.inf]` gives the spring
data bin 1 = 9 images below freezing, bin 2 = 391 above: the same two
categories, meaning the same two things, as the winter data. The factor becomes
comparable across collections because its vocabulary is now fixed by you rather
than derived per dataset.
[The bin count is a function of the data, not a setting](Binning.md#the-bin-count-is-a-function-of-the-data-not-a-setting)
has the measurements behind this.

### Other externally defined thresholds

The same shape recurs whenever a factor has a threshold set outside the data:
dawn and dusk in a time-of-day factor, a sensor's rated range in a distance
factor, the resolution below which an object is unrecognizable in a box-area
factor. In each case, a cut derived from the values is a good summary of the
distribution and not a summary of the boundary you care about, so that boundary
is worth declaring. These are the factors `continuous_factor_bins` exists for;
the rest are well served by the default.

## How the cut shows up in a score

The sections above describe where the edges land. This section describes the arithmetic that follows from having
edges: how much association a cut carries through, and how the score is normalized so that two different cuts can be
read on one scale.

### Binning reaches the three outputs differently

{class}`.Balance` returns three DataFrames, and binning does not reach them
equally. The difference is what each one divides by:

- `balance` and `classwise` divide by the entropy of the class label, which is
  never binned. Cutting a factor more finely moves the numerator toward the
  dependence the unbinned values carry and leaves the denominator alone, so the
  score converges. On a factor with a true dependence of 0.41, cutting it into
  2, 8, 32 and 256 bins gives 0.27, 0.39, 0.40 and 0.41. The score is settled by
  about 16 bins.
- `factors` compares two metadata factors, and both may be binned. There is no
  fixed reference, so a factor whose alphabet came out of the cuts contributes
  no entropy ceiling: such a pair is scored by the Linfoot transformation
  instead. Scoring it against a binned factor's entropy would make the reported
  association shrink as the same data is cut more finely: 0.40 at four bins down
  to 0.14 at 128, on identical data.

A factor whose values are its own alphabet, such as a category, a count or a
rating, keeps the entropy ceiling in `factors`, so a duplicated categorical
factor still reads 1.0. The two conventions are each correct in one regime,
which is why the choice is made per factor rather than globally.

Normalization does not add back resolution. A factor cut into four bins is
scored on four bins' worth of information whichever denominator is used. What
the Linfoot branch buys is dropping the entropy-ceiling artifact that grows with
bin count. The reported score no longer inflates for a bin count
[the data derived rather than the caller](Binning.md#the-bin-count-is-a-function-of-the-data-not-a-setting)
settled on, though the mutual information underneath still reflects what those
bins retained.

### The Linfoot branch has a ceiling of its own, and it is divided out

Dropping the entropy denominator does not by itself leave the Linfoot branch
scale-free. Mutual information between two factors cannot exceed the smaller of
their entropies whatever produced the codes, so the largest value the
transformation can return is bounded by their alphabets too, just in the
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
rather than only on how many there are: a lopsided binary split would be capped
near 0.47.

That ceiling is the mirror image of the artifact the Linfoot branch exists to
remove. The entropy denominator deflates a score as bins are added; this one
would deflate it as bins are removed. Both are properties of the cut rather than
of the data, so `factors` divides by the reachable maximum, as the entropy
branch divides by the entropy. A duplicated factor reads 1.0 on either branch,
at any cut, however unevenly its bins are filled.

What this does not do is add resolution the cut did not keep (see below).

### Effect of coarse binning on the correlation score

The practical consequence is that `factor_correlation_threshold`, 0.5 by
default, is crossed at very different true dependences depending on how finely
each side was cut. Measured on bivariate normal pairs where the true dependence
on the Linfoot scale equals ρ², at n = 5,000:

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

A pair with a true dependence of 0.64, a strong relationship, is
flagged at 16 bins and is not flagged at 2. The coarser the cut, the more
dependence a pair must carry before `is_correlated` fires.

What remains after the ceiling is divided out is the information the cut removed: two
binary variables cut from a strongly correlated pair are less dependent than the
values they came from, so a lower report is correct. It does mean a declared
cutoff, which is usually coarse (two or three bins), gains its meaning at the
cost of sensitivity, and that a pair sitting just under the threshold is worth
re-reading at a finer cut before concluding anything. The level counts are worth
reading beside the scores, and a factor's level count is the number of distinct
codes its `factor_data` column carries.

### Effect of fine binning on the correlation score

Cutting finer does not recover it either, because both tails lose the signal. A
cut so fine that the contingency table has more cells than the sample has rows
records which cells were hit rather than anything about the factor, and the
chance correction then subtracts nearly everything the counts found. Recovered
dependence for the same ρ = 0.9 pair (a true value of 0.810), cut into quantiles
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
  - 0.729
  - 0.606
  - 0.302
  - 0.072
  - 0.011
  - 20
- - 1,000
  - 0.576
  - 0.705
  - 0.761
  - 0.770
  - 0.667
  - 0.376
  - 0.135
  - 31
- - 5,000
  - 0.582
  - 0.705
  - 0.766
  - 0.788
  - 0.789
  - 0.713
  - 0.458
  - 70
:::

The count that reads best is k = 8 at n = 200, k = 16 at n = 1,000 and k = 32 at
n = 5,000. It moves with the sample size, so one setting is unlikely to suit two
datasets of different sizes. This is the same finding as
[the bin count being a function of the data](Binning.md#the-bin-count-is-a-function-of-the-data-not-a-setting),
arriving from the other direction: there it is the automatic path that cannot be
pinned, here it is the declared one that cannot be pinned once for all datasets.

The level budget is a ceiling, not a target. It is the last column above, and in
every row it sits past the peak: cutting at the budget reports 0.507 where 8
bins report 0.729 at n = 200, and 0.691 against 0.789 at n = 5,000. Staying
under the budget is what keeps a factor's table readable; it is not what makes
the cut a good one. DataEval warns when a declared encoding overruns the budget
and deliberately names no count in its place, because the number it would name
is this one.

A cut cannot avoid this trade: the sensitivity a fine cut buys is given back
through the sparsity it creates. The one read with no `k` to choose is the
unbinned one, which is what {attr}`~dataeval.bias.Balance.factor_source` selects
and why it prefers native values wherever nobody declared a cut.

For calibration at the other end, the largest score two independent factors
produced over 40 seeds, both carrying 16 levels (the level at which a reported
association means nothing). One seed set, all three regimes, so the columns are
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
on noise, but at n = 200 a spurious 0.137 is within reach, and that is a
sample-size limit rather than a binning one.

Reading the two tables together shows what is at stake in the choice that
{attr}`~dataeval.bias.Balance.factor_source` makes. A pair read as native
values reports 0.5 at a true dependence of 0.49 and carries the lowest null
floor of the three: the closest to calibrated, and the one that discards
nothing. Every cut under-reports, and the coarser the cut the further it
under-reports: 0.73 at two bins against 0.51 at sixteen. So the span the single
threshold has to cover is set by how coarsely factors were cut, not by which
channel read them; reading unbinned values moves the near edge of that span from
0.51 to 0.49, on a range already 0.24 wide.

That is why `factor_source="auto"` reads native values only where nobody
declared anything. A declared cutoff is a statement about which distinction matters, and the lower
reported score is the cost of honoring it: the question was about freezing and
not about temperature. Where no cutoff was declared, there is nothing to honor,
and the calibrated read is the better one.

## Related concept pages

- [Binning](Binning.md): what binning is, how DataEval places the edges, and the limits of the automatic path
- [Dataset bias and coverage](DatasetBias.md): the evaluators that consume binned factors
- [Parity](Parity.md): the contingency-table test that the temperature example uses

## See this in practice

### How-to guides

- [Control and reuse a factor's binning](../notebooks/h2_control_factor_binning.py)
