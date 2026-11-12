<!-- markdownlint-disable MD051 -->

# Outliers in metadata factors

{class}`.Outliers` accepts a {class}`.Metadata` as well as images. It then judges ordered metadata factors against the
rows at the level they were measured at, categorical factors by their rarity within a stratum, combinations of factors,
and values that depart from the rows next to them along a video or track. This page describes the method behind each of
these and the evidence behind its defaults. The call surface is documented on {meth}`.Outliers.evaluate`. Outlier
detection over image statistics and embeddings is described in
[Data integrity](DataIntegrity.md#outlier-detection-image-statistics-and-embeddings).

Extending {class}`.Outliers` beyond image statistics raises three questions the image path never faced: which columns
can be thresholded, how a categorical value can be an outlier, and how small a population can support a finding.

## Which columns can be thresholded

A {class}`.Metadata` holds whatever was attached to it. Several kinds of column
are numeric without being measurements, such as identifiers, booleans, and the
codes a categorical factor is stored as, and thresholding them produces findings
that are hard to act on.

One rule would be to threshold only factors that {class}`.Metadata`
classifies as continuous. That classification exists to choose bins, so it
answers a different question. It is a heuristic over the spacing of the values
(see [Binning](Binning.md)) that calls data discrete whenever the values sit on
a lattice or number fewer than 20:

| Column | Classified continuous? | Meaningful to threshold? |
| --- | --- | --- |
| gap count per track (Poisson counts, n = 3000) | no | yes |
| track duration in frames (integers, n = 3000) | no | yes |
| mean speed (floating point, n = 3000) | yes | yes |
| per-video factor, 12 videos | no (under 20 rows) | no (below the floor: listed in `result.populations`, not judged) |
| capture time in epoch seconds at 1 Hz | no (a lattice) | yes, sequentially |

Gating on that classification would silently exclude every integer count a
track produces and every factor at the sequence level. Eligibility is instead
decided by the raw column's type:

- Ordered: integers, floats, datetimes, durations. Thresholded on the raw values
  with the same tests the image path uses.
- Categorical: strings, booleans, and integers the caller has declared as a
  fixed vocabulary. Never given a location or scale; judged by rarity within a
  stratum (below).
- Ineligible: list-valued columns, and the reserved addressing columns (row
  indices, `track_id`, `item_id`), by construction rather than by heuristic.
  List-valued columns are still read when several columns are judged together
  (see [Combinations of factors](#combinations-of-factors)).

Integer identifiers are the one case a type check cannot catch: a numeric
`camera_serial` looks ordered. Declare it as a fixed vocabulary, or name the
factors to analyze explicitly. A cardinality heuristic is deliberately not used:
it would add a threshold to tune and defend.

## Why bins are never used

A factor's coded form (its bin indices) describes the cut, not the measurement:
a z-score over bin codes `[0, 1, 2, 3]` measures the bin edges. Two further
problems exist. A categorical factor's numeric reading is its codes, and a
missing categorical value is stored as one code past the last level rather than
as NaN, so a missing value would read as the highest level instead of being
excluded. Stratifying on a binned continuous factor makes every stratum depend
on the bin edges.

Detection therefore reads raw values only, and stratifies only on categorical
columns. A continuous conditioning variable is a joint relationship, not a
stratum, so you pass it to `given=` (see
[Combinations of factors](#combinations-of-factors)). As a result, re-binning a
metadata does not change an outlier result.

## Relative box geometry

Box width, height, area, and position are measured in pixels. Stratifying them
by class compares a box in a 4K frame against one in a 640-pixel frame, and
resolution dominates the comparison. Geometry relative to the image (area as a
fraction of the frame, offsets as fractions of width and height, distance from
the center as a fraction of the half-diagonal) turns "this class is usually a
small box near the corner" into a statement about composition, not the camera.
Absolute size remains its own question: it determines whether an object is
detectable at all.

## Small populations

A threshold fitted to a handful of rows describes the handful, not the distribution. {class}`.Outliers` therefore fits
a statistical threshold only to a population of at least 20 rows with a real value. A population is the rows of one
check at one level within one stratum, or the whole level when you pass no `strata=`. Below 20 rows, nothing is fitted
and no row is flagged from that population.

The floor comes from the share of clean rows the default adaptive threshold flags as the population shrinks. Each cell
below is the share of rows flagged on a single column of clean data, from 20,000 or more rows per cell:

| Rows | Normal | Gamma (shape 2) | Lognormal |
| --- | --- | --- | --- |
| 5 | 3.37 % | 3.46 % | 4.30 % |
| 10 | 0.17 % | 0.28 % | 0.98 % |
| 20 | 0.05 % | 0.19 % | 1.09 % |
| 30 | 0.04 % | 0.18 % | 1.06 % |
| 100 | 0.008 % | 0.12 % | 1.09 % |
| 1,000 | 0.003 % | 0.09 % | 1.04 % |

From 20 rows, the rate on normal and gamma data stays at or below about 0.2 % of clean rows. At 10 rows it is 0.17 to
0.28 %, and at 5 rows it is above 3 %. The rate rises fastest below 20 rows. Of the sizes in the table, 20 is the
smallest at which it is at or below about 0.2 % on both normal and gamma data. Lognormal data stays near 1 % from 10
rows on, so the floor does not govern heavy tails. The table measures about 1 % for lognormal values at 1,000 rows, and
the calibration paragraph under rarity measures 0.6 % at 5,000 rows. Neither rate is set by the floor.

The checks that score a combination have their own minimum, because each one fits more than a location and a scale:

| Check | Judged when the population has |
| --- | --- |
| single column, or a distribution of values | at least 20 rows with a real value |
| robust Mahalanobis distance over $d$ dimensions | at least $\max(20,\ 10d)$ scored rows |
| residual of one column given another | at least 31 rows |
| categorical rarity | at least 93 rows |

The clean-data rates of the score checks, under the same default threshold, are in the same range as a single
column's. The robust Mahalanobis distance flags 0.12 % of clean rows at 20 rows and 2 dimensions, 0.025 % at 100 rows
and 2 dimensions, and 0.01 % at 500 rows and 10 dimensions. The distribution distance flags 0.18 % at 20 rows and
0.08 % at 50 rows.

A population below the floor is skipped and listed in a table. No warning is raised and no ranking is returned. A
warning on every population below the floor would fire on every video dataset, where a per-video factor has a dozen
rows. A ranking of a small population gives no basis for action, because with no threshold the top of the ranking is
always some row. Skipping the population and reporting the skip states the limit of the evidence.

A `ConstantThreshold` is exempt. Its bounds do not depend on the data, so a small population supports it. The floor
does not apply to cluster-distance detection.

## Stratification

A check judges a row against its whole level by default. A frame of a clear day and a frame of a snowy day then share
one threshold, and a value that is ordinary for snow can look extreme beside the clear frames. Stratifying judges each
row against the rows that share its value of one or more categorical factors, and every flagged row names the stratum it
was judged in, such as `weather=rain`.

A stratum is a categorical factor, or one of the reserved columns `item_index`, `unit_index` and `track_id`, which are
paired with `item_index` so that a population is one frame or one track of one video. Several names cross, and a
population is one combination of their values. A continuous factor is refused as a stratum, because strata built from
its bins would depend on the bin edges (see [Why bins are never used](#why-bins-are-never-used)). To judge a value
against rows that are near it in a continuous factor, use a conditioning variable (see [Combinations of
factors](#combinations-of-factors)). A row takes the value of the ancestor that defines it: a detection takes its
frame's `weather`, and a frame takes its video's. A null stratum value is a population of its own and is scored like any
other. Stratifying also turns on rarity for the categorical factors at the levels the strata reach (next section), and
it is the condition for judging a group of categorical factors in a combination. Cluster-distance detection is not
stratified.

## Categorical rarity within a stratum

A categorical value is an outlier when it is **improbable given its stratum**:
one frame stamped `weather="snow"` inside an otherwise clear 300-frame
sequence, or three frames of a sequence reporting a different `sensor_id`.

No widely used data validation tool ships a default for this. Great
Expectations, TensorFlow Data Validation, Deequ, ydata-profiling, Evidently,
and whylogs flag values outside a known domain or shares the user sets; none
flags a rare value with a default frequency cutoff, and none uses a minimum
count. The default below is derived rather than borrowed.

For a stratum of $n$ rows in which level $v$ appears $k$ times (counting the row
being judged), the value is flagged when the exact one-sided binomial test
rejects "$v$ occurs in this stratum with probability at least $p$":

$$P(X \le k \mid X \sim \mathrm{Binomial}(n, p)) \le \alpha$$

Equivalently, the one-sided Clopper–Pearson upper confidence bound on
$P(v \mid s)$ falls below $p$ ([Clopper & Pearson, 1934](#ref1)); that bound is
the number reported, because it reads directly as "this level occurs in at most
this share of its stratum". Every row of a level within a stratum shares one
test, so the unit of the test is the (stratum, level) cell.

The test treats the rows of a stratum as independent draws. Correlated frames, such as a burst from one event, make it
count one event as several observations, so a bound computed over frames is narrower than the evidence supports. See
[Limitations](#limitations).

The defaults are rarity $p = 0.05$ at 95 % confidence ($\alpha = 0.05$). The
confidence level follows the conventional limit used for zero- and
small-numerator bounds ([Hanley & Lippman-Hand, 1983](#ref2)). The rarity is a
chosen value: of $p \in \{0.01, 0.02, 0.05\}$ it is the only value under which three
wrong-sensor frames in a 300-frame sequence are flagged.

The minimum group size: a singleton can only be flagged once the stratum is
large enough for $P(X \le 1) \le \alpha$, which at the defaults is $n \ge 93$,
approximately $4.74 / p$, the one-occurrence extension of the "rule of three"
for zero occurrences. [Das & Schneider (2007)](#ref3) derive the minimum support
of their categorical detector from its significance level in the same way. The
setting $p$ sets both the rarity and the minimum. Strata below the minimum are
listed in `result.populations` with the reason `below 93 rows` and without a
verdict, and are left out of the multiple-testing count, because a discrete test
that cannot reach significance need not count against the ones that can
([Tarone, 1990](#ref4)). Across the observed cells of the testable strata at a
level, the Benjamini–Hochberg procedure controls the false discovery rate at
0.05 ([Benjamini & Hochberg, 1995](#ref5)). The cells of a level are counted
together across all of its categorical checks, and a cell withheld as rare
everywhere (below) still counts, because it was tested.

Levels rare everywhere are not outliers. A class that is rare in every stratum
is a finding about the dataset's balance (see
[Dataset bias](DatasetBias.md)). [Das & Schneider (2007)](#ref3)
make the same argument (a pairing that is rare because both of its parts are
rare "can be explained") and divide by the marginal frequencies to remove it.
That correction fails a common video case: with 100 sequences each captured on
its own sensor, every sensor has a global share of about 1 %, so three frames of
sensor B inside sequence A look expected. The rule used instead: a level is
withheld as rare everywhere only when it is rare in every stratum large enough
to test. A stratum where the level does not occur at all counts as rare for this
rule, so a level seen in only one testable stratum is withheld: it is rare in
the dataset. A level common in at least one stratum and also seen in another
(the snow videos, sensor B's own sequence) stays flagged for its rare
appearances elsewhere. The rule is applied to the cells that survive the
Benjamini–Hochberg correction, and every row of a flagged cell is one finding.

| Case | Upper bound on the level's share | Flagged |
| --- | --- | --- |
| 1 snow frame in a clear 300-frame sequence | 0.016 | yes |
| 3 wrong-sensor frames in a 300-frame sequence | 0.026 | yes |
| 2 "truck" frames in a 60-frame "car" track | 0.101 | no (60 rows cannot support it) |

The last case is not a coverage gap. Label drift within a track is already
measured as an ordered, per-track quantity (the chosen label's share of the
track) and thresholded against other tracks. The categorical test covers strata
too large for such a summary to exist.

The rejected alternative: a conformal p-value (the share of the stratum made of
levels at least as rare as the row's own) is the other candidate, and is what
the "q-value" of [Das & Schneider (2007)](#ref3) computes. Its validity is exact
at any sample size ([Laxhammar, 2014](#ref6); [Bates et al., 2023](#ref7)),
and it can flag a singleton in a stratum of $1/\alpha$ rows. But scored within
the stratum it bounds the number of flags (at most $\lfloor \alpha n \rfloor$
rows per stratum) rather than giving evidence that the value is improbable.
[Laxhammar (2014)](#ref6) notes that a conformal anomaly may be "a relatively
rare … example generated from the same probability distribution".

| | Conformal p-value | Exact binomial |
| --- | --- | --- |
| What a flag claims | the row is in the rarest $\alpha$ of its stratum | its level's share is below $p$, with 95 % confidence |
| Smallest stratum that can flag a singleton | $1/\alpha$ (20 at $\alpha$ = 0.05) | 93 at $p$ = 0.05 |
| Rows flagged in a clean stratum of 20 classes with Zipf-distributed shares | about $\alpha$ of them (0.8 % at $\alpha$ = 0.01) | 0.07–0.26 % |
| A legitimate level with true share 0.5 %, $n$ = 300, $\alpha$ = 0.01 | flagged in 71 % of simulations | flagged only if its share is truly below $p$ |
| Benjamini–Hochberg across strata | loses power: no p-value can fall below $1/n$ | usable |

On a clean dataset with many rare classes, the binomial rule flags a small share of rows, as the table shows.

Calibration beside the ordered tests: on clean data the default adaptive
threshold flags about 0.0004 % of normal values, 0.09 % of gamma-distributed
values, and 0.6 % of log-normal values (5,000 values, 50 draws each). The
binomial rule's 0.07–0.26 % on a clean stratum is the same order, so ordered and
categorical findings have comparable false-alarm rates within one result.

## Combinations of factors

A row can be ordinary in every column and still be improbable as a combination: a large box near the horizon, where
boxes of that class are small; a frame stamped 02:00 with daylight brightness; a fast track with low straightness.
Judging one column at a time cannot find these. {func}`.combination_scores` scores each row once per group of columns,
within its stratum, and reports numbers rather than findings.

Each group is scored by the method that fits its columns:

| Columns | Method | Score |
| --- | --- | --- |
| numbers, vectors, samples | robust Mahalanobis distance | higher is more unusual |
| categories, tag sets | count of rows sharing the combination | lower is rarer |
| one sample of values per row | distance to the typical distribution, adjusted for sample length | higher is more unusual |
| one number given another | robust z against the nearest rows of the other | larger magnitude is more unusual |

The function returns scores and not findings. Deciding what counts as an outlier needs a threshold, a minimum
population and a correction for running many tests. Those choices belong to the evaluator that reads the scores, as
they do for single columns.

The tail of the distance: the textbook reference for a squared Mahalanobis distance is a chi-square distribution. With a
robust fit on a few dozen rows it is too liberal: on clean normal data, 19 % of rows fell below 0.01 at 10 rows and 2
columns, where 1 % should. The tail reported here uses the finite-sample distribution of reweighted minimum covariance
determinant distances ([Cerioli, 2010](#ref8)). From 50 rows per dimension, 0.5–2 % of clean rows fall below 0.01. A
vector adds one dimension per element. Below that, and whenever a sample column is in the group, no tail is reported and
only the distance is.

Curved relationships: one column given another is fitted locally, against the rows nearest in the conditioning column,
not with a straight line. Brightness rises and falls over a day, so a straight line through it is nearly flat and leaves
the daily curve in every residual. Each row is compared with its nearest 30 rows, or $\lceil\sqrt{n}\rceil$ when that is
larger, so a population needs at least 31 rows. Fewer neighbors make the spread too noisy and flag clean rows too often.
With `y = 0.5x + noise`, the share of clean rows flagged by the default threshold depends on the smallest window:

| Smallest window | 50 rows | 100 rows | 200 rows | 1,000 rows |
| --- | --- | --- | --- | --- |
| 10 neighbors | 0.50 % | 0.46 % | 0.22 % | 0.04 % |
| 20 neighbors | 0.14 % | 0.13 % | 0.10 % | 0.04 % |
| 30 neighbors | 0.075 % | 0.06 % | 0.04 % | 0.04 % |

With 10 neighbors the spread estimate is noisy and the z-scores are heavy-tailed
(0.89 % of rows at 50 rows on a cosine relationship). A window of 30 brings the
rate to the single-column level.

`Outliers` turns the scores into findings with the same machinery it uses for a column: it fits the threshold to the
scores of each population and flags the rows beyond it. The two distance methods flag only the upper side, because a
small distance is not unusual. The residual flags both sides of the signed z, and rarity flags cells with an
improbably low count, by the binomial test above. Each flagged row names its check and its group, such as
`rel_size, rel_offset_y` or `brightness given hour`, and the populations and floors are those in
[Small populations](#small-populations). A group of categorical columns is judged by rarity and needs strata that reach
its level. A group that mixes numbers and categories is refused, because it would only repeat the split that a stratum
makes, and so is a group whose columns do not lie on one chain of levels. To ask whether a value is unusual for its
category, use the category as a stratum.

Why the tail is not used: {func}`.combination_scores` also reports the tail of the distance, and it is the natural cut.
At 0.01 it meets its band: 1.3 to 1.8 % of clean rows fall below 0.01 at (100 rows, 2 dimensions), (250, 5) and (500,
10). Further out it is liberal, most of all near the edge of 50 rows per dimension. At (100 rows, 2 dimensions) the
share of clean rows below the nominal level is 1.82 times that level at 0.01, 2.98 times at 0.001 and 4.25 times at
0.0001. At (250, 5) it is 1.43, 1.64 and 1.80 times. `Outliers` fits its threshold to the distances of each population
instead, as it does for any column, so a finding never relies on that deep tail. The clean-data rates are in
[Small populations](#small-populations).

A combination finding is kept beside single-column findings. A row flagged on a combination is kept when one of the
group's columns is also flagged alone. There is no shared error budget between the checks, since each population has its
own threshold, so reporting both counts nothing twice. The two are separate rows in the result.

Rejected approaches: one model over every column at once, such as an isolation forest on a mixed-type distance, gives
scores without a calibrated meaning and cannot say which combination made a row unusual. Encoding categories as numbers
and fitting one distance makes the covariance singular and lets rare categories dominate it.

## Neighbors along a sequence

Some wrong values are ordinary over the whole dataset and only stand out against the rows next to them. A timestamp
three hours off sits inside the dataset's range of capture times. A GPS fix that jumps 40 km for one frame is a
plausible location. A box whose size doubles for one frame of its track is a plausible box size. Judging a column
against its level, or within a stratum, cannot find these. {func}`.sequence_residuals` compares each frame with the
frames before and after it in its video, and each detection with the detections before and after it in its track,
ordered by `unit_index`.

Where the neighbors put a value: each row is compared with up to five rows on each side. The candidates for where it
should sit are the median of the rows before it, the median of the rows after it, and the line through those two medians
read at the row's position. The candidate nearest the row's value is its expected value, so each kind of change matches
the candidate that follows it:

| Along the sequence | Left median | Right median | Line between them | Flagged |
| --- | --- | --- | --- | --- |
| a scene cut: the level jumps and stays | matches the frames before the cut | matches the frames after the cut | sits between the two levels | no |
| a fade: the level ramps over several frames | lags behind | runs ahead | follows the ramp | no |
| a one-frame glitch | far from it | far from it | far from it | yes |

The first and last row of a sequence have one side. For them, the line runs through the medians of the near half and the
far half of that side, so a sequence that ends on a trend does not flag its last frame.

The expected value is interpolated between the two sides and not extrapolated. The alternative is to project each
side's own slope into the row. That slope comes from a few steps between noisy rows, and projected several rows its
error is as large as the glitches the check looks for. In measurements, a spike of 8 standard deviations was caught in
62 to 100 % of runs by extrapolation and in 100 % of runs by the line on most shapes. A clock rounded to milliseconds
flagged 15 % of its clean frames by extrapolation and none by the line. The line through the two side medians uses no
slope estimated from steps, only the two medians the side candidates already compute.

One scale for every sequence: the residual is the value minus its expected value. Each residual is divided by its
sequence's spread, shrunk toward the spread of every sequence with a weight of 30 rows, and the scores of every sequence
are judged together. A quiet video is not judged by a noisy video's jitter, and a short track is scaled mostly by the
pooled spread instead of by a dozen rows. Pooling raw residuals instead flagged about 2 % of clean rows once videos
differed in noise; the shrunk scale flags 0.1 to 0.9 %.

Timestamps: a date, datetime or duration factor is detrended by its median interval per frame before the comparison, and
so are `time_s` and `pts`. A steady rise, a varying frame rate and a dropped frame then do not read as departures, and a
residual within the clock's own rounding is set to zero. Each timestamp is also checked for order. The fewest frames to
remove for the sequence to be non-decreasing are flagged when there are at most five, so one bad stamp flags only its
own frame. A larger set is a reset, and only the frame where the clock went back is flagged. A timestamp held as plain
numbers under another name is read as an ordinary factor, so cast it to a Datetime or Duration before you add it.

The threshold is fitted to the scores of every sequence at the level, and the order check is a rule and not a fitted
threshold. A pool needs at least 20 scored rows, and stratifying splits the pool as it does for any check.

## Interpretation

A flag on a metadata factor says that a value is extreme, improbable or out of place relative to the population it was
judged against: the rows at its level, the rows of its stratum, the rows with similar conditioning values, or its
neighbors in a sequence. It does not say that the value is wrong, because a rare but valid value is flagged in the same
way. A population below the floor is not judged, and the absence of a flag there is not evidence that its rows are
normal. A categorical flag reports an upper bound on the level's share of its stratum, and a level that is rare in every
stratum is a finding about the dataset's balance and not an outlier. Combination scores are numbers and not findings
until `Outliers` fits a threshold to them.

## Limitations

- Categorical rarity assumes the rows of a stratum are exchangeable. Consecutive video frames are not: a three-frame
  burst of a wrong value is closer to one event than to three independent ones. A stratum smaller than the minimum
  cannot support a verdict on its own. Borrowing strength across strata with an empirical-Bayes prior is one way to
  judge them, and is not part of the default.
- The conditioning column of a residual does not wrap, so hour 23 is not next to hour 0. Consecutive frames are not
  exchangeable, so a burst of rows inflates counts and fits, as it does for single columns.
- The neighbor check reads a run of more than five departing frames as a segment, which the per-level threshold or the
  sequence-level `variability` reduction judges. A run of five can still flag a frame in its middle. On a steep ramp, a
  glitch in the ramp's own direction must exceed the ramp's rise over about three frames: on a ramp of one standard
  deviation per frame, a spike of eight was caught in 60 to 90 % of runs. On a steep trend, a glitch can also flag the
  clean frames next to it, because it shifts the median of each nearby frame's side. A clock that runs backwards
  throughout flags every frame but the first. A video that is noisy throughout is not flagged frame by frame.

## When to use it

Use the factor path when a defect is more likely to sit in annotation or metadata than in pixels: a wrong timestamp, a
GPS fix that jumps, an implausible box, a track with an unusual gap count. Use strata when a factor is ordinary in one
condition and extreme in another, combinations when a row is unusual only as a combination, and neighbors when a value
is plausible over the whole dataset and wrong beside the rows next to it. The how-to guides below work through each.

## Related concept pages

- [Data integrity](DataIntegrity.md): the image-statistics path of `Outliers` and the other integrity evaluators
- [Binning](Binning.md): why this path reads raw values and not bins
- [Metadata levels](MetadataLevels.md): the levels at which factors are measured, and the ordering along a sequence
- [Acting on results](ActingOnResults.md): what to do with flagged samples

## See this in practice

### How-to guides

- [Find outliers in metadata factors](../notebooks/h2_find_factor_outliers.py)
- [Find outliers that depend on their context](../notebooks/h2_find_conditional_outliers.py)

## References

1. [Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial
   limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404–413. [paper](https://academic.oup.com/biomet/article-abstract/26/4/404/291538)]{#ref1}

2. [Hanley, J. A., & Lippman-Hand, A. (1983). If nothing goes wrong, is
   everything all right? Interpreting zero numerators. *JAMA*, 249(13),
   1743–1745. [paper](https://pubmed.ncbi.nlm.nih.gov/6827763/)]{#ref2}

3. [Das, K., & Schneider, J. (2007). Detecting anomalous records in categorical
   datasets. In *Proceedings of the 13th ACM SIGKDD International Conference on
   Knowledge Discovery and Data Mining* (pp. 220–229). [paper](https://dl.acm.org/doi/10.1145/1281192.1281219)]{#ref3}

4. [Tarone, R. E. (1990). A modified Bonferroni method for discrete data.
   *Biometrics*, 46(2), 515–522. [paper](https://pubmed.ncbi.nlm.nih.gov/2364136/)]{#ref4}

5. [Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
   a practical and powerful approach to multiple testing. *Journal of the Royal
   Statistical Society: Series B*, 57(1), 289–300. [paper](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)]{#ref5}

6. [Laxhammar, R. (2014). *Conformal anomaly detection: Detecting abnormal
   trajectories in surveillance applications.* PhD thesis, University of
   Skövde. [thesis](https://www.diva-portal.org/smash/get/diva2:690997/FULLTEXT02.pdf)]{#ref6}

7. [Bates, S., Candès, E., Lei, L., Romano, Y., & Sesia, M. (2023). Testing for
   outliers with conformal p-values. *The Annals of Statistics*, 51(1),
   149–178. [paper](https://arxiv.org/abs/2104.08279)]{#ref7}

8. [Cerioli, A. (2010). Multivariate outlier detection with high-breakdown
   estimators. *Journal of the American Statistical Association*, 105(489),
   147–156. [paper](https://doi.org/10.1198/jasa.2009.tm09147)]{#ref8}
