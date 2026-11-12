# Parity calibration and power

[Parity](Parity.md) describes the factor test and the cell tests, and states the assumptions behind their p-values in
[Assumptions and their limits](Parity.md#assumptions-and-their-limits). This page reports the null scenarios and
false-positive rates of the two tests, the checks that run in CI, and the power of each test.

All rates on this page were measured by simulation. They apply to the designs tested
and do not establish calibration for other designs. The nominal false-positive rate does
not hold in every design: a class that bursts at one level only is a contraindication, and
its measured false-alarm rate was 11–17% (see [A4](Parity.md#a4-the-grouped-cell-variance-model)).

## Factor test

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
[Why rows are not independent](Parity.md#why-rows-are-not-independent) are in that
section's table.

## Cell tests

At α = 0.05, the grouped cell tests had false-positive rates of 0–0.2% in six
simulated small-table scenarios with four classes and rare levels, on
heavy-tailed images or sequence-level factors.

Excluding the contraindicated case in [A4](Parity.md#a4-the-grouped-cell-variance-model),
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
  [A4](Parity.md#a4-the-grouped-cell-variance-model).

A 100×100 table over 100,000 images had no flagged cell.

## Checks in CI

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
(the contraindication in [A4](Parity.md#a4-the-grouped-cell-variance-model)) and assert
that its false-alarm rate stays below 25%. The measured rate is 11–17%, and the
sandwich variance alone gives about 18%. The 25% ceiling is a regression guard:
it fails only if a change makes this case worse than having no floor at all,
and it makes no calibration claim.

The 20×20 and 40×40 tables also run through the evaluator, 100 and 30 tables
respectively. The evaluator tests assert that the share of tables with any
flagged or actionable cell after the adjustment, and the cell false-positive
rate, stay within the same bound.

## Power

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

## Interpretation

The false-positive rates show that, in the designs simulated, the tests rejected a true null at or below the nominal rate
plus Monte Carlo error, and were conservative when few effective groups were available. They do not establish
calibration for designs outside those listed. They also exclude the contraindicated case in
[A4](Parity.md#a4-the-grouped-cell-variance-model), where a class that bursts at one level only had a measured
false-alarm rate of 11–17%. The power tables show that power depends on the number of independent groups and not on the
number of rows, so a non-significant result from a dataset with few effective groups does not show that the factor is
independent of the class.

## Related concept pages

- [Parity](Parity.md): what the tests answer, how groups are found, how to read the output, and the assumptions
- [Binning](Binning.md): the categorical codes the factor test reads
