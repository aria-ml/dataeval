<!-- markdownlint-disable MD051 -->

# Data Leakage

Data leakage is a leading cause of overoptimistic — and irreproducible —
machine-learning results. In a survey across 17 scientific fields,
[Kapoor & Narayanan (2022)](#ref1) found 329 papers whose reported findings did
not hold up once leakage was corrected. The pattern is consistent: information
that will not be available about the distribution the model is meant to serve
contaminates either training or evaluation, the reported metric is inflated, and
the gap appears only when the model meets genuinely new data.

This page organizes leakage using the taxonomy of
[Kapoor & Narayanan (2022)](#ref1) — three top-level categories (L1–L3)
spanning eight specific failure modes. Several of these types are
_methodological_: they are errors in how a pipeline is constructed rather than
properties of the data; others are properties of the data itself. The aim here
is to characterize each failure mode — what leaks, how it inflates the reported
metric, and why it matters — as the taxonomy to build from.

## What is it

Supervised learning rests on an assumption: that the training and test
partitions are drawn independently from the same distribution $P(X, Y)$ — over
an input space $\mathcal{X}$ and label space $\mathcal{Y}$ — and that this
distribution matches the one the model will face in deployment, the
_distribution of interest_. Every performance number a team reports —
validation accuracy, held-out mAP, a passing acceptance test — is only
meaningful to the extent that this assumption holds.

[Kapoor & Narayanan (2022)](#ref1) frame leakage as a violation along two axes:
either the training and test sets are not cleanly separated (so the model sees
test information at training time), or the test set is not drawn from the
distribution of interest (so its measured performance does not transfer to the
population the claim is about). Either way the reported metric is
systematically better than the model's true generalization, because the model
has been rewarded for a signal that will not exist at inference time. This
framing predates them — [Kaufman et al. (2012)](#ref2) gave the formative
account — but their taxonomy is the most actionable for a data-centric workflow.

## Taxonomy of leakage

The taxonomy has three top-level categories. **L1** concerns the integrity of
the train/test boundary; **L2** concerns whether the features are legitimately
available at prediction time; **L3** concerns whether the test set represents
the population the claim is about. The table below enumerates the eight failure
modes; each is described in the sections that follow.

| Type                                       | What leaks                                                       |
| ------------------------------------------ | ---------------------------------------------------------------- |
| **L1.1** No test set                       | model evaluated on its own training data                         |
| **L1.2** Pre-processing on train + test    | imputation / resampling / transform fit over the full dataset    |
| **L1.3** Feature selection on train + test | feature choice informed by the test set                          |
| **L1.4** Duplicates                        | the same sample lands in both partitions                         |
| **L2** Illegitimate features               | a feature is a proxy for, or unavailable before, the outcome     |
| **L3.1** Temporal leakage                  | the test set contains data predating the training set            |
| **L3.2** Non-independence                  | train and test share the same underlying units                   |
| **L3.3** Sampling bias                     | the test set is not representative of the population of interest |

### L1 — Lack of clean separation of training and test data

If the training set is not held fully apart from the test set through _every_
pre-processing, modeling, and evaluation step, the model gains access to test
information before it is graded, and the reported metric reflects memorization
rather than generalization ([Kapoor & Narayanan, 2022](#ref1)).

**L1.1 is purely methodological.** Evaluating a model on its own training data,
with no held-out set, leaves no trace in the data itself; it is caught by code
review and disciplined pipeline construction, where every fit happens inside the
training fold. The practical guards are _provenance artifacts_ — data cards and
model cards (or trained-model metadata) that record which samples a model was fit
on — so that test-set membership can be audited against the training record. When
the split is documented this way, the overlap that defines L1.1 becomes a check
anyone can run; when it is not, the error is invisible until results fail to
reproduce.

**L1.2 and L1.3 are wiring errors that nonetheless leave a measurable trace.**
Fitting imputation or over/under-sampling on the full dataset before splitting
(L1.2), or selecting features using the whole dataset (L1.3), are mistakes in how
the pipeline is wired — but in some cases they imprint test-set information into
the training representation. The classic case is an unsupervised transform — PCA
or SVD — fit on the entire dataset and then used as features: accuracy in the
transformed space jumps, and the jump is conventionally explained away as relief
from the curse of dimensionality. The real cause is often that the transform's
basis was computed with the test set in view, so the features quietly encode it,
and performance in the transformed space is inflated relative to what the raw
features legitimately support.

**L1.4 — Duplicates** is the one L1 failure that is a property of the data rather
than the pipeline. When the same sample, or a near-identical variant, appears on
both sides of the split, the held-out set is no longer held out. Exact overlap is
unambiguous; the harder cases are **near-duplication** — a crop, rescale,
recompression, brightness shift, or rotation of a training image landing in the
test set, invisible to byte comparison — and **semantic duplication**, where
distinct samples depict the same underlying content. Cross-split duplicates of
any kind are direct contamination of the boundary.

One caveat the L1 framing alone misses: clean separation is not only about
disjoint samples but about **temporal ordering**. A split that is sample-disjoint
can still leak the future into the past — see [L3.1](#l3--test-set-not-drawn-from-the-distribution-of-interest) below.

### L2 — Illegitimate features

L2 leakage occurs when the model uses a feature that should not legitimately be
available — most often a **proxy for the outcome** that would not be present at
prediction time ([Kapoor & Narayanan, 2022](#ref1)). In imagery, a sharp example
is a heads-up-display (HUD) overlay burned into frames captured from a tactical
or sensor system: reticles, range readouts, and target cues are rendered onto the
scene during collection, and because those markers tend to appear precisely when
an operator has already fixed on a target, they are accidentally captured as a
"feature" of the objects and targets in the scene. A model can learn to read the
HUD rather than the target itself — and the cue vanishes the moment imagery is
collected without that display, so the measured performance will not reproduce in
real use. Whether a feature is legitimate is a domain-knowledge judgment, which is
why Kapoor & Narayanan give L2 no sub-categories.

**Feasibility and sufficiency are _not_ leakage.** Two related conditions are easy
to misfile here. A task may simply be _hard_ — the irreducible (Bayes) error is
high given the features — or the training set may be too _small_, so that more
data would still help. Both are real and important, but neither involves
information crossing a boundary it should not, so neither is leakage.

(leakage-l3)=

### L3 — Test set not drawn from the distribution of interest

In L3, the train/test boundary may be clean, yet the test distribution differs
from the population the scientific or operational claim is about. Measured
performance is then valid for the test set and misleading for the deployment
target ([Kapoor & Narayanan, 2022](#ref1)).

**L3.1 — Temporal leakage.** When the task is to predict a future outcome, the
test set must not contain data from before the training set; otherwise the model
is built on information "from the future" it would not have at prediction time.
A sample-disjoint split can still commit this error. The remedy is structural —
time-based partitioning or block cross-validation — so that the chronological
boundary matches the prediction boundary.

**L3.2 — Non-independence between train and test.** When train and test samples
come from the same underlying units — the same patient, the same video burst,
the same scene under slightly different conditions — they are not independent,
and the split overstates generalization unless the claim itself concerns that
dependence structure. This is the same phenomenon as near-duplication in L1.4,
viewed at the level of units rather than individual samples: the two sides of the
split are not the distinct, independent draws the evaluation assumes.

**L3.3 — Sampling bias.** The test set is a non-representative subset of the
population of interest — a single geographic region standing in for many, or
borderline cases quietly excluded. This is also the root of **shortcut
learning** ([Geirhos et al., 2020](#ref3)): when the sampling process couples
the label with a non-causal attribute, the model learns the attribute. If
parent-and-child images were collected disproportionately in one store chain, a
model can "predict" the relationship from the storefront rather than anything
about the people — and the shortcut evaporates the moment the sampling changes.
That a test set can look faithful and still be subtly non-representative is not
hypothetical: [Recht et al. (2019)](#ref5) rebuilt new ImageNet and CIFAR-10 test
sets following the original collection protocols and saw accuracy drop 11–14
points across a wide range of models — the original sets had drifted from the
distribution they were assumed to represent.

## Leakage through repeated evaluation

There is a form of leakage that none of the above captures and that no static
analysis of a dataset can see, because it happens over _time_. A held-out test
set can stay sealed — its samples never inspected — and still leak, through the
performance scores themselves. Each time a team evaluates a model against the
same holdout and uses the score to choose the next model, a sliver of test-set
information passes into the modeling process. Over many iterations the holdout
is, in effect, fit to.

The effect is like tuning a robot batter against a pitcher it never sees, with
only the score of each at-bat revealed. No pitch is ever observed, yet after
enough rounds the pattern of scores betrays that the pitcher throws only a
curveball and a changeup — enough to tune against. The hidden information leaks
through the outcomes alone. Public leaderboards and long-lived benchmark test
sets behave the same way ([Recht et al., 2019](#ref5)).

No static analysis of a dataset can see adaptive overfitting — it is a property
of the evaluation _process_, not the data. The mitigations are procedural: limit
the number of evaluations against any fixed holdout, refresh test sets
periodically, and budget test-set access deliberately, as in the reusable-holdout
mechanism of [Dwork et al. (2015)](#ref4).

## References

1. [Kapoor, S., & Narayanan, A. (2022). Leakage and the reproducibility crisis
   in ML-based science. _arXiv preprint arXiv:2207.07048._
   [paper](https://arxiv.org/abs/2207.07048)]{#ref1}

2. [Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in
   data mining: Formulation, detection, and avoidance. _ACM Transactions on
   Knowledge Discovery from Data_, 6(4), 1–21.
   doi: 10.1145/2382577.2382579 [paper](https://dl.acm.org/doi/10.1145/2382577.2382579)]{#ref2}

3. [Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge,
   M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks.
   _Nature Machine Intelligence_, 2(11), 665–673.
   doi: 10.1038/s42256-020-00257-z [paper](https://arxiv.org/abs/2004.07780)]{#ref3}

4. [Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O., & Roth, A.
   (2015). The reusable holdout: Preserving validity in adaptive data analysis.
   _Science_, 349(6248), 636–638.
   doi: 10.1126/science.aaa9375 [paper](https://www.science.org/doi/10.1126/science.aaa9375)]{#ref4}

5. [Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). Do ImageNet
   classifiers generalize to ImageNet? In _Proceedings of the 36th International
   Conference on Machine Learning_ (pp. 5389–5400).
   [paper](https://arxiv.org/abs/1902.10811)]{#ref5}
