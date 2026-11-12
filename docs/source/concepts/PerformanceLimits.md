<!-- markdownlint-disable MD051 -->

# Performance limits

Before investing in model architecture, hyperparameter tuning, or additional data collection, it is useful to
establish what performance a dataset can support. Performance limits analysis addresses three questions: is a
performance requirement out of reach on this data, has some model already shown that it is reachable, and where do
additional samples stop producing meaningful improvement? {class}`.Feasibility` answers the first two with finite-sample
bounds and records the evidence behind each answer. The BER estimators estimate the first asymptotically for
classification, UAP shows what one crop classifier reaches on correctly localized detection boxes, and
{class}`.Sufficiency` answers the third.

These limits inform program decisions under the assumptions of each method. If a lower bound on the irreducible error
exceeds the allowance of an operational accuracy requirement, the requirement is out of reach for the models that bound
covers, which for BER are the models that read the same embedding (see Limitations). A change to the data or to the
requirement is then needed, and not a change to the model. If the fitted learning curve has saturated, additional data
of the same type is projected to yield diminishing returns, and engineering effort may be better spent elsewhere. A
finding that a requirement is out of reach is as informative as a finding that it is reachable, because it identifies
improvements that the data cannot support.

```{important}
The Bayes Error Rate (BER) metric is applicable to classification tasks only.
BER operates on class embeddings. The Upper-bound Average Precision (UAP) metric
reduces an object detection problem to a classification problem over bounding
box crops. See the UAP section below for details. The experimental
{class}`.Feasibility` evaluator supports image and video classification in this
release; object detection and tracking datasets raise `NotImplementedError`.
```

## How these tools relate

Performance limits analysis has four tools. Each answers a different question and rests on a different
construction, so the tools complement each other.

| Tool | Question it answers | What it rests on |
| --- | --- | --- |
| BER (`ber_mst`, `ber_knn`) | What is the lowest error any classifier reading this embedding could reach? | Asymptotic bounds from how class labels mismatch among neighbors in the embedding space |
| UAP | What average precision does one crop classifier reach when every bounding box is correctly localized? | The score of one classifier on the ground-truth bounding box crops |
| {class}`.Feasibility` | Is a stated requirement ruled out, already shown reachable, or neither, and what evidence says so? | Finite-sample confidence bounds from agreement among repeated or local labels (envelopes) and from frozen models scored on held-out items (witnesses) |
| {class}`.Sufficiency` | How does performance grow with the amount of training data, and how much more data would a target need? | A power law fitted to performance measured on growing subsets of the training data |

BER and Feasibility both speak to whether a requirement is out of reach, but they do not share a construction. BER
uses only the neighbor structure of the embeddings and gives asymptotic bounds, so it is a quick first read.
Feasibility's envelopes use agreement among repeated or local labels and its determinations carry a stated
confidence level and recorded assumptions, so it requires more setup and supports a claim with stated assumptions.

As a starting point, run BER on a classification dataset early, as a low-cost check. Use Feasibility when a
requirement needs a stated confidence level behind it. Use UAP when an object detection program is judged on mean
average precision. Use Sufficiency when the question is how much more data to collect.

## What they are

### Bayes error rate: the irreducible floor

The {term}`Bayes Error Rate (BER)` is the lowest possible error rate achievable
by any classifier on a given probability distribution. It is not a property of
any particular model. It is a property of the joint distribution of the inputs
and labels. No classifier on that distribution, however large or well tuned,
can achieve an error below the BER.

BER arises from **class overlap in the feature space**. When two classes share
regions of the input space, meaning the same image features are legitimately
associated with more than one class, no deterministic classifier can resolve the
ambiguity. The probability of misclassification in those regions is irreducible.

In computer vision, class overlap has several common sources.
**Label ambiguity** occurs when human annotators disagree about the correct
class for an image. **Sensor noise and resolution limits** cause images that are
semantically distinct to become perceptually identical at the pixel level.
**Incomplete feature sets** occur when the data does not contain enough
information to distinguish the classes, for example when trying to predict
vehicle type from low-resolution thumbnails where distinguishing features are
not resolvable. In all cases, the result is the same: overlapping class
distributions in embedding space, and a non-zero floor on error.

In operational contexts, BER is a pre-development diagnostic, and its two bounds answer different questions. The
lower bound estimates the least error any model reading the embedding can reach; if it exceeds the allowance of the
accuracy requirement, the estimate indicates that the requirement is out of reach for that representation. The upper
bound is an error a simple classifier already achieves; if it is within the allowance, the estimate indicates that the
requirement is reachable. If the performance requirement lies between the two bounds, the bounds do not settle whether
it is reachable, and the outcome depends on the model and its training.

### Upper-bound average precision: feasibility for object detection

{term}`Upper-bound Average Precision (UAP)` extends the feasibility concept
to object detection tasks, where the primary performance metric is mean average precision
(mAP) rather than classification accuracy. It is based on the empirical upper bound of
[Borji & Iranmanesh (2019)](#ref3).

Object detection combines two subtasks: **localization** (where is the object?) and **classification** (what is
it?). UAP isolates the classification component by treating the ground-truth bounding box crops as a classification
dataset and scoring them with a crop classifier, which removes localization error from the equation.

UAP is the AP that one crop classifier achieves on correctly localized boxes. That makes it evidence that a level is
reachable for the classification part of the problem, not an upper bound on what a detector can reach: a better crop
classifier, or a detector that uses context the crop leaves out, can score higher.

UAP is marked experimental and may change in future releases.

### Feasibility: determinations with bounds

Feasibility takes a requirement, such as "at least 90% accuracy", and compares it against two
kinds of evidence. One kind says how high performance could possibly go on this data. This is an **envelope**. The
other says how high performance has already been shown to go. This is a **witness**. If the envelope is below the
requirement, the requirement is out of reach under the stated assumptions. If a witness is above the requirement,
the requirement is reachable. If neither is true, the evidence does not settle the question, and Feasibility says
what additional evidence would. For a program, this separates "this requirement cannot be met with this data" from
"this requirement can be met" and from "we do not know yet", before model development begins.

The Feasibility sections use four terms:

- A **requirement** is a performance metric (for example accuracy) and a threshold the metric must reach, given as a
  {class}`.TargetSpec`.
- The **audit sample** is the set of labeled items Feasibility sets aside to score evidence on, kept separate from the
  items used to fit anything else in the analysis. Each independent item in it is an audit unit. For video data, frames
  from one video are grouped, because they are not independent of each other.
- A **frozen model** is a model whose training is finished before it is scored on the audit sample, so its score on
  those items measures how it performs on the population the data represents.
- A **bound** is one confidence limit on a performance metric: the upper limit of an envelope or the lower limit of a
  witness.

{class}`.Feasibility` asks, for each requirement (a performance metric and a threshold), whether the evidence
demonstrates that the threshold is reachable on the population the data represents, rules it out under recorded
assumptions, or does neither. It answers with a **determination** per requirement and the **evidence** behind it, of two
kinds:

- An **envelope** is an upper bound on the performance metric that any model in a stated class can reach on the data.
- A **witness** is a lower bound on the performance metric that one frozen baseline model reaches on the data.

Each bound is a finite-sample confidence bound: it is correct with high probability, not with certainty. One call
returns several bounds and spends a single error budget across all of them (see
[One error budget per call](#one-error-budget-per-call)).

Every determination marks the bounds it rests on, and every bound names its source, whose theorems are listed in
[Feasibility evidence](FeasibilityEvidence.md#the-basis-of-each-source). The feasibility result depends on the data, the
feature representation (the embeddings behind the local panels and the built-in witnesses), the models brought and
the evidence collected.

Feasibility does not ask whether training on this many examples will reach the threshold; that is the question
{class}`.Sufficiency` answers. Feasibility is experimental, and this release covers image and video classification:
accuracy, AP per class or macro-averaged over classes, and, for image classification, the recall of one class at a
stated false-positive limit.

### Sufficiency: the diminishing returns curve

Where Feasibility, BER and UAP concern what level the data allows or a model reaches, {class}`.Sufficiency`
characterizes how performance grows with the amount of training data. It addresses how much additional data
collection is warranted, given the data already available.

Sufficiency trains a model on progressively larger subsets of the training data
(at 10%, 25%, 50%, 75%, and 100% of currently available samples, for example), evaluates
it at each step, and fits a power law curve to the resulting learning
trajectory. That curve can then be used to project performance at larger dataset
sizes and to invert the projection, which gives the number of samples a target
accuracy requires under the fitted curve.

## Theory

### BER estimation: MST and KNN bounds

The formal definition of BER is the expected misclassification rate of the
optimal classifier:

$$\text{BER} = E_X\!\left[P\!\left(Y \neq \underset{i}{\arg\max}\; P(Y=i \mid X=x)\right)\right]$$

This expectation can only be estimated in practice: it requires knowing the true
conditional class distribution $P(Y \mid X)$, which is what the model is trying
to learn. DataEval instead computes **bounds** on BER using two graph-theoretic
estimators that operate on the {term}`embedding
<Embeddings>` space of the dataset.

Both estimators measure the rate of class label mismatches in the local
neighborhood structure of the embedding space. Samples
near the class boundary, where the same image features are associated with
multiple classes, tend to have neighbors with different labels. The frequency of
these mismatches in the neighborhood graph estimates the confusion in the
feature space.

**MST-based estimation** (`ber_mst`) constructs a minimum spanning tree over
the full set of embeddings. Edges in the MST connect each sample to its
globally nearest neighbor under the constraint that the tree spans all samples.
The number of MST edges crossing class boundaries (after subtracting the
$m - 1$ forced inter-class connections required to span $m$ classes) is used
to estimate the error rate. For a dataset of $n$ samples with $m$ classes, the
upper bound is:

$$\hat{e}_\text{upper} = \frac{2 \cdot \text{mismatches}}{n}$$

The lower bound is derived from the multi-class extension of the
Hellman-Devroye inequality (see [Tumer & Ghosh, 1996](#ref9); Th. 3 and Th. 4 in
[Sekeh et al., 2020](#ref2)):

$$\hat{e}_\text{lower} = \frac{m-1}{m}\left(1 - \sqrt{\max\!\left(0,\; 1 - \frac{2m}{m-1}\hat{e}_\text{upper}\right)}\right)$$

**KNN-based estimation** (`ber_knn`) constructs a $k$-nearest neighbor graph
instead. Each sample takes the plurality class among its $k$ nearest neighbors,
with a tie going to the tied class whose member is nearest. The fraction of
samples that vote misclassifies is the upper bound, an error a $k$-NN classifier
achieves:

$$\hat{e}_\text{upper} = \frac{\text{misclassified}}{n}$$

The lower bound inverts the asymptotic relation between the $k$-NN error and the
Bayes error. For two classes and an effective $k$ above 1 (under nearest-tie
voting an even $k$ votes like $k - 1$):

$$\hat{e}_\text{lower} = \frac{\hat{e}_\text{upper}}{1 + a_k}$$

where $a_k$ is the tighter of $1/\sqrt{k}$ ([Renggli et al., 2021](#ref7)) and, for odd
$k \ge 5$, [Devroye's (1981)](#ref1) constant. Otherwise, for $k$ of 1 or 2 or for more than
two classes, [Cover & Hart's](#ref6) inequality is inverted on the 1-NN error, the same
formula as the MST case. The lower bound never exceeds the upper bound.

The two estimators are complementary. The MST estimator uses global structure
and tends to be more stable on smaller datasets. The KNN estimator is sensitive to the
choice of $k$ and can provide tighter bounds when $k$ is well chosen. The
recommended default is $k = 5$. Adaptive choices such as $k = \sqrt{n}$ or
$k = c_0 n^{4/(4+d)}$ (where $d$ is the embedding dimension) have theoretical
support but are less practical for most users.

Both methods return an `upper_bound` and a `lower_bound`, and they point in different directions. The k-NN upper
bound is the error of a real classifier, so an upper bound within an operational allowance indicates that the
requirement is reachable. The lower bound is what indicates that a requirement is out of reach, and only for models
that read this embedding, since a feature map can only raise the Bayes error. Both are asymptotic: in high dimensions
the finite-sample neighbor error sits above its limit, which pushes the lower bound up.

### UAP estimation

UAP is computed directly as the weighted average precision score over the
bounding-box classification problem:

$$\text{UAP} = \sum_c w_c \cdot \text{AP}_c$$

where $w_c$ is the class weight (proportional to class frequency) and
$\text{AP}_c$ is the area under the precision-recall curve for class $c$.
This is scikit-learn's
[`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
with `average="weighted"`.

The reduction from object detection to classification removes the localization burden, but the score comes from one
classifier, so UAP shows what is achievable for correctly localized boxes rather than bounding what any detector can
reach.

### Feasibility: envelopes, witnesses and determinations

Feasibility builds envelopes from agreement among labels and witnesses from frozen models, then applies
five ordered rules to decide what the evidence shows about each requirement. The rules have four possible
determinations:

- `demonstrated_feasible`: a frozen model's lower bound reaches the threshold.
- `infeasible_under_recorded_assumptions`: an envelope's upper bound falls below the threshold.
- `not_ruled_out`: an envelope leaves room for the threshold, but no frozen model has shown it.
- `unresolved`: there is not enough usable evidence to say either way.

An envelope and a witness that contradict each other are a conflict. A conflict is `unresolved` with the limitation
`conflict`: no claim is made, and the cause needs inspection.

For example, suppose the requirement is 90% accuracy. An illustrative envelope upper bound of 85% means the requirement
is `infeasible_under_recorded_assumptions`. A witness lower bound of 93% means it is `demonstrated_feasible`. An
envelope of 97% with a best witness lower bound of 80% means it is `not_ruled_out`: the data does not exclude 90%, but
nothing has shown it either. An envelope of 85% together with a witness of 93% cannot both hold, so the determination is
`unresolved` with the limitation `conflict`. The numbers are illustrative only. The subsections below give the
constructions, and [Feasibility evidence](FeasibilityEvidence.md) gives the exact rules.

When a requirement is neither demonstrated feasible nor infeasible, Feasibility names the limitation, such as too few
independent units, no repeated labels, or no strong enough witness, and the evidence that would address it. [Acting on
results](ActingOnResults.md#performance-limits-findings) describes what to do with these findings.

#### Envelopes from consensus

Feasibility's envelopes rest on the consensus inequality of the Detection Feasibility Envelope (DFE) construction.
An envelope is an upper bound on the accuracy that models in a stated class can reach on this data. It is set by how
much labels disagree for the same input. If annotators label the same image "truck" 70% of the time and "SUV" 30% of
the time, and the labels are independent draws from one label distribution, no model can agree with the annotators more
than 70% of the time on that image.

Repeated labels (`replicas=`): when each input has several independent labels, those labels form its **panel**, a
group of $m$ labels treated as independent draws of the label for that input. The share of a panel's labels that
disagree with its most common label measures how ambiguous the input is. Averaging this share over inputs gives an
estimate of the lowest error any model could reach. Because the estimate comes from a sample, Feasibility
subtracts a confidence margin from it, and one minus the result is the accuracy envelope. Since the panel describes
the input itself, the envelope bounds every model that reads the input.

Local panels: most datasets have only one label per input, so there are no repeated labels to form a panel.
In that case Feasibility substitutes similar images for repeated labels of one image. It groups the images into
cells using unsupervised k-means on the embeddings, then forms panels from $m$ different
images within the same cell. The assumption is that images in one cell are similar enough to share one label
distribution, so their labels act like repeated labels. The cells are pooled by post-stratification, meaning each
cell is weighted by its share of the data. This pooling is DataEval's own construction on the DFE panels (C3); its
derivation is in the Notes of {func}`.local_panel_bound`.

The inequality: for either kind of panel, let $b_m = 1 - \max_k N_k / m$ be the share of the panel's $m$ labels
outside its most common class ($N_k$ is the number of labels in class $k$). Because the expected largest count is
at least the largest expected count,

$$E[b_m] \le 1 - \max_k P(Y = k \mid x),$$

where the right side is the error of the best answer to that input (or, for local panels, to that cell).
On average, a panel's disagreement does not exceed the irreducible error, so it underestimates it in expectation.
Averaged over inputs, the mean consensus value lower-bounds the Bayes risk, so one minus a lower bound on it is
an accuracy envelope.

What local panels change: local panels bound the best single answer per cell, not per image. A model that
tells apart images within a cell can exceed the envelope, so local-panel envelopes are `partition_restricted`: they
limit models at the resolution of the cells, not every model. A cell that mixes classes the embedding fails to
separate lowers the envelope, so a low envelope from local panels can reflect the partition more than the data.
Repeated labels on a random sample (`replicas=`) are therefore preferable.

#### Witnesses from frozen models

A witness is a model fitted without the audit units and scored once on them. Its accuracy bound is a one-sided
bound on the mean of per-unit correctness. Feasibility fits built-in witnesses itself and also accepts a frozen
model's class scores or class decisions as a witness, or the model itself. Where the trials are independent yes-or-no
outcomes, such as whether a witness classified an item correctly, bounds are exact binomial confidence bounds;
elsewhere, for example on consensus values and on grouped video data, they use concentration inequalities.

#### Recall at a false-positive limit

Programs often state a requirement for one class as an operating condition: identify tanks with probability at least
0.90 while labeling at most 5% of the other vehicles as tanks. Feasibility judges this as a recall requirement, which
always carries a false-positive limit, because any recall is reachable by always predicting the class.

#### One error budget per call

Because each bound is computed from a random sample, it can be wrong. A call therefore spends one error budget,
`alpha` (0.05 by default), across every bound of every requirement: with probability at least `1 - alpha`, every
bound from the call is correct at the same time, so you can read all of the call's determinations together without
discounting for how many requirements you asked about. Each pair of a requirement and a source is one claim, every
claim gets an equal share of the error budget, and inside a claim the share is split again across the bounds its
construction needs.
This follows from a union bound. The guarantee holds under the recorded assumptions and covers one call on one fixed
audit. Comparing several representations
or models across calls, rerunning with other seeds, or growing the audit and looking again each requires dividing the
budget further to avoid multiple-hypothesis problems.

### Power law curve fitting for sufficiency

Power-law learning curves have been reported empirically for deep learning
([Hestness et al., 2017](#ref4)). The learning curves produced by {class}`.Sufficiency` are modeled with a
three-parameter **inverse power law**:

$$f(n) = c \cdot n^{-m} + c_0$$

where $n$ is the training set size, $c$ is the scale coefficient, $m > 0$ is
the decay exponent, and $c_0$ is the asymptote. DataEval works internally with
the error form of this curve (higher is worse), then converts to the performance
form for output:

$$\text{performance}(n) = 1 - f(n) = 1 - c \cdot n^{-m} - c_0$$

The asymptote $c_0$ is what makes this model useful for program decisions. As
$n \to \infty$, $f(n) \to c_0$, meaning the fitted performance converges to $1 - c_0$. If
$c_0 > 0$, the fitted curve does not reach 1.0 at any dataset size. The asymptote
estimates an error floor by extrapolating the shape of the learning curve, whereas
BER bounds the floor from the geometry of the embedding space. The two need not agree.

The three parameters are fitted by minimizing the sum of squared residuals:

$$\min_{c, m, c_0} \sum_i \left(f(n_i) - p_i\right)^2$$

where $(n_i, p_i)$ are the observed (sample size, performance) pairs. Because
this loss surface is non-convex and has multiple local minima, DataEval uses
**basin-hopping** ([Wales & Doye, 1997](#ref5)) with L-BFGS-B as the local minimizer.
Initial parameters are set by log-linear regression on the observed points,
and the search runs for up to 1,000 iterations by default (early stopping at
200 iterations without improvement).

For unit-interval metrics (accuracy, precision, recall), the asymptote $c_0$
is constrained to $[0, 1]$. For unbounded metrics (loss), no constraint is
applied. The `unit_interval` parameter controls this.

**Forward projection** evaluates $\text{performance}(n)$ at arbitrary $n$ values
using the fitted parameters. It gives the performance expected with $N$ samples
under the fitted curve.

**Inverse projection** solves $\text{performance}(n) = t$ for $n$ given a
target $t$:

$$n = \left(\frac{t - (1 - c_0)}{-c}\right)^{-1/m}$$

It gives the number of samples needed to reach accuracy $t$ under the fitted curve. When the
target exceeds $1 - c_0$ (the asymptotic ceiling), the inverse projection
returns $-1$, which indicates that the fitted curve does not reach the target with more data of the same
type. This is the sufficiency counterpart of the BER lower bound, with the weaker basis that the asymptote is
extrapolated from the observed points.

Multiple independent runs (controlled by `runs`) average the learning curves
before fitting, reducing the variance introduced by random subset sampling.
The default schedule uses geometrically spaced evaluation points
(`np.geomspace`) across the full training set size, which provides more
resolution at small $n$ where the curve changes fastest.

## Interpretation

- BER reports bounds on the Bayes error for one embedding. A lower bound above the allowance of a requirement
  indicates that the requirement is out of reach for models that read that embedding, and an upper bound within the
  allowance indicates that it is reachable. The bounds are asymptotic estimates. They do not establish that a trained
  model will reach either bound, and they describe the embedding as much as the data.
- UAP reports the average precision of one crop classifier on correctly localized boxes. It is evidence that a level is
  reachable for the classification part of the problem. It is not an upper bound on what a detector can reach, and a
  UAP below the requirement does not rule the requirement out.
- Feasibility reports a determination for each requirement, with the bounds it rests on. A determination is a claim
  about the population the data represents, under the recorded assumptions, with one error budget per call.
  `demonstrated_feasible` and `infeasible_under_recorded_assumptions` settle a requirement. `not_ruled_out` and
  `unresolved` do not: the first means the evidence leaves room for the requirement without showing it, and the second
  means the evidence is insufficient or contradictory. An envelope from local panels limits models at the resolution of
  the cells and not every model.
- Sufficiency reports a fitted learning curve and projections from it. A projection is conditional on the fitted power
  law, and the inverse projection returns $-1$ when the target exceeds the fitted asymptote. If BER has also been run
  and its bounds are high, the two results are consistent with a class-separability limit in the data. If BER is low
  and the inverse projection returns $-1$, the limit may lie in the model, the training procedure, label noise, or a
  train-test mismatch. The two results do not by themselves identify which.

To decide what to do after assessing performance limits, see
[Performance limits findings](ActingOnResults.md#performance-limits-findings) in Acting on results.

## Limitations

BER and UAP estimation both depend on embedding quality. The estimated class
overlap in embedding space is only as meaningful as the embedding's ability to
represent the semantic distinctions relevant to the classification task. An
embedding model that is poorly suited to the target domain may project
separable classes into overlapping regions, producing an artificially high BER
estimate, or may separate classes that are confusable at the pixel level,
producing an artificially low estimate. A BER estimate is therefore a statement
about the embedding as much as about the data, and it carries no signal about
whether that embedding suited the task (see the [Embeddings](Embeddings.md)
concept page).

The MST and KNN estimators produce bounds, not point estimates. The gap
between upper and lower bounds can be wide, particularly for small datasets or
high-dimensional embeddings. A wide gap means the true BER could be anywhere in
a large range. The result is informative but not precise. Increasing sample
count narrows the bounds.

BER is a classification-only metric and UAP is an object detection metric that
reduces the problem to a classification metric. They do not assess feasibility
for regression tasks, generative tasks, or object detection beyond the UAP
reduction. For object detection, UAP scores the classification part only: a
program can see a high UAP and still face localization challenges that prevent
meeting its mAP requirement, and a UAP below the requirement does not rule the
requirement out, since a better crop classifier can score higher.

Feasibility is experimental, and its first release leaves out some tasks and options; the release notes and
[Feasibility evidence](FeasibilityEvidence.md) list them.

Its bounds can be wide. A panel's consensus sits below the risk it bounds, more so for small panels, and more data
does not remove that gap; only the confidence radii shrink with more units. Every claim is about the population of
labeled items the dataset represents, under its annotation protocol: unlabeled items are excluded, and nothing extends
the result to them or to physical truth.

Sufficiency curve fitting can fail or produce unreliable parameters when the
observed learning curve is non-monotonic (due to randomness in small-subset
training), when too few evaluation points are available, or when the data is
far from power-law behavior. The default of 5 substeps is a minimum; more
substeps and multiple runs produce more reliable curve fits. When fitting
produces an asymptote that seems implausibly high or low, inspect the raw
learning curve data before trusting the projection.

The power law model assumes that performance will continue improving
monotonically with more data of the same type. If the limiting factor is
something other than sample count, such as label noise, model capacity, or
distribution mismatch between train and test, the projection is likely to be optimistic.

## When to use it

BER should be run during dataset readiness assessment, before any significant model development investment.
Compare both bounds against the operational accuracy requirement: an upper bound within the allowance indicates the
requirement is reachable; a lower bound beyond it suggests the constraint is in the data, for that representation.

BER is also useful as a comparative metric across dataset versions. If a new
data collection campaign was intended to reduce class overlap, a lower BER
upper bound after the collection is quantitative evidence of reduced class overlap in that embedding.

UAP should be used in object detection programs where mAP is the
acceptance criterion, and where you want evidence of what the classification
part of the problem reaches before investing in detector training. Because it
requires running a classifier on bounding box crops rather than a full
detector, it is typically less costly to compute than training and evaluating
an object detector.

Feasibility is for classification decisions that need evidence with a stated confidence level behind them: before
model development, to learn whether
the evidence rules a requirement out or already shows it reachable, and once a model exists, to show on held-out data
that it meets the requirement. Without repeated labels or allowances, every envelope Feasibility builds is
`partition_restricted`, so a `full_input` requirement can be demonstrated but not ruled out; collect repeated labels on
a random sample of the data, passed as `replicas=`, when ruling out matters.

Sufficiency is most useful at two points: early in a program, when deciding how
much data to collect before beginning training; and after an initial training
run, when deciding whether more data will help. The inverse projection
(`inv_project`) is the primary tool for the first case: it translates a target
accuracy directly into a required sample count. The forward projection
(`project`) is the primary tool for the second case: it shows where the learning
curve is headed and whether the current trajectory will reach the operational
requirement.

## Related concept pages

- [Embeddings](Embeddings.md): the representation space that BER and UAP
  operate in; embedding quality affects the reliability of both estimates
- [Data integrity](DataIntegrity.md): label errors and sensor corruption
  can raise the estimated BER by adding noise at the class boundary; address
  integrity problems before interpreting BER results
- [Dataset bias and coverage](DatasetBias.md): poor coverage can raise the
  effective BER on held-out data even when the training distribution appears
  separable, and bias can cause the BER on the operational distribution to
  exceed the BER measured on the training set
- [Acting on results](ActingOnResults.md): what to do when BER is high or
  sufficiency projection falls short of the operational requirement
- [Feasibility evidence](FeasibilityEvidence.md): each source of bounds behind
  Feasibility's determinations, with its basis, assumptions and error budget

## See this in practice

### Tutorials

- [Assess whether a classification requirement is reachable](../notebooks/tt_assess_ic_feasibility.py): whether
  vehicle type, vehicle category and tank requirements are reachable on
  MilitaryVehicles, the evidence behind each determination, and how to change a
  question the evidence cannot answer

### How-to guides

- [Measure dataset sufficiency for image classification](../notebooks/h2_measure_ic_sufficiency.py)
- [Create embeddings](../notebooks/h2_create_embeddings.py)

## References

1. [Devroye, L. (1981). On the asymptotic probability of error in nonparametric
   discrimination. *The Annals of Statistics*, 9(6), 1320–1327. [paper](https://doi.org/10.1214/aos/1176345648)]{#ref1}

2. [Sekeh, S. Y., Oselio, B., & Hero, A. O. (2020). Learning to bound the
   multi-class Bayes error. *IEEE Transactions on Signal Processing*, 68,
   3793–3807. doi: 10.1109/TSP.2020.2994807 [paper](https://arxiv.org/abs/1811.06419)
   (Theorems 3 and 4 provide the MST and KNN bound derivations used in DataEval.)]{#ref2}

3. [Borji, A., & Iranmanesh, S. M. (2019). Empirical upper bound in object
   detection and more. *arXiv preprint arXiv:1911.12451.* [paper](https://arxiv.org/abs/1911.12451)]{#ref3}

4. [Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad,
   H., Patwary, M. A., Yang, Y., & Zhou, Y. (2017). Deep learning scaling is
   predictable, empirically. *arXiv preprint arXiv:1712.00409.* [paper](https://arxiv.org/abs/1712.00409)]{#ref4}

5. [Wales, D. J., & Doye, J. P. K. (1997). Global optimization by basin-hopping
   and the lowest energy structures of Lennard-Jones clusters containing up to
   110 atoms. *Journal of Physical Chemistry A*, 101(28), 5111–5116. [paper](https://arxiv.org/abs/cond-mat/9803344)
   (Basin-hopping algorithm used for power law curve fitting.)]{#ref5}

6. [Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information
   Theory*, 13(1), 21–27. [paper](https://doi.org/10.1109/TIT.1967.1053964)]{#ref6}

7. [Renggli, C., Rimanic, L., Hollenstein, N., & Zhang, C. (2021). Evaluating Bayes error estimators on real-world
   datasets with FeeBee. *NeurIPS 2021 Datasets and Benchmarks*. [paper](https://arxiv.org/abs/2108.13034)]{#ref7}

8. [Detection Feasibility Envelope (DFE) manuscript: the consensus inequality (replicated-consensus section), the
   disjoint within-cell panels (local-pooling section) and the operating-point inequalities for precision, recall
   and AP.
   (The constructions behind every Feasibility envelope and the AP witness, cited as "DFE" in the basis table of
   [Feasibility evidence](FeasibilityEvidence.md#the-basis-of-each-source). The post-stratified pooling of the local
   panels is DataEval's own, proved in `local_panel_bound`. The other sources are in the references of that
   page.)]{#ref8}

9. [Tumer, K., & Ghosh, J. (1996). Estimating the Bayes error rate through classifier combining. *Proceedings of the
   13th International Conference on Pattern Recognition*, 2, 695–699.
   [paper](https://doi.org/10.1109/ICPR.1996.546912)]{#ref9}
