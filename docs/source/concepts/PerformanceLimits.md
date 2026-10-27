<!-- markdownlint-disable MD051 -->

# Performance limits

Before investing in model architecture, hyperparameter tuning, or additional
data collection, a T&E engineer needs to answer a more fundamental question:
what is this dataset capable of producing? Performance Limits analysis addresses
three questions: is a requirement out of reach on this data, has some model
already shown that it is reachable, and where do additional samples stop
producing meaningful improvement? {class}`.Feasibility` answers the first two
with finite-sample bounds and records the evidence behind each answer. The BER
estimators estimate the first asymptotically for classification, UAP shows what
one crop classifier reaches on correctly localized detection boxes, and
{class}`.Sufficiency` answers the third.

Understanding these limits has direct program consequences. If the irreducible
error of a dataset exceeds the allowance of an operational accuracy requirement,
no model trained on that data can meet the requirement: the problem must change,
not the model. If the learning curve has already saturated, budget directed at
data collection will yield diminishing returns, and engineering effort is better
spent elsewhere. Both findings are as valuable as a positive result: they
prevent programs from pursuing improvements that are mathematically impossible
given the current data.

```{important}
The Bayes Error Rate (BER) metric is applicable to **classification tasks only**.
BER operates on class embeddings. The Upper-bound Average Precision (UAP) metric
reduces an object detection problem to a classification problem over bounding
box crops. See the UAP section below for details. The experimental
{class}`.Feasibility` evaluator supports image and video classification in this
release; object detection and tracking datasets raise `NotImplementedError`.
```

## What they are

### Bayes error rate: the irreducible floor

The {term}`Bayes Error Rate (BER)` is the lowest possible error rate achievable
by any classifier on a given probability distribution. It is not a property of
any particular model. It is a property of the data. No matter how large the
model, how long it trains, or how well it is tuned, it cannot achieve error
below the BER on that distribution.

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
accuracy requirement, the requirement is out of reach for that representation. The upper bound is an error a simple
classifier already achieves; if it is within the allowance, the requirement is reachable. Between the two, the data
alone does not decide.

### Upper-bound average precision: feasibility for object detection

{term}`Upper-bound Average Precision (UAP)` extends the feasibility concept
to object detection tasks, where the primary metric is mean average precision
(mAP) rather than classification accuracy.

Object detection combines two subtasks: **localization** (where is the object?) and **classification** (what is
it?). UAP isolates the classification component by treating the ground-truth bounding box crops as a classification
dataset and scoring them with a crop classifier, which removes localization error from the equation.

UAP is the AP that one crop classifier achieves on correctly localized boxes. That makes it evidence that a level is
reachable for the classification part of the problem, not an upper bound on what a detector can reach: a better crop
classifier, or a detector that uses context the crop leaves out, can score higher.

UAP is marked experimental and may change in future releases.

### Feasibility: determinations with evidence

{class}`.Feasibility` asks, for each target (a metric, its details and a threshold), whether the evidence demonstrates
that the threshold is reachable on the population the data represents, rules it out under recorded assumptions, or
does neither. It answers with a **determination** per target and the **evidence** behind it, of two kinds:

- A **ceiling** is an upper endpoint on the metric that any model in a stated class can reach on the population.
- A **witness** is a lower endpoint on the metric that one frozen model reaches on the population, scored once on
  held-out audit units.

Each endpoint is a finite-sample confidence bound, not an asymptotic estimate, and all the endpoints of one call hold
together with probability at least `1 - alpha`. Every determination names the evidence items it rests on, and every
evidence item names the theorems it rests on in its `basis`. Which determination comes back depends on the data, the
representation, the models brought and the evidence collected.

Feasibility does not ask whether training on this many examples will reach the threshold; that is the question
{class}`.Sufficiency` answers. Feasibility is experimental, and this release covers image and video classification:
accuracy, AP per class or macro-averaged over classes, and, for image classification, the recall of one class at a
stated false-positive limit.

### Sufficiency: the diminishing returns curve

Where Feasibility, BER and UAP concern what level the data allows or a model reaches, {class}`.Sufficiency`
characterizes how performance grows with the amount of training data. It answers the question: given the data you have,
how much of your budget should go toward collecting more?

Sufficiency trains a model on progressively larger subsets of the training data
(at 10%, 25%, 50%, 75%, and 100% of available samples, for example), evaluates
it at each step, and fits a power law curve to the resulting learning
trajectory. That curve can then be used to project performance at larger dataset
sizes and to invert the projection: given a target accuracy, how many samples
are required to reach it?

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
neighborhood structure of the embedding space. The intuition is that samples
near the class boundary, where the same image features are associated with
multiple classes, will have neighbors with different labels. The frequency of
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
Hellman-Devroye inequality (see Tumer & Ghosh, 1996; Th. 3 and Th. 4 in
Sekeh et al., 2020):

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

where $a_k$ is the tighter of $1/\sqrt{k}$ (Renggli et al., 2021) and, for odd
$k \ge 5$, Devroye's (1981) constant. Otherwise, for $k$ of 1 or 2 or for more than
two classes, Cover & Hart's inequality is inverted on the 1-NN error, the same
formula as the MST case. The lower bound never exceeds the upper bound.

The two estimators are complementary. The MST estimator uses global structure
and is more stable on smaller datasets. The KNN estimator is sensitive to the
choice of $k$ but provides tighter bounds when $k$ is well chosen. The
recommended default is $k = 5$. Adaptive choices such as $k = \sqrt{n}$ or
$k = c_0 n^{4/(4+d)}$ (where $d$ is the embedding dimension) have theoretical
support but are less practical for most users.

Both methods return an `upper_bound` and a `lower_bound`, and they point in different directions. The k-NN upper
bound is the error of a real classifier, so an upper bound within an operational allowance shows the requirement is
reachable. The lower bound is what shows a requirement is out of reach, and only for models that read this
embedding, since a feature map can only raise the Bayes error. Both are asymptotic: in high dimensions the
finite-sample neighbor error sits above its limit, which pushes the lower bound up.

### UAP estimation

UAP is computed directly as the weighted average precision score over the
bounding-box classification problem:

$$\text{UAP} = \sum_c w_c \cdot \text{AP}_c$$

where $w_c$ is the class weight (proportional to class frequency) and
$\text{AP}_c$ is the area under the precision-recall curve for class $c$.
This is scikit-learn's [`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
with `average="weighted"`.

The reduction from object detection to classification removes the localization burden, but the score comes from one
classifier, so UAP shows what is achievable for correctly localized boxes rather than bounding what any detector can
reach.

### Feasibility: ceilings, witnesses and determinations

#### Ceilings from consensus

Feasibility's ceilings rest on the consensus inequality of the Detection Feasibility Envelope (DFE) construction.
Draw $m$ labels for one input independently under the annotation protocol, and let $b_m = 1 - \max_k N_k / m$ be the
share of them outside the panel's most common class. Because the expected largest count is at least the largest
expected count,

$$E[b_m] \le 1 - \max_k P(Y = k \mid x),$$

the error of the best answer to that input. Averaged over inputs, the mean consensus value lower-bounds the Bayes
risk, so one minus a lower endpoint on it is an accuracy ceiling. Feasibility forms panels two ways:

- Repeated labels (`replicas=`): an input's own labels form its panel, so the ceiling bounds every model that
  reads the input.
- Local panels: on single-label data, Feasibility fits a label-blind k-means partition of the representation on
  a fit split, then draws disjoint panels of $m$ different audit images within each cell. Their labels are draws from
  the cell's label distribution, so their consensus bounds the error of the best single answer per cell. The cells
  are pooled by post-stratification. This pooling is DataEval's own construction on the DFE panels (C3), proved with
  two Hoeffding (1963) steps in the Notes of {func}`.local_panel_bound`:

  $$\ell = \sum_h \frac{n_h}{N}\left(\bar g_h - \beta_h\right) - r_1 - r_2,$$

  where $n_h$ of the $N$ audit units fall in cell $h$, $\bar g_h$ is the mean consensus value of the cell's panels (0
  without panels), $\beta_h$ is an optional allowance (below), and $r_1$ and $r_2$ are Hoeffding (1963) radii for the
  panels given the cells and for the cell shares. The accuracy ceiling is $1 - \max(0, \ell - \Gamma)$.

AP ceilings use the one-vs-rest consensus value $\min(k/m, \lambda(1 - k/m))$, where $k$ counts a class's labels in
the panel and $\lambda$ is the cost of a false alarm relative to a missed positive. Lower bounds on that cost-weighted
risk, at the costs where it changes slope, and an upper endpoint on the class's prevalence feed the risk-to-AP envelope
of Reid and Williamson (2011), an upper bound on any scorer's population AP. The raw `costs` and `risk_lower` of an AP
ceiling keep the cost 0, whose risk is trivially 0; a recall ceiling (below) drops it.

#### Witnesses from frozen models

A witness is a model fitted without the audit units and scored once on them. Its accuracy endpoint is a one-sided
bound on the mean of per-unit correctness. Its AP endpoint is a staircase under its precision-recall curve (Davis and
Goadrich, 2006), built from lower endpoints on true positives and upper endpoints on false positives at score
thresholds fixed from the fit split. Feasibility fits two built-in witnesses on the fit split, a 5-nearest-neighbor
classifier and a logistic-regression probe, and accepts a frozen model's class scores or class decisions as
`predictions=` (see [Decision arrays](#decision-arrays-and-folded-scores)).

#### Exact binomial endpoints

Where the trials are independent items, each indicator (correct or not, a class decision or not, a label equal to the
class or not) is a Bernoulli trial, and Feasibility bounds its mean with the exact one-sided Clopper–Pearson (1934)
endpoint, a beta quantile that holds at its stated confidence for any sample size ({func}`.binomial_endpoint`). It is
used for accuracy witnesses, the recall and false-positive endpoints of a recall witness, the true-positive and
false-positive endpoints and the prevalence endpoint of an AP witness, and the prevalence endpoints of local-panel
ceilings. It is not used on grouped (video) data, where items are not independent, or for the consensus values and
risk lower bounds, which are bounded variables rather than Bernoulli trials; those keep the hybrid of a Hoeffding
(1963) radius and an empirical Bernstein radius (Maurer and Pontil, 2009, Theorem 4), each at half the endpoint's
error share, or Hoeffding alone where a source documents it. Each endpoint records the `method` it used. Because
Clopper–Pearson endpoints are asymmetric, the suggested minimum inverts the endpoint itself rather than a radius
formula.

#### Recall at a false-positive limit

Programs often state a requirement for one class as an operating condition: identify tanks with probability at least
0.90 while labeling at most 5% of the other vehicles as tanks. Feasibility judges it as a recall target, spelled
`TargetSpec(task="IC", metric="recall", threshold=r0, class_name=c, max_false_positive_rate=f0)`, with `threshold` the
minimum recall $r_0$ in $(0, 1]$ and $f_0$ in $(0, 1)$. Both numbers are part of the requirement's identity, and
recall without a limit is refused, since any recall is reachable by always predicting the class. The requirement holds
for a model class when some model in it has recall $P(h(x) = c \mid y = c) \ge r_0$ and false-positive rate
$P(h(x) = c \mid y \ne c) \le f_0$. The negatives are `false_positive_scope="pooled"`, the population's mixture of every
other declared class. In this release a recall target is refused on grouped (video) data, with `false_positive_scope`
`"each_class"`, and as a mapping key, which cannot carry two numbers. `class_name` names the class in the dataset's own
vocabulary, so a requirement on a coarser class follows a `Relabel`
(see [Decision arrays and folded scores](#decision-arrays-and-folded-scores)).

A frozen witness gives a decision $d_i$ for each audit item: the top class of its
scores, or the supplied class decision. With $P$ the audit items labeled $c$ and $N$ the others, it gets two exact
binomial endpoints, each at half the (target, source) claim's share of `alpha`: a lower endpoint on recall, from the
decisions $d_i = c$ among $P$, and an upper endpoint on the false-positive rate, from those among $N$. Given the labels,
the indicators within $P$ and within $N$ are independent Bernoulli trials, because the audit items are independent and
the policy is frozen. A witness is **certified** when its own false-positive upper endpoint is at or below $f_0$, and
only a certified witness counts: its recall lower endpoint is the one the determination rules compare with the
threshold, and one witness must supply both bounds, so two witnesses that each give one favorable bound demonstrate
nothing. The false-positive endpoint itself is never compared with the threshold or a ceiling. An uncertified witness
supports neither a demonstration nor an `evidence_conflict`, but both of its bounds stay on the evidence and in
`aggregate_by_evidence()`. Without audit positives or without audit negatives, the corresponding endpoint is
unavailable.

The cost-weighted consensus constructions that bound AP bound a recall target too. For a positive cost
$\lambda_j$ at a slope change of the consensus, a local panel or replica route gives $\ell_j$, a lower bound on
$\inf_h [P(y = c, h \ne c) + \lambda_j P(y \ne c, h = c)]$. A model with recall $r \ge r_0$ and false-positive rate
$f \le f_0$ has $\pi(1 - r) + \lambda_j (1 - \pi) f \le \pi(1 - r_0) + \lambda_j (1 - \pi) f_0$, where $\pi$ is the
class's prevalence. With a two-sided confidence interval $[\pi^-, \pi^+]$ on $\pi$ from the same ledger, Feasibility
reports the ceiling $U_r$ as an upper bound on the recall any model in the ceiling's scope reaches at a false-positive
rate of at most $f_0$:

$$
U_r = \min_j \max_{\pi \in [\pi^-, \pi^+]} \left[\, 1 - \lambda_j f_0 - \frac{\ell_j - \lambda_j f_0}{\pi} \,\right],
\quad \text{clipped to } [0, 1]
$$

Each bracket increases in $\pi$ when $\ell_j > \lambda_j f_0$, so its maximum sits at $\pi^+$; it decreases when
$\ell_j < \lambda_j f_0$, so the maximum sits at $\pi^-$ (and is unbounded, reported as 1, when $\pi^- = 0$); and it is
the constant $1 - \lambda_j f_0$ when they are equal. Which prevalence end is conservative depends on the cost, so the
lower prevalence endpoint is required, where an AP ceiling needs only the upper one. The target is excluded if and only
if $U_r < r_0$, and then the five determination rules apply unchanged. The claim's share of `alpha` is split equally
over the positive costs and the two prevalence endpoints. The raw `costs` and `risk_lower` of a recall ceiling drop the
cost 0 that an AP ceiling keeps, since a cost-0 risk is trivially 0 and bounds nothing.

A recall ceiling inherits its route's scope: local panels bound only models that give one answer per cell, so the
ceiling is `partition_restricted` (in the `local_panel_ceiling/v1:cost_weighted` item when allowances split the
local-panel item), and replicas bound every model reading the full input, so it is `full_input`. It never says anything
about one model. A certified witness whose recall lower endpoint exceeds $U_r$ is an `evidence_conflict`, whether or
not it reaches $r_0$.

#### Decision arrays and folded scores

`predictions=` takes a frozen model's class scores, whose top class is the decision, or a 1-D integer array of class
decisions, one per dataset item. Decisions are indices of the dataset's declared classes, validated against its
vocabulary: an index outside it is refused, and so is a boolean or float array, which cast to integers would silently
mean classes 0 and 1. Feasibility has no decision threshold of its own and does not search for one on the audit. To
judge a different operating point, such as a threshold on one class's score, apply it before the call as a complete
multiclass policy (the class when its score reaches the threshold, otherwise the highest-scoring other class) and pass
the decisions. `provenance["decision_policy"]` records that rule, accepted only with decisions, and the evidence
records it as an assumption: the decisions follow a policy frozen before the audit. One call's decisions feed every
target in it, so a different operating point needs its own call, with the error budget split across the planned calls.
Decisions support accuracy and recall targets; they rank nothing, so an AP target gets no witness from them, and the
endpoint is unavailable with that reason.

A model's scores follow the vocabulary it was trained on. When a requirement is stated in a coarser vocabulary,
relabel the dataset with {class}`.Relabel` (for instance with an ontology that maps vehicle types to categories) and
fold the model's scores through the same map with `Relabel.fold_scores(scores, view=view)`. Columns of source classes
that map to one target class are summed, columns of classes the relabel drops are discarded, nothing is renormalized
(as `Relabel` folds a classification datum's own score vector), and with `view` the rows of dropped items are
removed so the result lines up with the relabeled view's items. The result has one column per target class index.

#### One error budget per call

A call spends one error budget, `alpha` (0.05 by default), across every endpoint of every target. Each pair of a
target and an evidence source is one claim, every claim gets an equal share, and inside a claim the share is split
again across the endpoints its construction needs. By a union bound, every endpoint of the call holds at once with
probability at least `1 - alpha`. The guarantee covers one call on one fixed audit:

- comparing $k$ representations or models across calls needs `alpha / k` per call;
- rerunning with $k$ other seeds and keeping the best result is a selection among those $k$ calls, and needs
  `alpha / k` per call as well;
- growing the audit and looking again needs a {class}`.SamplingPlan` declared before the first look. Look $t$ uses the
  first `sizes[t - 1]` units of one seeded order and spends $\alpha / (t(t + 1))$, which sums to $\alpha$ over all
  looks.

#### The five determinations

For each target, Feasibility considers only the endpoints for that target on the call's population. Ceilings count at
the judged information scope or broader, and witnesses where they are eligible (both defined below). The first rule
that matches decides:

| Order | Determination | When | What it claims |
| --- | --- | --- | --- |
| 1 | `evidence_conflict` | A ceiling's upper endpoint is below the lower endpoint of a witness eligible at that ceiling's scope | No claim is made: the determination is suspended. The two endpoints cannot both hold, so either a recorded assumption failed or this call fell in its failure event, which has probability at most `alpha` and in which some endpoint does not hold. Likely causes of a failed assumption: audit units in the witness's training data, ungrouped duplicates or sequences, or a witness reading more than the ceiling's scope covers. Basis and scope: those of the two conflicting endpoints, named in the reason. |
| 2 | `demonstrated_feasible` | The largest lower endpoint of a witness eligible at the judged scope is at or above the threshold | The frozen witness reaches the threshold on the population, at the call's confidence. Basis: the witness's concentration endpoint. Scope: every judged scope whose model class contains the witness. It assumes the witness's recorded provenance: trained without the audit units and frozen before scoring them. |
| 3 | `infeasible_under_recorded_assumptions` | The smallest ceiling upper endpoint is strictly below the threshold | No model in the ceiling's class reaches the threshold on the population, at the call's confidence, under the ceiling's recorded assumptions. Basis: the ceiling's consensus construction. Scope: the ceiling's information scope, which is the judged scope or broader. |
| 4 | `not_ruled_out` | An informative ceiling upper endpoint is at or above the threshold, and no eligible witness reaches it | The deciding ceiling does not exclude the threshold at its information scope, under its recorded assumptions. Basis: its consensus construction. Scope: the ceiling's information scope, the judged scope or broader. It is not evidence of feasibility. |
| 5 | `unresolved` | Otherwise | No claim is made. No ceiling at the judged scope or broader is available and informative, and no eligible witness reaches the threshold; the reason says which, and ceilings narrower than the judged scope remain evidence only. |

Rules 2 and 3 cannot both match without rule 1 matching first: a witness at or above the threshold would exceed a
ceiling below it. An endpoint is **unavailable** when it cannot be computed (too few units, no complete panel, a
missing declared bound), and **uninformative** when it is valid but trivial: a ceiling at the metric's maximum of 1, or
a witness at its minimum of 0. Neither can make a ceiling decide rule 4. For a recall target, a witness counts in these
rules only when it is certified at the false-positive limit, and the endpoint compared with the threshold or a ceiling
is its recall endpoint (see [Recall at a false-positive limit](#recall-at-a-false-positive-limit)).

The **recorded assumptions** are these; each evidence item lists the ones it relies on in `details["assumptions"]`:

- Labels follow the annotation protocol. Every claim, ceiling or witness, concerns the labels the annotation
  protocol produces, not physical truth. A replica ceiling records that the replicas and the primary labels follow one
  protocol, each an independent label of the item, so that it and the witnesses, scored on primary labels, concern the
  same target.
- Replicas are a random sample. The items with replicas are a random sample of the population, so a replica ceiling
  describes the population and not only those items.
- The representation is independent of the audit labels. The extractor or embeddings were fitted and selected
  without the audit units' labels. You declare this; Feasibility records it for local panels and built-in witnesses.
- The allowances hold for their scopes. β and Γ, when given, are bounds you supply. Feasibility assumes them and
  does not estimate them.

Grouped (video) data and sampling plans record more: annotators labeled the replicas with the same video context;
for `sequence_mean` targets, the replicated sequences are a random sample with the replicated frames uniform within
each; for `frame_pooled` targets, the replicated frames are a uniform sample of all frames (whole random sequences, or
each frame independently with one probability); and with a sampling plan, the replica set was fixed before the first
look.

#### Information scopes

Each ceiling and each witness belongs to a class of models, its **information scope**:

| Scope | Models in the class |
| --- | --- |
| `full_input` | Any model that reads everything the eventual model is allowed to read |
| `fixed_representation` | Models that read only the given representation (the embeddings) |
| `partition_restricted` | Models that give one fixed answer per cell of the frozen partition |

The classes nest: `full_input` ⊃ `fixed_representation` ⊃ `partition_restricted`. A ceiling bounds every model in its
class, so it decides a target judged at its own scope or any narrower one, never a broader one. A target names the
scope it is judged at, `full_input` by default (the key `"accuracy/fixed_representation"` judges another). A ceiling
narrower than the judged scope is listed in the row's `evidence_ids` under its own scope, but it never decides and never
fills the `ceiling` column: a restricted ceiling is always reported as restricted.

| Ceiling | Information scope |
| --- | --- |
| Repeated labels | `full_input` |
| Local panels, accuracy, image data, no allowances | `partition_restricted` |
| Local panels, accuracy, image data, with β | `fixed_representation` |
| Local panels, accuracy, image data, with β and Γ | `full_input` |
| Local panels, AP or recall, or any local panel on grouped (video) data | `partition_restricted` |

The allowances bridge the scopes, in risk units:

- β (`allowances={"beta": ...}`, a scalar) bounds how much a model reading the representation can improve on one
  fixed answer inside a cell. It must be a justified bound: β = 0 asserts that no model reading the representation
  beats one fixed answer per cell. A per-cell β would refer to the cells of the internal partition, which are not
  exposed, so this release rejects it.
- Γ (`"gamma"`, a scalar, given with β) bounds how much reading the full input can lower the best risk below what
  the representation allows.

A witness can demonstrate a target only if it belongs to the judged scope's class. The built-in witnesses read the
representation, so they are eligible at `fixed_representation` and `full_input`. A user witness declares what it
reads in `provenance["reads"]`: `"full_input"` (the default) is eligible at `full_input`, and `"representation"` at
`fixed_representation` and `full_input`. A witness giving one answer per cell would be eligible at every scope, but
it needs the internal partition, which this release does not expose: `"partition"` is rejected, and no witness is
eligible for a `partition_restricted` target, which only a ceiling can decide.

#### Evidence

Each evidence item is one source run on one sample. It carries a fixed spine: its `kind`, whether it `supports` a
ceiling or a witness, its `basis`, `population`, `sample`, `information_scope`, independent `unit`, `provenance`, and
one validated endpoint per target it bounds, with the endpoint's value, direction and share of `alpha`. The
determination rules read only the spine. Free-form `details` carry the assumptions and the raw core results behind
each endpoint, for programmatic access. A call can produce four kinds of evidence:

- `local_panel_ceiling/v1`, from local panels on the dataset's own labels;
- `replica_ceiling/v1`, from repeated labels passed as `replicas=`;
- `builtin_witness/v1`, the k-NN and probe witnesses;
- `user_witness/v1`, from a frozen model's `predictions=`, class scores or class decisions.

[Feasibility evidence](FeasibilityEvidence.md) documents each kind's basis, `details` and error budget.

#### Recommendations

A final determination, `demonstrated_feasible` or `infeasible_under_recorded_assumptions`, needs no recommendation.
`evidence_conflict` gets the limiting factor `conflict` and a recommendation to inspect the likely causes. Any other
determination names the first limiting factor that applies and the evidence that addresses it:

| Order | Limiting factor | When | Evidence to collect |
| --- | --- | --- | --- |
| 1 | `scope_restricted` | No ceiling at the judged scope or broader, but a narrower one is available | Repeated labels on a randomly sampled subset of items, for a `full_input` ceiling; a justified β/Γ allowance; or accept the narrower claim |
| 2 | `missing_bound` | A ceiling is unavailable for lack of a declared bound such as `max_sequence_length` | Declare the bound |
| 3 | `sampling_width` | Every ceiling is unavailable for lack of units or complete panels | More independent units |
| 4 | `no_witness` | No eligible witness is available | A frozen model's predictions on held-out units, from the judged scope's model class; at `partition_restricted` none can be eligible in this release |
| 5 | `sampling_width` | The deciding ceiling's point estimate is below the threshold, or the best witness's above it, but the endpoint is not | More independent units |
| 6 | `no_replicas` | Every ceiling, if any, is `partition_restricted` and no replica evidence exists | Repeated labels on a randomly sampled subset of items |
| 7 | `no_witness` | Otherwise: the best witness is weak | A stronger model's predictions on held-out units |

For a recall target, the factor follows the witness's observed rates:

- `sampling_width` when the observed recall is at or above the minimum and the observed false-positive rate at or below
  the limit, but a confidence bound falls short: the recall lower bound, the false-positive upper bound or both. The
  suggested minimum counts the positives the recall bound needs and the negatives the false-positive bound needs,
  converts each to evaluation images through the estimated prevalence (rounded up, the prevalence being the audit's
  share of positives) and takes the larger. It assumes the observed rates persist, and the printout says which bound
  would cross and in what units.
- `no_witness` when the observed recall is below the minimum or the observed false-positive rate above the limit: a
  stronger model or a different frozen operating point is needed, and there is no suggested minimum.

When unlabeled items were excluded, the evidence covers labeled items only. That is a property of the population, not
of the evidence, so it never displaces the factor: the recommendation ends with a note to label the excluded items to
extend the result to the full dataset. The `labels_conditioned` value stays in the column's enum but is not produced.

When the deciding ceiling's point estimate is already below the threshold, or the best witness's
above it, the endpoint misses only by its confidence radius. `suggested_minimum` inverts that radius at the observed
mean and variance: it is the number of independent units behind the endpoint at which it would cross the threshold if
those rates hold. For local panels and witnesses those are audit-split units (the audit is half of the independent units
in this release), and the number is rounded up to the next size of a declared sampling plan. For a replica ceiling they
are units with replicas, which are not plan units, so it is not rounded. It is set for such a crossing whatever the
limiting factor. For a recall target only a witness projects: the minimum counts the positives its recall bound needs
and the negatives its false-positive bound needs, converts each to evaluation images through the estimated prevalence
and takes the larger. It is null when the determination is final or a conflict, when no endpoint's point estimate is
past the threshold (for recall, when no witness's observed rates meet the requirement), when the endpoint is a recall
ceiling (it has no single point estimate to project), when the endpoint has no closed-form radius to invert (AP
endpoints and `frame_pooled` ratios), when no size up to $10^7$ suffices, or when the size lies beyond the sampling
plan. An exact binomial endpoint is inverted directly: {func}`.units_to_cross` finds the smallest number of trials at
which the Clopper–Pearson bound, evaluated at the observed rate projected to that size, clears the gap, from the lower
bound for a lower endpoint and the upper bound for an upper one.

### Power law curve fitting for sufficiency

The learning curves produced by {class}`.Sufficiency` are modeled with a
three-parameter **inverse power law**:

$$f(n) = c \cdot n^{-m} + c_0$$

where $n$ is the training set size, $c$ is the scale coefficient, $m > 0$ is
the decay exponent, and $c_0$ is the asymptote. DataEval works internally with
the error form of this curve (higher is worse), then converts to the performance
form for output:

$$\text{performance}(n) = 1 - f(n) = 1 - c \cdot n^{-m} - c_0$$

The asymptote $c_0$ is what makes this model useful for program decisions. As
$n \to \infty$, $f(n) \to c_0$, meaning performance converges to $1 - c_0$. If
$c_0 > 0$, performance will never reach 1.0 regardless of dataset size,
reflecting the same irreducible error that BER measures directly, now estimated
from the shape of the learning curve rather than from the embedding space
geometry.

The three parameters are fitted by minimizing the sum of squared residuals:

$$\min_{c, m, c_0} \sum_i \left(f(n_i) - p_i\right)^2$$

where $(n_i, p_i)$ are the observed (sample size, performance) pairs. Because
this loss surface is non-convex and has multiple local minima, DataEval uses
**basin-hopping** (Wales & Doye, 1997) with L-BFGS-B as the local minimizer.
Initial parameters are set by log-linear regression on the observed points,
and the search runs for up to 1,000 iterations by default (early stopping at
200 iterations without improvement).

For unit-interval metrics (accuracy, precision, recall), the asymptote $c_0$
is constrained to $[0, 1]$. For unbounded metrics (loss), no constraint is
applied. The `unit_interval` parameter controls this.

**Forward projection** evaluates $\text{performance}(n)$ at arbitrary $n$ values
using the fitted parameters. This answers: if you had $N$ samples, what accuracy
would you expect?

**Inverse projection** solves $\text{performance}(n) = t$ for $n$ given a
target $t$:

$$n = \left(\frac{t - (1 - c_0)}{-c}\right)^{-1/m}$$

This answers: how many samples do you need to reach accuracy $t$? When the
target exceeds $1 - c_0$ (the asymptotic ceiling), the inverse projection
returns $-1$, indicating the target is unachievable with more data of the same
type. This is the sufficiency equivalent of the BER bound: the learning curve
has told you that this requirement cannot be met by scaling data alone.

Multiple independent runs (controlled by `runs`) average the learning curves
before fitting, reducing the variance introduced by random subset sampling.
The default schedule uses geometrically spaced evaluation points
(`np.geomspace`) across the full training set size, which provides more
resolution at small $n$ where the curve changes fastest.

## When to use it

BER should be run during dataset readiness assessment, before any significant model development investment.
Compare both bounds against the operational accuracy requirement: an upper bound within the allowance shows the
requirement is reachable; a lower bound beyond it suggests the constraint is in the data, for that representation.

BER is also useful as a comparative metric across dataset versions. If a new
data collection campaign was intended to reduce class overlap, a lower BER
upper bound after the collection is quantitative evidence that it succeeded.

UAP should be used in object detection programs where mAP is the
acceptance criterion, and where you want evidence of what the classification
part of the problem reaches before investing in detector training. Because it
requires running a classifier on bounding box crops rather than a full
detector, it is substantially cheaper to compute than training and evaluating
an object detector.

Feasibility is for classification decisions that need a proof behind them: before model development, to learn
whether the evidence rules a requirement out or already shows it reachable, and once a model exists, to show on held-out
data that it meets the requirement. Put every target into one call, so they share one error budget, and pass your
model's class scores, or its class decisions, as `predictions=` with its provenance. Without repeated labels or
allowances, every ceiling Feasibility builds is `partition_restricted`, so a `full_input` target can be demonstrated but
not ruled out; collect repeated labels on a random sample of the data, passed as `replicas=`, when ruling out matters.

For a per-class operating condition, state it as a recall target with its false-positive limit, freeze the decision
policy before the audit, and put that policy's decisions in its own call. When the requirement is about a coarser class
than the model was trained on, relabel the dataset and fold the model's scores through the same `Relabel`.

Sufficiency is most useful at two points: early in a program, when deciding how
much data to collect before beginning training; and after an initial training
run, when deciding whether more data will help. The inverse projection
(`inv_project`) is the primary tool for the first case: it translates a target
accuracy directly into a required sample count. The forward projection
(`project`) is the primary tool for the second case: it shows where the learning
curve is headed and whether the current trajectory will reach the operational
requirement.

When the inverse projection returns $-1$ for a target's sample count, the interpretation
depends on context. If BER has already been run and returns a high upper bound,
the two findings are consistent: the data has a fundamental separability
problem. If BER is low but sufficiency projects $-1$, the issue is more likely
the model architecture or training procedure rather than the data itself.

To better understand what to do after assessing performance limits, review the
[Performance Limits section in the Acting on Results explanation page](ActingOnResults.md#performance-limits-findings).

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

Feasibility is experimental, and its first release leaves out several things:

- Object detection and tracking datasets raise `NotImplementedError`.
- `frame_pooled` video targets get no local-panel ceiling yet; their ceilings come only from repeated labels.
- Feasibility always fits its own label-blind k-means partition; a partition you supply is not accepted.
- The allowances β and Γ are taken as given and recorded as assumed, never estimated. A per-cell β and a witness that
  reads the partition (`reads="partition"`) are rejected, since the partition is internal.
- The panel size and cell count are provisional defaults; the evidence records both as provisional.
- A recall target is for image classification only: it is refused on grouped (video) data, with
  `false_positive_scope="each_class"` (one limit per other class) and as a mapping key. It covers one class at one
  operating point; a second operating point needs its own call. Feasibility judges whether the requirement is
  reachable; it does not evaluate or accept a model, and an uncertified witness is not evidence for it.

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
distribution mismatch between train and test, the projection will be optimistic.

## Related concept pages

- [Embeddings](Embeddings.md) — the representation space that BER and UAP
  operate in; embedding quality directly affects the reliability of both
  estimates
- [Data integrity](DataIntegrity.md) — label errors and sensor corruption
  raise BER by introducing noise into the class boundary; fix integrity
  problems before interpreting BER results
- [Dataset bias and coverage](DatasetBias.md) — poor coverage raises
  effective BER on held-out data even when the training distribution appears
  separable; bias causes BER on the operational distribution to exceed BER
  measured on the training set
- [Acting on results](ActingOnResults.md) — what to do when BER is high or
  sufficiency projection falls short of the operational requirement
- [Feasibility evidence](FeasibilityEvidence.md) — each kind of evidence behind
  Feasibility's determinations, with its basis, details and error budget

## See this in practice

### Tutorials

- [Assess whether a classification requirement is reachable](../notebooks/tt_assess_ic_feasibility.py) — whether
  vehicle type, vehicle category and tank requirements are reachable on
  MilitaryVehicles, the evidence behind each determination, and how to change a
  question the evidence cannot answer

### How-to guides

- [How to determine image classification feasibility](../notebooks/h2_measure_ic_feasibility.py)
- [How to measure dataset sufficiency for image classification](../notebooks/h2_measure_ic_sufficiency.py)
- [How to create embeddings](../notebooks/h2_create_embeddings.py)

## References

1. Devroye, L. (1981). On the asymptotic probability of error in nonparametric
   discrimination. *The Annals of Statistics*, 9(6), 1320–1327. [paper](https://doi.org/10.1214/aos/1176345648)

2. Sekeh, S. Y., Oselio, B., & Hero, A. O. (2020). Learning to bound the
   multi-class Bayes error. *IEEE Transactions on Signal Processing*, 68,
   3793–3807. doi: 10.1109/TSP.2020.2994807 [paper](https://arxiv.org/abs/1811.06419)  
   (Theorems 3 and 4 provide the MST and KNN bound derivations used in DataEval.)

3. Borji, A., & Iranmanesh, S. M. (2019). Empirical upper bound in object
   detection and more. *arXiv preprint arXiv:1911.12451.* [paper](https://arxiv.org/abs/1911.12451)

4. Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad,
   H., Patwary, M. A., Yang, Y., & Zhou, Y. (2017). Deep learning scaling is
   predictable, empirically. *arXiv preprint arXiv:1712.00409.* [paper](https://arxiv.org/abs/1712.00409)

5. Wales, D. J., & Doye, J. P. K. (1997). Global optimization by basin-hopping
   and the lowest energy structures of Lennard-Jones clusters containing up to
   110 atoms. *Journal of Physical Chemistry A*, 101(28), 5111–5116. [paper](https://arxiv.org/abs/cond-mat/9803344)  
   (Basin-hopping algorithm used for power law curve fitting.)

6. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information
   Theory*, 13(1), 21–27. [paper](https://doi.org/10.1109/TIT.1967.1053964)

7. Renggli, C., Rimanic, L., Hollenstein, N., & Zhang, C. (2021). Evaluating Bayes error estimators on real-world
   datasets with FeeBee. *NeurIPS 2021 Datasets and Benchmarks*. [paper](https://arxiv.org/abs/2108.13034)

8. Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. *Journal of the American
   Statistical Association*, 58(301), 13–30. [paper](https://doi.org/10.1080/01621459.1963.10500830)  
   (Gives every Hoeffding radius in Feasibility's endpoints, including both steps of the local-panel pooling.)

9. Maurer, A., & Pontil, M. (2009). Empirical Bernstein bounds and sample variance penalization. *Proceedings of
   the 22nd Annual Conference on Learning Theory (COLT 2009)*. [paper](https://arxiv.org/abs/0907.3740)  
   (Theorem 4 gives the empirical Bernstein radius of the hybrid endpoints.)

10. Reid, M. D., & Williamson, R. C. (2011). Information, divergence and risk for binary experiments. *Journal of
    Machine Learning Research*, 12, 731–817. [paper](https://jmlr.org/papers/v12/reid11a.html)  
    (The basis of the risk-to-AP envelope behind Feasibility's AP ceilings.)

11. Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *Proceedings of the
    23rd International Conference on Machine Learning*, 233–240. [paper](https://doi.org/10.1145/1143844.1143874)  
    (The precision-recall interpolation behind the AP witness staircase and the AP envelope.)

12. Detection Feasibility Envelope (DFE) manuscript: the consensus inequality (replicated-consensus section), the
    disjoint within-cell panels (local-pooling section) and the operating-point inequalities for precision, recall
    and AP.  
    (The constructions behind every Feasibility ceiling and the AP witness, cited as "DFE" in each evidence item's
    `basis`. The post-stratified pooling of the local panels is DataEval's own, proved in `local_panel_bound`.)

13. Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the
    binomial. *Biometrika*, 26(4), 404–413. [paper](https://doi.org/10.1093/biomet/26.4.404)  
    (The exact binomial endpoints of Feasibility's Bernoulli witnesses and prevalence bounds.)
