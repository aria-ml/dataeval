<!-- markdownlint-disable MD051 -->

# Feasibility evidence

```{important}
`Feasibility` and its evidence types are experimental and may change in future releases. This page describes the
first release, which covers image and video classification.
```

This page is the practical companion to [Performance limits](PerformanceLimits.md).
It covers how to supply evidence, how each kind of evidence is built, and how Feasibility turns evidence into determination.
Parameter-by-parameter details are in the {class}`.Feasibility` API reference.

{class}`.Feasibility` compares each performance requirement (a **target**) against evidence and returns a
**determination** for it. Every determination cites the evidence items behind it in `evidence_ids`, and every evidence
item names the theorems it rests on in its `basis`.

Terms used on this page, each defined in more depth on the concept page:

- **Ceiling:** an upper confidence bound on what any model in a stated class can reach on the data.
- **Witness:** a lower confidence bound on what one frozen model reaches on the data.
- **Endpoint:** the upper or lower end of one confidence bound.
- **Fit split and audit split:** Feasibility divides the labeled items in half, keeping each group whole. Anything that
  is
  fitted (a partition, a built-in model, a score threshold) uses the fit split only. Endpoints are computed on the audit
  split, and each independent item in it is an audit unit.
- **Information scope:** the class of models a bound covers (see [below](#information-scope-which-models-a-bound-covers)).
- **DFE:** the Detection Feasibility Envelope manuscript, the source of the consensus constructions behind every ceiling.

## Read a result

Start with `print(result)`. It shows:

1. A header stating the judged scope and the call's joint confidence.
2. A table with one row per requirement: its determination, the deciding lower confidence bound and the deciding upper
   confidence bound, each with its source. A recall requirement shows both of its witness's bounds in the lower-bound
   cell.
3. "Next": the recommended action for each undecided requirement.
4. "Notes": the observed scores, ceilings narrower than the judged scope, and conflicts.

For the full picture, `result.aggregate_by_evidence()` lays out every bound of the call as one table, with one row per
evidence item, target and bounded quantity. It gives a readable source name (such as "your model" or "group-based
ceiling"), the bound, the observed score of a witness, its information scope and its independent units. The `basis` and
share of `alpha` of each bound stay on `result.evidence`. With `dataeval-plots` installed,
`dataeval_plots.plot(result)` draws one band per quantity, grouped into requirements the evidence demonstrates, leaves
unknown, or rules out under the recorded assumptions. `plot_evidence=True` draws each source's observed score and bound
instead.

## Choose what evidence to supply

Feasibility always builds local-panel ceilings and fits two built-in witnesses from the dataset and its embeddings. The
other kinds of evidence come from what you pass in.

| If you have | Pass | You get | What it adds |
| --- | --- | --- | --- |
| Only the dataset and a representation | nothing extra | Local-panel ceilings, built-in witnesses | A requirement can be demonstrated. It can be ruled out only for models that give one answer per cell, unless allowances are supplied. |
| Repeated labels on a random sample of items | `replicas=` | A replica ceiling | A requirement can be ruled out for any model reading the full input. |
| A frozen model's class scores or class decisions | `predictions=`, with `provenance=` | A user witness | A requirement can be demonstrated by your own model. |
| A justified bound on how much a broader model could improve on the local panels | `allowances=` | A wider local-panel accuracy ceiling | Rules out at a broader scope, but only if the bound holds. |
| An audit you plan to grow | `sampling=` | Looks that share one budget | You can add data and look again without invalidating earlier looks. |

Practical rules:

- **Put every target in one call.** They then share one error budget (see
  [How the error budget is spent](#how-the-error-budget-is-spent)).
- **Describe your model.** Pass its class scores or class decisions as `predictions=` and declare
  `provenance`, including `training_items` so Feasibility can check that no audit unit was used for training.
- **Fold scores when the requirement is coarser than the model.** If a requirement is about a coarser class than the model
  was trained on, relabel the dataset with {class}`.Relabel` and fold the model's scores through the same map with
  `Relabel.fold_scores(scores, view=view)`. Columns of source classes that map to one target class are summed,
  columns of dropped classes are discarded, nothing is renormalized, and with `view` the rows of dropped items are
  removed so the result lines up with the relabeled items.
- **Freeze other operating points before the call.** Feasibility has no decision threshold of its own and does not search
  for one on the audit. To judge a different operating point, such as a threshold on one class's score, apply it before
  the call as a complete multiclass policy (the class when its score reaches the threshold, otherwise the highest-scoring
  other class), pass the resulting decisions, and record the rule in `provenance["decision_policy"]`. Each operating point
  needs its own call, with the error budget split across the planned calls.
- **Collect repeated labels when ruling out matters.** Without `replicas=` or allowances, every ceiling covers only
  `partition_restricted` models, so a `full_input` requirement can be demonstrated but not ruled out.

## How a determination is made

For each target, Feasibility considers only the endpoints for that target on the call's population. Ceilings count at
the judged information scope or broader, and witnesses count where they are eligible (see
[Information scope](#information-scope-which-models-a-bound-covers)). The first rule that matches decides:

| Order | Determination | When | What it claims |
| --- | --- | --- | --- |
| 1 | `evidence_conflict` | A ceiling's upper endpoint is below the lower endpoint of a witness eligible at that ceiling's scope | No claim. The two endpoints cannot both hold, so either a recorded assumption failed or this call fell in its failure event (probability at most `alpha`). Likely causes of a failed assumption: audit units in the witness's training data, ungrouped duplicates or sequences, or a witness reading more than the ceiling's scope covers. |
| 2 | `demonstrated_feasible` | The largest lower endpoint of an eligible witness is at or above the threshold | The frozen witness reaches the threshold on the population, at the call's confidence, assuming it was trained without the audit units and frozen before scoring them. |
| 3 | `infeasible_under_recorded_assumptions` | The smallest ceiling upper endpoint is strictly below the threshold | No model in the ceiling's class reaches the threshold on the population, at the call's confidence, under the ceiling's recorded assumptions. |
| 4 | `not_ruled_out` | An informative ceiling is at or above the threshold, and no eligible witness reaches it | The deciding ceiling does not exclude the threshold at its scope. This is not evidence of feasibility. |
| 5 | `unresolved` | Otherwise | No claim. No informative ceiling at the judged scope or broader is available and no eligible witness reaches the threshold. Narrower ceilings remain evidence only. |

Rules 2 and 3 cannot both match without rule 1 matching first, because a witness at or above the threshold would exceed
a ceiling below it. Each determination row also records the basis and scope of the endpoints that decided it.

An endpoint has one of three states:

- `unavailable`: it could not be computed (too few units, no complete panel, a missing declared bound such as
  `max_sequence_length`). Its `value` is null and `reason` says why.
- `uninformative`: it is valid but trivial. A ceiling at the metric's maximum of 1, a witness at its minimum of 0, or a
  false-positive upper endpoint at 1. It says nothing about any threshold and cannot make a ceiling decide rule 4.
- `informative`: valid and not trivial.

## Information scope: which models a bound covers

Each ceiling and witness belongs to a class of models, its **information scope**. The classes nest:

| Scope | Models in the class |
| --- | --- |
| `full_input` | Any model that reads everything the eventual model is allowed to read |
| `fixed_representation` | Models that read only the given representation (the embeddings) |
| `partition_restricted` | Models that give one fixed answer per cell of the frozen partition |

`full_input` ⊃ `fixed_representation` ⊃ `partition_restricted`. A ceiling bounds every model in its class, so it decides
a target judged at its own scope or any narrower one, never a broader one. A target is judged at `full_input` by default;
the key `"accuracy/fixed_representation"` judges another scope. A ceiling narrower than the judged scope is listed in the
row's `evidence_ids` under its own scope, but it never decides and never fills the `ceiling` column.

| Ceiling | Information scope |
| --- | --- |
| Repeated labels | `full_input` |
| Local panels, accuracy, image data, no allowances | `partition_restricted` |
| Local panels, accuracy, image data, with β | `fixed_representation` |
| Local panels, accuracy, image data, with β and Γ | `full_input` |
| Local panels, AP or recall, or any local panel on video data | `partition_restricted` |

**Allowances** are bounds you supply, in risk units, that widen a local-panel accuracy ceiling on image data:

- β (`allowances={"beta": ...}`) bounds how much a model reading the representation can improve on one fixed answer inside
  a cell. With it the ceiling covers `fixed_representation`. β = 0 asserts that no model reading the representation beats
  one fixed answer per cell.
- Γ (`"gamma"`, given with β) bounds how much reading the full input can lower the best risk below what the representation
  allows. With both, the ceiling covers `full_input`.

A larger allowance gives a higher, weaker ceiling. Feasibility does not estimate or check an allowance. It records it as
an assumption, and every claim of the extended ceiling, including a target ruled out, holds only if the allowance does.
The justification must come from outside the call: an argument or separate evidence, kept with the result. Without one,
repeated labels give a `full_input` ceiling that needs no allowance. Both allowances are scalars, since a per-cell β would
refer to the internal partition's cells.

**Witness eligibility.** A witness can demonstrate a target only if it belongs to the judged scope's class. The built-in
witnesses read the representation, so they are eligible at `fixed_representation` and `full_input`. A user witness declares
what it reads in `provenance["reads"]`: `"full_input"` (the default) is eligible at `full_input`, and `"representation"`
at `fixed_representation` and `full_input`. No witness is eligible for a `partition_restricted` target, which only a ceiling
can decide in this release.

## The four kinds of evidence

Each evidence item is one source run on one sample. Its fixed **spine** is what the determination rules read: its `kind`,
whether it `supports` a ceiling or a witness, its `basis` (the theorems and citations behind it, never empty),
`population`, `sample`, `information_scope`, independent `unit`, `provenance`, and one {class}`.Endpoint` per target it
bounds. `details` holds the assumptions, settings and raw results behind each endpoint for programmatic access, and
`Evidence.to_dict()` serializes the item. The fields are listed in the {class}`.Evidence` and {class}`.Endpoint`
references.

A ceiling and a witness are compared only when their endpoints name the same target and their items the same
`population`. `population` is `"items of <dataset id>"`, or `"labeled items of <dataset id>"` when unlabeled items were
excluded.

| Evidence id | `kind` | Supports | Information scope | Comes from |
| --- | --- | --- | --- | --- |
| `local_panel_ceiling/v1` | `local_panel_ceiling/v1` | ceiling | `partition_restricted` | The dataset's own labels |
| `local_panel_ceiling/v1:accuracy` | `local_panel_ceiling/v1` | ceiling | `fixed_representation` (β) or `full_input` (β and Γ) | The dataset's own labels, with allowances |
| `local_panel_ceiling/v1:cost_weighted` | `local_panel_ceiling/v1` | ceiling | `partition_restricted` | AP and recall targets, when allowances split the item |
| `replica_ceiling/v1` | `replica_ceiling/v1` | ceiling | `full_input` | `replicas=` |
| `builtin_witness/v1:knn` | `builtin_witness/v1` | witness | `fixed_representation` | A built-in 5-nearest-neighbor classifier |
| `builtin_witness/v1:probe` | `builtin_witness/v1` | witness | `fixed_representation` | A built-in logistic-regression probe |
| `user_witness/v1` | `user_witness/v1` | witness | From `provenance["reads"]` | `predictions=` |

### Local-panel ceilings (`local_panel_ceiling/v1`)

**What it bounds.** The accuracy, AP, or recall at a false-positive limit that a model in its scope can reach. It needs
only the dataset's own labels.

**How it is built.** Feasibility fits a k-means partition of the representation on the fit split only, so labels never
enter the partition. Within each cell it draws random disjoint panels of `panel_size` ($m$) audit units. On grouped data
it first draws one item per group, uniformly. The labels of a panel come from different images in one cell, so they are
draws from the cell's label distribution, and their consensus value lower-bounds the risk of the best single answer for
that cell (the DFE consensus inequality). The cells' panel means are then pooled with the audit's own cell shares. The disjoint
panels are from DFE. This post-stratified pooling of them is DataEval's own construction, proved with two Hoeffding (1963)
steps in the Notes of {func}`.local_panel_bound`:

$$\ell = \sum_h \frac{n_h}{N}\left(\bar g_h - \beta_h\right) - r_1 - r_2$$

where $n_h$ of the $N$ audit units fall in cell $h$, $\bar g_h$ is the mean consensus value of the cell's panels (0 for
a cell without panels), $\beta_h$ is the heterogeneity allowance (0 without one), and $r_1$ and $r_2$ are Hoeffding radii
for the panels given the cells and for the cell shares. $\ell$ is floored at 0.

- **Accuracy:** the consensus value is the share of a panel's labels outside its most common class. The ceiling is
  $1 - \max(0, \ell - \Gamma)$, with $\Gamma$ the information-loss allowance (0 without one).
- **AP, per class:** the one-vs-rest consensus value is $\min(k/m, \lambda(1 - k/m))$, where $k$ counts the class's labels
  in a panel and $\lambda$ is the cost of a false alarm relative to a missed positive. It is pooled at each positive cost
  where it changes slope. Those risk lower bounds and an upper endpoint on the class's prevalence feed the
  Reid–Williamson (2011) risk-to-AP envelope, an upper bound on any scorer's population AP. Allowances are not applied to
  AP. Macro AP is the mean over every declared class index, so a class absent from the data still counts with its trivial
  endpoint.
- **Recall:** see [Recall targets](#recall-targets-recall-at-a-false-positive-limit).

**Defaults.** `panel_size` is 5, and `n_cells` is 10 below 20,000 planned audit units, 30 below 50,000 and 300 from 50,000.
Both are provisional and recorded in `details["provisional_defaults"]` when you leave them unset.

**Unavailable when** the fit split has fewer items than cells, the audit has fewer than 2 units, or no cell holds $m$ audit
units.

**Assumptions.** The representation was fitted and selected independently of the audit units' labels (declared by you).
β and Γ are supplied by you and assumed, not estimated. On video data, single-frame local panels stay
`partition_restricted`.

### Replica ceilings (`replica_ceiling/v1`)

**What it bounds.** The accuracy, AP, or recall at a false-positive limit that any model reading the full input can reach.

**How it is built.** `replicas=` maps datum ids to extra, independent class labels of those items. Each such item's panel
is its primary label plus its replicas, so $m_i = 1 + r_i$, and items without replicas are not used. Because a panel holds
labels of one input, its consensus value lower-bounds the error of the best answer to that input, whatever the model reads.
The accuracy ceiling is one minus a lower endpoint on the mean consensus value. AP and recall use the cost-weighted
consensus as local panels do, at the slope-change costs of the most common panel size. Video data use a group's mean
(`sequence_mean`) or total (`frame_pooled`) as the observation. A `frame_pooled` accuracy is a ratio of per-group totals,
bounded with simultaneous numerator and denominator endpoints whose support is the declared `max_sequence_length`.

**Assumptions.**

- Items with replicas are a random sample of the population, so the ceiling describes the population, not only those
  items.
- Replicas and primary labels follow one annotation protocol, each an independent label of the item. Witnesses are
  scored on primary labels, so the ceiling and the witnesses then concern the same target.
- On video data, annotators labeled the replicas with the same video context as the primary labels. For `sequence_mean`
  targets, the replicated sequences are a random sample with replicated frames uniform within each. For `frame_pooled`
  targets, the replicated frames are a uniform sample of all frames.
- With a sampling plan, the replica set was fixed before the first look.

### Built-in witnesses (`builtin_witness/v1`)

**What it bounds.** A lower bound on the accuracy or AP that a frozen model reaches. For a recall target, a lower bound
on its recall together with an upper bound on its false-positive rate. Two items run by default (`builtin_witnesses=True`):
a 5-nearest-neighbor classifier and a logistic-regression probe.

**How it is built.** Each model is fitted on the fit split's representation and labels only, then frozen and scored once
on the audit split. Accuracy uses each unit's correctness (a group's mean or total on video data). AP for a class uses a
staircase under the model's precision–recall curve (Davis and Goadrich, 2006): at score thresholds fixed from the fit split,
it combines a lower endpoint on true positives and an upper endpoint on false positives with an upper endpoint on the
prevalence. The thresholds are the 5% to 95% quantiles of the fit-split positives' scores when the class's fit-split
prevalence is below 0.15 and the fit split holds positives, and of all fit-split scores otherwise.

**Unavailable when** a model cannot be fitted on the fit split or cannot score the audit, for example with a single class
in the fit split.

**Assumptions.** The representation was fitted and selected independently of the audit units' labels (declared by you),
and the model was fitted on the fit split only and frozen before the audit.

### User witnesses (`user_witness/v1`)

**What it bounds.** The same quantities as a built-in witness, for your own frozen model.

**How to supply it.** `predictions=` is either class scores, shape `(len(dataset), n_classes)`, aligned with the dataset
and finite, or class decisions: a 1-D integer array with one declared-class index per item. An index outside the declared
classes is refused, and so is a boolean or float array, which would silently be read as classes 0 and 1. The scores on the
fit split set the AP thresholds, and the audit split gives the endpoints, as for the built-in witnesses. Decisions rank
nothing, so an AP target gets no endpoint from them (the endpoint is unavailable, with the reason `"predictions are class
decisions, not scores; AP needs scores"`).

**Provenance.** The accepted `provenance` keys are `reads`, `training_items`, `frozen_before_audit`, `selection`,
`preprocessing` and `decision_policy`, and only together with `predictions`. Two are checked, and a violation raises a
`ValueError`: `reads` must be one of its two values, and `training_items` (the datum ids of items the model trained on)
must be of the datum ids' type and share no physical group with an audit unit. The rest are recorded, not verified.
`decision_policy` is accepted only with class decisions.

**Assumptions.** A demonstration assumes the model was trained without the audit units and frozen before scoring them. With
class decisions, the evidence adds that the decisions follow a policy frozen before the audit.

## Recall targets: recall at a false-positive limit

Programs often state a requirement for one class as an operating condition: identify tanks with probability at least 0.90
while labeling at most 5% of the other vehicles as tanks. Feasibility judges it as a recall target,
`TargetSpec(task="IC", metric="recall", threshold=r0, class_name=c, max_false_positive_rate=f0)`. The false-positive limit
is always required, because any recall is reachable by always predicting the class. A model class meets the requirement
when some model in it has recall $P(h(x) = c \mid y = c) \ge r_0$ and false-positive rate
$P(h(x) = c \mid y \ne c) \le f_0$, with the negatives being the pooled mixture of every other declared class. `class_name`
uses the dataset's own vocabulary, so a requirement on a coarser class follows a {class}`.Relabel`. See {class}`.TargetSpec`
for the fields.

### Witnesses for a recall target

A witness's decision for each audit item is the top class of its scores, or the supplied class decision. With $P$ the audit
items labeled $c$ and $N$ the others, it gets two exact binomial endpoints, each at half of the claim's share of `alpha`:
a lower endpoint on recall from the decisions $d_i = c$ among $P$, and an upper endpoint on the false-positive rate from
those among $N$. A witness is **certified** when its own false-positive upper endpoint is at or below $f_0$.

- Only a certified witness counts. Its recall lower endpoint is the one the determination rules compare with the threshold.
  One witness must supply both bounds, so two witnesses that each give one favorable bound demonstrate nothing.
- The false-positive endpoint itself is never compared with the threshold or a ceiling.
- An uncertified witness supports neither a demonstration nor an `evidence_conflict`, but both of its bounds stay on the
  evidence and in `aggregate_by_evidence()`.
- Without audit positives or without audit negatives, the corresponding endpoint is unavailable.

### Ceilings for a recall target

The cost-weighted consensus constructions that bound AP bound recall too. For a positive cost $\lambda_j$ at a slope change
of the consensus, a local-panel or replica route gives $\ell_j$, a lower bound on
$\inf_h [P(y = c, h \ne c) + \lambda_j P(y \ne c, h = c)]$. A model with recall $r \ge r_0$ and false-positive rate
$f \le f_0$ satisfies $\pi(1 - r) + \lambda_j (1 - \pi) f \le \pi(1 - r_0) + \lambda_j (1 - \pi) f_0$, where $\pi$ is the
class's prevalence. With a two-sided confidence interval $[\pi^-, \pi^+]$ on $\pi$, Feasibility reports the ceiling $U_r$
as an upper bound on the recall any model in the ceiling's scope reaches at a false-positive rate of at most $f_0$:

$$
U_r = \min_j \max_{\pi \in [\pi^-, \pi^+]} \left[\, 1 - \lambda_j f_0 - \frac{\ell_j - \lambda_j f_0}{\pi} \,\right],
\quad \text{clipped to } [0, 1]
$$

Which end of the prevalence interval is conservative depends on the cost:

- When $\ell_j > \lambda_j f_0$, the bracket increases in $\pi$, so the maximum sits at $\pi^+$.
- When $\ell_j < \lambda_j f_0$, it decreases, so the maximum sits at $\pi^-$ (unbounded, reported as 1, when $\pi^- = 0$).
- When they are equal, it is the constant $1 - \lambda_j f_0$.

This is why a recall ceiling needs the lower prevalence endpoint, where an AP ceiling needs only the upper one. The target
is excluded if and only if $U_r < r_0$, and then the five determination rules apply unchanged. A recall ceiling inherits
its route's scope: local panels bound only models that give one answer per cell (`partition_restricted`), and replicas bound
every model reading the full input (`full_input`). It never says anything about one model. A certified witness whose recall
lower endpoint exceeds $U_r$ is an `evidence_conflict`, whether or not it reaches $r_0$. Allowances are not applied to recall.

### Read a recall target

For `Tank` at least 80% recall with at most 10% false positives, one call on a synthetic dataset with class decisions and
`provenance={"decision_policy": "top class of the frozen classifier"}` prints:

```text
Feasibility of 1 requirement, judged for models that may use the whole image.
All bounds hold together at 95% confidence, for new images from the same population.
They assume no evaluation image was used to train or choose a model, or to set a requirement.

Requirement                               Determination    Lower confidence bound                Upper confidence bound
Tank: ≥80% recall, ≤10% false positives   demonstrated ✓   recall 93.5%, FP ≤2.9% (your model)   —

Notes
  Your model scored 96.9% recall on 261 Tank images and 1.2% false positives on 489 other images.
```

The witness is certified (its false-positive upper endpoint, 2.9%, is within 10%) and its recall lower endpoint, 93.5%,
reaches 80%, so rule 2 decides. `aggregate_by_evidence()` lists the two bounds as separate rows with `quantity` `recall`
and `false_positive_rate`, `bound_type` `lower` and `upper`, and the counts behind each in `n` (261 positives, 489 negatives).
The upper cell shows `—` because the only ceiling, the group-based one, bounds a narrower scope than the one judged and
is uninformative here (it is 1). Had the false-positive upper endpoint exceeded 10%, the lower cell would still show both
bounds, marked `(your model, not counted)`, and a note would say no tested model counts.

## How endpoints are computed

Where the trials are independent items, each indicator (correct or not, a class decision or not, a label equal to the class
or not) is a Bernoulli trial. Feasibility bounds its mean with the exact one-sided Clopper–Pearson (1934) endpoint, a beta
quantile that holds at its stated confidence for any sample size ({func}`.binomial_endpoint`). This is used for:

- accuracy witnesses,
- the recall and false-positive endpoints of a recall witness,
- the true-positive, false-positive and prevalence endpoints of an AP witness,
- the prevalence endpoints of local-panel ceilings.

It is not used on grouped (video) data, where items are not independent, or for consensus values and risk lower bounds,
which are bounded variables rather than Bernoulli trials. Those use a hybrid of a Hoeffding (1963) radius and an empirical
Bernstein radius (Maurer and Pontil, 2009, Theorem 4), each at half the endpoint's error share, or Hoeffding alone where
a source documents it. Each endpoint records the `method` it used.

## How the error budget is spent

A call spends one budget, `alpha` (0.05 by default), across every endpoint of every target. By a union bound, all the
endpoints of a call hold together with probability at least `1 - alpha`. With a sampling plan, look $t$ spends
$\alpha / (t(t+1))$.

Each pair of a target and an evidence source is one **claim**, and every claim gets an equal share `alpha / n_claims`. Per
target, local panels count as one claim (none for a `frame_pooled` target), replicas as one, the built-in witnesses as two
and the user witness as one, each when present. Every endpoint records its claim's share in `alpha`, and inside a claim
the share is split again:

| Source and quantity | Split inside the claim |
| --- | --- |
| Local panel, accuracy | Two Hoeffding steps, half each |
| Local panel, AP (one class) | Equally over the $m - 1$ positive costs and one prevalence upper endpoint; each cost's pooling halves its part again |
| Local panel, recall | Equally over the $m - 1$ positive costs and two prevalence endpoints; each cost's pooling halves its part again |
| Replica, accuracy | One hybrid endpoint, or Hoeffding alone at the full share when items' panel sizes differ; a `frame_pooled` ratio spends half on numerator and half on denominator |
| Replica, AP and recall | Equally over the positive costs and the prevalence endpoint(s), using the replica endpoint method, never exact binomial |
| Witness, accuracy | One endpoint: exact binomial on ungrouped data, else hybrid |
| Witness, AP (one class, $T$ thresholds) | $2T + 1$ endpoints, equal parts |
| Witness, recall | The recall and false-positive endpoints, half each |

Macro AP first divides the claim's share equally among the classes. `details["results"]` records the split in `deltas`.

The guarantee covers one call on one fixed audit. Further calls need the budget divided again:

- Comparing $k$ representations or models across calls needs `alpha / k` per call.
- Rerunning with $k$ other seeds and keeping the best result is a selection among those calls and also needs `alpha / k`.
- Growing the audit and looking again needs a {class}`.SamplingPlan` declared before the first look. Look $t$ uses the first
  `sizes[t - 1]` units of one seeded order and spends $\alpha / (t(t+1))$, which sums to $\alpha$ over all looks. Keep the
  seed fixed across looks.

## What to do about a non-final result

A final determination, `demonstrated_feasible` or `infeasible_under_recorded_assumptions`, needs no recommendation.
`evidence_conflict` gets the limiting factor `conflict` and a recommendation to inspect the likely causes listed under
[rule 1](#how-a-determination-is-made). Any other determination names the first limiting factor that applies and the
evidence that addresses it:

| Order | Limiting factor | When | Evidence to collect |
| --- | --- | --- | --- |
| 1 | `scope_restricted` | No ceiling at the judged scope or broader, but a narrower one is available | Repeated labels on a randomly sampled subset of items for a `full_input` ceiling, a justified β/Γ allowance, or accept the narrower claim |
| 2 | `missing_bound` | A ceiling is unavailable for lack of a declared bound such as `max_sequence_length` | Declare the bound |
| 3 | `sampling_width` | Every ceiling is unavailable for lack of units or complete panels | More independent units |
| 4 | `no_witness` | No eligible witness is available | A frozen model's predictions on held-out units, from the judged scope's model class (none can be eligible at `partition_restricted`) |
| 5 | `sampling_width` | The deciding ceiling's point estimate is below the threshold, or the best witness's above it, but the endpoint is not | More independent units |
| 6 | `no_replicas` | Every ceiling, if any, is `partition_restricted` and no replica evidence exists | Repeated labels on a randomly sampled subset of items |
| 7 | `no_witness` | Otherwise: the best witness is weak | A stronger model's predictions on held-out units |

For a recall target, the factor follows the witness's observed rates:

- `sampling_width` when observed recall is at or above the minimum and the observed false-positive rate at or below the
  limit, but a confidence bound falls short.
- `no_witness` when observed recall is below the minimum or the observed false-positive rate is above the limit. A stronger
  model or a different frozen operating point is needed, and there is no suggested minimum.

When unlabeled items were excluded, the evidence covers labeled items only. That never displaces the limiting factor, but
the recommendation ends with a note to label the excluded items to extend the result to the full dataset.

### The suggested minimum

When the deciding ceiling's point estimate is already below the threshold, or the best witness's already above it, the
endpoint misses only by its confidence radius. `suggested_minimum` is the number of independent units behind the endpoint
at which it would cross the threshold, if the observed rates hold.

- For local panels and witnesses these are audit-split units (the audit is half of the independent units in this release),
  rounded up to the next size of a declared sampling plan. For a replica ceiling they are units with replicas, which are
  not plan units, so it is not rounded.
- An exact binomial endpoint is inverted directly: {func}`.units_to_cross` finds the smallest number of trials at which
  the Clopper–Pearson bound, evaluated at the observed rate projected to that size, clears the gap.
- For a recall target only a witness projects. The minimum counts the positives its recall bound needs and the negatives
  its false-positive bound needs, converts each to evaluation images through the estimated prevalence (rounded up), and
  takes the larger. The printout says which bound would cross and in what units.
- It is null when the determination is final or a conflict, when no endpoint's point estimate is past the threshold, when
  the endpoint is a recall ceiling (no single point estimate to project), when the endpoint has no closed-form radius to
  invert (AP endpoints and `frame_pooled` ratios), when no size up to $10^7$ suffices, or when the size lies beyond the
  sampling plan.

[Acting on results](ActingOnResults.md#performance-limits-findings) describes what to do with these findings.

## Limitations of the first release

- Object detection and tracking datasets raise `NotImplementedError`.
- `frame_pooled` video targets get no local-panel ceiling and spend no budget on one. Their ceilings come only from
  replicas.
- Feasibility always fits its own label-blind k-means partition. A partition you supply is not accepted, and because the
  partition is internal, a per-cell β and a witness that reads it (`reads="partition"`) are rejected.
- β and Γ are taken as given and recorded as assumed, never estimated.
- The panel size and cell count are provisional defaults, recorded as such in the evidence.
- A recall target is for image classification only: one class per target, one frozen operating point per call, with the
  false-positive limit on the pooled negatives. It is refused on grouped (video) data, with
  `false_positive_scope="each_class"`, and as a mapping key, which cannot carry two numbers. Feasibility judges whether
  the requirement is reachable. It does not evaluate or accept a model, and an uncertified witness is not evidence for it.

The conceptual limitations (bounds can be wide, and claims are about labeled items under the annotation protocol) are on
the [Performance limits](PerformanceLimits.md#limitations) page.

## Related concept pages

- [Performance limits](PerformanceLimits.md) — what Feasibility answers, why, and how it relates to BER, UAP and
  Sufficiency
- [Embeddings](Embeddings.md) — the representation that local panels partition and the built-in witnesses read
- [Acting on results](ActingOnResults.md) — what to do with a non-final determination

## See this in practice

- [Assess whether a classification requirement is reachable](../notebooks/tt_assess_ic_feasibility.py) — vehicle type,
  vehicle category and tank recall requirements on MilitaryVehicles, the evidence behind each determination read as a table
  and a chart, and how to change a question the evidence cannot answer

## References

1. Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. *Journal of the American
   Statistical Association*, 58(301), 13–30. [paper](https://doi.org/10.1080/01621459.1963.10500830)  
   (Every Hoeffding radius in Feasibility's endpoints, including both steps of the local-panel pooling.)

2. Maurer, A., & Pontil, M. (2009). Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd
   Annual Conference on Learning Theory (COLT 2009)*. [paper](https://arxiv.org/abs/0907.3740)  
   (Theorem 4 gives the empirical Bernstein radius of the hybrid endpoints.)

3. Reid, M. D., & Williamson, R. C. (2011). Information, divergence and risk for binary experiments. *Journal of Machine
   Learning Research*, 12, 731–817. [paper](https://jmlr.org/papers/v12/reid11a.html)  
   (The risk-to-AP envelope behind Feasibility's AP ceilings.)

4. Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *Proceedings of the 23rd
   International Conference on Machine Learning*, 233–240. [paper](https://doi.org/10.1145/1143844.1143874)  
   (The precision–recall interpolation behind the AP witness staircase.)

5. Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the
   binomial. *Biometrika*, 26(4), 404–413. [paper](https://doi.org/10.1093/biomet/26.4.404)  
   (The exact binomial endpoints of Feasibility's Bernoulli witnesses and prevalence bounds.)

6. Detection Feasibility Envelope (DFE) manuscript: the consensus inequality (replicated-consensus section), the disjoint
   within-cell panels (local-pooling section) and the operating-point inequalities for precision, recall and AP.  
   (The constructions behind every Feasibility ceiling and the AP witness. The post-stratified pooling of the local panels
   is DataEval's own, proved in `local_panel_bound`.)
