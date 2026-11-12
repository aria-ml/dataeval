<!-- markdownlint-disable MD051 -->

# Feasibility evidence

```{important}
`Feasibility` and its types are experimental and may change in future releases. This page describes the
first release, which covers image and video classification.
```

This page is the practical companion to [Performance limits](PerformanceLimits.md).
It covers how to supply evidence, how each kind of evidence is built, and how Feasibility turns evidence into a determination.
Parameter-by-parameter details are in the {class}`.Feasibility` API reference.

{class}`.Feasibility` compares each performance requirement (a {class}`.TargetSpec`) against evidence and returns a
**determination** for it. Every determination marks the bounds it rests on, and every bound names its source, whose
theorems are listed in [the basis of each source](#the-basis-of-each-source).

Terms used on this page, each defined in more depth on the concept page:

- Envelope: an upper confidence bound on what any model in a stated class can reach on the data.
- Witness: a lower confidence bound on what one frozen model reaches on the data.
- Bound: one confidence limit on a performance metric, one row of the `bounds` table.
- Fit split and audit split: Feasibility divides the labeled items in half, keeping each group whole. Anything that
  is
  fitted (a partition, a built-in model, a score threshold) uses the fit split only. Bounds are computed on the audit
  split, and each independent item in it is an audit unit.
- Information scope: the class of models a bound covers (see [below](#information-scope-which-models-a-bound-covers)).
- DFE: the Detection Feasibility Envelope manuscript, the source of the consensus constructions behind every envelope.

## Read a result

Start with `print(result)`. It shows one line per requirement, the call's joint confidence, the assumptions the bounds
rest on, which DataEval declares and does not verify, and what a minimum means: the independent units that would cross
the requirement if the observed rates hold.

`result.bounds` lays out every bound of the call as one table, with one row per source, requirement and quantity, and
`is_determining` marks the bounds a determination rests on. `result.determinations` holds the printed rows as a table.
With `dataeval-plots` installed, `dataeval_plots.plot(result)` draws one band per quantity, grouped into requirements
the evidence demonstrates, leaves unknown, or rules out under the recorded assumptions. `plot_evidence=True` draws each
source's observed score and bound instead.

## Choose what evidence to supply

Feasibility builds local-panel envelopes and fits two built-in witnesses from the dataset and its embeddings, or from an
extractor. The other kinds of evidence come from what you pass in.

| If you have | Pass | You get | What it adds |
| --- | --- | --- | --- |
| Only the dataset and a representation | `embeddings=` (or an extractor) | Local-panel envelopes, built-in witnesses | A requirement can be demonstrated. It can be ruled out only for models that give one answer per cell, unless allowances are supplied. |
| Repeated labels on a random sample of items | `replicas=` | A replica envelope | A requirement can be ruled out for any model reading the full input. |
| A frozen model's class scores or class decisions | `predictions=` | A user witness | A requirement can be demonstrated by your own model. |
| A frozen MAITE model | `model=` | A user witness | The same, and DataEval runs the model over the dataset for you, in batches. |
| A justified bound on how much a broader model could improve on the local panels | `allowances=Allowances(beta=..., gamma=...)` | A wider local-panel accuracy envelope | Rules out at a broader scope, but only if the bound holds. |
| Only a model or its predictions | nothing else | A user witness and no envelope | A requirement can be demonstrated. One it cannot demonstrate stays `unresolved` with the limitation `no_envelope`. |

Practical rules:

- Put every requirement in one call. They then share one error budget (see
  [How the error budget is spent](#how-the-error-budget-is-spent)).
- Describe your model. DataEval records that it was trained without the audit units and frozen before it scored them,
  and does not check either, so train it on items that are not in the dataset you pass.
- Collect repeated labels when ruling out matters. Without `replicas=` or allowances, every envelope covers only
  `partition_restricted` models, so a `full_input` requirement can be demonstrated but not ruled out.

## How a determination is made

For each requirement, Feasibility considers only the bounds for that requirement on the call's population. Envelopes
count at the judged information scope or broader, and witnesses count where they are eligible (see
[Information scope](#information-scope-which-models-a-bound-covers)). The first rule that matches decides:

| Order | Determination | When | What it claims |
| --- | --- | --- | --- |
| 1 | `unresolved` with limitation `conflict` | An envelope's upper bound is below the lower bound of a witness eligible at that envelope's scope | No claim. The two bounds cannot both hold, so either a recorded assumption failed or this call fell in its failure event (probability at most `alpha`). The conflicting pair sits in the `envelope` and `witness` columns. Likely causes of a failed assumption: audit units in the witness's training data, ungrouped duplicates or sequences, or a witness reading more than the envelope's scope covers. |
| 2 | `demonstrated_feasible` | The largest lower bound of an eligible witness is at or above the threshold | The frozen witness reaches the threshold on the population, at the call's confidence, assuming it was trained without the audit units and frozen before scoring them. |
| 3 | `infeasible_under_recorded_assumptions` | The smallest envelope upper bound is strictly below the threshold | No model in the envelope's class reaches the threshold on the population, at the call's confidence, under the envelope's recorded assumptions. |
| 4 | `not_ruled_out` | An informative envelope is at or above the threshold, and no eligible witness reaches it | The determining envelope does not exclude the threshold at its scope. This is not evidence of feasibility. |
| 5 | `unresolved` | Otherwise | No claim. No informative envelope at the judged scope or broader is available and no eligible witness reaches the threshold. Narrower envelopes remain evidence only. The limitation says which. |

Rules 2 and 3 cannot both match without rule 1 matching first, because a witness at or above the threshold would exceed
an envelope below it. Each row of `determinations` also records the information scope of the envelope and the source
of the witness that decided it.

A call with a witness but neither embeddings nor replicas has no envelope of any scope, so only rules 2 and 5 can apply:
the determination is `demonstrated_feasible`, or `unresolved` with the limitation `no_envelope`.

A bound has one of three states:

- `unavailable`: it could not be computed (too few units, no complete panel, a missing declared bound such as
  `max_sequence_length`). Its `bound` is null and `reason` says why: `too_few_units`, `fit_split_smaller_than_cells`,
  `no_complete_panel`, `max_sequence_length_missing`, `no_positives`, `no_negatives`, `needs_scores`,
  `witness_not_fitted`, or `no_positives_possible` (a recall envelope whose prevalence upper bound is 0).
- `uninformative`: it is valid but trivial. An envelope at the metric's maximum of 1 (`bound_at_maximum`), a witness at
  its minimum of 0 (`bound_at_minimum`), or a false-positive upper bound at 1. It says nothing about any threshold and
  cannot make an envelope decide rule 4.
- `informative`: valid and not trivial.

The `assumptions` column of `bounds` lists what each bound rests on. Each source below names the assumptions it records.
DataEval declares them and does not verify them.

## Information scope: which models a bound covers

Each envelope and witness belongs to a class of models, its **information scope**. The classes nest:

| Scope | Models in the class |
| --- | --- |
| `full_input` | Any model that reads everything the eventual model is allowed to read |
| `fixed_representation` | Models that read only the given representation (the embeddings) |
| `partition_restricted` | Models that give one fixed answer per cell of the frozen partition |

`full_input` ⊃ `fixed_representation` ⊃ `partition_restricted`. An envelope bounds every model in its class, so it
decides a requirement judged at its own scope or any narrower one, never a broader one. A requirement is judged at
`full_input` by default; `TargetSpec(scope="fixed_representation")` judges another scope. An envelope narrower than the
judged scope stays in `bounds` under its own scope, but it never decides and never fills the `envelope` column.

| Envelope | Information scope |
| --- | --- |
| Repeated labels | `full_input` |
| Local panels, accuracy, image data, no allowances | `partition_restricted` |
| Local panels, accuracy, image data, with β | `fixed_representation` |
| Local panels, accuracy, image data, with β and Γ | `full_input` |
| Local panels, AP or recall, or any local panel on video data | `partition_restricted` |

**Allowances** are bounds you supply, in risk units, that widen a local-panel accuracy envelope on image data:

- β (`Allowances(beta=...)`) bounds how much a model reading the representation can improve on one fixed answer inside
  a cell. With it the envelope covers `fixed_representation`. β = 0 asserts that no model reading the representation beats
  one fixed answer per cell.
- Γ (`gamma`, given with β) bounds how much reading the full input can lower the best risk below what the representation
  allows. With both, the envelope covers `full_input`.

A larger allowance gives a higher, weaker envelope. Feasibility does not estimate or check an allowance. It records it as
an assumption, and every claim of the extended envelope, including a requirement ruled out, holds only if the allowance
does. The justification must come from outside the call: an argument or separate evidence, kept with the result. Without
one, repeated labels give a `full_input` envelope that needs no allowance. Both allowances are scalars, since a per-cell
β would refer to the internal partition's cells.

Witness eligibility: a witness can demonstrate a requirement only if it belongs to the judged scope's class. The
built-in witnesses read the representation, so they are eligible at `fixed_representation` and `full_input`. Your model
may read the whole input, so it is eligible at `full_input` only. No witness is eligible for a `partition_restricted`
requirement, which only an envelope can decide in this release.

## The four kinds of evidence

Each bound is one source run on one sample, for one requirement and one quantity. Its row in `result.bounds` carries what
the determination rules read: its `source`, whether it is an envelope or a witness (`role`), `quantity`, `bound_type`,
`bound`, `population`, `sample`, `information_scope` and independent `unit`. The other columns say how the bound was
computed (`method`, `radius`, `n`, `variance`, `alpha`) and what it rests on (`assumptions`). The fields are listed in
the {class}`.FeasibilityOutput` reference.

An envelope and a witness are compared only when their bounds name the same requirement and the same `population`.
`population` is `items`, or `labeled_items` when unlabeled items were excluded.

| Source | Supports | Information scope | Comes from |
| --- | --- | --- | --- |
| `local_panels` | envelope | `partition_restricted`, or for accuracy `fixed_representation` (β) or `full_input` (β and Γ) | The dataset's own labels, with allowances for the broader scopes |
| `replicas` | envelope | `full_input` | `replicas=` |
| `knn` | witness | `fixed_representation` | A built-in 5-nearest-neighbor classifier |
| `linear_probe` | witness | `fixed_representation` | A built-in logistic-regression probe |
| `user_model` | witness | `full_input` | `predictions=` or `model=` |

### The basis of each source

The theorems and citations each source rests on are listed here and not in the output, so one Enum value identifies them.

| Source | Basis |
| --- | --- |
| `local_panels` | DataEval's post-stratified pooling (C3) of DFE disjoint within-cell panels, proved in `local_panel_bound`'s Notes with two Hoeffding (1963) steps; DFE consensus inequality; AP via the Reid-Williamson (2011) risk-to-AP envelope (DFE operating-point inequalities); recall at a false-positive limit via DFE operating-point inequalities over a two-sided prevalence interval; prevalence: Clopper-Pearson (1934) exact binomial endpoints on ungrouped Bernoulli trials, else hybrid |
| `replicas` | DFE replicated consensus (consensus inequality); Hoeffding (1963) and Maurer-Pontil (2009, Theorem 4) hybrid endpoints; `frame_pooled` as a ratio of per-group totals with simultaneous numerator and denominator endpoints; AP via the Reid-Williamson (2011) risk-to-AP envelope (DFE operating-point inequalities); recall at a false-positive limit via DFE operating-point inequalities over a two-sided prevalence interval |
| `knn`, `linear_probe`, `user_model` | Frozen witness scored once on the audit; Clopper-Pearson (1934) exact binomial endpoints on ungrouped Bernoulli trials, else Hoeffding (1963) and Maurer-Pontil (2009, Theorem 4) hybrid endpoints; `frame_pooled` as a ratio of per-group totals with simultaneous numerator and denominator endpoints; AP via the DFE witness staircase (Davis-Goadrich 2006) |

The raw core results behind a bound, such as the per-class risk curves of an AP envelope, are not carried: they are
large and grow with the number of classes. Reproduce them with the `dataeval.core` function that computes them and the
seed in `result.meta().state`.

### Local-panel envelopes (`local_panels`)

What it bounds: the accuracy, AP, or recall at a false-positive limit that a model in its scope can reach. It needs
only the dataset's own labels, and it needs embeddings.

How it is built: Feasibility fits a k-means partition of the representation on the fit split only, so labels never
enter the partition. Within each cell it draws random disjoint panels of $m$ audit units. On grouped data
it first draws one item per group, uniformly. The labels of a panel come from different images in one cell, so they are
draws from the cell's label distribution, and their consensus value lower-bounds the risk of the best single answer for
that cell (the DFE consensus inequality). The cells' panel means are then pooled with the audit's own cell shares. The disjoint
panels are from DFE. This post-stratified pooling of them is DataEval's own construction, proved with two Hoeffding (1963)
steps in the Notes of {func}`.local_panel_bound`:

$$\ell = \sum_h \frac{n_h}{N}\left(\bar g_h - \beta_h\right) - r_1 - r_2$$

where $n_h$ of the $N$ audit units fall in cell $h$, $\bar g_h$ is the mean consensus value of the cell's panels (0 for
a cell without panels), $\beta_h$ is the heterogeneity allowance (0 without one), and $r_1$ and $r_2$ are Hoeffding radii
for the panels given the cells and for the cell shares. $\ell$ is floored at 0.

- Accuracy: the consensus value is the share of a panel's labels outside its most common class. The envelope is
  $1 - \max(0, \ell - \Gamma)$, with $\Gamma$ the information-loss allowance (0 without one).
- AP, per class: the one-vs-rest consensus value is $\min(k/m, \lambda(1 - k/m))$, where $k$ counts the class's labels
  in a panel and $\lambda$ is the cost of a false alarm relative to a missed positive. It is pooled at each positive cost
  where it changes slope. Those risk lower bounds and an upper bound on the class's prevalence feed the
  Reid–Williamson (2011) risk-to-AP envelope, an upper bound on any scorer's population AP. Allowances are not applied to
  AP. Macro AP is the mean over every declared class index, so a class absent from the data still counts with its trivial
  bound.
- Recall: see [Recall requirements](#recall-requirements-recall-at-a-false-positive-limit).

Defaults: the panel size $m$ is 5, and the partition has 10 cells below 20,000 planned audit units, 30 below 50,000
and 300 from 50,000. These are provisional internal settings.

Unavailable when the fit split has fewer items than cells (`fit_split_smaller_than_cells`), the audit has fewer than
2 units (`too_few_units`), or no cell holds $m$ audit units (`no_complete_panel`).

Assumptions: the representation was fitted and selected independently of the audit units' labels (declared by you,
`representation_independent_of_audit_labels`). β and Γ are supplied by you and assumed, not estimated (`allowance_beta`
and `allowance_gamma`). On video data, single-frame local panels stay `partition_restricted`.

### Replica envelopes (`replicas`)

What it bounds: the accuracy, AP, or recall at a false-positive limit that any model reading the full input can reach.

How it is built: `replicas=` maps datum ids to extra, independent class labels of those items. Each such item's panel
is its primary label plus its replicas, so $m_i = 1 + r_i$, and items without replicas are not used. Because a panel holds
labels of one input, its consensus value lower-bounds the error of the best answer to that input, whatever the model reads.
The accuracy envelope is one minus a lower bound on the mean consensus value. AP and recall use the cost-weighted
consensus as local panels do, at the slope-change costs of the most common panel size. Video data use a group's mean
(`sequence_mean`) or total (`frame_pooled`) as the observation. A `frame_pooled` accuracy is a ratio of per-group totals,
bounded with simultaneous numerator and denominator bounds whose support is the declared `max_sequence_length`.

Assumptions:

- Items with replicas are a random sample of the population, so the envelope describes the population, not only those
  items (`replicas_random_sample`).
- Replicas and primary labels follow one annotation protocol, each an independent label of the item. Witnesses are
  scored on primary labels, so the envelope and the witnesses then concern the same requirement
  (`replicas_one_protocol`).
- On video data, annotators labeled the replicas with the same video context as the primary labels
  (`replicas_same_video_context`). For `sequence_mean` requirements, the replicated sequences are a random sample with
  replicated frames uniform within each (`replicated_sequences_random_sample`). For `frame_pooled` requirements, the
  replicated frames are a uniform sample of all frames (`replicated_frames_uniform_sample`).

### Built-in witnesses (`knn` and `linear_probe`)

What it bounds: a lower bound on the accuracy or AP that a frozen model reaches. For a recall requirement, a lower
bound on its recall together with an upper bound on its false-positive rate. Two sources run by default whenever there
are embeddings: `knn`, a 5-nearest-neighbor classifier, and `linear_probe`, a logistic-regression probe.

How it is built: each model is fitted on the fit split's representation and labels only, then frozen and scored once
on the audit split. Accuracy uses each unit's correctness (a group's mean or total on video data). AP for a class uses a
staircase under the model's precision–recall curve (Davis and Goadrich, 2006): at score thresholds fixed from the fit split,
it combines a lower bound on true positives and an upper bound on false positives with an upper bound on the
prevalence. The thresholds are the 5% to 95% quantiles of the fit-split positives' scores when the class's fit-split
prevalence is below 0.15 and the fit split holds positives, and of all fit-split scores otherwise.

Unavailable when a model cannot be fitted on the fit split or cannot score the audit, for example with a single class
in the fit split (`witness_not_fitted`).

Assumptions: the representation was fitted and selected independently of the audit units' labels (declared by you,
`representation_independent_of_audit_labels`), and the model was fitted on the fit split only and frozen before the
audit.

### User witnesses (`user_model`)

What it bounds: the same quantities as a built-in witness, for your own frozen model.

How to supply it: `predictions=` is either class scores, shape `(len(dataset), n_classes)`, aligned with the dataset
and finite, or class decisions: a 1-D integer array with one declared-class index per item. An index outside the declared
classes is refused, and so is a boolean or float array, which would silently be read as classes 0 and 1. `model=` takes
a MAITE model in its place. DataEval runs it over the whole dataset in batches of `batch_size`, which you set with
`Feasibility(batch_size=...)` or `dataeval.config.set_batch_size` (a `model=` call raises without one), and a torch
model runs without autograd. The scores on the fit split set the AP thresholds, and the audit split gives the bounds,
as for the built-in witnesses. Decisions rank nothing, so an AP requirement gets no bound from them (the bound is
unavailable, with the reason `needs_scores`).

The scores follow the dataset's declared class order. That is an assumption (`class_order_matches_dataset`) unless a
`model=` brings its own `index2label` in its `metadata`. If both declare one and they differ, the call raises a
`ValueError`, because DataEval never remaps one to the other.

Assumptions: a demonstration assumes the model was trained without the audit units and frozen before scoring them
(`model_independent_of_audit_items`). With class decisions, it adds that the decisions follow a policy frozen before the
audit.

## Recall requirements: recall at a false-positive limit

Programs often state a requirement for one class as an operating condition: identify tanks with probability at least 0.90
while labeling at most 5% of the other vehicles as tanks. Feasibility judges it as a recall requirement,
`TargetSpec(task="IC", metric="recall", threshold=r0, class_name=c, max_false_positive_rate=f0)`. The false-positive limit
is always required, because any recall is reachable by always predicting the class. A model class meets the requirement
when some model in it has recall $P(h(x) = c \mid y = c) \ge r_0$ and false-positive rate
$P(h(x) = c \mid y \ne c) \le f_0$, with the negatives being the pooled mixture of every other declared class. `class_name`
uses the dataset's own vocabulary, so a requirement on a coarser class follows a {class}`.Relabel`. See {class}`.TargetSpec`
for the fields.

### Witnesses for a recall requirement

A witness's decision for each audit item is the top class of its scores, or the supplied class decision. With $P$ the audit
items labeled $c$ and $N$ the others, it gets two exact binomial bounds, each at half of the claim's share of `alpha`:
a lower bound on recall from the decisions $d_i = c$ among $P$, and an upper bound on the false-positive rate from
those among $N$. A witness is **certified** when its own false-positive upper bound is at or below $f_0$.

- Only a certified witness counts. Its recall lower bound is the one the determination rules compare with the threshold.
  One witness must supply both bounds, so two witnesses that each give one favorable bound demonstrate nothing.
- The false-positive bound itself is never compared with the threshold or an envelope.
- An uncertified witness supports neither a demonstration nor a conflict, but both of its bounds stay in `bounds`.
- Without audit positives or without audit negatives, the corresponding bound is unavailable (`no_positives` or
  `no_negatives`).

### Envelopes for a recall requirement

The cost-weighted consensus constructions that bound AP bound recall too. For a positive cost $\lambda_j$ at a slope change
of the consensus, a local-panel or replica route gives $\ell_j$, a lower bound on
$\inf_h [P(y = c, h \ne c) + \lambda_j P(y \ne c, h = c)]$. A model with recall $r \ge r_0$ and false-positive rate
$f \le f_0$ satisfies $\pi(1 - r) + \lambda_j (1 - \pi) f \le \pi(1 - r_0) + \lambda_j (1 - \pi) f_0$, where $\pi$ is the
class's prevalence. With a two-sided confidence interval $[\pi^-, \pi^+]$ on $\pi$, Feasibility reports the envelope $U_r$
as an upper bound on the recall any model in the envelope's scope reaches at a false-positive rate of at most $f_0$:

$$
U_r = \min_j \max_{\pi \in [\pi^-, \pi^+]} \left[\, 1 - \lambda_j f_0 - \frac{\ell_j - \lambda_j f_0}{\pi} \,\right],
\quad \text{clipped to } [0, 1]
$$

Which end of the prevalence interval is conservative depends on the cost:

- When $\ell_j > \lambda_j f_0$, the bracket increases in $\pi$, so the maximum sits at $\pi^+$.
- When $\ell_j < \lambda_j f_0$, it decreases, so the maximum sits at $\pi^-$ (unbounded, reported as 1, when $\pi^- = 0$).
- When they are equal, it is the constant $1 - \lambda_j f_0$.

This is why a recall envelope needs the lower prevalence bound, where an AP envelope needs only the upper one. The
requirement is excluded if and only if $U_r < r_0$, and then the five determination rules apply unchanged. A recall envelope
inherits its route's scope: local panels bound only models that give one answer per cell (`partition_restricted`), and
replicas bound every model reading the full input (`full_input`). It never says anything about one model. A certified
witness whose recall lower bound exceeds $U_r$ is a conflict, whether or not it reaches $r_0$. Allowances are not applied
to recall.

### Read a recall requirement

For `Tank` at least 90% recall with at most 10% false positives, one call on a synthetic five-class dataset, with class
decisions from a classifier and nothing else, prints:

```text
#   requirement               determination           limitation   envelope   witness            minimum
0   recall:Tank ≥0.9 fp≤0.1   demonstrated_feasible   —            —          96.7% user_model   —
joint confidence: 95%
assumes: model_independent_of_audit_items, class_order_matches_dataset
minimum: independent units to cross the requirement if the observed rates hold
```

The witness is certified (its false-positive upper bound, 7.0%, is within 10%) and its recall lower bound, 96.7%,
reaches 90%, so rule 2 decides. `result.bounds` lists the two bounds as separate rows with `quantity` `recall` and
`false_positive_rate`, `bound_type` `lower` and `upper`, and the counts behind each in `n` (1,200 positives, 800
negatives). The envelope cell shows `—` because the call has no embeddings and no replicas, so there is no envelope.
Had the false-positive upper bound exceeded 10%, the witness would not count: `witness` would be null, and both bounds
would stay in `bounds` with `is_determining` false.

## How bounds are computed

Where the trials are independent items, each indicator (correct or not, a class decision or not, a label equal to the class
or not) is a Bernoulli trial. Feasibility bounds its mean with the exact one-sided Clopper–Pearson (1934) endpoint, a beta
quantile that holds at its stated confidence for any sample size ({func}`.binomial_endpoint`). This is used for:

- accuracy witnesses,
- the recall and false-positive bounds of a recall witness,
- the true-positive, false-positive and prevalence bounds of an AP witness,
- the prevalence bounds of local-panel envelopes.

It is not used on grouped (video) data, where items are not independent, or for consensus values and risk lower bounds,
which are bounded variables rather than Bernoulli trials. Those use a hybrid of a Hoeffding (1963) radius and an empirical
Bernstein radius (Maurer and Pontil, 2009, Theorem 4), each at half the bound's error share, or Hoeffding alone where
a source documents it. Each bound records the `method` it used.

## How the error budget is spent

A call spends one budget, `alpha` (0.05 by default), across every bound of every requirement. By a union bound, all the
bounds of a call hold together with probability at least `1 - alpha`.

Each pair of a requirement and a source is one **claim**, and every claim gets an equal share `alpha / n_claims`. Per
requirement, local panels count as one claim (none for a `frame_pooled` requirement), replicas as one, the built-in
witnesses as two and your model as one, each when present. A call with no embeddings has no local-panel or built-in
claim. Every bound records its claim's share in `alpha`, and inside a claim the share is split again:

| Source and quantity | Split inside the claim |
| --- | --- |
| Local panel, accuracy | Two Hoeffding steps, half each |
| Local panel, AP (one class) | Equally over the $m - 1$ positive costs and one prevalence upper bound; each cost's pooling halves its part again |
| Local panel, recall | Equally over the $m - 1$ positive costs and two prevalence bounds; each cost's pooling halves its part again |
| Replica, accuracy | One hybrid bound, or Hoeffding alone at the full share when items' panel sizes differ; a `frame_pooled` ratio spends half on numerator and half on denominator |
| Replica, AP and recall | Equally over the positive costs and the prevalence bound(s), using the replica bound method, never exact binomial |
| Witness, accuracy | One bound: exact binomial on ungrouped data, else hybrid |
| Witness, AP (one class, $T$ thresholds) | $2T + 1$ bounds, equal parts |
| Witness, recall | The recall and false-positive bounds, half each |

Macro AP first divides the claim's share equally among the classes.

The guarantee covers one call on one fixed audit. Comparing $k$ representations or models across calls, keeping the
best of $k$ seeds, or growing the audit and looking again is a selection among calls, and each call then needs
`alpha / k`.

## What to do about a non-final determination

A final determination, `demonstrated_feasible` or `infeasible_under_recorded_assumptions`, has no limitation. A
conflict has the limitation `conflict`: inspect the likely causes listed under
[rule 1](#how-a-determination-is-made). Any other determination names the first limitation that applies and the
evidence that addresses it:

| Order | Limitation | When | Evidence to collect |
| --- | --- | --- | --- |
| 1 | `no_envelope` | No envelope of any scope exists | Embeddings (or an extractor) for local panels, or repeated labels for a `full_input` envelope |
| 2 | `scope_restricted` | No envelope at the judged scope or broader, but a narrower one is available | Repeated labels on a randomly sampled subset of items for a `full_input` envelope, a justified β/Γ allowance, or accept the narrower claim |
| 3 | `missing_bound` | An envelope is unavailable because `max_sequence_length` was not declared | Declare the bound |
| 4 | `envelope_unavailable` | Every envelope is unavailable for lack of units or complete panels | More independent units |
| 5 | `no_eligible_witness` | No available witness is eligible at the judged scope | A frozen model's predictions on held-out units, from the judged scope's model class (none can be eligible at `partition_restricted`) |
| 6 | `within_radius` | The determining envelope's point estimate is below the threshold, or the best witness's above it, but the bound is not | More independent units |
| 7 | `weak_witness` | Otherwise: the best witness is weak | A stronger model's predictions on held-out units |

For a recall requirement, the limitation follows the witness's observed rates:

- `within_radius` when observed recall is at or above the minimum and the observed false-positive rate at or below the
  limit, but a confidence bound falls short.
- `weak_witness` when observed recall is below the minimum or the observed false-positive rate is above the limit. A
  stronger model or a different frozen operating point is needed, and there is no suggested minimum.

When unlabeled items were excluded, the bounds cover labeled items only, and `population` is `labeled_items`. That never
displaces the limitation, so label the excluded items to extend the result to the full dataset.

### The suggested minimum

When the determining envelope's point estimate is already below the threshold, or the best witness's already above it, the
bound misses only by its confidence radius. `suggested_minimum` is the number of independent units behind the bound
at which it would cross the threshold, if the observed rates hold.

- `minimum_source` and `minimum_quantity` name the `bounds` row it projects from.
- For local panels and witnesses these are audit-split units (the audit is half of the independent units in this release).
  For a replica envelope they are units with replicas.
- An exact binomial bound is inverted directly: {func}`.units_to_cross` finds the smallest number of trials at which
  the Clopper–Pearson bound, evaluated at the observed rate projected to that size, clears the gap.
- For a recall requirement only a witness projects. The minimum counts the positives its recall bound needs and the
  negatives its false-positive bound needs, converts each to evaluation images through the estimated prevalence (rounded
  up), and takes the larger. `minimum_quantity` says which bound would cross.
- It is null when the determination is final or a conflict, when no bound's point estimate is past the threshold, when
  the bound is a recall envelope (no single point estimate to project), when the bound has no closed-form radius to
  invert (AP bounds and `frame_pooled` ratios), or when no size up to $10^7$ suffices.

## Limitations of the first release

- `frame_pooled` video requirements get no local-panel envelope and spend no budget on one. Their envelopes come only from
  replicas.
- Feasibility always fits its own label-blind k-means partition. A partition you supply is not accepted, and because the
  partition is internal, a per-cell β is rejected and no witness can read it.
- The panel size and cell count are provisional internal settings.
- A recall requirement is for image classification only: one class per requirement, one frozen operating point per call,
  with the false-positive limit on the pooled negatives. It is refused on grouped (video) data and with
  `false_positive_scope="each_class"`. Feasibility judges whether the requirement is reachable. It does not evaluate or
  accept a model, and an uncertified witness is not evidence for it.

The conceptual limitations (bounds can be wide, and claims are about labeled items under the annotation protocol) are on
the [Performance limits](PerformanceLimits.md#limitations) page.

## Related concept pages

- [Performance limits](PerformanceLimits.md): what Feasibility answers, why, and how it relates to BER, UAP and
  Sufficiency
- [Embeddings](Embeddings.md): the representation that local panels partition and the built-in witnesses read
- [Acting on results](ActingOnResults.md#performance-limits-findings): what to do with the BER, UAP and Sufficiency
  findings that complement a determination

## See this in practice

- [Assess whether a classification requirement is reachable](../notebooks/tt_assess_ic_feasibility.py): vehicle type,
  vehicle category and tank recall requirements on MilitaryVehicles, the evidence behind each determination read as a table
  and a chart, and how to change a question the evidence cannot answer

## References

1. Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. *Journal of the American
   Statistical Association*, 58(301), 13–30. [paper](https://doi.org/10.1080/01621459.1963.10500830)  
   (Every Hoeffding radius in Feasibility's bounds, including both steps of the local-panel pooling.)

2. Maurer, A., & Pontil, M. (2009). Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd
   Annual Conference on Learning Theory (COLT 2009)*. [paper](https://arxiv.org/abs/0907.3740)  
   (Theorem 4 gives the empirical Bernstein radius of the hybrid bounds.)

3. Reid, M. D., & Williamson, R. C. (2011). Information, divergence and risk for binary experiments. *Journal of Machine
   Learning Research*, 12, 731–817. [paper](https://jmlr.org/papers/v12/reid11a.html)  
   (The risk-to-AP envelope behind Feasibility's AP envelopes.)

4. Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *Proceedings of the 23rd
   International Conference on Machine Learning*, 233–240. [paper](https://doi.org/10.1145/1143844.1143874)  
   (The precision–recall interpolation behind the AP witness staircase.)

5. Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the
   binomial. *Biometrika*, 26(4), 404–413. [paper](https://doi.org/10.1093/biomet/26.4.404)  
   (The exact binomial bounds of Feasibility's Bernoulli witnesses and prevalence bounds.)

6. Detection Feasibility Envelope (DFE) manuscript: the consensus inequality (replicated-consensus section), the disjoint
   within-cell panels (local-pooling section) and the operating-point inequalities for precision, recall and AP.  
   (The constructions behind every Feasibility envelope and the AP witness. The post-stratified pooling of the local panels
   is DataEval's own, proved in `local_panel_bound`.)
