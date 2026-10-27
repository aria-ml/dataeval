# Feasibility evidence

A determination of {class}`.Feasibility` cites the evidence items behind it in `evidence_ids`, and every evidence item
names the theorems it rests on. This page describes the evidence in `FeasibilityOutput.evidence`: the spine that every
item shares, and then each evidence kind, with what it bounds, its information scope, its `basis`, its `details` keys
and how it spends its share of the error budget. For the determinations, scopes and recommendations themselves, see
[Feasibility](PerformanceLimits.md#feasibility-determinations-with-evidence).

To read a result, start with `print(result)`: a header that states the judged scope and the call's joint confidence,
then a table with one row per requirement (its determination, the deciding lower confidence bound and the deciding upper
confidence bound, each with its source), then "Next", the plain action for each undecided requirement, and "Notes",
which give the observed scores, ceilings narrower than the judged scope and conflicts. A recall requirement shows both
of its witness's bounds in its lower-bound cell. `result.aggregate_by_evidence()` lays out every bound of the call as
one table, one row per evidence item, target and bounded quantity, with a readable source name (such as "your model" or
"group-based ceiling"), the quantity, the bound, the observed score of a witness, its information scope and its
independent units; each bound's `basis` and share of `alpha` stay on `result.evidence`. The items behind both are the
ones this page describes. With `dataeval-plots` installed, `dataeval_plots.plot(result)` draws the same answer as one
band per quantity, split into the requirements the evidence demonstrates, those it leaves unknown and those it rules out
under the recorded assumptions. `plot_evidence=True` draws each source's observed score and confidence bound instead.

```{important}
`Feasibility` and its evidence types are experimental and may change in future releases. This page describes the
first release, which covers image and video classification.
```

## The evidence spine

`FeasibilityOutput.evidence` maps an evidence id to an {class}`.Evidence` item. An item is one source run on one
sample, and it carries one endpoint per target it bounds. Its fields:

| Field | Meaning |
| --- | --- |
| `kind` | The construction and its version, such as `local_panel_ceiling/v1`. A new theory is a new kind or version. |
| `supports` | `"ceiling"` or `"witness"`: which side of a determination it backs. |
| `basis` | The theorems and citations its endpoints rest on. Never empty. |
| `population` | What the claims are about: `"items of <dataset id>"`, or `"labeled items of <dataset id>"` when unlabeled items were excluded. |
| `sample` | Which units this source used, such as `"audit split"` or `"120 items with replicas"`. |
| `information_scope` | `full_input`, `fixed_representation` or `partition_restricted`: the model class its claims cover. |
| `unit` | The independent unit: `"item"`, or `"group"` when `groups=` was given. |
| `provenance` | How the source was built: the route for a ceiling, the model and what it reads for a witness. |
| `endpoints` | One {class}`.Endpoint` per target this source bounds. |
| `details` | Free-form scientific content: assumptions, settings and the raw core results behind each endpoint. |

The determination rules read only the spine; the recommendation also reads `details["endpoints"]`. A witness and a
ceiling are compared only when their endpoints name the same target and their items the same `population`; the
samples may differ. `Evidence.to_dict()` serializes the spine and adds `"schema": "feasibility-evidence/1"`.

Evidence ids and kinds differ where one kind runs more than once in a call:

| Evidence id | `kind` | `supports` | `information_scope` |
| --- | --- | --- | --- |
| `local_panel_ceiling/v1` | `local_panel_ceiling/v1` | ceiling | `partition_restricted` |
| `local_panel_ceiling/v1:accuracy` | `local_panel_ceiling/v1` | ceiling | `fixed_representation` (β) or `full_input` (β and Γ) |
| `local_panel_ceiling/v1:cost_weighted` | `local_panel_ceiling/v1` | ceiling | `partition_restricted` |
| `replica_ceiling/v1` | `replica_ceiling/v1` | ceiling | `full_input` |
| `builtin_witness/v1:knn` | `builtin_witness/v1` | witness | `fixed_representation` |
| `builtin_witness/v1:probe` | `builtin_witness/v1` | witness | `fixed_representation` |
| `user_witness/v1` | `user_witness/v1` | witness | from the declared `provenance["reads"]` |

### Endpoints

An {class}`.Endpoint` is one bound on one target:

| Field | Meaning |
| --- | --- |
| `target_spec` | The target's canonical, versioned JSON, the same string as the `target_spec` column of `determinations`. |
| `value` | The endpoint, on the metric's scale; null when the endpoint is unavailable. |
| `direction` | `"upper"` for a ceiling and for a false-positive-rate endpoint, `"lower"` for a witness's metric endpoint. |
| `alpha` | The claim's share of the call's error budget that this endpoint spends. A witness on a recall target spends half of its claim's share on each of its two endpoints. |
| `quantity` | What the endpoint bounds: `"metric"` (the target's metric; recall for a recall target) or `"false_positive_rate"` (the false-positive rate a recall target limits). Only a witness has a false-positive-rate endpoint, and it is always an upper endpoint. |
| `state` | `unavailable`, `uninformative` or `informative`. |
| `reason` | Why the endpoint is unavailable or uninformative; null when informative. |

The three states are kept distinct:

- `unavailable`: no endpoint could be computed, for example too few units, no complete panel, or a missing
  `max_sequence_length`. The value is null and `reason` says why.
- `uninformative`: computed and valid, but at the metric's trivial value: a ceiling at the maximum, 1 (reason
  `"a ceiling at the metric's maximum (1)"`), a witness at the minimum, 0, or a false-positive upper endpoint at 1
  (reason `"a false-positive bound at its maximum (1)"`). It says nothing about any threshold.
- `informative`: valid and not trivial.

### Details shared by every kind

| Key | Contents |
| --- | --- |
| `endpoints` | For each endpoint's key, the summary statistics a recommendation reads: `mean` (the point estimate on the metric's scale), `variance`, `n` (independent units), `support`, `delta` and `method`. The key is the target's canonical spec, and `"<spec>#false_positive_rate"` for a witness's false-positive endpoint, so the two endpoints of a recall target stay apart. None where the endpoint is unavailable or has no closed-form radius to invert: AP endpoints, recall ceilings and `frame_pooled` ratios. |
| `results` | For each target's canonical spec, the raw core results the endpoint was computed from, or None where it is unavailable. Each carries `deltas`, the error shares it spent. Raw results are on the observation's scale: an accuracy ceiling's results bound a risk, and its endpoint is one minus that bound. |
| `assumptions` | The assumptions the item's claims rest on, as sentences. |
| `seed` | The seed of the fit/audit split, the partition and the panels. |
| `call_alpha` | The error budget of the call (of the look, with a sampling plan). |
| `allocation` | How the budget is split over claims: `"equal_per_claim"`, a provisional rule. |
| `order_seed` | The sampling plan's order seed, or None without a plan. |
| `sampling_plan`, `look` | With a sampling plan only: its sizes, and which look this is. |

### How the error budget is split

A call spends one budget, `alpha`, across every endpoint of every target; with a sampling plan, look `t` spends
`alpha / (t (t + 1))`. Each pair of a target and an evidence source is one claim, and every claim gets an equal share
`alpha / n_claims`. Per target, local panels count as one claim (none for a `frame_pooled` target), replicas as one,
the built-in witnesses as two and the user witness as one, each when present. Every endpoint records its claim's share
in `alpha`; inside the claim the share is split again, as each section below describes, and `deltas` in
`details["results"]` records the split. A witness on a recall target splits its claim in half between its recall and
false-positive endpoints. By a union bound, all the endpoints of a call hold together with probability
at least `1 - alpha`.

## `local_panel_ceiling/v1`

This kind gives an upper bound on the accuracy, the AP, or the recall at a false-positive limit, that a model in its
information scope can reach on the population. It needs only the dataset's own labels.

Feasibility splits the units in half, keeping each group whole: a fit split and an audit split. It
fits a k-means partition of the representation, with `n_cells` cells, on the fit split only, so labels never enter the
partition. Within each cell it draws random disjoint panels of `panel_size` (m) audit units; on grouped data it first
draws one item per group, uniformly. The labels of a panel come from different images in one cell, so they are draws
from the cell's label distribution, and their consensus value lower-bounds the risk of the best single answer for the
cell (the consensus inequality of the Detection Feasibility Envelope construction, DFE). The panel means of the cells
are then pooled with the audit's own cell shares. The disjoint within-cell panels are DFE's; this post-stratified
pooling of them is DataEval's own construction (C3). Its proof, two Hoeffding (1963) steps, is in the Notes of
{func}`.local_panel_bound`:

$$\ell = \sum_h \frac{n_h}{N}\left(\bar g_h - \beta_h\right) - r_1 - r_2,
\qquad r_1 = v\sqrt{\frac{\ln(1/\delta_w)}{2}\sum_{h:\,k_h>0}\frac{n_h^2}{N^2 k_h}},
\qquad r_2 = (v + \beta_{\max} - \beta_{\min})\sqrt{\frac{\ln(1/\delta_b)}{2N}}$$

where $n_h$ of the $N$ audit units fall in cell $h$, $\bar g_h$ is the mean consensus value of its $k_h$ panels (0
for a cell without panels), $v$ is the largest value the consensus can take and $\beta_h$ is the heterogeneity
allowance (0 without one). $r_1$ covers the panels given the cells, and $r_2$ covers the cell shares; both are
Hoeffding (1963) steps, valid for independent terms that are not identically distributed, and $\ell$ is floored at 0.

- Accuracy: the 0-1 consensus value is $1 - \max_k N_k / m$, the share of a panel's labels outside its most common
  class, with $v = 1 - \lceil m / M \rceil / m$ for $M$ classes. The ceiling is $1 - \max(0, \ell - \Gamma)$, with
  $\Gamma$ the information-loss allowance (0 without one).
- AP, per class: the one-vs-rest consensus value is $\min(k/m, \lambda(1 - k/m))$, where $k$ counts the class's
  labels in a panel and $\lambda$ is the cost of a false alarm relative to a missed positive. It is pooled as above at
  each positive cost where it changes slope, $\lambda = k/(m-k)$ for $k = 1, \dots, m - 1$ (at $m = 5$: 1/4, 2/3, 3/2
  and 4). Those risk lower bounds and an upper endpoint on the class's prevalence in the audit feed the
  Reid–Williamson (2011) risk-to-AP envelope, an upper bound on any scorer's population AP. Allowances are not applied
  to AP. Macro AP is the mean over every class index from 0 to the largest declared index, so a class absent from the
  data, or an index missing from a sparse `index2label`, still counts with its trivial endpoint.
- Recall, per class: a recall target (a minimum recall $r_0$ for a class at a false-positive limit $f_0$) uses the
  same one-vs-rest consensus values and risk lower bounds $\ell_j$ at the positive kink costs $\lambda_j$, with a
  two-sided interval $[\pi^-, \pi^+]$ on the class's prevalence in the audit. The ceiling $U_r$ bounds the recall any
  model that gives one answer per cell reaches at a false-positive rate of at most $f_0$:
  $U_r = \min_j \max_{\pi \in [\pi^-, \pi^+]} \left[1 - \lambda_j f_0 - (\ell_j - \lambda_j f_0)/\pi\right]$, each term
  clipped to $[0, 1]$ (a term whose maximum is unbounded, because $\ell_j < \lambda_j f_0$ and $\pi^- = 0$, is 1). The
  derivation and the three cases are in [Performance limits](PerformanceLimits.md#recall-at-a-false-positive-limit).
  The target is excluded if and only if $U_r < r_0$. Allowances are not applied to recall.

Its information scope depends on the allowances. Without allowances, the consensus bounds only models that give one
fixed answer per cell, so the item is `partition_restricted`. On image data, `allowances={"beta": ...}` makes the
accuracy endpoints `fixed_representation` and adding `"gamma"` makes them `full_input`; the item then splits into
`local_panel_ceiling/v1:accuracy` at that scope and `local_panel_ceiling/v1:cost_weighted` (AP and recall) at
`partition_restricted`, each present when the call has such targets. On grouped (video) data, local panels read single
frames and stay `partition_restricted` whatever the allowances. β is a scalar, and it must be a justified bound: β = 0
asserts that no model reading the representation beats one fixed answer per cell. A per-cell β is rejected, since the
partition's cells are internal.

The `basis` string is:

```text
DataEval's post-stratified pooling (C3) of DFE disjoint within-cell panels, proved in local_panel_bound's Notes with two
Hoeffding (1963) steps; DFE consensus inequality; AP via the Reid-Williamson (2011) risk-to-AP envelope (DFE
operating-point inequalities); recall at a false-positive limit via DFE operating-point inequalities over a two-sided
prevalence interval; prevalence: Clopper-Pearson (1934) exact binomial endpoints on ungrouped Bernoulli trials, else
hybrid
```

In the spine, `sample` is `"audit split"`, or `"audit split, one item drawn uniformly per group"` on grouped data.
`provenance` is `{"route": "post-stratified local panels (C3)", "partition": "k-means fitted on the fit split"}`.

Besides the shared keys, `details` holds:

| Key | Contents |
| --- | --- |
| `panel_size` | m, the labels per panel. |
| `n_cells` | The partition's cell count. |
| `provisional_defaults` | Provisional items: `n_cells` and `panel_size`, each listed when left unset. |
| `cost_grid` | The AP and recall ceilings' cost grid: `"kinks"`, where the consensus changes slope (provisional). |
| `allowances` | The allowances you passed. |
| `allowances_applied` | `beta` and `gamma` as applied to accuracy endpoints: None and 0 at `partition_restricted`. |
| `panel_items` | The positions of the audit units the panels were drawn from. |

`details["results"]` holds, per accuracy target, the {func}`.local_panel_bound` result (`bound`, `within_radius`,
`between_radius`, `n_units`, `n_panels`, `panels_by_cell`, `deltas`, `value_max`) plus `gamma`; the ceiling is `1 -
max(0, bound - gamma)`. Per AP class it holds `value`, `costs`, `risk_lower`, `risk_results` (one `local_panel_bound`
result per positive cost, and None at the cost 0 that `costs` and `risk_lower` keep), `mu_upper` and `deltas`; a macro
AP target holds `value`, `per_class` and `deltas`. Per recall target it holds `value` (that is $U_r$), `costs`,
`risk_lower`, `risk_results`, `pi_lower`, `pi_upper` and `deltas`; unlike an AP ceiling's, its `costs` and `risk_lower`
drop the cost 0, whose risk is trivially 0 and bounds nothing, so every entry has a `risk_results` item. The
`details["endpoints"]` entry of an accuracy target has `method` `"hoeffding"`, `n` equal to the audit units and a
`support` that is an effective constant: it reproduces $r_1 + r_2$, so the suggested minimum tracks this construction
rather than a single Hoeffding step.

The claim's share is spent as follows. Accuracy: two Hoeffding steps, $\delta_w = \delta_b$ = half the claim's share.
AP for one class: the share is split equally over the $m - 1$ positive costs and one prevalence upper endpoint (5 parts
at $m = 5$), and each cost's pooling halves its part again. Macro AP: the share is first divided equally among the
classes. Recall: the share is split equally over the $m - 1$ positive costs and two prevalence endpoints, lower and
upper (6 parts at $m = 5$), and each cost's pooling halves its part again. The prevalence endpoints are exact binomial
on ungrouped data (the panel units' 0/1 labels), hybrid on grouped data, where recall targets are refused anyway.

It records these assumptions: the representation was fitted and selected independently of the audit units' labels
(declared by you); β and Γ are user-supplied and assumed, not estimated; on grouped data, single-frame local panels stay
`partition_restricted`.

The defaults are `panel_size` 5 (at least 2 when you set it); `n_cells` 10 below 20,000 planned audit units, 30 below
50,000, and 300 from 50,000. Both are provisional and recorded in `provisional_defaults` when you leave them unset.

An endpoint is unavailable when the fit split has fewer items than cells, the audit has fewer than 2 units, or no cell
holds m audit units.

### Allowances

Without allowances, the local-panel accuracy ceiling bounds only models that give one fixed answer per cell. On image
data, an allowance extends it to a broader scope by asserting, in risk units, how much better a broader model can do.
On grouped (video) data, local panels read single frames and stay `partition_restricted` whatever the allowances.

- β (`allowances={"beta": ...}`) bounds how much a model reading the representation can improve on one fixed
  answer per cell. With it, the ceiling covers `fixed_representation`. β = 0 asserts that no model reading the
  representation beats one fixed answer per cell.
- Γ (`"gamma"`, given with β) bounds how much reading the full input can lower the best risk below what the
  representation allows. With both, the ceiling covers `full_input`. Γ = 0 asserts that reading the full input cannot
  lower it at all.

A larger allowance gives a higher, weaker ceiling. Feasibility does not estimate or check an allowance: it records it
as an assumption, and every claim of the extended ceiling, including a target ruled out under the recorded assumptions,
holds only if the allowance does. The justification must therefore come from outside the call: an argument or separate
evidence, kept with the result, that each allowance bounds the improvement it describes on this population. Without
one, repeated labels (`replica_ceiling/v1`, below) give a `full_input` ceiling that needs no allowance.

## `replica_ceiling/v1`

This kind gives an upper bound on the accuracy, the AP, or the recall at a false-positive limit, that any model
reading the full input can reach on the population, against one label drawn under the annotation protocol.

`replicas=` maps datum ids to extra, independent class labels of those items. Each such item's panel
is its primary label plus its replicas, so $m_i = 1 + r_i$; items without replicas are not used. Because a panel holds
labels of one input, its consensus value lower-bounds the error of the best answer to that input, whatever the model
reads (the DFE consensus inequality). The accuracy ceiling is one minus a lower endpoint on the mean consensus value. AP
uses the cost-weighted consensus at the slope-change costs of the most common panel size and the class's prevalence
upper endpoint through the Reid–Williamson (2011) envelope, as for local panels. A recall target uses the same costs and
risk lower bounds with a two-sided interval on the class's prevalence, and reports $U_r$ as for local panels, here at
`full_input`; its prevalence is the mean label share over the items with replicas. Grouped data use a group's mean
(`sequence_mean`) or total (`frame_pooled`) as its observation; a `frame_pooled` accuracy is a ratio of per-group totals
bounded with simultaneous numerator and denominator endpoints, whose support is the declared `max_sequence_length`.

Its information scope is `full_input`.

The `basis` string is:

```text
DFE replicated consensus (consensus inequality); Hoeffding (1963) and Maurer-Pontil (2009, Theorem 4) hybrid endpoints;
frame_pooled as a ratio of per-group totals with simultaneous numerator and denominator endpoints; AP via the
Reid-Williamson (2011) risk-to-AP envelope (DFE operating-point inequalities); recall at a false-positive limit via DFE
operating-point inequalities over a two-sided prevalence interval
```

In the spine, `sample` is `"<n> items with replicas"`. `provenance` is
`{"route": "replicated consensus", "replicas": "replicas= sidecar plus the primary label"}`.

Besides the shared keys, `details` holds `replica_items`, the positions of the items with replicas, and
`cost_grid`, as for local panels.
`details["results"]` holds, per accuracy target, the {func}`.mean_endpoint` result on the consensus loss (`n`, `mean`,
`variance`, `radius_hoeffding`, `radius_bernstein`, `endpoint`, `method`, `side`, `support`, `delta`) plus `deltas`;
for `frame_pooled`, `ratio_endpoint`, `num_support`, `den_support` and `deltas`. Per AP class it holds `value`,
`costs`, `risk_lower`, `risk_results` (one `mean_endpoint` result per positive cost, and None at the cost 0 that `costs`
and `risk_lower` keep), `mu_upper` and `deltas`. Per recall target it holds `value` ($U_r$), `costs` and `risk_lower`
(positive costs only, the cost 0 dropped), `risk_results`, `pi_lower`, `pi_upper` and `deltas`.

The claim's share is spent as follows. Accuracy: one hybrid endpoint at the claim's share. The hybrid method computes a
Hoeffding radius and an empirical Bernstein radius, each at half that share, and uses the smaller. When the items'
panel sizes differ, their consensus values are not identically distributed, so every replica endpoint uses Hoeffding
alone, at the full share; Hoeffding needs only independent observations with a common support. `method` in
`details["endpoints"]` and in the raw results records which. A `frame_pooled` ratio spends half on the numerator and
half on the denominator. AP for one class: the share is split equally over the positive costs and one prevalence upper
endpoint; macro AP first divides it among the classes. Recall: the share is split equally over the positive costs and
two prevalence endpoints (lower and upper). A label share is not a Bernoulli trial, so these endpoints use the same
method as the item's other replica endpoints (hybrid, or Hoeffding alone when panel sizes differ), never exact
binomial.

It records these assumptions:

- Items with replicas are a random sample of the population, so the ceiling describes the population, not only those
  items.
- Replicas and primary labels follow one annotation protocol, each an independent label of the item. Witnesses are
  scored on primary labels, so the ceiling and the witnesses then concern the same target.
- On grouped data, annotators labeled the replicas with the same video context as the primary labels.
- For `sequence_mean` targets, replicated groups are a random sample of sequences, with the replicated frames uniform
  within each.
- For `frame_pooled` targets, replicated frames are a uniform sample of all frames (whole random sequences, or each
  frame independently with one probability).
- With a sampling plan, the replica set was fixed before the first look.

## `builtin_witness/v1`

Two items of this kind run by default (`builtin_witnesses=True`): `builtin_witness/v1:knn`, a 5-nearest-neighbor
classifier, and `builtin_witness/v1:probe`, a logistic-regression probe.

This kind gives a lower bound on the accuracy, or the AP, that the frozen model reaches on the population, and, for
a recall target, a lower bound on its recall together with an upper bound on its false-positive rate.

Each model is fitted on the fit split's representation and labels only, then frozen and scored once on
the audit split. Accuracy uses each unit's correctness (a group's mean or total on grouped data). A recall target for
class $c$ uses the model's top class as its decision: a lower endpoint on the share of decisions $c$ among the audit
items labeled $c$ (recall), and an upper endpoint on the share among the other audit items (false-positive rate), each
exact binomial and at half the claim's share. AP for a class uses
predeclared score thresholds fixed from the fit split: the 5% to 95% quantiles of the fit-split positives' scores when
the class's fit-split prevalence is below 0.15 and the fit split holds positives, and of all fit-split scores otherwise.
At each threshold a lower endpoint on true positives and an upper endpoint on false positives, with an upper endpoint on
the prevalence, give a staircase that lies under the model's precision-recall curve (Davis and Goadrich, 2006). On
ungrouped data all of these AP endpoints are exact binomial.

Its information scope is `fixed_representation`: the models read only the representation. They are eligible to
demonstrate `fixed_representation` and `full_input` targets.

The `basis` string is shared with `user_witness/v1`:

```text
Frozen witness scored once on the audit; Clopper-Pearson (1934) exact binomial endpoints on ungrouped Bernoulli trials,
else Hoeffding (1963) and Maurer-Pontil (2009, Theorem 4) hybrid endpoints; frame_pooled as a ratio of per-group totals
with simultaneous numerator and denominator endpoints; AP via the DFE witness staircase (Davis-Goadrich 2006)
```

In the spine, `sample` is `"audit split"`. `provenance` is `{"reads": "representation", "model": ..., "params":
..., "fit": "fit split"}`, with the model's class name and its parameters.

`details["results"]` holds, per accuracy target, the `mean_endpoint` result on per-unit correctness (the
`binomial_endpoint` result on ungrouped data, with `method` `"exact_binomial"`) plus `deltas` (or the ratio form for
`frame_pooled`). Per AP class it holds `value`, `thresholds`, `tp_lower`, `fp_upper`, `mu_upper` and `deltas`; a macro
AP target holds `value`, `per_class` and `deltas`. Per recall target it holds the recall endpoint's `value`, `stats` and
`population` (such as `audit items labeled 'Tank'`), a `false_positive_rate` entry of the same form for the other audit
items, and `deltas`.

The claim's share is spent as follows. Accuracy: one endpoint at the claim's share: exact binomial on ungrouped data,
else hybrid (Hoeffding and empirical Bernstein at half each). AP for one class with $T$ distinct thresholds: $2T + 1$
endpoints at an equal part each. Macro AP first divides the share among the classes. Recall: the recall and
false-positive endpoints at half the claim's share each.

It records these assumptions: the representation was fitted and selected independently of the audit units' labels
(declared by you); the model was fitted on the fit split only and frozen before the audit.

An endpoint is unavailable when a model cannot be fitted on the fit split or score the audit, for example with a single
class in the fit split. A recall endpoint is unavailable when the audit holds no item of the class, and the
false-positive endpoint when it holds no other item.

## `user_witness/v1`

This kind gives a lower bound on the accuracy, or the AP, that your frozen model reaches on the population, and, for
a recall target, a lower bound on its recall together with an upper bound on its false-positive rate.

`predictions=` holds your model's class scores, shape `(len(dataset), n_classes)`, aligned with the
dataset and finite, or class decisions (below). The scores on the fit split set the AP thresholds; the scores on the
audit split give the endpoints, as for the built-in witnesses.

You can also pass class decisions. A 1-D integer array of length `len(dataset)` is read as one class decision per item,
indices of the dataset's declared classes. An index outside them is refused, and so is a boolean or float array (cast
to integers, a boolean would silently mean classes 0 and 1). Decisions are turned into one-hot scores, so the decision
is the given class and accuracy and recall targets get their endpoints from it. They rank nothing, so an AP target gets
no endpoint from them: it is unavailable with the reason
`"predictions are class decisions, not scores; AP needs scores"`. Use decisions for any policy other than the top
class, such as a threshold on one class's score applied as a complete multiclass policy before the call. One call's
decisions feed every target in it, so a second operating point needs its own call with the error budget split across
the planned calls. Scores for a relabeled dataset come from `Relabel.fold_scores`, which sums the columns of source
classes that map to one target class (see
[Performance limits](PerformanceLimits.md#decision-arrays-and-folded-scores)).

The information scope comes from `provenance["reads"]`, the model class the witness belongs to:

| `reads` | `information_scope` | Eligible to demonstrate |
| --- | --- | --- |
| `"full_input"` (default) | `full_input` | `full_input` targets |
| `"representation"` | `fixed_representation` | `fixed_representation` and `full_input` targets |

A witness giving one answer per cell (`"partition"`) would be eligible at every scope, but it needs the internal
partition, which this release does not expose, so it is rejected.

The `basis` string is the same as for `builtin_witness/v1`.

In the spine, `sample` is `"audit split"`. `provenance` is your declared provenance, with `reads` filled in. Its
accepted keys are `reads`, `training_items`, `frozen_before_audit`, `selection`, `preprocessing` and `decision_policy`;
others are rejected, and `provenance` is accepted only together with `predictions`. Two are checked: `reads` must be one
of its two values, and `training_items`, the datum ids of this dataset's items that the model trained on (any iterable,
a set too), must be of the datum ids' type and share no physical group with an audit unit; any violation raises a
`ValueError`. The rest are recorded, not checked. `decision_policy` is the rule that turned scores into the class
decisions, frozen before the audit; it is accepted only when `predictions=` are class decisions, and a call that gives
it with scores raises a `ValueError`.

`details` holds `assumptions` and `results` as for `builtin_witness/v1`.

The error budget is spent as for `builtin_witness/v1`.

It records these assumptions: your declared provenance is recorded, not verified. A demonstration by this witness
assumes the model was trained without the audit units and frozen before scoring them. With class decisions, the
evidence adds that the decisions follow a policy frozen before the audit (declared).

### Read a recall target

For `Tank` at least 80% recall with at most 10% false positives, one call on a synthetic
dataset with class decisions and `provenance={"decision_policy": "top class of the frozen classifier"}` prints:

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
reaches 80%. `aggregate_by_evidence()` lists the two bounds as separate rows with `quantity` `recall` and
`false_positive_rate`, `bound_type` `lower` and `upper`, and the counts behind each in `n` (261 positives, 489
negatives); the `threshold` column repeats the target's threshold on both rows. The upper cell shows `—` because the
only ceiling, the group-based one, bounds a narrower scope than the one judged and is uninformative here (it is 1).
Had its false-positive upper endpoint exceeded 10%, the lower cell would still show both bounds, marked
`(your model, not counted)`, and a note would say no tested model counts.

## What the first release does not do

- Object detection and tracking: `evaluate` raises `NotImplementedError` for detection and tracking datasets.
- Frame-weighted local panels: `frame_pooled` targets get no local-panel ceiling yet, and spend no budget on one;
  their ceilings come only from replicas.
- User-supplied partitions: Feasibility always fits its own label-blind k-means partition on the fit split. Since
  that partition is internal, a per-cell β and a witness reading the partition (`reads="partition"`) are rejected.
- Estimated allowances: β and Γ are taken as given and recorded as assumed; Feasibility does not estimate them.
- Recall beyond one pooled operating condition: a recall target is for image classification, one class per target
  and one frozen operating point per call, with the false-positive limit on the pooled negatives. It is refused on
  grouped data, with `false_positive_scope="each_class"`, and as a mapping key.

## Related concept pages

- [Performance limits](PerformanceLimits.md) — the determinations, information scopes and recommendations these
  evidence items feed
- [Embeddings](Embeddings.md) — the representation that local panels partition and the built-in witnesses read

## See this in practice

- [Assess whether a classification requirement is reachable](../notebooks/tt_assess_ic_feasibility.py) — the evidence
  behind a tank recall requirement, read as a table and a chart
