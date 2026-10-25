"""
Known-answer validation suites for Feasibility: each bound against the quantity it estimates, known in closed form
or by Monte Carlo.

Every image target is analytic (gauss2's and rare5's closed forms through Phi and Bin(m, p), or a quad integral to
1e-9), so no seed is ambiguous. The video targets are Monte Carlo truths over 400,000 sequences, and a seed within
3·SE of one is ambiguous: neither covered nor missed, and counted against the claim. A coverage claim runs
``COVERAGE_SEEDS`` seeds and fails at ``COVERAGE_CRITICAL`` or fewer covered: the exact one-sided binomial test of
coverage >= 0.95 at level 0.01 (spike report §6). The same count bounds a determination error rate at 0.05. The
suites validate the implementation and its empirical coverage; the proofs are the cited theorems.

Claims, each at alpha 0.05 (gauss2: classes 0 and 1 with means +e1 and -e1 in 8-D, sigma 1, posterior
p = sigmoid(2 x1); rare5: priors 0.60, 0.20, 0.10, 0.07, 0.03 with means 3 times random unit vectors in 16-D; labels
are drawn from the posterior, so (x, label) has the generative joint and a policy's recall and false-positive rate are
measured against a posterior label draw, as replicas are):

- replica accuracy ceiling (primary + 2 replicas, 2,000 items) >= 1 - E_x[p(1-p)] = 0.88760, the expected 3-label
  consensus; it implies the weaker claim >= Bayes accuracy Phi(1) = 0.84134;
- accuracy witness of the Bayes scorer (the posterior, through ``_Scores.bound``: exact binomial lower endpoint, 2,000
  items) <= Phi(1);
- negative control: the same witness's point estimate, with no radius, must fail the harness (covered in at most
  ``COVERAGE_CRITICAL`` seeds), which shows the harness has the power to reject;
- video accuracy witness of the Bayes scorer through ``_Scores.bound`` on ``make_sequences`` gauss2 (500 sequences of
  1 to 8 frames, rho 0.8, declared max_sequence_length 8): the ``sequence_mean`` endpoint <= the mean per-sequence
  accuracy (0.82115, SE 0.00018) and the ``frame_pooled`` ratio endpoint <= the pooled frame accuracy (0.82123,
  SE 0.00018);
- C3 risk lower bound (m = 5, 20,000 units) on a fixed 11-cell partition of x1 <= sum_h w_h E[min(K, 5-K)] / 5
  with K ~ Bin(5, p_h(0)) and population w_h, p_h (0.13417); the cell counts' sampling is the between-cell radius's.
  It implies the weaker claim <= the partition risk sum_h w_h (1 - max_c p_h(c)) = 0.16614;
- C3 class-0 AP ceiling on the same panels and cells >= the population AP of the best scorer that reads only the cell,
  cells ranked by p_h(0) and each emitted whole (0.89956);
- AP witness staircase of the posterior scorer (1,000 fit, 1,000 audit) <= its population AP (quad, 0.92177). This
  checks validity only: the slack belongs to the bound, 39 Bonferroni'd endpoints (19 thresholds) at 1,000 audit
  items, while the population staircase on the same thresholds sits only about 0.01 below the AP;
- kink cost grid vs G_DFE (spike report §8): C3 AP ceilings on the same gauss2 panels and cells (20,000 units, 50
  seeds, sampled endpoints); the mean paired difference kink - G_DFE is at most 3 SE, per class;
- determination errors of a representative call (gauss2, 600 items, 2 replicas on 300, built-in witnesses):
  false ``demonstrated_feasible`` at accuracy 0.95 (above Phi(1)) and false
  ``infeasible_under_recorded_assumptions`` at 0.70 (below it);
- recall witness (class 0 at false-positive limit F0 = 0.1) of the Bayes Neyman-Pearson rule, class 0 when
  x1 > 0.28155, through ``_Scores.recall`` on class decisions (2,000 items): jointly, the exact binomial recall lower
  endpoint <= its recall Phi(2 + Phi^-1(F0)) = 0.76376 and the false-positive upper endpoint >= its rate F0;
- negative control: the same recall point estimate, with no radius, must fail the harness;
- C3 recall ceiling U_r of class 0 at F0 on the same panels and cells >= U_r at the expected panel consensus and the
  true prevalence (0.81458): min over the positive kink costs lam of 1 - (E_lam - lam (1 - pi) F0) / pi with
  E_lam = sum_h w_h E[min(K/5, lam (1 - K/5))], K ~ Bin(5, p_h(0)), pi = 0.5. Each E_lam is at most the best
  cost-weighted risk, so it implies the weaker claim >= the Neyman-Pearson recall at F0 of randomized
  one-answer-per-cell policies (cells by likelihood ratio, the boundary cell randomized; 0.75830);
- replica recall ceiling (primary + 2 replicas, 2,000 items) >= U_r at the expected 3-label consensus (quad over x1,
  K ~ Bin(3, p)) and pi = 0.5 (0.86280); it implies >= the Bayes Neyman-Pearson recall at F0, 0.76376;
- rare5 recall ceiling: class 4 (prior 0.03) on a fixed 11-cell partition of the projection on its mean direction
  (20,000 units, F0 = 0.005) >= U_r at its expected panel consensus (0.67612), so >= the Neyman-Pearson cell recall
  (0.59654). The cheapest cost's l_j straddles lam_j F0 across seeds, and the test asserts that pi_lower sets U_r in
  some seeds (U_r above what pi_upper alone gives);
- recall determination errors (gauss2, 2,000 items; a replica ceiling and two decision witnesses, the NP rule and the
  top class with recall Phi(1) at false-positive rate Phi(-1) > F0, on one ledger of alpha / 6 per claim): false
  ``demonstrated_feasible`` 0.01 above the Bayes NP recall and false ``infeasible_under_recorded_assumptions`` 0.01
  below it.

Observed on 2026-10-21, covered/seeds with the min and mean margin to the target: replica 346/346 (0.0010, 0.0103);
witness 328/346 (-0.0130, 0.0142; the exact bound is tight, so about alpha of the seeds miss, where the hybrid
endpoint covered 346/346 at (0.0014, 0.0294)); point estimate 175/346 (-0.0262, 0.0001); video
``sequence_mean`` 346/346 (0.0225, 0.0527) and ``frame_pooled`` 346/346 (0.1627, 0.1824, the ratio's support is the
declared length), 0 ambiguous; C3 346/346 (0.0063, 0.0126); C3 AP ceiling 346/346 (0.0614, 0.0650); AP witness
346/346 (0.189, 0.230; hybrid: 0.273, 0.318). The witness and AP witness were re-measured on 2026-10-23 after the
exact binomial switch. Kink - G_DFE: mean -0.00095 (SE 6e-6) in both classes, negative in every seed.

Recall claims, observed on 2026-10-23: witness 331/346 jointly (-0.0109, 0.0169; recall alone 339, false positives
alone 338); recall point estimate 173/346 (-0.0373, -0.0005); C3 recall 346/346 (0.0301, 0.0460; to the cell NP recall
0.0864, 0.1023), U_r set at pi_lower in 337 seeds; replica recall 346/346 (0.0263, 0.0448; to the Bayes NP recall
0.1254, 0.1438), at pi_lower in 243; rare5 346/346 (0.2815, 0.3185), at pi_lower in 13 (U_r < 1 in 166). U_r at its
point estimates (no radius) covers the C3 and replica targets in 191 and 181 seeds (mean differences under 0.001), so
the targets are what the ceilings estimate. Recall determinations: ``not_ruled_out`` in every seed at both thresholds
(the recall radius at 1,000 positives is about 0.035, the ceiling about 0.15 above the truth); counting an uncertified
witness would demonstrate the upper threshold in 345/346. Runtime about 20 s for the module on one core
(single-threaded BLAS): the two determination tests 9 s, the recall claims 8 s.
"""

import math
from collections.abc import Callable
from functools import cache, partial
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import binom, norm

from dataeval.core import disjoint_panels, kink_costs
from dataeval.performance import Feasibility, TargetSpec
from dataeval.performance import _feasibility_classification as fc
from dataeval.performance._feasibility_constants import COVERAGE_CRITICAL, COVERAGE_SEEDS, PANEL_SIZE
from dataeval.performance._feasibility_decide import decide
from tests.data.conftest import MockICDataset
from tests.performance.feasibility_helpers import Population, draw_labels, make_population, make_sequences, make_units

pytestmark = [pytest.mark.performance, pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")]

ALPHA = 0.05
GAUSS2 = make_population("gauss2")
MEANS_X1 = np.array([1.0, -1.0])  # x1 means of classes 0 and 1
BAYES_ACCURACY = float(norm.cdf(1.0))


def _density(t: float) -> float:  # x1 is the mixture 0.5 N(1, 1) + 0.5 N(-1, 1)
    return float(0.5 * (norm.pdf(t - 1) + norm.pdf(t + 1)))


def _replica_target() -> float:
    """1 - E_x[p(1-p)] with p = sigmoid(2 x1), the class-0 posterior: three labels have E[b_3 | p] = p(1 - p)."""
    value, error = quad(lambda t: _density(t) * expit(2 * t) * expit(-2 * t), -np.inf, np.inf, epsabs=1e-12)
    assert error <= 1e-9
    return 1.0 - value


REPLICA_TARGET = _replica_target()  # 0.88760, above Phi(1): E[b_3] <= Bayes risk


def _cell_mass(edges: NDArray[np.float64], centers: NDArray[np.float64], priors: NDArray[np.float64]) -> NDArray:
    """
    P(label c, projection in cell h), cell h being (edges[h-1], edges[h]], for a projection on a unit direction.

    Labels follow p(c|x), so (x, label) has the generative joint: class c has prior priors[c] and its projection is
    N(centers[c], 1).
    """
    lo, hi = np.r_[-np.inf, edges][:, None], np.r_[edges, np.inf][:, None]
    return priors * (norm.cdf(hi - centers) - norm.cdf(lo - centers))


# Fixed partition of x1: cell h is (EDGES[h-1], EDGES[h]]; the cell around 0 straddles the Bayes boundary.
EDGES = np.linspace(-2.25, 2.25, 10)
_MASS = _cell_mass(EDGES, MEANS_X1, GAUSS2.priors)  # P(label c, x1 in cell h)
_W, _K = _MASS.sum(axis=1), np.arange(PANEL_SIZE + 1)  # population cell weights w_h; panel positives K
_PMF = binom.pmf(_K, PANEL_SIZE, _MASS[:, :1] / _W[:, None])  # K ~ Bin(m, p_h(0)) per cell
# sum_h w_h E[min(K, m - K)] / m: the expected 0-1 consensus of a within-cell panel (0.13417); partition risk 0.16614
C3_TARGET = float(_W @ _PMF @ np.minimum(_K, PANEL_SIZE - _K)) / PANEL_SIZE


def _cell_ap() -> float:
    """Class-0 AP of the best scorer reading only the cell: cells ranked by p_h(0), each emitted whole."""
    order = np.argsort(-_MASS[:, 0] / _W)
    tp, fp = np.cumsum(_MASS[order, 0]), np.cumsum(_MASS[order, 1])
    return float(np.diff(np.r_[0.0, tp / tp[-1]]) @ (tp / (tp + fp)))


CELL_AP = _cell_ap()


def _cell_np_recall(pos: NDArray[np.float64], neg: NDArray[np.float64], f0: float) -> float:
    """
    Neyman-Pearson recall at false-positive rate f0 over randomized one-answer-per-cell policies, from the cell masses.

    Cells enter by likelihood ratio pos/neg, the boundary cell answered positive with the probability that spends
    the rest of f0: linear interpolation along the cells' ROC polyline.
    """
    order = np.argsort(-pos / neg)
    tpr, fp_rate = np.cumsum(pos[order]) / pos.sum(), np.cumsum(neg[order]) / neg.sum()
    return float(np.interp(f0, np.r_[0.0, fp_rate], np.r_[0.0, tpr]))


def _positive_kinks(m: int) -> NDArray[np.float64]:
    """The positive kink costs k / (m - k), k = 1..m-1: the recall ceiling's cost grid for panels of m."""
    k = np.arange(1, m)
    return k / (m - k)


def _cell_consensus(pos: NDArray[np.float64], neg: NDArray[np.float64]) -> list[float]:
    """Per positive kink cost lam: sum_h w_h E[min(K/m, lam (1 - K/m))], K ~ Bin(m, p_h), the panels' expectation."""
    w, k = pos + neg, _K / PANEL_SIZE
    pmf = binom.pmf(_K, PANEL_SIZE, (pos / w)[:, None])
    return [float(w @ pmf @ np.minimum(k, lam * (1 - k))) for lam in _positive_kinks(PANEL_SIZE)]


def _recall_target(expected: list[float], pi: float, f0: float, m: int) -> float:
    """
    U_r at the expected cost consensus values and the true prevalence: min over the positive kink costs lam_j of
    1 - (E_j - lam_j (1 - pi) f0) / pi, clipped to [0, 1].

    The ceiling covers it when every risk endpoint l_j <= E_j and pi_lower <= pi <= pi_upper. It is at least the
    Neyman-Pearson recall at f0 of the ceiling's scope: E_j is at most the best cost-weighted risk
    pi (1 - r) + lam_j (1 - pi) f of any policy, which at f <= f0 bounds r by the term.
    """
    terms = [1 - (e - lam * (1 - pi) * f0) / pi for e, lam in zip(expected, _positive_kinks(m), strict=True)]
    return float(np.clip(min(terms), 0.0, 1.0))


def _replica_consensus(lam: float) -> float:
    """E_x E[min(K/3, lam (1 - K/3))], K ~ Bin(3, sigmoid(2 x1)): primary + 2 replicas, all from the posterior."""
    k = np.arange(4)
    loss = np.minimum(k / 3, lam * (1 - k / 3))
    value, error = quad(lambda t: _density(t) * binom.pmf(k, 3, expit(2 * t)) @ loss, -np.inf, np.inf, epsabs=1e-12)
    assert error <= 1e-9
    return value


# Recall of class 0 at false-positive rate <= F0. The Bayes Neyman-Pearson rule thresholds the likelihood ratio,
# monotone in x1: class 0 when x1 > NP_CUT, with recall Phi(1 - NP_CUT) and false-positive rate Phi(-1 - NP_CUT) = F0
# against a label drawn from the posterior (the generative joint).
F0 = 0.1
NP_CUT = float(-1 - norm.ppf(F0))
BAYES_NP_RECALL = float(norm.cdf(1 - NP_CUT))  # Phi(2 + Phi^-1(F0)) = 0.76376
CELL_NP_RECALL = _cell_np_recall(_MASS[:, 0], _MASS[:, 1], F0)  # 0.75830
C3_RECALL_TARGET = _recall_target(_cell_consensus(_MASS[:, 0], _MASS[:, 1]), 0.5, F0, PANEL_SIZE)  # 0.81458
REPLICA_RECALL_TARGET = _recall_target([_replica_consensus(c) for c in _positive_kinks(3)], 0.5, F0, 3)  # 0.86282
RECALL_TARGETS = (BAYES_NP_RECALL + 0.01, BAYES_NP_RECALL - 0.01)  # just above and just below the reachable recall

# rare5's class 4 (prior 0.03) on a fixed partition of the projection on its mean direction, at a false-positive limit
# where the cheapest cost's l_j straddles lam_j f0 across seeds, so the lower prevalence end sets U_r in some.
RARE5 = make_population("rare5")
RARE, RARE5_F0 = 4, 0.005
RARE5_AXIS = RARE5.means[RARE] / np.linalg.norm(RARE5.means[RARE])
RARE5_EDGES = np.linspace(-1.0, 4.0, 10)
_RARE5_MASS = _cell_mass(RARE5_EDGES, RARE5.means @ RARE5_AXIS, RARE5.priors)
_RARE5_POS = _RARE5_MASS[:, RARE]
_RARE5_NEG = _RARE5_MASS.sum(axis=1) - _RARE5_POS
RARE5_NP_RECALL = _cell_np_recall(_RARE5_POS, _RARE5_NEG, RARE5_F0)  # 0.59654
RARE5_TARGET = _recall_target(_cell_consensus(_RARE5_POS, _RARE5_NEG), RARE5.priors[RARE], RARE5_F0, PANEL_SIZE)


def _ap_truth() -> float:
    """Class-0 AP of the posterior (monotone in x1): emitting x1 >= t has recall Phi(1-t) at precision below."""

    def precision(t: float) -> float:  # Phi(1-t) / (Phi(1-t) + Phi(-1-t)), on the log scale against underflow
        return 1.0 / (1.0 + math.exp(norm.logcdf(-1 - t) - norm.logcdf(1 - t)))

    value, error = quad(lambda t: precision(t) * norm.pdf(1 - t), -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)
    assert error <= 1e-9
    return value


AP_TRUTH = _ap_truth()
G_DFE = np.array([0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
TARGETS = (0.95, 0.70)  # accuracy thresholds above and below Phi(1)


def _gauss2(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x, y, post = GAUSS2.sample(n, rng)
    return x, y, post, rng


def _call(
    y: NDArray[np.intp],
    x: NDArray[np.float64],
    *,
    fit=None,
    audit=None,
    groups=None,
    max_len=None,
    n_classes: int = 2,
    alpha: float = ALPHA,
) -> fc._Call:
    """One source's call at the claim's alpha; every item is audited unless told otherwise."""
    fit = np.empty(0, dtype=np.intp) if fit is None else fit
    audit = np.arange(len(y)) if audit is None else audit
    return fc._Call(make_units(y, groups=groups, n_classes=n_classes), x, audit, fit, alpha, max_len, {})


def _value(bound: fc._Bound) -> float:
    """A bound's value, which these sources always make available."""
    assert bound[0] is not None
    return bound[0]


def _recall(threshold: float, f0: float = F0, class_name: str = "0") -> TargetSpec:
    return TargetSpec("IC", "recall", threshold, class_name=class_name, max_false_positive_rate=f0)


def _np_rule(x: NDArray[np.float64]) -> NDArray[np.intp]:
    """Class decisions of the Bayes Neyman-Pearson rule at F0: class 0 when x1 > NP_CUT."""
    return np.where(x[:, 0] > NP_CUT, 0, 1)


def _panels(cells: NDArray[np.intp], rng: np.random.Generator) -> dict[int, NDArray[np.intp]]:
    return {h: ix for h, ix in disjoint_panels(cells, PANEL_SIZE, rng).items() if len(ix)}


def _covered(margin: Callable[[int], float], se: float = 0.0) -> tuple[int, int]:
    """
    Covered and ambiguous seeds, by the margin from the bound to its target (positive on the covered side).

    A Monte Carlo truth with standard error ``se`` makes a seed within 3·SE of it ambiguous, neither covered nor
    missed (spike report §6). Acceptance counts covered seeds against all ``COVERAGE_SEEDS``, so an ambiguous seed
    weighs as a miss: conservative, since the critical count only falls with fewer seeds.
    """
    margins = np.array([margin(seed) for seed in range(COVERAGE_SEEDS)])
    ambiguous = np.abs(margins) <= 3 * se if se else np.zeros(len(margins), dtype=bool)
    return int(((margins >= 0) & ~ambiguous).sum()), int(ambiguous.sum())


def _accept(counts: tuple[int, int], claim: str) -> None:
    covered, ambiguous = counts
    assert covered > COVERAGE_CRITICAL, (
        f"{claim}: {covered}/{COVERAGE_SEEDS} covered ({ambiguous} ambiguous), fails at <= {COVERAGE_CRITICAL}"
    )


def _replica_ceiling(seed: int, spec: TargetSpec) -> float:
    """The replica ceiling (primary + 2 replicas on 2,000 gauss2 items); the threshold enters no bound."""
    x, y, post, rng = _gauss2(2_000, seed)
    rows = np.c_[y, draw_labels(post, 2, rng)]
    source = fc._Replicas(_call(y, x), list(rows), np.full(len(y), 3), np.arange(len(y)))
    return _value(source.bound(spec))


def _bayes_witness(seed: int) -> tuple[float, dict[str, Any]]:
    """The production witness path on the Bayes scorer (the posterior: class 0 when x1 > 0), 2,000 items audited."""
    x, y, post, _ = _gauss2(2_000, seed)
    bound = fc._Scores(_call(y, x), post[:0], post).bound(TargetSpec("IC", "accuracy", 0.5))
    assert bound[1] is not None
    return _value(bound), bound[1]


def _witness_margin(seed: int) -> float:
    return BAYES_ACCURACY - _bayes_witness(seed)[0]


def _point_estimate_margin(seed: int) -> float:
    return BAYES_ACCURACY - _bayes_witness(seed)[1]["mean"]  # no radius: covered about half the time


def _c3(
    seed: int, pop: Population = GAUSS2, axis: NDArray[np.float64] = GAUSS2.means[0], edges: NDArray = EDGES
) -> tuple[fc._Panels, NDArray[np.intp], dict[int, NDArray[np.intp]]]:
    """Disjoint panels of 20,000 units within fixed cells of their projection on ``axis`` (gauss2: x1 cells)."""
    rng = np.random.default_rng(seed)
    x, y, _ = pop.sample(20_000, rng)
    cells = np.searchsorted(edges, x @ axis).astype(np.intp)
    panels = _panels(cells, rng)
    n_by_cell = np.bincount(cells, minlength=len(edges) + 1)
    return fc._Panels(_call(y, x, n_classes=pop.n_classes), y, n_by_cell, panels, PANEL_SIZE, None, 0.0), y, panels


def _c3_margin(seed: int) -> float:
    return C3_TARGET - _c3(seed)[0].accuracy()[2]["bound"]


def _ap_ceiling_margin(seed: int) -> float:
    ap = TargetSpec("IC", "ap", 0.5, class_name="0", ap_functional="population_ap_integral")
    return _value(_c3(seed)[0].bound(ap)) - CELL_AP


def _c3_recall_margin(seed: int) -> float:
    return _value(_c3(seed)[0].bound(_recall(0.5))) - C3_RECALL_TARGET


@cache
def _rare5_ceiling(seed: int) -> tuple[float, dict[str, Any]]:
    """The local-panel recall ceiling of rare5's class 4 at RARE5_F0, with its raw results."""
    source = _c3(seed, RARE5, RARE5_AXIS, RARE5_EDGES)[0]
    bound = source.bound(_recall(0.5, RARE5_F0, str(RARE)))
    return _value(bound), bound[2]


def _lower_prevalence_sets(seed: int) -> bool:
    """Whether U_r is above what pi_upper alone gives: a cost with l_j < lam_j f0 set its term at pi_lower."""
    value, raw = _rare5_ceiling(seed)
    return value > fc._recall_upper(raw["risk_lower"], raw["costs"], RARE5_F0, raw["pi_upper"], raw["pi_upper"])


def _np_witness(seed: int) -> dict[str, Any]:
    """The production recall witness path (``_Scores.recall`` on class decisions) of the NP rule, 2,000 items."""
    x, y, _, _ = _gauss2(2_000, seed)
    decisions = np.eye(2)[_np_rule(x)]
    return fc._Scores(_call(y, x), decisions[:0], decisions, decisions=True).bound(_recall(0.5))[2]


def _recall_witness_margin(seed: int) -> float:
    """Joint: the recall lower endpoint under the rule's recall and the false-positive upper endpoint over F0."""
    raw = _np_witness(seed)
    return min(BAYES_NP_RECALL - raw["value"], raw["false_positive_rate"]["value"] - F0)


def _recall_point_estimate_margin(seed: int) -> float:
    return BAYES_NP_RECALL - _np_witness(seed)["stats"]["mean"]  # no radius: covered about half the time


VIDEO_SEQUENCES, VIDEO_MAX_LEN = 500, 8  # per seed; the declared max_sequence_length


@cache
def _video_truths() -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Monte Carlo truths, with their SEs, of the Bayes scorer's accuracy on ``make_sequences`` gauss2 video.

    Correctness enters through its conditional mean max_c p(c|x), so the only noise is the frames'. Returns the mean
    over sequences of the per-sequence mean (``sequence_mean``) and the pooled frame accuracy, a ratio of sums
    (``frame_pooled``; delta-method SE), over 400,000 sequences.
    """
    rng = np.random.default_rng(10**6)  # no test seed reuses it
    means, sums, lengths = [], [], []
    for _ in range(20):
        _, _, post, seq, length = make_sequences(GAUSS2, 20_000, max_len=VIDEO_MAX_LEN, rng=rng)
        total = np.bincount(seq, weights=post.max(axis=1))
        means.append(total / length)
        sums.append(total)
        lengths.append(length)
    mean, s, n = np.concatenate(means), np.concatenate(sums), np.concatenate(lengths).astype(np.float64)
    pooled = s.sum() / n.sum()
    pooled_se = float(np.std(s - pooled * n, ddof=1) / math.sqrt(len(n)) / n.mean())
    return (float(mean.mean()), float(mean.std(ddof=1) / math.sqrt(len(mean)))), (float(pooled), pooled_se)


def _video_witness(seed: int, aggregation: Any) -> float:
    """The production witness path (``_Scores.bound``) on the Bayes scorer over 500 sequences of 1 to 8 frames."""
    rng = np.random.default_rng(seed)
    x, y, post, seq, _ = make_sequences(GAUSS2, VIDEO_SEQUENCES, max_len=VIDEO_MAX_LEN, rng=rng)
    call = _call(y, x, groups=seq, max_len=VIDEO_MAX_LEN)
    return _value(fc._Scores(call, post[:0], post).bound(TargetSpec("IC", "accuracy", 0.5, aggregation=aggregation)))


def _ap_witness_margin(seed: int) -> float:
    x, y, post, _ = _gauss2(2_000, seed)
    fit, audit = np.arange(1_000), np.arange(1_000, 2_000)
    source = fc._Scores(_call(y, x, fit=fit, audit=audit), post[fit], post[audit])
    return AP_TRUTH - source.ap_class(None, 0, ALPHA)[0]


def _kink_minus_g_dfe(seed: int) -> NDArray[np.float64]:
    """Per class, the C3 AP ceiling with the kink grid minus the one with G_DFE, on the same gauss2 panels."""
    source, y, panels = _c3(seed)
    out = np.zeros(2)
    for c in range(2):
        risk = partial(source.risk, {h: (y[ix] == c).sum(axis=1) for h, ix in panels.items()})
        prevalence = (y == c).astype(np.float64)
        kink, g_dfe = (fc._ap_ceiling(g, risk, prevalence, 1, ALPHA)[0] for g in (kink_costs(PANEL_SIZE), G_DFE))
        out[c] = kink - g_dfe
    return out


def _determinations(seed: int) -> dict[float, str]:
    x, y, post, rng = _gauss2(600, seed)
    replicas = dict(enumerate(draw_labels(post[:300], 2, rng)))
    dataset: Any = MockICDataset(y.tolist(), {0: "a", 1: "b"})
    out = Feasibility(seed=seed).evaluate(
        dataset, target=[TargetSpec("IC", "accuracy", t) for t in TARGETS], embeddings=x, replicas=replicas
    )
    return dict(out.determinations.select("threshold", "determination").iter_rows())


def _recall_determinations(seed: int) -> list[str]:
    """
    Determinations of RECALL_TARGETS on 2,000 gauss2 items: a replica ceiling and two decision witnesses, the NP rule
    and the uncertified top class (recall Phi(1), false-positive rate Phi(-1) > F0), on one ledger.
    """
    x, y, post, rng = _gauss2(2_000, seed)
    call = _call(y, x, alpha=ALPHA / 6)  # 3 sources x 2 targets, each claim an equal share as allocate splits a call
    specs = [_recall(r0) for r0 in RECALL_TARGETS]
    evidence = fc._replica(call, specs, dict(enumerate(draw_labels(post, 2, rng))))
    for name, decisions in (("np_rule", _np_rule(x)), ("top_class", post.argmax(axis=1))):
        evidence[name] = fc._user(call, specs, decisions, None)[fc._USER]
    return [decide(spec, evidence, call.units.population).determination for spec in specs]


def test_replica_accuracy_ceiling_covers_its_consensus_target():
    accuracy = TargetSpec("IC", "accuracy", 0.5)
    _accept(
        _covered(lambda seed: _replica_ceiling(seed, accuracy) - REPLICA_TARGET),
        "replica accuracy ceiling >= 1 - E[p(1-p)]",
    )


def test_accuracy_witness_of_the_bayes_rule_covers_its_accuracy():
    _accept(_covered(_witness_margin), "accuracy witness <= Phi(1)")


def test_the_point_estimate_without_a_radius_fails_the_harness():
    covered, _ = _covered(_point_estimate_margin)
    assert covered <= COVERAGE_CRITICAL, f"negative control: {covered}/{COVERAGE_SEEDS} covered"


@pytest.mark.parametrize("aggregation", ["sequence_mean", "frame_pooled"])
def test_video_accuracy_witness_covers_its_monte_carlo_truth(aggregation):
    truth, se = _video_truths()[aggregation == "frame_pooled"]
    _accept(_covered(lambda seed: truth - _video_witness(seed, aggregation), se), f"{aggregation} witness <= truth")


def test_c3_bound_covers_its_consensus_target():
    _accept(_covered(_c3_margin), "C3 risk lower bound <= expected within-cell panel consensus")


def test_c3_ap_ceiling_covers_the_best_cell_scorer_ap():
    _accept(_covered(_ap_ceiling_margin), "C3 AP ceiling >= the AP of cells ranked by p_h(0)")


def test_ap_witness_staircase_covers_the_posterior_ap():
    _accept(_covered(_ap_witness_margin), "AP witness staircase <= population AP")


def test_recall_witness_of_the_np_rule_covers_its_recall_and_false_positive_rate():
    _accept(_covered(_recall_witness_margin), "recall witness <= Phi(1 - cut) and false-positive bound >= F0")


def test_the_recall_point_estimate_without_a_radius_fails_the_harness():
    covered, _ = _covered(_recall_point_estimate_margin)
    assert covered <= COVERAGE_CRITICAL, f"negative control: {covered}/{COVERAGE_SEEDS} covered"


def test_c3_recall_ceiling_covers_the_best_cell_policy_recall():
    assert C3_RECALL_TARGET >= CELL_NP_RECALL  # so covering the target covers the Neyman-Pearson cell recall
    _accept(_covered(_c3_recall_margin), "C3 recall ceiling >= U_r at the expected panel consensus")


def test_replica_recall_ceiling_covers_the_bayes_np_recall():
    assert REPLICA_RECALL_TARGET >= BAYES_NP_RECALL  # so covering the target covers the Bayes Neyman-Pearson recall
    recall = _recall(0.5)
    margin = _covered(lambda seed: _replica_ceiling(seed, recall) - REPLICA_RECALL_TARGET)
    _accept(margin, "replica recall ceiling >= U_r at the expected 3-label consensus")


def test_rare_class_recall_ceiling_covers_its_target_with_the_lower_prevalence_end():
    assert RARE5_TARGET >= RARE5_NP_RECALL
    _accept(_covered(lambda seed: _rare5_ceiling(seed)[0] - RARE5_TARGET), "rare5 class 4 recall ceiling >= target")
    lower = sum(_lower_prevalence_sets(seed) for seed in range(COVERAGE_SEEDS))
    assert lower > 0, "no seed's U_r was set at pi_lower: the claim does not exercise the lower prevalence end"


def test_kink_grid_is_no_looser_than_g_dfe():
    diff = np.array([_kink_minus_g_dfe(seed) for seed in range(50)])
    mean, se = diff.mean(axis=0), diff.std(axis=0, ddof=1) / math.sqrt(len(diff))
    assert (mean <= 3 * se).all(), f"kink - G_DFE AP ceiling per class: mean {mean}, SE {se}"


def test_determination_error_rates_on_gauss2():
    """
    False feasible on 0.95 and false infeasible on 0.70 each pass the critical-count rule for a rate <= 0.05.

    Observed: 0/346 false feasible and 0/346 false infeasible. Informativeness (not asserted): 0.95 was
    ``infeasible_under_recorded_assumptions`` in 345/346 seeds (1 ``not_ruled_out``) and 0.70
    ``demonstrated_feasible`` in 324/346 (22 ``not_ruled_out``).
    """
    runs = [_determinations(seed) for seed in range(COVERAGE_SEEDS)]
    hard, easy = [run[TARGETS[0]] for run in runs], [run[TARGETS[1]] for run in runs]
    _accept((COVERAGE_SEEDS - hard.count("demonstrated_feasible"), 0), "accuracy 0.95 not demonstrated_feasible")
    _accept((COVERAGE_SEEDS - easy.count("infeasible_under_recorded_assumptions"), 0), "accuracy 0.70 not infeasible")


def test_recall_determination_error_rates_on_gauss2():
    """False feasible just above the Bayes NP recall at F0 and false infeasible just below it each pass the rule."""
    above, below = zip(*(_recall_determinations(seed) for seed in range(COVERAGE_SEEDS)), strict=True)
    _accept((COVERAGE_SEEDS - above.count("demonstrated_feasible"), 0), "recall above the NP recall not feasible")
    _accept((COVERAGE_SEEDS - below.count("infeasible_under_recorded_assumptions"), 0), "recall below not infeasible")
