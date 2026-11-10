"""
Pin Feasibility's bounds and determinations: golden values captured from the code before the typed rewrite.

Each scenario runs ``gather_evidence`` and ``decide`` at a fixed seed on a synthetic population and is normalized to
records that do not depend on the internal types (``feasibility_pin.json``). A change that moves a bound or a
determination fails here. ``_run`` and ``_observed`` are the only parts that follow the internals.

Rows that depend only on the scores and labels (``user_model``, ``replicas``) compare at ``rel=1e-9``. The rows fitted
with scikit-learn (the local-panel partition's KMeans, the built-in k-NN and probe) and every determination compare at
``rel=1e-6`` and skip when numpy or scikit-learn moves a minor version from the one that wrote the golden values.
"""

import functools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sklearn
from numpy.typing import NDArray

from dataeval.performance._feasibility_classification import count_claims, gather_evidence
from dataeval.performance._feasibility_decide import decide
from dataeval.performance._feasibility_evidence import allocate
from dataeval.performance._feasibility_target import Aggregation, Allowances, Scope, TargetSpec
from dataeval.performance._feasibility_units import Units, split_units
from tests.performance.feasibility_helpers import draw_labels, make_population, make_sequences, make_units

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

GOLDEN = Path(__file__).with_name("feasibility_pin.json")
ALPHA = 0.05
SOURCES = ("local_panels", "replicas", "knn", "linear_probe", "user_model")  # the bounds row order
FITTED = {"local_panels", "knn", "linear_probe"}


@dataclass
class Scenario:
    specs: list[TargetSpec]
    units: Units
    features: NDArray[Any]
    replicas: dict[int, Any] | None = None
    predictions: Any = None
    allowances: tuple[float, float | None] | None = None  # (beta, gamma)
    max_sequence_length: int | None = None
    builtin: bool = True
    n_cells: int | None = None
    panel_size: int | None = None
    seed: int = 0


def _acc(threshold: float = 0.8, aggregation: Aggregation | None = None, scope: Scope = "full_input") -> TargetSpec:
    return TargetSpec(task="IC", metric="accuracy", threshold=threshold, aggregation=aggregation, scope=scope)


def _ap(cls: str | None = None, aggregation: Aggregation | None = None) -> TargetSpec:
    return TargetSpec(
        task="IC",
        metric="ap",
        threshold=0.5,
        class_name=cls,
        ap_functional="population_ap_integral",
        aggregation=aggregation,
    )


def _recall(cls: str, f0: float, threshold: float = 0.9) -> TargetSpec:
    return TargetSpec(task="IC", metric="recall", threshold=threshold, class_name=cls, max_false_positive_rate=f0)


def _gauss2(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x, y, post = make_population("gauss2").sample(n, rng)
    return x, y, post, rng


def _video(seed: int):
    rng = np.random.default_rng(seed)
    x, y, post, seq, _ = make_sequences(make_population("gauss2"), 600, max_len=8, rng=rng)
    return x, y, post, seq, rng


def _acc_ap_scores() -> Scenario:
    x, y, post, _ = _gauss2(2000, 1)
    units = make_units(y, class_names=["a", "b"])
    return Scenario([_acc(0.8), _acc(0.95), _ap("b")], units, x, predictions=post)


def _acc_beta() -> Scenario:
    x, y, post, _ = _gauss2(2000, 1)
    units = make_units(y, class_names=["a", "b"])
    return Scenario([_acc(0.8), _ap("b")], units, x, predictions=post, allowances=(0.01, None))


def _acc_beta_gamma() -> Scenario:
    x, y, post, _ = _gauss2(2000, 1)
    units = make_units(y, class_names=["a", "b"])
    return Scenario([_acc(0.8), _acc(0.95), _ap("b")], units, x, predictions=post, allowances=(0.01, 0.02))


def _scopes() -> Scenario:
    x, y, post, _ = _gauss2(2000, 1)
    units = make_units(y, class_names=["a", "b"])
    specs = [
        _acc(0.8, scope="fixed_representation"),
        _acc(0.95, scope="fixed_representation"),
        _acc(0.8, scope="partition_restricted"),
    ]
    return Scenario(specs, units, x, predictions=post, allowances=(0.01, None))


def _video_sequence_mean() -> Scenario:
    x, y, post, seq, _ = _video(3)
    return Scenario([_acc(0.8, "sequence_mean")], make_units(y, groups=seq, n_classes=2), x, predictions=post)


def _video_beta_gamma() -> Scenario:
    x, y, post, seq, _ = _video(7)
    units = make_units(y, groups=seq, n_classes=2)
    return Scenario([_acc(0.8, "sequence_mean")], units, x, predictions=post, allowances=(0.01, 0.02))


def _video_frame_pooled() -> Scenario:
    x, y, post, seq, rng = _video(4)
    units = make_units(y, groups=seq, n_classes=2, class_names=["a", "b"])
    specs = [_acc(0.8, "frame_pooled"), _ap("b", "frame_pooled")]
    replicas = dict(enumerate(draw_labels(post, 2, rng)))
    return Scenario(specs, units, x, replicas=replicas, predictions=post, max_sequence_length=8)


def _rare5() -> tuple[Any, ...]:
    rng = np.random.default_rng(2)
    x, y, post = make_population("rare5").sample(4000, rng)
    return x, y, post, make_units(y, n_classes=5, class_names=["a", "b", "c", "d", "e"])


def _ap_macro_class_scores() -> Scenario:
    x, _, post, units = _rare5()
    return Scenario([_ap(), _ap("c")], units, x, predictions=post)


def _class_decisions() -> Scenario:
    x, y, post, _ = _gauss2(2000, 5)
    units = make_units(y, class_names=["a", "b"])
    return Scenario([_acc(0.8), _ap("a")], units, x, predictions=post.argmax(axis=1))


def _recall_scores() -> Scenario:
    x, _, post, units = _rare5()
    specs = [_recall("a", 0.5, 0.5), _recall("a", F0_CROSSING, 0.5), _recall("a", 0.001)]
    return Scenario(specs, units, x, predictions=post)


def _replicas_unequal() -> Scenario:
    x, y, post, rng = _gauss2(2000, 6)
    units = make_units(y, class_names=["a", "b"])
    replicas = {i: draw_labels(post[i : i + 1], 2 + i % 2, rng)[0] for i in range(500)}  # panels of 3 and 4
    specs = [_acc(0.8), _ap("a"), _recall("a", 0.3, 0.6)]
    return Scenario(specs, units, x, replicas=replicas, builtin=False)


def _replicas_equal() -> Scenario:
    x, y, post, rng = _gauss2(2000, 6)
    units = make_units(y, class_names=["a", "b"])
    replicas = {i: draw_labels(post[i : i + 1], 2, rng)[0] for i in range(500)}
    return Scenario([_acc(0.8), _acc(0.95), _ap("b")], units, x, replicas=replicas, builtin=False)


def _conflict() -> Scenario:
    """A witness that copies the labels (accuracy about 1) against a full_input replica envelope near Bayes."""
    x, y, post, rng = _gauss2(2000, 8)
    replicas = {i: draw_labels(post[i : i + 1], 2, rng)[0] for i in range(500)}
    return Scenario([_acc(0.8)], make_units(y, n_classes=2), x, replicas=replicas, predictions=y, builtin=False)


def _explicit_knobs() -> Scenario:
    x, y, post, _ = _gauss2(2000, 9)
    units = make_units(y, class_names=["a", "b"])
    return Scenario([_acc(0.8), _ap("a")], units, x, predictions=post, builtin=False, n_cells=7, panel_size=3)


F0_CROSSING = 0.06  # a false-positive limit between the witness's observed rate (0.054) and its upper bound (0.077)
SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "acc_ap_scores": _acc_ap_scores,
    "acc_beta": _acc_beta,
    "acc_beta_gamma": _acc_beta_gamma,
    "scopes": _scopes,
    "video_sequence_mean": _video_sequence_mean,
    "video_beta_gamma": _video_beta_gamma,
    "video_frame_pooled": _video_frame_pooled,
    "ap_macro_class_scores": _ap_macro_class_scores,
    "class_decisions": _class_decisions,
    "recall_scores": _recall_scores,
    "replicas_unequal": _replicas_unequal,
    "replicas_equal": _replicas_equal,
    "conflict": _conflict,
    "explicit_knobs": _explicit_knobs,
}


def _run(sc: Scenario):
    """Run one scenario through ``gather_evidence`` and ``decide``."""
    fit, audit = split_units(sc.units, fit_fraction=0.5, seed=sc.seed)
    n_claims = count_claims(
        sc.specs,
        has_replicas=bool(sc.replicas),
        has_predictions=sc.predictions is not None,
        builtin_witnesses=sc.builtin,
    )
    allowances = None if sc.allowances is None else Allowances(*sc.allowances)
    bounds = gather_evidence(
        sc.specs,
        units=sc.units,
        features=sc.features,
        audit=audit,
        fit=fit,
        replicas=sc.replicas,
        predictions=sc.predictions,
        allowances=allowances,
        alpha_claim=allocate(ALPHA, n_claims),
        seed=sc.seed,
        max_sequence_length=sc.max_sequence_length,
        n_cells=sc.n_cells,
        panel_size=sc.panel_size,
        builtin_witnesses=sc.builtin,
    )
    return bounds, [decide(spec, i, bounds) for i, spec in enumerate(sc.specs)]


def _observed(sc: Scenario) -> dict[str, Any]:
    """Normalize a run to the golden records: bounds and determinations, free of the internal types."""
    bounds, determinations = _run(sc)
    return {
        "bounds": sorted(
            (
                {
                    "requirement": b.requirement,
                    "source": b.source,
                    "scope": b.information_scope,
                    "quantity": b.quantity,
                    "bound_type": b.bound_type,
                    "state": b.state,
                    "value": b.bound,
                    "alpha": b.alpha,
                    # a bound with no closed form (AP, a recall envelope, frame_pooled) records no method and no stats
                    "stats": None
                    if b.method is None
                    else {
                        "mean": b.mean,
                        "variance": b.variance,
                        "n": b.n,
                        "support": b.support,
                        "delta": b.delta,
                        "method": b.method,
                    },
                }
                for b in bounds
            ),
            key=lambda b: (b["requirement"], SOURCES.index(b["source"]), b["quantity"] == "false_positive_rate"),
        ),
        "determinations": [
            {
                "requirement": i,
                # the golden values keep the old name for a suspended conclusion
                "determination": "evidence_conflict" if d.limitation == "conflict" else d.determination,
                "envelope": d.envelope,
                "envelope_scope": d.envelope_scope,
                "witness": d.witness,
                "witness_source": d.witness_source,
                "suggested_minimum": d.suggested_minimum,
            }
            for i, d in enumerate(determinations)
        ],
    }


@functools.cache
def _observed_scenario(name: str) -> dict[str, Any]:
    return _observed(SCENARIOS[name]())


def record() -> dict[str, Any]:
    """Return the golden document for the current code; ``make_pin.py`` writes it."""
    import polars as pl

    return {
        "generated_from": None,
        "versions": {
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "polars": pl.__version__,
        },
        "scenarios": {name: _observed_scenario(name) for name in SCENARIOS},
    }


def _minor(version: str) -> tuple[str, ...]:
    return tuple(version.split(".")[:2])


def _assert_same(got: Any, want: Any, rel: float, where: str) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict), where
        assert got.keys() == want.keys(), where
        for key in want:
            _assert_same(got[key], want[key], rel, f"{where}.{key}")
    elif isinstance(want, list):
        assert isinstance(got, list), where
        assert len(got) == len(want), where
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _assert_same(g, w, rel, f"{where}[{i}]")
    elif isinstance(want, float):
        assert got == pytest.approx(want, rel=rel, abs=1e-12), where
    else:
        assert got == want, where


def _key(bound: dict[str, Any]) -> tuple[int, str, str]:
    return bound["requirement"], bound["source"], bound["quantity"]


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("name", SCENARIOS)
def test_pin_rows_exist_and_independent_rows_match(name, golden):
    want, got = golden["scenarios"][name]["bounds"], _observed_scenario(name)["bounds"]
    assert [_key(b) for b in got] == [_key(b) for b in want]  # the same rows, in the same order
    for g, w in zip(got, want, strict=True):
        if w["source"] not in FITTED:
            _assert_same(g, w, 1e-9, f"{name}{_key(w)}")


@pytest.mark.parametrize("name", SCENARIOS)
def test_pin_fitted_stages_and_determinations_match(name, golden):
    versions = golden["versions"]
    if (_minor(np.__version__), _minor(sklearn.__version__)) != (
        _minor(versions["numpy"]),
        _minor(versions["scikit-learn"]),
    ):
        pytest.skip("KMeans, k-NN and the probe can differ across numpy and scikit-learn minor versions")
    want, got = golden["scenarios"][name], _observed_scenario(name)
    for g, w in zip(got["bounds"], want["bounds"], strict=True):
        if w["source"] in FITTED:
            _assert_same(g, w, 1e-6, f"{name}{_key(w)}")
    _assert_same(got["determinations"], want["determinations"], 1e-6, f"{name}.determinations")
