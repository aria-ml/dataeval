"""Tests for the Feasibility evaluator end to end: what it decides, what it refuses and how it warns."""

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

import dataeval.performance._feasibility as feasibility_module
from dataeval._experimental import experimental, quiet_experimental
from dataeval.data import Limit, View
from dataeval.exceptions import ExperimentalWarning, ShapeMismatchError
from dataeval.extractors import FlattenExtractor
from dataeval.performance import Allowances, Feasibility, FeasibilityOutput, TargetSpec
from dataeval.performance._feasibility_constants import FIT_FRACTION
from dataeval.performance._feasibility_units import resolve_units, split_units
from tests.data.conftest import MockICDataset, MockODDataset
from tests.performance.feasibility_helpers import draw_labels, make_population, targets

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

AB = {0: "a", 1: "b"}
TARGET = {"accuracy": 0.6, "accuracy/fixed_representation": 0.97, "ap:a": 0.5}
P40 = np.full((40, 2), 0.5)
# A fresh interpreter, so no earlier call has used up a function's one warning: only the evaluator's own may remain.
_ONE_CALL = """
import warnings
import numpy as np
from dataeval.exceptions import ExperimentalWarning
from dataeval.performance import Feasibility
from tests.data.conftest import MockICDataset
from tests.performance.feasibility_helpers import targets
y = [i % 2 for i in range(200)]
x = np.random.default_rng(0).normal(size=(200, 4)) + np.array(y)[:, None]
target = targets({"accuracy": 0.6, "ap:a": 0.5})  # outside the recorded call: a caller's TargetSpec warns once
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Feasibility().evaluate(
        MockICDataset(y, {0: "a", 1: "b"}), target=target, embeddings=x,
        replicas={0: [0], 1: [1]}, predictions=np.eye(2)[y],
    )
print(sorted({str(w.message).split("'")[1] for w in caught if issubclass(w.category, ExperimentalWarning)}))
"""


class _Images(MockICDataset):
    """A classification dataset whose images are the given feature vectors."""

    def __init__(self, x, labels):
        super().__init__([int(v) for v in labels], AB)
        self._x = np.asarray(x, dtype=np.float32)

    def __getitem__(self, index: int):
        _, target, meta = super().__getitem__(index)
        return self._x[index], target, meta


def _ds(labels, index2label=AB) -> Any:
    return MockICDataset([int(v) for v in labels], index2label)


@contextmanager
def _without_builtin_witnesses():
    """Run the evaluator with the built-in witnesses off, as the removed constructor knob did.

    ``gather_evidence`` keeps ``builtin_witnesses`` as a private parameter, so the call is routed through it here.
    """
    gather, count = feasibility_module.gather_evidence, feasibility_module.count_claims
    with (
        patch.object(feasibility_module, "gather_evidence", lambda *a, **kw: gather(*a, **kw, builtin_witnesses=False)),
        patch.object(
            feasibility_module, "count_claims", lambda *a, **kw: count(*a, **{**kw, "builtin_witnesses": False})
        ),
    ):
        yield


ABT = {0: "a", 1: "b", 2: "Tank"}  # the target class's index (2) is not 1
with quiet_experimental():  # at import, outside the module's warning filter
    TANK_R80_F10 = TargetSpec(task="IC", metric="recall", threshold=0.8, class_name="Tank", max_false_positive_rate=0.1)


def _three_classes(n: int):
    """Labels cycling over ABT and features that separate them."""
    y = np.arange(n) % 3
    return y, np.eye(3)[y] + np.random.default_rng(0).normal(scale=0.1, size=(n, 3))


def _gauss2(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x, y, post = make_population("gauss2").sample(n, rng)
    return x, y, post, rng


@pytest.fixture(scope="module")
def gauss2_run():
    x, y, post, rng = _gauss2(2000)
    replicas = dict(enumerate(draw_labels(post, 4, rng)))
    return Feasibility(seed=3).evaluate(_ds(y), target=targets(TARGET), embeddings=x, replicas=replicas)


def _user_bounds(out: FeasibilityOutput) -> pl.DataFrame:
    return out.bounds.filter(pl.col("source") == "user_model")


@pytest.mark.required
class TestFeasibility:
    def test_end_to_end_gauss2_demonstrates_an_easy_target(self, gauss2_run):
        easy = gauss2_run.determinations.row(0, named=True)
        assert easy["determination"] == "demonstrated_feasible"
        assert easy["witness"] >= 0.6
        assert easy["witness_source"] in ("knn", "linear_probe")
        assert easy["limitation"] is None
        # Bayes accuracy is Phi(1) ~ 0.841: the replica envelope rules out 0.97 at fixed_representation.
        hard = gauss2_run.determinations.row(1, named=True)
        assert hard["determination"] == "infeasible_under_recorded_assumptions"
        assert hard["envelope_scope"] == "full_input"
        assert 0.841 <= hard["envelope"] < 0.97

    def test_the_minimal_call_is_a_dataset_predictions_and_a_requirement(self):
        _, y, post, _ = _gauss2(1000)
        out = Feasibility().evaluate(_ds(y), predictions=post, target=[TargetSpec(metric="accuracy", threshold=0.6)])
        assert out.determinations["determination"].to_list() == ["demonstrated_feasible"]

    def test_a_witness_only_call_has_no_envelope(self):
        _, y, post, _ = _gauss2(1000)
        target = targets({"accuracy": 0.6, "accuracy/fixed_representation": 0.99})
        out = Feasibility(seed=1).evaluate(_ds(y), target=target, predictions=post)
        assert set(out.bounds["source"]) == {"user_model"}
        pairs = list(zip(out.determinations["determination"], out.determinations["limitation"], strict=True))
        # only an envelope can bound what is unreachable: a witness alone demonstrates or leaves it unresolved
        assert pairs == [("demonstrated_feasible", None), ("unresolved", "no_envelope")]

    def test_a_witness_only_call_spends_alpha_on_the_witness_alone(self):
        _, y, post, _ = _gauss2(1000)
        out = Feasibility(seed=1).evaluate(_ds(y), target=targets({"accuracy": 0.6, "ap:a": 0.5}), predictions=post)
        assert out.bounds["alpha"].to_list() == pytest.approx([0.05 / 2] * 2)  # two requirements, one source

    def test_replicas_alone_give_an_envelope_and_no_witness(self):
        _, y, post, rng = _gauss2(1000)
        replicas = dict(enumerate(draw_labels(post[:500], 3, rng)))
        target = targets({"accuracy": 0.6, "accuracy/fixed_representation": 0.99})
        out = Feasibility(seed=1).evaluate(_ds(y), target=target, replicas=replicas)
        assert set(out.bounds["source"]) == {"replicas"}
        assert out.determinations["witness"].null_count() == 2
        # an envelope can rule a requirement out or leave it open, and never demonstrate it
        assert out.determinations["determination"].to_list() == [
            "not_ruled_out",
            "infeasible_under_recorded_assumptions",
        ]

    def test_no_source_raises_and_names_the_inputs(self):
        _, y, _, _ = _gauss2(100)
        with pytest.raises(ValueError, match="predictions, a model, replicas, embeddings or an extractor"):
            Feasibility().evaluate(_ds(y), target=targets({"accuracy": 0.6}))

    def test_tiny_audit_is_unresolved_not_an_error(self):
        x, y, _, _ = _gauss2(12)
        out = Feasibility().evaluate(_ds(y), target=targets({"accuracy": 0.9}), embeddings=x)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("unresolved", "envelope_unavailable")
        local = out.bounds.filter(pl.col("source") == "local_panels")
        assert local["state"].to_list() == ["unavailable"]
        assert local["reason"].to_list() == ["fit_split_smaller_than_cells"]

    def test_od_dataset_is_rejected_clearly(self):
        dataset: Any = MockODDataset([[0, 1], [1]], AB)
        msg = "Feasibility supports image and video classification in this release; OD is planned."
        target = [TargetSpec(task="OD", metric="ap", threshold=0.5, iou=0.5, ap_functional="population_ap_integral")]
        with pytest.raises(NotImplementedError, match=msg):
            Feasibility().evaluate(dataset, target=target, embeddings=np.zeros((2, 2)))

    def test_recall_with_no_audit_positives_is_unresolved(self):
        y, x = _three_classes(300)
        _, audit = split_units(resolve_units(_ds(y, ABT), groups=None), fit_fraction=FIT_FRACTION, seed=0)
        y[audit] = np.where(y[audit] == 2, 0, y[audit])  # Tank only in the fit split
        out = Feasibility(seed=0).evaluate(_ds(y, ABT), target=[TANK_R80_F10], embeddings=x, predictions=y)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["witness"]) == ("unresolved", None)
        recall = _user_bounds(out).filter(pl.col("quantity") == "recall").row(0, named=True)
        assert (recall["state"], recall["reason"]) == ("unavailable", "no_positives")
        assert "recall:Tank ≥0.8 fp≤0.1" in str(out)

    def test_a_recall_requirement_gets_a_full_input_envelope_from_repeated_labels(self):
        y, x = _three_classes(600)
        i = np.arange(600)  # two replicas per image, each disagreeing with the primary label on every other image
        replicas = dict(enumerate(np.c_[np.where(i % 2, y, (y + 1) % 3), np.where(i // 2 % 2, y, (y + 2) % 3)]))
        with _without_builtin_witnesses():
            out = Feasibility(seed=0).evaluate(_ds(y, ABT), target=[TANK_R80_F10], embeddings=x, replicas=replicas)
        (row,) = out.bounds.filter(pl.col("source") == "replicas").iter_rows(named=True)
        assert (row["information_scope"], row["bound_type"], row["state"]) == ("full_input", "upper", "informative")
        assert 0 <= row["bound"] < TANK_R80_F10.threshold <= 1
        assert out.determinations["determination"][0] == "infeasible_under_recorded_assumptions"
        assert out.bounds["alpha"].sum() <= 0.05 + 1e-12

    def test_class_decisions_drive_a_recall_witness_only_when_certified(self):
        y, x = _three_classes(600)
        decisions = np.where(np.arange(600) % 20 == 0, (y + 1) % 3, y)  # 95% agreement
        run = Feasibility(seed=0)
        with _without_builtin_witnesses():
            out = run.evaluate(_ds(y, ABT), target=[TANK_R80_F10], embeddings=x, predictions=decisions)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["witness_source"]) == ("demonstrated_feasible", "user_model")
        always_tank = np.full(600, 2)  # recall 1 and false-positive rate 1
        with _without_builtin_witnesses():
            out = run.evaluate(_ds(y, ABT), target=[TANK_R80_F10], embeddings=x, predictions=always_tank)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["witness_source"]) == ("unresolved", None)
        recall, false_positives = _user_bounds(out).iter_rows(named=True)
        assert (recall["bound"] or 0) > 0.8  # the recall bound clears the requirement
        assert (false_positives["bound"] or 0) > 0.1  # the false-positive bound does not

    def test_class_decisions_are_a_user_witness(self):
        y, x = _three_classes(300)
        decisions = np.where(np.arange(300) % 10 == 0, (y + 1) % 3, y)  # 90% agreement, integer class indices
        with _without_builtin_witnesses():
            out = Feasibility(seed=0).evaluate(
                _ds(y, ABT), target=targets({"accuracy": 0.6}), embeddings=x, predictions=decisions
            )
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["witness_source"]) == ("demonstrated_feasible", "user_model")

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            (np.array([True, False] * 50), "pass integer class indices"),  # as integers: classes 0 and 1
            (np.zeros(100), "pass integer class indices"),
            (np.full(100, 7), "outside"),
            (np.full(100, -1), "outside"),
            (np.zeros(99, dtype=int), "one per dataset item"),
        ],
    )
    def test_decision_arrays_are_validated(self, bad, match):
        y, _ = _three_classes(100)
        with pytest.raises(ValueError, match=match):  # before the embedding pass, which would fail on no extractor
            Feasibility(seed=0).evaluate(_ds(y, ABT), target=targets({"accuracy": 0.6}), predictions=bad)

    def test_ap_with_class_decisions_only_is_unavailable(self):
        y, x = _three_classes(300)
        with _without_builtin_witnesses():
            out = Feasibility(seed=0).evaluate(
                _ds(y, ABT), target=targets({"ap:Tank": 0.5}), embeddings=x, predictions=y
            )
        (b,) = _user_bounds(out).iter_rows(named=True)
        assert (b["state"], b["reason"]) == ("unavailable", "needs_scores")

    def test_a_subset_view_takes_plain_targets(self):
        x, y, _, _ = _gauss2(200)
        view: Any = View(_ds(y), Limit(size=120))  # straight over a plain dataset: grouped by source index
        out = Feasibility().evaluate(view, target=targets({"accuracy": 0.6}), embeddings=x[:120])
        assert out.determinations["determination"][0] == "demonstrated_feasible"
        assert set(out.bounds["unit"]) == {"image"}

    def test_the_extractor_warns_outside_the_quiet_scope(self):
        @experimental
        def flatten(batch):
            return FlattenExtractor()(batch)

        x, y, _, _ = _gauss2(200)
        dataset: Any = _Images(x, y)
        with pytest.warns(ExperimentalWarning, match="flatten"):
            Feasibility(extractor=flatten).evaluate(dataset, target=targets({"accuracy": 0.6}))

    def test_extractor_matches_given_embeddings(self):
        x, y, _, _ = _gauss2(200)
        x = x.astype(np.float32)  # what the extractor sees, so both paths read identical features
        dataset: Any = _Images(x, y)
        given = Feasibility().evaluate(dataset, target=targets({"accuracy": 0.6}), embeddings=x)
        extracted = Feasibility(extractor=FlattenExtractor()).evaluate(dataset, target=targets({"accuracy": 0.6}))
        assert extracted.determinations.equals(given.determinations)
        assert extracted.bounds.equals(given.bounds)

    @pytest.mark.parametrize("cls", [Feasibility, FeasibilityOutput, Allowances, TargetSpec])
    def test_experimental_warning(self, cls):
        assert cls.__name__ in getattr(cls, "__experimental__", "")

    def test_a_call_warns_for_the_evaluator_only(self):
        root = Path(__file__).parents[2]
        out = subprocess.run([sys.executable, "-c", _ONE_CALL], cwd=root, capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "['Feasibility']"

    def test_plot_type_identifies_the_output_for_plotting(self, gauss2_run):
        """`plot_type` is the discriminator dataeval-plots dispatches on."""
        assert gauss2_run.plot_type == "feasibility"

    def test_repr_is_not_the_str(self, gauss2_run):
        assert repr(gauss2_run).startswith("FeasibilityOutput(requirements=")
        assert "DataFrame(shape=(3, 18))" in repr(gauss2_run)
        assert str(gauss2_run) != repr(gauss2_run)

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"embeddings": None}, ValueError, "extractor"),
            ({"embeddings": np.zeros((5, 2))}, ShapeMismatchError, "embeddings"),
            ({"allowances": {"beta": 0.1}}, ValueError, "Allowances"),
            # checked before the embedding pass, which would fail first on the missing extractor
            ({"embeddings": None, "predictions": np.zeros((3, 2))}, ValueError, "predictions"),
            ({"embeddings": None, "predictions": P40 * np.nan}, ValueError, "finite"),
            ({"embeddings": None, "replicas": {999: [0]}}, ValueError, "not a datum id"),
        ],
    )
    def test_arguments_are_checked_at_the_boundary(self, kwargs, error, match):
        x, y, _, _ = _gauss2(40)
        with pytest.raises(error, match=match):
            Feasibility().evaluate(_ds(y), target=targets({"accuracy": 0.6}), **({"embeddings": x} | kwargs))

    def test_class_names_are_checked_before_the_embedding_pass(self):
        x, y, _, _ = _gauss2(40)
        undeclared = _ds(y)
        undeclared.metadata.pop("index2label")
        for name in ("cat", "7"):  # no index2label: only an index below n_classes names a class
            with pytest.raises(ValueError, match=f"Cannot map class '{name}'"):
                Feasibility().evaluate(undeclared, target=targets({f"ap:{name}": 0.6}))

    def test_one_class_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2 classes"):
            Feasibility().evaluate(
                _ds([0] * 10, {0: "a"}), target=targets({"accuracy": 0.6}), embeddings=np.zeros((10, 2))
            )
