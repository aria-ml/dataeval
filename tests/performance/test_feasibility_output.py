"""Tests for FeasibilityOutput: the frozen ``determinations`` and ``bounds`` schemas, their contents and the render."""

import pickle
from dataclasses import replace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from dataeval.config import set_seed
from dataeval.performance import Allowances, Feasibility, FeasibilityOutput, TargetSpec
from tests.data.conftest import MockICDataset
from tests.performance.feasibility_helpers import draw_labels, make_population

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

AB = {0: "a", 1: "b"}
_SCOPE = pl.Enum(["full_input", "fixed_representation", "partition_restricted"])
_SOURCE = pl.Enum(["local_panels", "replicas", "knn", "linear_probe", "user_model"])
_QUANTITY = pl.Enum(["accuracy", "ap", "recall", "recall_at_fp", "mota", "false_positive_rate"])
_ASSUMPTION = pl.Enum([
    "representation_independent_of_audit_labels",
    "allowance_beta",
    "allowance_gamma",
    "model_independent_of_audit_items",
    "class_order_matches_dataset",
    "replicas_random_sample",
    "replicas_one_protocol",
    "replicas_same_video_context",
    "replicated_sequences_random_sample",
    "replicated_frames_uniform_sample",
])
DETERMINATIONS_SCHEMA: dict[str, Any] = {
    "requirement": pl.Int64,
    "task": pl.Enum(["IC", "OD", "MOT"]),
    "metric": pl.Enum(["accuracy", "ap", "recall", "recall_at_fp", "mota"]),
    "threshold": pl.Float64,
    "judged_scope": _SCOPE,
    "class_name": pl.String,
    "aggregation": pl.Enum(["frame_pooled", "sequence_mean"]),
    "ap_functional": pl.Enum(["population_ap_integral", "population_ap_101"]),
    "max_false_positive_rate": pl.Float64,
    "determination": pl.Enum([
        "demonstrated_feasible",
        "infeasible_under_recorded_assumptions",
        "not_ruled_out",
        "unresolved",
    ]),
    "limitation": pl.Enum([
        "conflict",
        "no_envelope",
        "scope_restricted",
        "missing_bound",
        "envelope_unavailable",
        "no_eligible_witness",
        "within_radius",
        "weak_witness",
    ]),
    "suggested_minimum": pl.Int64,
    "minimum_source": _SOURCE,
    "minimum_quantity": _QUANTITY,
    "envelope": pl.Float64,
    "envelope_scope": _SCOPE,
    "witness": pl.Float64,
    "witness_source": _SOURCE,
}
BOUNDS_SCHEMA: dict[str, Any] = {
    "requirement": pl.Int64,
    "source": _SOURCE,
    "role": pl.Enum(["envelope", "witness"]),
    "quantity": _QUANTITY,
    "bound_type": pl.Enum(["lower", "upper"]),
    "information_scope": _SCOPE,
    "population": pl.Enum(["items", "labeled_items"]),
    "sample": pl.Enum(["audit_split", "audit_split_one_per_group", "replica_items"]),
    "unit": pl.Enum(["image", "sequence"]),
    "state": pl.Enum(["unavailable", "uninformative", "informative"]),
    "reason": pl.Enum([
        "too_few_units",
        "max_sequence_length_missing",
        "no_positives_possible",
        "no_positives",
        "no_negatives",
        "needs_scores",
        "fit_split_smaller_than_cells",
        "no_complete_panel",
        "witness_not_fitted",
        "bound_at_maximum",
        "bound_at_minimum",
    ]),
    "bound": pl.Float64,
    "observed": pl.Float64,
    "n": pl.Int64,
    "variance": pl.Float64,
    "radius": pl.Float64,
    "method": pl.Enum(["hoeffding", "hybrid", "exact_binomial"]),
    "alpha": pl.Float64,
    "assumptions": pl.List(_ASSUMPTION),
    "is_determining": pl.Boolean,
}


def _ds(labels, names: dict[int, str] = AB) -> Any:
    return MockICDataset([int(v) for v in labels], names)


def _gauss2(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x, y, post = make_population("gauss2").sample(n, rng)
    return x, y, post, rng


def _specs() -> list[TargetSpec]:
    """Accuracy at two scopes, one class's AP and one class's recall: every kind of requirement in one call."""
    return [
        TargetSpec(metric="accuracy", threshold=0.6),
        TargetSpec(metric="accuracy", threshold=0.97, scope="fixed_representation"),
        TargetSpec(metric="ap", threshold=0.5, class_name="a"),
        TargetSpec(metric="recall", threshold=0.6, class_name="a", max_false_positive_rate=0.3),
    ]


@pytest.fixture(scope="module")
def run() -> FeasibilityOutput:
    x, y, post, rng = _gauss2(1000)
    replicas = dict(enumerate(draw_labels(post[:400], 3, rng)))
    return Feasibility(seed=3).evaluate(
        _ds(y),
        target=_specs(),
        embeddings=x,
        replicas=replicas,
        predictions=post,
        allowances=Allowances(beta=0.1, gamma=0.05),
    )


@pytest.mark.required
class TestSchemas:
    def test_determinations_schema_is_frozen(self, run):
        assert list(run.determinations.schema.items()) == list(DETERMINATIONS_SCHEMA.items())

    def test_bounds_schema_is_frozen(self, run):
        assert list(run.bounds.schema.items()) == list(BOUNDS_SCHEMA.items())

    def test_no_string_columns_except_class_name(self, run):
        strings = {
            (name, frame)
            for frame in ("determinations", "bounds")
            for name, dtype in getattr(run, frame).schema.items()
            if dtype == pl.String
        }
        assert strings == {("class_name", "determinations")}  # a name from the dataset's own labels

    def test_the_frames_stay_typed_when_a_column_is_empty(self):
        x, y, _, _ = _gauss2(12)
        out = Feasibility().evaluate(_ds(y), target=_specs()[:1], embeddings=x)
        assert list(out.determinations.schema.items()) == list(DETERMINATIONS_SCHEMA.items())
        assert list(out.bounds.schema.items()) == list(BOUNDS_SCHEMA.items())


@pytest.mark.required
class TestDeterminations:
    def test_one_row_per_requirement_in_order(self, run):
        assert run.determinations["requirement"].to_list() == [0, 1, 2, 3]
        assert run.determinations["metric"].to_list() == ["accuracy", "accuracy", "ap", "recall"]
        assert (
            run.determinations["judged_scope"].to_list() == ["full_input", "fixed_representation"] + ["full_input"] * 2
        )
        assert run.determinations["class_name"].to_list() == [None, None, "a", "a"]
        assert run.determinations["max_false_positive_rate"].to_list() == [None, None, None, 0.3]
        assert run.determinations["ap_functional"].to_list() == [None, None, "population_ap_integral", None]

    def test_requirements_are_the_targets_as_given(self, run):
        assert run.requirements == tuple(_specs())

    def test_rows_stay_distinct_across_judged_scopes(self):
        x, y, _, _ = _gauss2(400)
        specs = [TargetSpec(metric="accuracy", threshold=0.6, scope=s) for s in ("full_input", "fixed_representation")]
        out = Feasibility().evaluate(_ds(y), target=specs, embeddings=x)
        assert out.bounds["requirement"].n_unique() == 2
        assert out.determinations["judged_scope"].to_list() == ["full_input", "fixed_representation"]

    def test_every_decided_requirement_rests_on_a_bound(self, run):
        decided = run.determinations.filter(pl.col("determination") != "unresolved")
        assert decided.height >= 2
        for row in decided.iter_rows(named=True):
            determining = run.bounds.filter((pl.col("requirement") == row["requirement"]) & pl.col("is_determining"))
            assert determining.height >= 1
            assert determining["bound"].null_count() == 0

    def test_the_determining_bounds_are_the_ones_the_determination_quotes(self, run):
        for row in run.determinations.iter_rows(named=True):
            determining = run.bounds.filter((pl.col("requirement") == row["requirement"]) & pl.col("is_determining"))
            if row["determination"] == "demonstrated_feasible":
                assert row["witness"] in determining.filter(pl.col("role") == "witness")["bound"].to_list()
            if row["determination"] == "infeasible_under_recorded_assumptions":
                assert row["envelope"] in determining.filter(pl.col("role") == "envelope")["bound"].to_list()

    def test_a_conflict_is_unresolved_with_limitation_conflict(self):
        x, y, post, rng = _gauss2(2000, 8)
        replicas = {i: draw_labels(post[i : i + 1], 2, rng)[0] for i in range(500)}
        out = Feasibility(seed=0).evaluate(
            _ds(y),
            target=[TargetSpec(metric="accuracy", threshold=0.8)],
            embeddings=x,
            replicas=replicas,
            predictions=y,
        )
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limitation"]) == ("unresolved", "conflict")
        assert (row["witness_source"], row["envelope_scope"]) == ("user_model", "full_input")
        assert row["witness"] > row["envelope"]  # the conflicting pair sits in these columns
        determining = out.bounds.filter(pl.col("is_determining"))
        assert sorted(zip(determining["source"], determining["role"], strict=True)) == [
            ("replicas", "envelope"),
            ("user_model", "witness"),
        ]

    def test_the_minimum_names_the_bounds_row_it_projects_from(self):
        rng = np.random.default_rng(2)
        x, y, post = make_population("rare5").sample(4000, rng)
        names = dict(enumerate("abcde"))
        spec = TargetSpec(metric="recall", threshold=0.5, class_name="a", max_false_positive_rate=0.06)
        out = Feasibility(seed=0).evaluate(_ds(y, names), target=[spec], embeddings=x, predictions=post)
        row = out.determinations.row(0, named=True)
        assert row["witness_source"] is None  # the user model is uncertified: its false-positive bound is over 0.06
        assert row["suggested_minimum"] is not None
        assert (row["minimum_source"], row["minimum_quantity"]) == ("user_model", "false_positive_rate")
        joined = out.bounds.filter(
            (pl.col("requirement") == 0)
            & (pl.col("source") == row["minimum_source"])
            & (pl.col("quantity") == row["minimum_quantity"])
        )
        assert joined.height == 1
        assert joined["unit"][0] == "image"
        assert joined["sample"][0] == "audit_split"


@pytest.mark.required
class TestBounds:
    def test_rows_are_ordered_by_requirement_then_source(self, run):
        sources = _SOURCE.categories.to_list()
        keys = [
            (r, sources.index(s), q == "false_positive_rate")
            for r, s, q in zip(run.bounds["requirement"], run.bounds["source"], run.bounds["quantity"], strict=True)
        ]
        assert keys == sorted(keys)
        assert set(run.bounds["source"]) == set(sources)
        recall = run.bounds.filter((pl.col("requirement") == 3) & (pl.col("source") == "user_model"))
        assert recall["quantity"].to_list() == [
            "recall",
            "false_positive_rate",
        ]  # the metric row before its false-positive row

    def test_observed_is_null_for_every_envelope_row(self, run):
        envelopes = run.bounds.filter(pl.col("role") == "envelope")
        assert envelopes.height > 0
        assert envelopes["observed"].null_count() == envelopes.height
        witnesses = run.bounds.filter((pl.col("role") == "witness") & (pl.col("quantity") != "ap"))
        assert witnesses["observed"].null_count() == 0  # a witness's estimate is recorded, except where AP has none

    def test_the_alpha_shares_sum_to_the_call_budget(self, run):
        assert run.bounds["alpha"].sum() == pytest.approx(0.05)

    def test_bounds_that_use_an_allowance_say_so_and_no_other_does(self, run):
        for name in ("allowance_beta", "allowance_gamma"):
            using = [r for r in run.bounds.iter_rows(named=True) if name in r["assumptions"]]
            assert {(r["source"], r["quantity"]) for r in using} == {("local_panels", "accuracy")}
            assert {r["information_scope"] for r in using} == {"full_input"}
        for r in run.bounds.iter_rows(named=True):
            if r["source"] == "user_model":
                assert {"model_independent_of_audit_items", "class_order_matches_dataset"} <= set(r["assumptions"])
            if r["source"] == "replicas":
                assert {"replicas_random_sample", "replicas_one_protocol"} <= set(r["assumptions"])

    def test_radius_is_null_where_the_method_has_none(self, run):
        exact = run.bounds.filter(pl.col("method") == "exact_binomial")
        assert exact.height > 0
        assert exact["radius"].null_count() == exact.height  # Clopper-Pearson intervals have no radius
        no_form = run.bounds.filter(pl.col("method").is_null())
        assert no_form.height > 0  # AP and recall envelopes, and AP witnesses
        assert no_form["radius"].null_count() == no_form.height
        panels = run.bounds.filter((pl.col("source") == "local_panels") & (pl.col("quantity") == "accuracy"))
        assert panels.height > 0
        assert panels["radius"].null_count() == 0  # `.all()` below would skip nulls
        assert (panels["radius"] > 0).all()

    def test_population_and_unit_name_what_the_bound_covers(self):
        x, y, post, _ = _gauss2(300)
        labels: list[int | None] = [int(v) for v in y]
        labels[::10] = [None] * len(labels[::10])  # an unlabeled item has no target
        dataset = cast(Any, _Unlabeled(labels))  # the fake does not satisfy the dataset protocol's metadata type
        out = Feasibility(seed=0).evaluate(dataset, target=_specs()[:1], embeddings=x, predictions=post)
        assert set(out.bounds["population"]) == {"labeled_items"}
        assert set(out.bounds["unit"]) == {"image"}
        assert set(out.bounds["sample"]) <= {"audit_split"}


class _Unlabeled(MockICDataset):
    """A two-class dataset in which a ``None`` label is an unlabeled item (an empty target)."""

    def __init__(self, labels: list[int | None]):
        super().__init__([0] * len(labels), AB)
        self._labels = labels

    def __getitem__(self, i: int) -> Any:
        image, target, meta = super().__getitem__(i)
        label = self._labels[i]
        return image, np.zeros(0) if label is None else np.eye(2, dtype=np.float32)[label], meta


@pytest.mark.required
class TestOutputObject:
    def test_allowances_round_trip(self, run):
        assert run.allowances == Allowances(beta=0.1, gamma=0.05)
        assert run.data()["allowances"] == Allowances(beta=0.1, gamma=0.05)
        assert run.data()["requirements"] == tuple(_specs())
        again = pickle.loads(pickle.dumps(run))
        assert again.allowances == run.allowances
        assert again.requirements == run.requirements
        assert again.determinations.equals(run.determinations)
        assert again.bounds.equals(run.bounds)

    def test_no_allowances_stay_none(self):
        x, y, _, _ = _gauss2(200)
        assert Feasibility().evaluate(_ds(y), target=_specs()[:1], embeddings=x).allowances is None

    def test_meta_state_records_alpha_and_the_resolved_seed(self, run):
        state = run.meta().state
        assert (state["alpha"], state["resolved_seed"]) == (0.05, 3)
        x, y, _, _ = _gauss2(200)
        try:
            set_seed(11)
            assert Feasibility(alpha=0.1).evaluate(_ds(y), target=_specs()[:1], embeddings=x).meta().state == {
                **state,
                "alpha": 0.1,
                "resolved_seed": 11,
                "max_sequence_length": "NoneType",
            } | {"max_sequence_length": run.meta().state["max_sequence_length"]}
            set_seed(None)
            assert Feasibility().evaluate(_ds(y), target=_specs()[:1], embeddings=x).meta().state["resolved_seed"] == 0
        finally:
            # the suite's own seed (tests/conftest.py): set_seed(None) would leave NumPy and torch reseeded at random
            set_seed(0, all_generators=True)

    def test_reordering_the_requirements_permutes_the_rows_and_changes_nothing_else(self):
        x, y, post, _ = _gauss2(800)
        specs = _specs()
        kw: dict[str, Any] = {"embeddings": x, "predictions": post}
        forward = Feasibility(seed=1).evaluate(_ds(y), target=specs, **kw)
        backward = Feasibility(seed=1).evaluate(_ds(y), target=specs[::-1], **kw)
        n = len(specs)
        flip = ((n - 1) - pl.col("requirement")).alias("requirement")
        assert backward.requirements == tuple(specs[::-1])
        assert backward.determinations.with_columns(flip).sort("requirement").equals(forward.determinations)
        assert backward.bounds.with_columns(flip).sort("requirement", maintain_order=True).equals(forward.bounds)

    def test_str_is_terse(self, run):
        lines = str(run).splitlines()
        assert len(lines) == 1 + len(run.determinations) + 3  # header, one line per requirement, three footer lines
        assert lines[0].split() == ["#", "requirement", "determination", "limitation", "envelope", "witness", "minimum"]
        for line, row in zip(lines[1:5], run.determinations.iter_rows(named=True), strict=True):
            assert row["determination"] in line
            assert (row["limitation"] or "—") in line
        assert lines[-3] == "joint confidence: 95%"
        assert lines[-2].startswith("assumes: representation_independent_of_audit_labels, allowance_beta")
        assert lines[-1] == "minimum: independent units to cross the requirement if the observed rates hold"

    def test_str_names_the_bounds_and_the_minimum(self):
        x, y, post, _ = _gauss2(2000, 1)
        specs = [TargetSpec(metric="accuracy", threshold=0.8), TargetSpec(metric="accuracy", threshold=0.95)]
        replicas = dict(enumerate(draw_labels(post[:500], 3, np.random.default_rng(0))))
        out = Feasibility(seed=3).evaluate(_ds(y), target=specs, embeddings=x, replicas=replicas, predictions=post)
        lines = str(out).splitlines()
        assert "demonstrated_feasible" in lines[1]
        assert "user_model" in lines[1]
        assert "infeasible_under_recorded_assumptions" in lines[2]
        assert "replicas" in lines[2]  # the determining envelope's source

    def test_str_without_execution_metadata_still_has_its_footer(self, run):
        bare = replace(run)  # a copy that never went through evaluate, so it has no metadata
        lines = str(bare).splitlines()
        assert lines[-3] == "joint confidence: —"
        assert len(lines) == 1 + len(run.determinations) + 3

    def test_str_rounds_a_bound_the_conservative_way(self, run):
        row = run.determinations.row(0, named=True)
        witness_pct = f"{int(row['witness'] * 1000) / 10:.1f}%"  # a lower bound rounds down
        assert witness_pct in str(run).splitlines()[1]
