"""Tests for the Feasibility evaluator: the frozen output schema, end-to-end determinations and boundary checks."""

import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from dataeval._experimental import experimental, quiet_experimental
from dataeval.data import Limit, View
from dataeval.exceptions import ExperimentalWarning, ShapeMismatchError
from dataeval.extractors import FlattenExtractor
from dataeval.performance import Endpoint, Evidence, Feasibility, FeasibilityOutput, SamplingPlan, TargetSpec
from dataeval.performance._feasibility import _SCHEMA, _metric_words, _pct, _printout, _source_name, _unit_noun
from dataeval.performance._feasibility import _row as _determination
from dataeval.performance._feasibility_constants import FIT_FRACTION
from dataeval.performance._feasibility_decide import decide
from dataeval.performance._feasibility_evidence import check_ledger, stats_key
from dataeval.performance._feasibility_target import Scope
from dataeval.performance._feasibility_units import resolve_units, split_units
from tests.data.conftest import MockICDataset, MockODDataset
from tests.performance.feasibility_helpers import draw_labels, make_population
from tests.performance.test_feasibility_decide import R90_F05, R90_F10, P, _witness_for

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

AB = {0: "a", 1: "b"}
TARGET = {"accuracy": 0.6, "accuracy/fixed_representation": 0.97, "ap:a": 0.5}
BOTH_WAYS = "to show it, test a stronger model; to rule it out, collect repeated labels for a whole-image ceiling"
RECALL_BOTH_WAYS = BOTH_WAYS.replace("model;", "model or a different frozen operating point;")
NO_GAIN = "more evaluation images would not help at the observed scores; "
ASSUMES = "They assume no evaluation image was used to train or choose a model, or to set a requirement."
P40 = np.full((40, 2), 0.5)
# A fresh interpreter, so no earlier call has used up a function's one warning: only the evaluator's own may remain.
_ONE_CALL = """
import warnings
import numpy as np
from dataeval.exceptions import ExperimentalWarning
from dataeval.performance import Feasibility
from tests.data.conftest import MockICDataset
y = [i % 2 for i in range(200)]
x = np.random.default_rng(0).normal(size=(200, 4)) + np.array(y)[:, None]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Feasibility().evaluate(
        MockICDataset(y, {0: "a", 1: "b"}), target={"accuracy": 0.6, "ap:a": 0.5}, embeddings=x,
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


ABT = {0: "a", 1: "b", 2: "Tank"}  # the target class's index (2) is not 1
with quiet_experimental():  # at import, outside the module's warning filter
    TANK_R80_F10 = TargetSpec("IC", "recall", 0.8, class_name="Tank", max_false_positive_rate=0.1)


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
    return Feasibility(seed=3).evaluate(_ds(y), target=TARGET, embeddings=x, replicas=replicas)


def _row(out: FeasibilityOutput, target_spec: str) -> dict:
    return next(r for r in out.determinations.iter_rows(named=True) if r["target_spec"] == target_spec)


@pytest.fixture(scope="module")
def user_run():
    x, y, post, rng = _gauss2(1000)
    replicas = dict(enumerate(draw_labels(post, 4, rng)))
    return Feasibility(seed=3).evaluate(
        _ds(y),
        target=TARGET,
        embeddings=x,
        replicas=replicas,
        predictions=np.eye(2)[y],
        provenance={"reads": "representation"},
    )


def _undecided(run: FeasibilityOutput, **over: Any) -> dict:
    """The first determinations row of a run, turned into an unresolved row with the given fields replaced."""
    row = run.determinations.row(0, named=True)
    return {
        **row,
        "determination": "unresolved",
        "witness": None,
        "witness_source": None,
        "limiting_factor": "sampling_width",
        "suggested_minimum": 1234,
        **over,
    }


def _printed(row: dict, evidence: dict) -> str:
    """The printout of a hand-built output with one determinations row."""
    return str(FeasibilityOutput(pl.DataFrame([row], schema=_SCHEMA), evidence))


def _table(text: str) -> list[list[str]]:
    """The printout's table, header first, each row split into its cells at the column gaps."""
    return [re.split(r" {3,}", line) for line in text.split("\n\n")[1].splitlines()]


def _section(text: str, title: str) -> list[str]:
    """The lines of the printout's Next or Notes section without their indent, rejoined where they broke after a
    semicolon; empty when the section is absent."""
    blocks = [re.sub(r";\n +", "; ", block).splitlines() for block in text.split("\n\n")[2:]]
    return next(([line[2:] for line in b[1:]] for b in blocks if b[0] == title), [])


def _steps(text: str) -> dict[str, str]:
    """The Next section, by requirement label."""
    return {label: step for label, step in (re.split(r" {2,}", line, maxsplit=1) for line in _section(text, "Next"))}


TANK_SCORED = "Your model scored 95.0% recall on 200 Tank images and 2.0% false positives on 800 other images."
TANK_GROUP = (
    "Group-based ceiling (97.0% recall for Tank with at most 5% false positives) bounds only models that give one "
    "answer per group."
)
TANK_UNCERTIFIED = (
    "Tank: no tested model counts, since none keeps false positives within 5%; your model's upper confidence bound on "
    "false positives is 6.0%."
)
with quiet_experimental():  # at import, outside the module's warning filter
    A95, A99 = (TargetSpec("IC", "accuracy", t) for t in (0.95, 0.99))
    AP_A = TargetSpec("IC", "ap", 0.6, class_name="a", ap_functional="population_ap_integral")
    MACRO_AP = TargetSpec("IC", "ap", 0.5, ap_functional="population_ap_integral")
_SEEN = {"mean": 0.969, "variance": 0.03, "n": 5000, "support": 1.0, "delta": 0.001, "method": "exact_binomial"}
ACCURACY = {A95: _SEEN, A99: _SEEN}  # your model's observed accuracy, the same for both thresholds


def _source(eid: str, bounds: dict, stats: dict | None = None, *, alpha: float = 0.05) -> Evidence:
    """Evidence from source ``eid`` with one metric endpoint per target and the given stats per target, else None."""
    ceiling = "ceiling" in eid
    scope: Scope = "fixed_representation" if eid.startswith("builtin") else "full_input"
    return Evidence(
        kind=eid.split(":")[0],
        supports="ceiling" if ceiling else "witness",
        basis="a basis",
        population=P,
        sample="100 items with replicas" if eid.startswith("replica") else "audit split",
        information_scope="partition_restricted" if eid.startswith("local_panel") else scope,
        unit="item",
        provenance={},
        endpoints=tuple(
            Endpoint(s.canonical(), v, "upper" if ceiling else "lower", 0.001, "informative") for s, v in bounds.items()
        ),
        details={"endpoints": {s.canonical(): (stats or {}).get(s) for s in bounds}, "call_alpha": alpha},
    )


def _output(specs: list[TargetSpec], evidence: dict[str, Evidence]) -> FeasibilityOutput:
    """The output the evaluator builds for ``specs`` from ``evidence``, with the real determinations."""
    rows = [_determination(spec, decide(spec, evidence, P)) for spec in specs]
    return FeasibilityOutput(pl.DataFrame(rows, schema=_SCHEMA), evidence)


def _mixed_output() -> FeasibilityOutput:
    """Every kind of row: demonstrated, ruled out, not ruled out (AP), recall with a size, and a conflict."""
    recall = _witness_for([R90_F05], recall_lb=0.84, recall_mean=0.93, fpr_ub=0.031, fpr_mean=0.02)
    user = _source("user_witness/v1", {A95: 0.956, A99: 0.956, AP_A: 0.544}, ACCURACY)
    user = replace(
        user,
        endpoints=user.endpoints + recall.endpoints,
        details={**user.details, "endpoints": {**user.details["endpoints"], **recall.details["endpoints"]}},
    )
    evidence = {
        "user_witness/v1": user,
        "builtin_witness/v1:knn": _source("builtin_witness/v1:knn", {MACRO_AP: 0.99}),
        "replica_ceiling/v1": _source("replica_ceiling/v1", {A95: 0.98, A99: 0.98, AP_A: 0.974, MACRO_AP: 0.98}),
        "local_panel_ceiling/v1": _source("local_panel_ceiling/v1", {A95: 0.721, A99: 0.721}),
    }
    return _output([A95, A99, AP_A, R90_F05, MACRO_AP], evidence)


def _recall_ceiling(eid: str, value: float) -> Evidence:
    """A replica (full_input) or local-panel (partition_restricted) ceiling on R90_F05; like the real ones, no stats."""
    key, replica = R90_F05.canonical(), eid.startswith("replica")
    return Evidence(
        kind=eid,
        supports="ceiling",
        basis="a basis",
        population=P,
        sample="100 items with replicas" if replica else "audit split",
        information_scope="full_input" if replica else "partition_restricted",
        unit="item",
        provenance={},
        endpoints=(Endpoint(key, value, "upper", 0.01, "informative"),),
        details={"endpoints": {key: None}},
    )


def _recall_run(
    recall: tuple[float, float], fp_rate: tuple[float, float], ceilings: dict | None = None, others: dict | None = None
):
    """The output the evaluator builds for R90_F05 from a user witness's (bound, observed) pairs and other evidence."""
    witness = _witness_for(
        [R90_F05], recall_lb=recall[0], recall_mean=recall[1], fpr_ub=fp_rate[0], fpr_mean=fp_rate[1]
    )
    evidence = {"user_witness/v1": witness} | {k: _recall_ceiling(k, v) for k, v in (ceilings or {}).items()}
    return _output([R90_F05], evidence | (others or {}))


@pytest.fixture(scope="module")
def howto_run():
    """A how-to-like call: a realistic user model's predictions next to the built-in witnesses and repeated labels."""
    x, y, post, rng = _gauss2(1000)
    replicas = dict(enumerate(draw_labels(post, 4, rng)))
    # AP at 60%: the exact binomial staircase (lower bound 54.4%) clears 50%, so an undecided AP line needs a higher bar
    return Feasibility(seed=3).evaluate(
        _ds(y),
        target={**TARGET, "ap:a": 0.6},
        embeddings=x,
        replicas=replicas,
        predictions=post,
        provenance={"reads": "representation"},
    )


@pytest.fixture(scope="module")
def allowance_run():
    """The group-based ceiling decides every requirement: no replicas, lab-declared allowances."""
    x, y, _, _ = _gauss2(1000)
    specs = [
        TargetSpec("IC", "accuracy", t, scope=s) for t in (0.6, 0.99) for s in ("full_input", "partition_restricted")
    ]
    return Feasibility(seed=3, alpha=0.025).evaluate(
        _ds(y), target=specs, embeddings=x, allowances={"beta": 0.01, "gamma": 0.02}
    )


@pytest.fixture(scope="module")
def no_bound_run():
    """Every endpoint unavailable: too few units for any panel and no built-in witness."""
    x, y, _, _ = _gauss2(12)
    return Feasibility(panel_size=5, n_cells=10, builtin_witnesses=False).evaluate(
        _ds(y), target={"accuracy": 0.9, "ap:a": 0.5}, embeddings=x
    )


@pytest.mark.required
class TestFeasibility:
    def test_determination_schema_is_frozen(self, gauss2_run):
        scope = pl.Enum(["full_input", "fixed_representation", "partition_restricted"])
        expected = {
            "task": pl.Enum(["IC", "OD", "MOT"]),
            "metric": pl.Enum(["accuracy", "ap", "recall", "recall_at_fp", "mota"]),
            "threshold": pl.Float64,
            "judged_scope": scope,
            "target_spec": pl.String,
            "determination": pl.Enum([
                "evidence_conflict",
                "demonstrated_feasible",
                "infeasible_under_recorded_assumptions",
                "not_ruled_out",
                "unresolved",
            ]),
            "ceiling": pl.Float64,
            "ceiling_scope": scope,
            "witness": pl.Float64,
            "witness_source": pl.String,
            "reason": pl.String,
            "limiting_factor": pl.Enum([
                "sampling_width",
                "no_replicas",
                "scope_restricted",
                "no_witness",
                "missing_bound",
                "labels_conditioned",
                "conflict",
            ]),
            "recommendation": pl.String,
            "suggested_minimum": pl.Int64,
            "evidence_ids": pl.List(pl.String),
        }
        schema = gauss2_run.determinations.schema
        assert list(schema) == list(expected)
        assert dict(schema) == expected
        assert gauss2_run.determinations.height == len(TARGET)

    def test_end_to_end_gauss2_demonstrates_an_easy_target(self, gauss2_run):
        easy = TargetSpec("IC", "accuracy", 0.6).canonical()
        row = _row(gauss2_run, easy)
        assert row["determination"] == "demonstrated_feasible"
        assert row["witness"] >= 0.6
        assert row["witness_source"].startswith("builtin_witness/v1:")
        assert row["limiting_factor"] is None
        assert TargetSpec.from_canonical(row["target_spec"]) == TargetSpec("IC", "accuracy", 0.6)
        # Bayes accuracy is Phi(1) ~ 0.841: the replica ceiling rules out 0.97 at fixed_representation.
        hard = _row(gauss2_run, TargetSpec("IC", "accuracy", 0.97, scope="fixed_representation").canonical())
        assert hard["determination"] == "infeasible_under_recorded_assumptions"
        assert hard["ceiling_scope"] == "full_input"
        assert 0.841 <= hard["ceiling"] < 0.97

    def test_every_decided_row_cites_evidence_with_a_basis(self, gauss2_run):
        decided = gauss2_run.determinations.filter(pl.col("determination") != "unresolved")
        assert decided.height >= 2
        for row in decided.iter_rows(named=True):
            cited = [gauss2_run.evidence[eid] for eid in row["evidence_ids"]]
            assert any(
                ev.basis.strip()
                and any(e.target_spec == row["target_spec"] and e.value is not None for e in ev.endpoints)
                for ev in cited
            ), row

    def test_call_ledger_within_alpha(self, gauss2_run):
        spent = sum(e.alpha for ev in gauss2_run.evidence.values() for e in ev.endpoints)
        assert spent <= 0.05 + 1e-12
        assert {ev.details["seed"] for ev in gauss2_run.evidence.values()} == {3}
        assert {ev.details["allocation"] for ev in gauss2_run.evidence.values()} == {"equal_per_claim"}

    def test_a_look_spends_its_share_on_its_prefix(self):
        x, y, _, _ = _gauss2(800)
        plan = SamplingPlan((100, 200, 400), order_seed=1)
        out = Feasibility(alpha=0.06).evaluate(_ds(y), target={"accuracy": 0.6}, embeddings=x, sampling=plan, look=2)
        spent = sum(e.alpha for ev in out.evidence.values() for e in ev.endpoints)
        assert spent <= 0.06 / 6 + 1e-12
        witness = out.evidence["builtin_witness/v1:probe"]
        assert witness.details["look"] == 2
        assert witness.details["call_alpha"] == pytest.approx(0.01)
        assert witness.details["order_seed"] == 1
        assert next(iter(witness.details["endpoints"].values()))["n"] == 200

    def test_a_grouped_look_takes_whole_units(self):
        x, y, _, _ = _gauss2(800)
        groups = np.arange(800) // 3  # units of 3 items (the last of 2)
        plan = SamplingPlan((20, 40, 80), order_seed=1)
        out = Feasibility().evaluate(
            _ds(y), target={"accuracy|sequence_mean": 0.6}, embeddings=x, groups=groups, sampling=plan, look=2
        )
        stats = next(iter(out.evidence["builtin_witness/v1:knn"].details["endpoints"].values()))
        assert stats["n"] == 40

    def test_tiny_audit_is_unresolved_not_an_error(self):
        x, y, _, _ = _gauss2(12)
        out = Feasibility(panel_size=5, n_cells=10).evaluate(_ds(y), target={"accuracy": 0.9}, embeddings=x)
        row = out.determinations.row(0, named=True)
        assert row["determination"] == "unresolved"
        assert row["limiting_factor"] == "sampling_width"
        local = out.evidence["local_panel_ceiling/v1"]
        assert [e.state for e in local.endpoints] == ["unavailable"]
        assert "the fit split has 6 items, fewer than the 10 partition cells" in str(local.endpoints[0].reason)

    def test_od_dataset_is_rejected_clearly(self):
        dataset: Any = MockODDataset([[0, 1], [1]], AB)
        msg = "Feasibility supports image and video classification in this release; OD is planned."
        with pytest.raises(NotImplementedError, match=msg):
            Feasibility().evaluate(dataset, target={"ap@0.5": 0.5}, embeddings=np.zeros((2, 2)))

    def test_recall_with_no_audit_positives_is_unresolved(self):
        y, x = _three_classes(300)
        _, audit = split_units(resolve_units(_ds(y, ABT), groups=None), fit_fraction=FIT_FRACTION, seed=0)
        y[audit] = np.where(y[audit] == 2, 0, y[audit])  # Tank only in the fit split
        out = Feasibility(seed=0).evaluate(_ds(y, ABT), target=[TANK_R80_F10], embeddings=x, predictions=y)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["witness"]) == ("unresolved", None)
        recall = out.evidence["user_witness/v1"].endpoints[0]
        assert (recall.state, recall.reason) == ("unavailable", "no audit items labeled 'Tank'")
        assert _table(str(out))[1][:2] == ["Tank: ≥80% recall, ≤10% false positives", "insufficient evidence ?"]

    def test_a_recall_target_gets_a_full_input_ceiling_from_repeated_labels(self):
        y, x = _three_classes(600)
        i = np.arange(600)  # two replicas per image, each disagreeing with the primary label on every other image
        replicas = dict(enumerate(np.c_[np.where(i % 2, y, (y + 1) % 3), np.where(i // 2 % 2, y, (y + 2) % 3)]))
        run = Feasibility(seed=0, builtin_witnesses=False)
        out = run.evaluate(_ds(y, ABT), target=[TANK_R80_F10], embeddings=x, replicas=replicas)
        (row,) = out.aggregate_by_evidence().filter(pl.col("source") == "repeated labels").iter_rows(named=True)
        assert (row["information_scope"], row["bound_type"], row["state"]) == ("full_input", "upper", "informative")
        assert 0 <= row["bound"] < TANK_R80_F10.threshold <= 1
        determination = out.determinations.row(0, named=True)["determination"]
        assert determination == "infeasible_under_recorded_assumptions"
        check_ledger(out.evidence.values(), 0.05)

    def test_class_decisions_drive_a_recall_witness_only_when_certified(self):
        y, x = _three_classes(600)
        decisions = np.where(np.arange(600) % 20 == 0, (y + 1) % 3, y)  # 95% agreement
        run = Feasibility(seed=0, builtin_witnesses=False)
        out = run.evaluate(_ds(y, ABT), target=[TANK_R80_F10], embeddings=x, predictions=decisions)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["witness_source"]) == ("demonstrated_feasible", "user_witness/v1")
        always_tank = np.full(600, 2)  # recall 1 and false-positive rate 1
        out = run.evaluate(_ds(y, ABT), target=[TANK_R80_F10], embeddings=x, predictions=always_tank)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["witness_source"]) == ("unresolved", None)
        recall, false_positives = out.evidence["user_witness/v1"].endpoints
        assert (recall.value or 0) > 0.8  # the recall bound clears the requirement
        assert (false_positives.value or 0) > 0.1  # the false-positive bound does not

    def test_class_decisions_are_a_user_witness(self):
        y, x = _three_classes(300)
        decisions = np.where(np.arange(300) % 10 == 0, (y + 1) % 3, y)  # 90% agreement, integer class indices
        policy = "the true class, except every tenth item, which gets the next class index (synthetic)"
        out = Feasibility(seed=0, builtin_witnesses=False).evaluate(
            _ds(y, ABT),
            target={"accuracy": 0.6},
            embeddings=x,
            predictions=decisions,
            provenance={"decision_policy": policy},
        )
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["witness_source"]) == ("demonstrated_feasible", "user_witness/v1")
        assert out.evidence["user_witness/v1"].provenance["decision_policy"] == policy

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
            Feasibility(seed=0).evaluate(_ds(y, ABT), target={"accuracy": 0.6}, predictions=bad)

    def test_ap_with_class_decisions_only_is_unavailable(self):
        y, x = _three_classes(300)
        out = Feasibility(seed=0, builtin_witnesses=False).evaluate(
            _ds(y, ABT), target={"ap:Tank": 0.5}, embeddings=x, predictions=y
        )
        (ep,) = out.evidence["user_witness/v1"].endpoints
        assert (ep.state, ep.reason) == ("unavailable", "predictions are class decisions, not scores; AP needs scores")

    def test_a_subset_view_takes_plain_targets(self):
        x, y, _, _ = _gauss2(200)
        view: Any = View(_ds(y), Limit(size=120))  # straight over a plain dataset: grouped by source index
        out = Feasibility().evaluate(view, target={"accuracy": 0.6}, embeddings=x[:120])
        assert out.determinations["determination"][0] == "demonstrated_feasible"
        assert {ev.unit for ev in out.evidence.values()} == {"item"}

    def test_the_extractor_warns_outside_the_quiet_scope(self):
        @experimental
        def flatten(batch):
            return FlattenExtractor()(batch)

        x, y, _, _ = _gauss2(200)
        dataset: Any = _Images(x, y)
        with pytest.warns(ExperimentalWarning, match="flatten"):
            Feasibility(extractor=flatten).evaluate(dataset, target={"accuracy": 0.6})

    def test_extractor_matches_given_embeddings(self):
        x, y, _, _ = _gauss2(200)
        x = x.astype(np.float32)  # what the extractor sees, so both paths read identical features
        dataset: Any = _Images(x, y)
        given = Feasibility().evaluate(dataset, target={"accuracy": 0.6}, embeddings=x)
        extracted = Feasibility(extractor=FlattenExtractor()).evaluate(dataset, target={"accuracy": 0.6})
        assert extracted.determinations.equals(given.determinations)

    @pytest.mark.parametrize("cls", [Feasibility, FeasibilityOutput, SamplingPlan, TargetSpec, Evidence, Endpoint])
    def test_experimental_warning(self, cls):
        assert cls.__name__ in getattr(cls, "__experimental__", "")

    def test_a_call_warns_for_the_evaluator_only(self):
        root = Path(__file__).parents[2]
        out = subprocess.run([sys.executable, "-c", _ONE_CALL], cwd=root, capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "['Feasibility']"

    def test_repr_summarizes_the_evidence(self, gauss2_run):
        text = repr(gauss2_run)
        assert "DataFrame(shape=(3, 15))" in text
        assert "4 evidence items" in text
        assert "builtin_witness/v1:probe" in text
        assert len(text) < 400

    def test_plot_type_identifies_the_output_for_plotting(self, gauss2_run):
        """`plot_type` is the discriminator dataeval-plots dispatches on."""
        assert gauss2_run.plot_type == "feasibility"

    def test_repr_unchanged(self, gauss2_run):
        assert repr(gauss2_run).startswith("FeasibilityOutput(determinations=DataFrame(shape=")
        assert str(gauss2_run) != repr(gauss2_run)

    def test_str_lays_out_a_two_requirement_call(self):
        # The tutorial's call: your model demonstrates 95%, and only a group-based ceiling bounds 99%.
        out = _output(
            [A95, A99],
            {
                "user_witness/v1": _source("user_witness/v1", {A95: 0.956, A99: 0.956}, ACCURACY, alpha=0.025),
                "local_panel_ceiling/v1": _source("local_panel_ceiling/v1", {A95: 0.721, A99: 0.721}, alpha=0.025),
            },
        )
        text = str(out)
        assert text == (
            "Feasibility of 2 requirements, judged for models that may use the whole image.\n"
            "All bounds hold together at 97.5% confidence, for new images from the same population.\n"
            "They assume no evaluation image was used to train or choose a model, or to set a requirement.\n"
            "\n"
            "Requirement    Determination             Lower confidence bound   Upper confidence bound\n"
            "95% accuracy   demonstrated ✓            95.6% (your model)       —\n"
            "99% accuracy   insufficient evidence ?   95.6% (your model)       —\n"
            "\n"
            "Next\n"
            "  99% accuracy  more evaluation images would not help at the observed scores;\n"
            "                to show it, test a stronger model;\n"
            "                to rule it out, collect repeated labels for a whole-image ceiling\n"
            "\n"
            "Notes\n"
            "  Your model scored 96.9% accuracy on 5,000 evaluation images.\n"
            "  Group-based ceiling (72.1% accuracy) bounds only models that give one answer per group."
        )
        assert max(len(line) for line in text.splitlines()) <= 140

    def test_str_lays_out_every_kind_of_row(self):
        out = _mixed_output()
        n = out.determinations["suggested_minimum"][3]
        assert n is not None
        assert str(out) == (
            "Feasibility of 5 requirements, judged for models that may use the whole image.\n"
            "All bounds hold together at 95% confidence, for new images from the same population.\n"
            "They assume no evaluation image was used to train or choose a model, or to set a requirement.\n"
            "\n"
            "Requirement                              Determination             Lower confidence bound"
            "                Upper confidence bound\n"
            "95% accuracy                             demonstrated ✓            95.6% (your model)"
            "                    98.0% (repeated labels)\n"
            "99% accuracy                             ruled out ✗               95.6% (your model)"
            "                    98.0%* (repeated labels)\n"
            "60% AP for class a                       not ruled out ?           54.4% (your model)"
            "                    97.4% (repeated labels)\n"
            "Tank: ≥90% recall, ≤5% false positives   insufficient evidence ?   recall 84.0%, FP ≤3.1% (your model)"
            "   —\n"
            "50% macro AP                             evidence conflict !       99.0% (built-in k-NN)"
            "                 98.0% (repeated labels)\n"
            "\n"
            "Next\n"
            "  60% AP for class a  test a stronger model\n"
            f"  Tank                about {n:,} evaluation images in all, for your model's lower confidence bound on "
            "recall to cross if the observed rates hold\n"
            "\n"
            "Notes\n"
            "  Your model scored 96.9% accuracy on 5,000 evaluation images;\n"
            "    93.0% recall on 200 Tank images and 2.0% false positives on 800 other images.\n"
            "  Group-based ceiling (72.1% accuracy) bounds only models that give one answer per group.\n"
            "  50% macro AP: the evidence conflicts; check the assumptions, including whether evaluation images "
            "entered training.\n"
            "  * ruled out under the recorded assumptions"
        )

    def test_str_demonstrated_row(self, howto_run):
        text = str(howto_run)
        assert _table(text)[1] == [
            "60% accuracy (models that may use the whole image)",
            "demonstrated ✓",
            "79.3% (your model)",
            "88.9% (repeated labels)",
        ]
        assert "Your model scored 84.2% accuracy on 500 evaluation images." in _section(text, "Notes")

    def test_str_header_table_then_next_and_notes(self, howto_run):
        text = str(howto_run)
        blocks = text.split("\n\n")
        assert blocks[0].splitlines() == [
            "Feasibility of 3 requirements, judged at several scopes.",
            "All bounds hold together at 95% confidence, for new images from the same population.",
            ASSUMES,
        ]
        table = _table(text)
        assert table[0] == ["Requirement", "Determination", "Lower confidence bound", "Upper confidence bound"]
        assert len(table) == 1 + howto_run.determinations.height
        assert [block.splitlines()[0] for block in blocks[2:]] == ["Next", "Notes"]
        assert not any("AP" in note for note in _section(text, "Notes") if "scored" in note)  # AP records no mean

    def test_str_unresolved_row(self):
        x, y, _, _ = _gauss2(30)
        text = str(Feasibility().evaluate(_ds(y), target=TARGET, embeddings=x))
        row = _table(text)[1]
        assert (row[1], row[3]) == ("insufficient evidence ?", "—")
        step = _steps(text)[row[0]]
        assert step.startswith("built-in linear probe's lower confidence bound would cross at about ")
        assert step.endswith(f" evaluation images if the observed rates hold; {BOTH_WAYS}")
        # the group-based ceilings here are uninformative (100%), so no note names them
        assert _section(text, "Notes") == ["Built-in linear probe scored 80.0% accuracy on 15 evaluation images."]
        assert "beta" not in text
        assert "None" not in text

    def test_str_never_overstates(self, gauss2_run, user_run, howto_run, allowance_run, no_bound_run):
        for run in (gauss2_run, user_run, howto_run, allowance_run, no_bound_run):
            text = str(run)
            for word in ("guarantee", "maximum", "exactly", "None"):
                assert word not in text
            # only a row that rules a requirement out must say on what basis: its * and the note
            ruled_out = [row for row in _table(text)[1:] if row[1] == "ruled out ✗"]
            assert all("*" in row[3] for row in ruled_out)
            assert ("* ruled out under the recorded assumptions" in _section(text, "Notes")) == bool(ruled_out)

    def test_str_header_states_joint_confidence(self, gauss2_run, allowance_run):
        assert str(gauss2_run).splitlines()[1] == (
            "All bounds hold together at 95% confidence, for new images from the same population."
        )
        assert str(allowance_run).splitlines()[1].startswith("All bounds hold together at 97.5% confidence,")

    def test_str_header_rounds_the_joint_confidence_down(self):
        # alpha = 0.05 / 3, three planned calls: 98.3333...% is stated as 98.3%, never rounded up to overstate it
        source = _source("user_witness/v1", {A95: 0.956}, ACCURACY, alpha=0.05 / 3)
        assert str(_output([A95], {"user_witness/v1": source})).splitlines()[1] == (
            "All bounds hold together at 98.3% confidence, for new images from the same population."
        )

    def test_no_sign_off_caveat_in_the_printout_or_evidence(
        self, gauss2_run, user_run, howto_run, allowance_run, no_bound_run
    ):
        for run in (gauss2_run, user_run, howto_run, allowance_run, no_bound_run):
            for text in (str(run), *(repr(ev) for ev in run.evidence.values())):
                assert not any(word in text for word in ("sign-off", "lead-scientist", "lead scientist"))
        # the group-based ceiling decides every row of this call, with no caveat
        assert [row[3] for row in _table(str(allowance_run))[1:]] == [
            "96.4% (group-based ceiling)",
            "96.4% (group-based ceiling)",
            "96.4%* (group-based ceiling)",
            "96.4%* (group-based ceiling)",
        ]

    def test_str_without_any_bound(self, no_bound_run):
        text = str(no_bound_run)
        assert [row[1:] for row in _table(text)[1:]] == [["insufficient evidence ?", "—", "—"]] * 2
        assert _steps(text) == dict.fromkeys(["90% accuracy", "50% AP for class a"], "collect more labeled images")
        assert _section(text, "Notes") == []
        assert "None" not in text

    def test_str_omits_observed_when_absent(self, howto_run):
        witness = howto_run.evidence["user_witness/v1"]
        hollow = replace(witness, details={**witness.details, "endpoints": dict.fromkeys(witness.details["endpoints"])})
        text = str(FeasibilityOutput(howto_run.determinations, {**howto_run.evidence, "user_witness/v1": hollow}))
        assert _table(text)[1][2] == "79.3% (your model)"
        assert not any(note.startswith("Your model scored") for note in _section(text, "Notes"))
        assert "observed" not in text
        assert "would not help" not in text  # unrecorded scores: never claim more images would not help
        assert "None" not in text

    def test_str_names_both_bounds_when_not_ruled_out(self, howto_run):
        text = str(howto_run)
        ap = "60% AP for class a (models that may use the whole image)"
        assert _table(text)[3] == [ap, "not ruled out ?", "54.4% (built-in linear probe)", "97.4% (repeated labels)"]
        assert _steps(text) == {ap: "test a stronger model"}

    def test_str_macro_ap(self):
        x, y, _, _ = _gauss2(30)
        assert _table(str(Feasibility().evaluate(_ds(y), target={"ap": 0.5}, embeddings=x)))[1][0] == "50% macro AP"

    def test_str_survives_an_unavailable_ceiling_beside_an_available_one(self):
        x, y, post, rng = _gauss2(300)
        replicas = dict(enumerate(draw_labels(post, 4, rng)))
        out = Feasibility(seed=3, panel_size=5, n_cells=1000, builtin_witnesses=False).evaluate(
            _ds(y), target={"accuracy": 0.6, "accuracy/fixed_representation": 0.99}, embeddings=x, replicas=replicas
        )
        states = {k: [e.state for e in ev.endpoints] for k, ev in out.evidence.items()}
        assert states == {"local_panel_ceiling/v1": ["unavailable"] * 2, "replica_ceiling/v1": ["informative"] * 2}
        text = str(out)
        first, second = _table(text)[1:]
        assert first[1:3] == ["not ruled out ?", "—"]  # no witness
        assert first[3].endswith("% (repeated labels)")
        assert second[1:] == ["ruled out ✗", "—", "91.1%* (repeated labels)"]
        assert _steps(text) == {first[0]: "test a stronger model"}

    def test_str_names_only_the_ceiling_of_the_deciding_scope(self, allowance_run):
        # an equal-valued but narrower-scope ceiling must not be named for a broader requirement
        bounds = allowance_run._bounds().rows(named=True)
        row = allowance_run.determinations.row(2, named=True)
        assert row["determination"] == "infeasible_under_recorded_assumptions"
        narrower = [{**b, "information_scope": "partition_restricted", "source": "narrow source"} for b in bounds]
        assert _table(_printout([row], [*narrower, *bounds], {}))[1][3] == "96.4%* (group-based ceiling)"

    def test_str_unresolved_row_shows_both_bounds(self, gauss2_run):
        row = _undecided(gauss2_run, witness=0.5, witness_source="builtin_witness/v1:knn")
        cells = _table(_printed(row, gauss2_run.evidence))[1]
        assert cells == ["60% accuracy", "insufficient evidence ?", "50.0% (built-in k-NN)", "88.4% (repeated labels)"]

    def test_str_unresolved_with_a_ceiling_but_no_witness(self, gauss2_run):
        row = _undecided(gauss2_run, limiting_factor="no_witness", suggested_minimum=None)
        text = _printed(row, gauss2_run.evidence)
        assert _table(text)[1][2:] == ["—", "88.4% (repeated labels)"]
        assert _steps(text) == {"60% accuracy": "test a stronger model"}
        assert "None" not in text

    def test_str_falls_back_when_no_row_matches_the_ceiling(self, gauss2_run):
        text = _printed(_undecided(gauss2_run, ceiling=0.123), {})
        assert _table(text)[1][3] == "12.3%"
        assert text.splitlines()[1] == "Bounds are for new images from the same population."  # no call_alpha
        row = _undecided(gauss2_run, determination="infeasible_under_recorded_assumptions", ceiling=0.123)
        assert _table(_printed(row, {}))[1][1:] == ["ruled out ✗", "—", "12.3%*"]

    @pytest.mark.parametrize(
        ("factor", "action"),
        [
            ("sampling_width", "collect more evaluation images"),
            ("no_replicas", BOTH_WAYS),
            ("scope_restricted", BOTH_WAYS),
            ("no_witness", "test a stronger model"),
            ("missing_bound", "supply the input named in the recommendation column"),
        ],
    )
    def test_str_maps_each_limiting_factor_to_a_plain_action(self, gauss2_run, factor, action):
        row = _undecided(gauss2_run, ceiling=None, ceiling_scope=None, limiting_factor=factor, suggested_minimum=None)
        assert _steps(_printed(row, {})) == {"60% accuracy": action}

    def test_str_no_replicas_keeps_its_action_outside_unresolved(self, gauss2_run):
        row = _undecided(
            gauss2_run, determination="not_ruled_out", limiting_factor="no_replicas", suggested_minimum=None
        )
        text = _printed(row, gauss2_run.evidence)
        assert _steps(text) == {"60% accuracy": "have several annotators independently label the same images"}

    @pytest.mark.parametrize(
        ("factor", "step"),
        [
            (
                "sampling_width",
                (
                    "about 12,345 evaluation images in all, for built-in linear probe's lower confidence bound to "
                    "cross if the observed rates hold"
                ),
            ),
            *(
                (
                    factor,
                    (
                        "built-in linear probe's lower confidence bound would cross at about 12,345 evaluation images "
                        f"if the observed rates hold; {action}"
                    ),
                )
                for factor, action in (
                    ("no_replicas", BOTH_WAYS),
                    ("scope_restricted", BOTH_WAYS),
                    ("no_witness", "test a stronger model"),
                )
            ),
        ],
    )
    def test_str_states_the_crossing_size_for_any_factor(self, gauss2_run, factor, step):
        row = _undecided(gauss2_run, ceiling=None, ceiling_scope=None, limiting_factor=factor, suggested_minimum=12345)
        assert _steps(_printed(row, gauss2_run.evidence)) == {"60% accuracy": step}

    def test_str_crossing_without_its_evidence_names_no_bound(self, gauss2_run):
        row = _undecided(gauss2_run, ceiling=None, ceiling_scope=None, suggested_minimum=12345)
        assert _steps(_printed(row, {})) == {
            "60% accuracy": "collect more evaluation images; the bound would cross at about 12,345 images if the "
            "observed rates hold"
        }

    def test_accuracy_suggested_minimum_survives_exact_binomial(self):
        x, y, post, rng = _gauss2(400)
        replicas = dict(enumerate(draw_labels(post, 4, rng)))
        out = Feasibility(seed=0).evaluate(_ds(y), target={"accuracy": 0.75}, embeddings=x, replicas=replicas)
        witness = out.evidence["builtin_witness/v1:knn"]
        (stats,) = witness.details["endpoints"].values()
        assert stats["method"] == "exact_binomial"
        bound = witness.endpoints[0].value
        assert bound is not None
        assert stats["mean"] > 0.75 > bound  # observed above the requirement, bound below it
        row = out.determinations.row(0, named=True)
        assert row["limiting_factor"] == "sampling_width"
        assert row["suggested_minimum"] is not None

    def test_str_names_a_replica_crossing_in_images_with_repeated_labels(self):
        x, y, post, rng = _gauss2(2000)
        replicas = dict(enumerate(draw_labels(post, 4, rng)[:150]))
        out = Feasibility(seed=3).evaluate(_ds(y), target={"accuracy": 0.921}, embeddings=x, replicas=replicas)
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limiting_factor"]) == ("not_ruled_out", "sampling_width")
        n = row["suggested_minimum"]
        assert n is not None
        assert _steps(str(out)) == {
            "92.1% accuracy": f"about {n:,} images with repeated labels in all, for the upper confidence bound from "
            "the repeated labels to cross if the observed rates hold"
        }

    def test_str_keeps_the_aggregation(self):
        specs = [
            TargetSpec("IC", "accuracy", 0.6, aggregation="sequence_mean"),
            TargetSpec("IC", "accuracy", 0.6, aggregation="frame_pooled"),
            TargetSpec("IC", "ap", 0.6, ap_functional="population_ap_integral", aggregation="frame_pooled"),
            TargetSpec("IC", "ap", 0.6, class_name="a", ap_functional="population_ap_integral"),
        ]
        assert [_metric_words(spec) for spec in specs] == [
            "per-sequence mean accuracy",
            "frame-pooled accuracy",
            "frame-pooled macro AP",
            "AP for class a",
        ]

    @pytest.mark.parametrize(
        ("value", "direction", "text"),
        [
            (0.95996, "lower", "95.9%"),
            (0.95996, None, "96.0%"),
            (0.72001, "upper", "72.1%"),
            (0.957, "lower", "95.7%"),
            (0.957, "upper", "95.7%"),
        ],
    )
    def test_pct_rounds_bounds_conservatively(self, value, direction, text):
        assert _pct(value, direction) == text

    @pytest.mark.parametrize("factor", [None, "conflict", "labels_conditioned"])
    def test_str_omits_next_without_an_action(self, gauss2_run, factor):
        row = _undecided(gauss2_run, ceiling=None, ceiling_scope=None, limiting_factor=factor, suggested_minimum=None)
        text = _printed(row, {})
        assert _section(text, "Next") == []
        assert "\nNext\n" not in text

    def test_str_next_action_uses_sequences_for_grouped_evidence(self, gauss2_run):
        row = _undecided(gauss2_run, witness=0.7, witness_source="user_witness/v1", ceiling=None, ceiling_scope=None)
        spec = row["target_spec"]
        grouped = Evidence(
            kind="user_witness/v1",
            supports="witness",
            basis="b",
            population="p",
            sample="audit split",
            information_scope="fixed_representation",
            unit="group",
            provenance={},
            endpoints=(Endpoint(spec, 0.7, "lower", 0.01, "informative"),),
            details={
                "endpoints": {
                    spec: {"mean": 0.8, "variance": 0.16, "n": 150, "support": 1.0, "delta": 0.01, "method": "hybrid"}
                },
                "call_alpha": 0.05,
            },
        )
        row["limiting_factor"] = "no_replicas"
        row["evidence_ids"] = ["user_witness/v1"]
        text = _printed(row, {"user_witness/v1": grouped})
        assert text.splitlines()[:2] == [
            "Feasibility of 1 requirement, judged for models that may use the whole sequence.",
            "All bounds hold together at 95% confidence, for new sequences from the same population.",
        ]
        assert _table(text)[1][2] == "70.0% (your model)"
        assert _steps(text) == {
            "60% accuracy": "your model's lower confidence bound would cross at about 1,234 evaluation sequences if "
            f"the observed rates hold; {BOTH_WAYS.replace('image', 'sequence')}"
        }
        assert _section(text, "Notes") == ["Your model scored 80.0% accuracy on 150 evaluation sequences."]

    def test_str_conflict_row(self):
        spec = TargetSpec("IC", "accuracy", 0.95, scope="fixed_representation")
        row = {
            "task": "IC",
            "metric": "accuracy",
            "threshold": 0.95,
            "judged_scope": "fixed_representation",
            "target_spec": spec.canonical(),
            "determination": "evidence_conflict",
            "ceiling": 0.9,
            "ceiling_scope": "full_input",
            "witness": 0.96,
            "witness_source": "user_witness/v1",
            "reason": "r",
            "limiting_factor": "conflict",
            "recommendation": "Check for leakage. More text.",
            "suggested_minimum": None,
            "evidence_ids": ["user_witness/v1"],
        }
        text = _printed(row, {})
        assert text.splitlines()[:2] == [
            "Feasibility of 1 requirement, judged for models that may use only the embeddings.",
            "Bounds are for new images from the same population.",
        ]
        assert _table(text)[1] == ["95% accuracy", "evidence conflict !", "96.0% (user_witness/v1)", "90.0%"]
        assert _section(text, "Next") == []
        assert _section(text, "Notes") == [
            (
                "95% accuracy: the evidence conflicts; check the assumptions, including whether evaluation images "
                "entered training."
            )
        ]
        assert "Check for leakage" not in text

    @pytest.mark.parametrize(
        ("recall", "fp_rate", "ceilings", "cells", "steps", "notes", "advice"),
        [
            pytest.param(
                (0.92, 0.95),
                (0.04, 0.02),
                {},
                ["demonstrated ✓", "recall 92.0%, FP ≤4.0% (your model)", "—"],
                {},
                [TANK_SCORED],
                None,
                id="demonstrated",
            ),
            pytest.param(
                (0.7, 0.75),
                (0.04, 0.02),
                {},
                ["insufficient evidence ?", "recall 70.0%, FP ≤4.0% (your model)", "—"],
                {"Tank": NO_GAIN + RECALL_BOTH_WAYS},
                [TANK_SCORED.replace("95.0%", "75.0%")],
                "No ceiling broader than partition_restricted bounds this target",
                id="certified-witness-short",
            ),
            pytest.param(
                (0.7, 0.75),
                (0.04, 0.02),
                {"replica_ceiling/v1": 0.95},
                ["not ruled out ?", "recall 70.0%, FP ≤4.0% (your model)", "recall 95.0% (repeated labels)"],
                {"Tank": f"{NO_GAIN}test a stronger model or a different frozen operating point"},
                [TANK_SCORED.replace("95.0%", "75.0%")],
                "supply a stronger model or a different frozen operating point",
                id="not-ruled-out",
            ),
            pytest.param(
                # The uncertified witness's false-positive bound equals the ceiling: only the ceiling is named.
                (0.5, 0.55),
                (0.6, 0.55),
                {"local_panel_ceiling/v1": 0.97, "replica_ceiling/v1": 0.6},
                ["ruled out ✗", "recall 50.0%, FP ≤60.0% (your model, not counted)", "recall 60.0%* (repeated labels)"],
                {},
                [
                    TANK_SCORED.replace("95.0%", "55.0%").replace("2.0%", "55.0%"),
                    TANK_GROUP,
                    "* ruled out under the recorded assumptions",
                ],
                None,
                id="ruled-out-by-repeated-labels",
            ),
        ],
    )
    def test_str_recall_row(self, recall, fp_rate, ceilings, cells, steps, notes, advice):
        out = _recall_run(recall, fp_rate, ceilings)
        text = str(out)
        assert _table(text)[1] == ["Tank: ≥90% recall, ≤5% false positives", *cells]
        assert (_steps(text), _section(text, "Notes")) == (steps, notes)
        recommendation = out.determinations["recommendation"][0]
        assert recommendation is None if advice is None else advice in recommendation

    def test_str_recall_without_a_certified_witness(self):
        # Only a partition_restricted ceiling, and a witness whose false-positive bound misses the limit.
        out = _recall_run((0.92, 0.95), (0.06, 0.03), {"local_panel_ceiling/v1": 0.97})
        row = out.determinations.row(0, named=True)
        assert (row["determination"], row["limiting_factor"]) == ("unresolved", "scope_restricted")
        n = row["suggested_minimum"]
        assert n is not None
        text = str(out)
        # the tested witness's bounds show, marked as not counted, and its observed rates are a note
        assert _table(text)[1][1:] == [
            "insufficient evidence ?",
            "recall 92.0%, FP ≤6.0% (your model, not counted)",
            "—",
        ]
        crossing = (
            f"your model's upper confidence bound on false positives would cross at about {n:,} evaluation images if "
            "the observed rates hold"
        )
        assert _steps(text) == {"Tank": f"{crossing}; {RECALL_BOTH_WAYS}"}  # more images would help: no NO_GAIN
        assert _section(text, "Notes") == [TANK_SCORED.replace("2.0%", "3.0%"), TANK_UNCERTIFIED, TANK_GROUP]
        # Lines over the printout width break after their semicolons, Next's under its step column
        assert "\n  " + TANK_UNCERTIFIED.replace("; ", ";\n    ") + "\n" in text
        assert f"\n  Tank  {crossing};\n        to show it, " in text
        assert row["recommendation"].startswith(
            "Only narrower ceilings bound this full_input target (local_panel_ceiling/v1)."
        )
        assert (
            "not yet on its confidence bounds: its upper confidence bound on false positives would cross the "
            f"requirement at about {n} audit-split units"
        ) in row["recommendation"]

    def test_str_names_one_source_for_an_uncertified_recall_requirement(self):
        # k-NN's false-positive bound is nearer the limit, but only your model's observed rates meet the requirement:
        # Next and Notes both name your model, the witness the crossing names.
        knn = replace(_witness_for([R90_F05], recall_lb=0.9, fpr_ub=0.055, fpr_mean=0.07), kind="builtin_witness/v1")
        out = _recall_run((0.92, 0.95), (0.06, 0.03), {}, {"builtin_witness/v1:knn": knn})
        text = str(out)
        assert _section(text, "Notes") == [TANK_SCORED.replace("2.0%", "3.0%"), TANK_UNCERTIFIED]
        assert _table(text)[1][2] == "recall 92.0%, FP ≤6.0% (your model, not counted)"
        assert _steps(text)["Tank"].startswith("about ")
        assert "your model's upper confidence bound on false positives to cross" in _steps(text)["Tank"]
        assert "k-NN" not in text

    def test_str_recall_without_any_false_positive_bound(self):
        # No negatives in the audit: the tested model has a recall bound but no false-positive bound.
        witness = _witness_for([R90_F05], recall_lb=0.92, fpr_ub=0.03)
        key = R90_F05.canonical()
        reason = "no audit items labeled other than 'Tank'"
        fp = Endpoint(key, None, "upper", 0.005, "unavailable", reason, quantity="false_positive_rate")
        stats = {**witness.details["endpoints"], stats_key(key, "false_positive_rate"): None}
        bare = replace(witness, endpoints=(witness.endpoints[0], fp), details={"endpoints": stats})
        evidence = {"user_witness/v1": bare}
        row = _determination(R90_F05, decide(R90_F05, evidence, P))
        text = str(FeasibilityOutput(pl.DataFrame([row], schema=_SCHEMA), evidence))
        assert _table(text)[1][1:] == ["insufficient evidence ?", "—", "—"]
        assert _section(text, "Notes") == [
            "Tank: no tested model has an available upper confidence bound on false positives."
        ]
        assert "lower confidence bound on recall is available" not in text

    def test_str_names_recall_requirements_of_one_class_in_full(self):
        witness = _witness_for([R90_F05, R90_F10], recall_lb=0.7, recall_mean=0.75, fpr_ub=0.04, fpr_mean=0.02)
        evidence = {"user_witness/v1": witness}
        rows = [_determination(s, decide(s, evidence, P)) for s in (R90_F05, R90_F10)]
        text = str(FeasibilityOutput(pl.DataFrame(rows, schema=_SCHEMA), evidence))
        assert list(_steps(text)) == [
            "Tank: ≥90% recall, ≤5% false positives",
            "Tank: ≥90% recall, ≤10% false positives",
        ]

    @pytest.mark.parametrize(
        ("recall_bound", "fpr_bound", "short"),
        [
            (0.85, 0.04, "lower confidence bound on recall"),
            (0.92, 0.06, "upper confidence bound on false positives"),
            (0.85, 0.06, "lower confidence bound on recall and upper confidence bound on false positives"),
        ],
    )
    def test_str_recall_crossing_names_the_bounds_short_of_the_requirement(self, recall_bound, fpr_bound, short):
        out = _recall_run((recall_bound, 0.95), (fpr_bound, 0.03))
        row = out.determinations.row(0, named=True)
        n = row["suggested_minimum"]
        assert (row["limiting_factor"], n is not None) == ("sampling_width", True)
        assert _steps(str(out)) == {
            "Tank": f"about {n:,} evaluation images in all, for your model's {short} to cross if the observed rates "
            "hold"
        }
        assert f"its {short} would cross the requirement at about {n} audit-split units" in row["recommendation"]

    def test_aggregate_by_evidence_has_the_column_contract(self, gauss2_run):
        table = gauss2_run.aggregate_by_evidence()
        assert dict(table.schema) == {
            "source": pl.String,
            "metric": pl.Enum(["accuracy", "ap", "recall", "recall_at_fp", "mota"]),
            "quantity": pl.String,
            "threshold": pl.Float64,
            "bound_type": pl.Enum(["lower", "upper"]),
            "bound": pl.Float64,
            "observed": pl.Float64,
            "information_scope": pl.Enum(["full_input", "fixed_representation", "partition_restricted"]),
            "state": pl.String,
            "n": pl.Int64,
            "evidence_id": pl.String,
            "target_spec": pl.String,
        }
        assert list(table.schema) == [
            "source", "metric", "quantity", "threshold", "bound_type", "bound", "observed", "information_scope",
            "state", "n", "evidence_id", "target_spec",
        ]  # fmt: skip
        assert table["quantity"].to_list() == table["metric"].cast(pl.String).to_list()  # metric bounds only here
        assert set(gauss2_run._bounds()["unit"]) == {"image"}  # the printout's noun, not a reader column

    def test_aggregate_by_evidence_has_one_row_per_endpoint(self, gauss2_run):
        table = gauss2_run.aggregate_by_evidence()
        assert len(table) == sum(len(e.endpoints) for e in gauss2_run.evidence.values())
        assert not table.select("evidence_id", "target_spec", "quantity").is_duplicated().any()
        assert (
            table["target_spec"].unique(maintain_order=True).to_list()
            == gauss2_run.determinations["target_spec"].to_list()
        )

    def test_aggregate_by_evidence_matches_evidence(self, gauss2_run):
        for row in gauss2_run.aggregate_by_evidence().iter_rows(named=True):
            ev = gauss2_run.evidence[row["evidence_id"]]
            e = next(e for e in ev.endpoints if e.target_spec == row["target_spec"])
            assert (row["bound"], row["state"], row["bound_type"]) == (e.value, e.state, e.direction)
            assert row["information_scope"] == ev.information_scope
            target = _row(gauss2_run, row["target_spec"])
            assert (row["metric"], row["threshold"]) == (target["metric"], target["threshold"])

    def test_observed_is_null_for_ceilings(self, user_run):
        table = user_run.aggregate_by_evidence()
        assert table.filter(pl.col("bound_type") == "upper")["observed"].null_count() > 0
        assert table.filter((pl.col("bound_type") == "upper") & pl.col("observed").is_not_null()).is_empty()
        witnesses = table.filter((pl.col("bound_type") == "lower") & (pl.col("metric") == "accuracy"))
        assert witnesses.height == 3 * 2  # k-NN, probe and your model, on the two accuracy targets
        assert (witnesses["observed"] >= witnesses["bound"]).all()

    def test_source_names_are_readable(self, gauss2_run, user_run):
        assert {"built-in k-NN", "built-in linear probe", "group-based ceiling"} <= set(
            gauss2_run.aggregate_by_evidence()["source"]
        )
        assert "repeated labels" in set(gauss2_run.aggregate_by_evidence()["source"])
        users = user_run.aggregate_by_evidence().filter(pl.col("evidence_id") == "user_witness/v1")
        assert set(users["source"]) == {"your model"}

    def test_a_missing_stats_entry_gives_null_observed_and_n(self, user_run):
        spec = user_run.determinations["target_spec"][0]
        witness = Evidence(
            kind="user_witness/v1",
            supports="witness",
            basis="a basis",
            population="p",
            sample="s",
            information_scope="fixed_representation",
            unit="item",
            provenance={},
            endpoints=(Endpoint(spec, 0.7, "lower", 0.01, "informative"),),
            details={"endpoints": {spec: None}},
        )
        row = (
            FeasibilityOutput(user_run.determinations, {"user_witness/v1": witness})
            .aggregate_by_evidence()
            .row(0, named=True)
        )
        assert (row["observed"], row["n"], row["bound"]) == (None, None, 0.7)

    def test_bounds_name_each_bound_quantity(self):
        # An uncertified witness (false positives over the limit) keeps both of its measured bounds in the table.
        out = _recall_run((0.92, 0.95), (0.06, 0.03), {"replica_ceiling/v1": 0.97})
        table = out.aggregate_by_evidence()
        assert table.columns[:3] == ["source", "metric", "quantity"]
        assert table.select("source", "metric", "quantity", "bound_type", "bound", "observed", "n").rows() == [
            ("your model", "recall", "recall", "lower", 0.92, 0.95, 200),
            ("your model", "recall", "false_positive_rate", "upper", 0.06, 0.03, 800),
            ("repeated labels", "recall", "recall", "upper", 0.97, None, None),
        ]

    def test_unknown_source_kind_keeps_its_evidence_id(self):
        assert _source_name("x/v9", "x/v9") == "x/v9"

    def test_unit_noun(self):
        assert (_unit_noun("group"), _unit_noun("item")) == ("sequence", "image")

    def test_rows_stay_distinct_across_judged_scopes(self):
        x, y, _, _ = _gauss2(400)
        specs = [TargetSpec("IC", "accuracy", 0.6, scope=s) for s in ("full_input", "fixed_representation")]
        table = Feasibility().evaluate(_ds(y), target=specs, embeddings=x).aggregate_by_evidence()
        assert table["target_spec"].n_unique() == 2
        assert table["threshold"].n_unique() == 1
        assert not table.select("evidence_id", "target_spec").is_duplicated().any()

    def test_ap_targets_have_null_observed(self, user_run):
        ap = user_run.aggregate_by_evidence().filter(pl.col("metric") == "ap")
        assert ap.height > 0
        assert ap["observed"].null_count() == ap.height

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"embeddings": None}, ValueError, "extractor"),
            ({"embeddings": np.zeros((5, 2))}, ShapeMismatchError, "embeddings"),
            ({"look": 2}, ValueError, "sampling plan"),
            ({"allowances": {"delta": 0.1}}, ValueError, "'beta' and 'gamma'"),
            ({"allowances": {"beta": -0.1}}, ValueError, "non-negative"),
            ({"allowances": {"beta": [0.1, 0.2]}}, ValueError, "scalar in this release"),
            ({"allowances": {"gamma": 0.1}}, ValueError, "beta"),
            ({"allowances": {"beta": 0.1, "gamma": [0.1]}}, ValueError, "scalar"),
            # checked before the embedding pass, which would fail first on the missing extractor
            (
                {"embeddings": None, "predictions": P40, "provenance": {"training_item": [0]}},
                ValueError,
                "training_item",
            ),
            ({"embeddings": None, "predictions": P40, "provenance": {"reads": "pixels"}}, ValueError, "reads"),
            ({"embeddings": None, "predictions": P40, "provenance": {"reads": "partition"}}, ValueError, "partition"),
            ({"embeddings": None, "provenance": {"reads": "full_input"}}, ValueError, "predictions="),
            (
                {"embeddings": None, "predictions": P40, "provenance": {"decision_policy": "top class"}},
                ValueError,
                "1-D predictions= array",
            ),
            ({"embeddings": None, "predictions": np.zeros((3, 2))}, ValueError, "predictions"),
            ({"embeddings": None, "predictions": P40 * np.nan}, ValueError, "finite"),
            ({"embeddings": None, "replicas": {999: [0]}}, ValueError, "not a datum id"),
        ],
    )
    def test_arguments_are_checked_at_the_boundary(self, kwargs, error, match):
        x, y, _, _ = _gauss2(40)
        with pytest.raises(error, match=match):
            Feasibility().evaluate(_ds(y), target={"accuracy": 0.6}, **({"embeddings": x} | kwargs))

    def test_a_plan_beyond_the_audit_is_rejected(self):
        x, y, _, _ = _gauss2(40)
        with pytest.raises(ValueError, match="1000"):
            Feasibility().evaluate(
                _ds(y), target={"accuracy": 0.6}, embeddings=x, sampling=SamplingPlan((5, 10, 1000), order_seed=0)
            )

    def test_per_cell_beta_is_rejected(self):
        x, y, _, _ = _gauss2(40)
        with pytest.raises(ValueError, match="3 values"):
            Feasibility(n_cells=3).evaluate(
                _ds(y), target={"accuracy": 0.6}, embeddings=x, allowances={"beta": [0.1, 0.1, 0.1]}
            )

    def test_settings_and_class_names_are_checked_before_the_embedding_pass(self):
        x, y, _, _ = _gauss2(40)
        evaluator = Feasibility()
        evaluator.panel_size = 1  # past the Config check
        with pytest.raises(ValueError, match="panel_size must be at least 2"):
            evaluator.evaluate(_ds(y), target={"accuracy": 0.6})
        undeclared = _ds(y)
        undeclared.metadata.pop("index2label")
        for name in ("cat", "7"):  # no index2label: only an index below n_classes names a class
            with pytest.raises(ValueError, match=f"Cannot map class '{name}'"):
                Feasibility().evaluate(undeclared, target={f"ap:{name}": 0.6})

    def test_one_class_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2 classes"):
            Feasibility().evaluate(_ds([0] * 10, {0: "a"}), target={"accuracy": 0.6}, embeddings=np.zeros((10, 2)))
