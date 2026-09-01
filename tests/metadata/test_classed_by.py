"""The class axis a caller defines, and what a result has to say about it.

``class_labels`` is the dataset's own labels until :meth:`~dataeval.Metadata.classed_by`
makes it something else. These pin both halves: that a pivot answers the same question
``label=`` already answered, and that nothing downstream can mistake a derived axis for
ground truth.
"""

import random

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._helpers import resolve_label_axis
from dataeval.bias import Balance, Diversity, Parity

EVALUATORS = (Balance, Diversity, Parity)


def _od(get_od_dataset, items: int = 60, targets: int = 3, **kwargs) -> Metadata:
    random.seed(0)
    metadata = [
        {"weather": random.choice(["rain", "clear"]), "alt": random.choice([10.0, 20.0, 30.0])} for _ in range(items)
    ]
    return Metadata(get_od_dataset(items, targets_per_image=targets, metadata=metadata), **kwargs)


def _flat(items: int = 200) -> Metadata:
    rng = np.random.default_rng(0)
    return Metadata.from_factors(
        {
            "weather": np.where(rng.random(items) < 0.5, "rain", "clear"),
            "shift": np.where(rng.random(items) < 0.5, "day", "night"),
            "alt": rng.integers(0, 4, items),
        },
        class_labels=rng.integers(0, 3, items),
        index2label={0: "cat", 1: "dog", 2: "bird"},
    )


@pytest.mark.required
class TestTheAxisMoves:
    def test_class_labels_become_the_factor_codes(self):
        pivot = _flat().classed_by("weather")
        assert sorted(set(pivot.class_labels.tolist())) == [0, 1]
        assert sorted(pivot.index2label.values()) == ["clear", "rain"]

    def test_the_axis_factor_leaves_the_factor_set(self):
        """A factor left in place correlates perfectly with itself and reports 1.0."""
        pivot = _flat().classed_by("weather")
        assert "weather" not in pivot.factor_names
        assert "alt" in pivot.factor_names

    def test_a_composite_axis_crosses_its_factors(self):
        pivot = _flat().classed_by("weather", "shift")
        assert pivot.class_axis == "weather × shift"
        assert len(set(pivot.class_labels.tolist())) == 4
        assert all(" × " in name for name in pivot.index2label.values())

    def test_the_source_is_left_alone(self):
        metadata = _flat()
        metadata.classed_by("weather")
        assert metadata.class_axis == "class_label"
        assert "weather" in metadata.factor_names

    def test_the_axis_is_disclosed_by_repr(self):
        assert "classed_by=['weather']" in repr(_flat().classed_by("weather"))


@pytest.mark.required
class TestTheClassLabelBecomesAFactor:
    def test_it_joins_the_factor_set(self):
        pivot = _flat().classed_by("weather")
        assert "class" in pivot.factor_names

    def test_it_is_named_from_index2label(self):
        pivot = _flat().classed_by("weather")
        assert sorted(pivot.code_names("class").values()) == ["bird", "cat", "dog"]

    def test_it_is_not_dropped_as_an_identifier(self):
        """One instance per class reads as near-unique, and is still a class."""
        metadata = Metadata.from_factors(
            {"weather": np.array(["rain", "clear"] * 6)},
            class_labels=np.arange(12),
            index2label={i: f"class_{i}" for i in range(12)},
        )
        assert "class" in metadata.classed_by("weather").factor_names

    def test_it_is_absent_above_the_label_level(self, get_od_dataset):
        """A frame has no single class, and inventing one is what class_labels refuses."""
        pivot = _od(get_od_dataset).at("unit").classed_by("weather")
        assert "class" not in pivot.factor_names
        assert pivot.class_labels.shape[0] == pivot.level_counts["unit"]

    def test_it_is_what_makes_the_association_measurable(self):
        """The first question anybody asks of a pivot: is my axis a proxy for the class?"""
        result = Balance().evaluate(_flat().classed_by("weather"))
        assert "class" in result.balance["factor_name"].to_list()


@pytest.mark.required
class TestItAgreesWithLabel:
    """A pivot and the equivalent ``label=`` are the same question."""

    @pytest.mark.parametrize("evaluator", EVALUATORS)
    def test_the_shared_factors_score_the_same(self, evaluator, get_od_dataset):
        metadata = _od(get_od_dataset)
        named = evaluator(label="weather").evaluate(metadata).data()
        pivoted = evaluator().evaluate(metadata.classed_by("weather")).data()
        frames = [(v, pivoted[k]) for k, v in named.items() if isinstance(v, pl.DataFrame)]
        assert frames
        compared = 0
        for left, right in frames:
            keys = [c for c in left.columns if not left.schema[c].is_numeric()]
            numeric = [c for c in left.columns if left.schema[c].is_numeric()]
            if not numeric or not keys:
                continue
            cast = [pl.col(c).cast(pl.String) for c in keys]
            l_rows = left.with_columns(cast).rows_by_key(keys, named=True)
            r_rows = right.with_columns(cast).rows_by_key(keys, named=True)
            for key in set(l_rows) & set(r_rows):
                for column in numeric:
                    assert l_rows[key][0][column] == pytest.approx(r_rows[key][0][column]), (key, column)
                    compared += 1
        assert compared

    def test_resolution_reports_the_same_axis(self, get_od_dataset):
        metadata = _od(get_od_dataset)
        named = resolve_label_axis(metadata, "weather")
        pivoted = resolve_label_axis(metadata.classed_by("weather"), None)
        assert np.array_equal(named.values, pivoted.values)
        assert dict(named.names) == dict(pivoted.names)
        assert named.label == pivoted.label == "weather"
        assert named.source == pivoted.source == "derived"


@pytest.mark.required
class TestTheDefaultIsUnchanged:
    @pytest.mark.parametrize("evaluator", EVALUATORS)
    def test_an_unpivoted_run_is_untouched(self, evaluator, get_od_dataset):
        metadata = _od(get_od_dataset)
        assert evaluator().evaluate(metadata) is not None
        assert "class" not in metadata.factor_names

    def test_the_ground_truth_axis_names_itself(self, get_od_dataset):
        metadata = _od(get_od_dataset)
        assert metadata.class_axis == "class_label"
        assert metadata.class_axis_source == "ground_truth"
        assert metadata.class_axis_info.level == metadata.label_level

    def test_the_record_answers_where_class_labels_refuses(self, get_od_dataset):
        """class_axis_info must be cheap and total, since every evaluation reads it."""
        coarse = _od(get_od_dataset).at("unit")
        with pytest.raises(ValueError, match="class_labels is defined at"):
            _ = coarse.class_labels
        assert coarse.class_axis_info.source == "ground_truth"


@pytest.mark.required
class TestTheAxisSurvivesDerivation:
    def test_at(self, get_od_dataset):
        pivot = _od(get_od_dataset).classed_by("weather")
        moved = pivot.at("unit")
        assert moved.class_axis == "weather"
        assert moved.class_labels.shape[0] == moved.level_counts["unit"]

    def test_where(self, get_od_dataset):
        pivot = _od(get_od_dataset).classed_by("weather")
        kept = pivot.where(pl.col("alt") > 10.0, level="unit")
        assert kept.class_axis == "weather"
        assert kept.class_labels.shape[0] == kept.factor_data.shape[0]

    def test_having(self, get_od_dataset):
        pivot = _od(get_od_dataset).classed_by("weather")
        assert pivot.having(pl.col("class_label") == 1, level="instance").class_axis == "weather"

    def test_agg(self, get_od_dataset):
        pivot = _od(get_od_dataset).at("unit").classed_by("weather")
        rolled = pivot.agg("instance", "unit", pl.len().alias("n_detections"))
        assert rolled.class_axis == "weather"
        assert "n_detections" in rolled.factor_names

    def test_it_is_not_written_by_save(self, get_od_dataset, tmp_path):
        """Like view, include and exclude: how a reader asks, not what the rows are."""
        pivot = _od(get_od_dataset).classed_by("weather")
        pivot.save(tmp_path / "md.zip")
        restored = Metadata.load(tmp_path / "md.zip", dataset=pivot._dataset)
        assert restored.class_axis == "class_label"
        assert restored.class_axis_source == "ground_truth"


@pytest.mark.required
class TestFanOut:
    def test_a_coarse_axis_is_replicated_onto_finer_rows(self, get_od_dataset):
        metadata = _od(get_od_dataset)
        instance = metadata.classed_by("weather")
        unit = metadata.at("unit").classed_by("weather")
        assert instance.class_labels.shape[0] == 3 * unit.class_labels.shape[0]

    def test_the_fan_out_is_reported_rather_than_hidden(self, get_od_dataset):
        metadata = _od(get_od_dataset)
        assert metadata.classed_by("weather").class_axis_info.rows_per_group_entity == pytest.approx(3.0)
        assert metadata.at("unit").classed_by("weather").class_axis_info.rows_per_group_entity == pytest.approx(1.0)

    def test_the_axis_level_is_reported(self, get_od_dataset):
        assert _od(get_od_dataset).classed_by("weather").class_axis_level == "unit"

    def test_inherited_false_refuses_a_coarse_axis(self, get_od_dataset):
        """`inherited` is the existing declaration that ancestor values stay off these rows."""
        metadata = _od(get_od_dataset, inherited=False)
        with pytest.raises(ValueError, match="inherited=False"):
            metadata.classed_by("weather")


@pytest.mark.required
class TestVocabulary:
    def test_a_declared_alphabet_is_reported_as_declared(self):
        rng = np.random.default_rng(0)
        metadata = Metadata.from_factors(
            {"weather": np.where(rng.random(60) < 0.5, "rain", "clear"), "alt": rng.integers(0, 3, 60)},
            class_labels=rng.integers(0, 2, 60),
            factor_levels={"weather": ["clear", "fog", "rain"]},
        )
        assert metadata.classed_by("weather").class_axis_info.vocabulary == "declared"

    def test_an_observed_alphabet_says_so(self):
        assert _flat().classed_by("weather").class_axis_info.vocabulary == "observed"

    def test_a_composite_is_always_observed(self):
        assert _flat().classed_by("weather", "shift").class_axis_info.vocabulary == "observed"


@pytest.mark.required
class TestRefusals:
    def test_naming_nothing(self):
        with pytest.raises(ValueError, match="names the factor"):
            _flat().classed_by()

    def test_naming_an_unknown_factor(self):
        with pytest.raises(ValueError, match="not among this metadata's factors"):
            _flat().classed_by("nope")

    def test_naming_every_factor_leaves_the_promoted_class_label(self):
        """The class label is a factor of the pivoted instance, so it is what remains."""
        pivot = _flat().classed_by("weather", "shift", "alt")
        assert list(pivot.factor_names) == ["class"]

    def test_naming_every_factor_with_nothing_to_promote(self, get_od_dataset):
        """Above the label level there is no class per row, so nothing takes their place."""
        coarse = _od(get_od_dataset).at("unit")
        with pytest.raises(ValueError, match="nothing left to measure"):
            coarse.classed_by(*coarse.factor_names)

    def test_pivoting_twice(self):
        with pytest.raises(ValueError, match="already classed by"):
            _flat().classed_by("weather").classed_by("shift")

    def test_label_against_a_pivot_is_two_answers_to_one_question(self):
        with pytest.raises(ValueError, match="already classed by"):
            Balance(label="shift").evaluate(_flat().classed_by("weather"))


@pytest.mark.required
class TestTraceability:
    """A result has to be able to say which variable produced it."""

    @pytest.mark.parametrize("evaluator", EVALUATORS)
    def test_the_ground_truth_default_names_itself(self, evaluator):
        result = evaluator().evaluate(_flat())
        assert result.class_axis is not None
        assert result.class_axis.name == "class_label"
        assert result.class_axis.source == "ground_truth"
        assert result.meta().state["class_axis_source"] == "ground_truth"

    @pytest.mark.parametrize("evaluator", EVALUATORS)
    def test_a_pivot_is_recorded_even_though_label_is_none(self, evaluator):
        """The regression this exists to prevent: `state` reads the evaluator, not the metadata."""
        result = evaluator().evaluate(_flat().classed_by("weather"))
        assert result.meta().state["class_axis"] == "weather"
        assert result.meta().state["class_axis_source"] == "derived"

    @pytest.mark.parametrize("evaluator", EVALUATORS)
    def test_label_records_the_effective_axis(self, evaluator):
        """`label=` names an axis the evaluator chose; the metadata still says class_label."""
        metadata = _flat()
        result = evaluator(label="weather").evaluate(metadata)
        assert metadata.class_axis == "class_label"
        assert result.class_axis is not None
        assert result.class_axis.name == "weather"
        assert result.meta().state["class_axis_source"] == "derived"

    def test_the_record_survives_the_state_formatter(self):
        """`set_metadata` flattens anything that is not a scalar, so these must be strings."""
        state = Balance().evaluate(_flat()).meta().state
        for key in ("class_axis", "class_axis_source", "class_axis_level"):
            assert isinstance(state[key], str)
            assert state[key] not in {"NoneType", "ClassAxis"}


@pytest.mark.required
class TestConsumersOutsideBias:
    def test_coverage_follows_the_axis_and_says_so(self):
        from dataeval.scope import Coverage

        rng = np.random.default_rng(0)
        metadata, embeddings = _flat(), rng.normal(size=(200, 8))
        pivoted = Coverage().evaluate(metadata.classed_by("weather"), embeddings=embeddings)
        assert sorted(pivoted.data()["class"].to_list()) == ["clear", "rain"]
        assert pivoted.class_axis is not None
        assert pivoted.class_axis.source == "derived"
        plain = Coverage().evaluate(metadata, embeddings=embeddings)
        assert plain.class_axis is not None
        assert plain.class_axis.source == "ground_truth"

    def test_split_dataset_stratifies_on_the_axis_and_says_so(self):
        from dataeval.data import split_dataset

        splits = split_dataset(_flat().classed_by("weather"), num_folds=2, stratify=True)
        assert splits.class_axis is not None
        assert splits.class_axis.name == "weather"
        plain = split_dataset(_flat(), num_folds=2, stratify=True)
        assert plain.class_axis is not None
        assert plain.class_axis.source == "ground_truth"

    def test_representation_refuses_a_derived_axis(self):
        """Its label names are resolved against an ontology, and `rain` is not a concept."""
        from dataeval import Ontology
        from dataeval.scope import Representation
        from dataeval.types import OntologyConcept

        ontology = Ontology([OntologyConcept(id=name, label=name) for name in ("cat", "dog", "bird")])
        with pytest.raises(ValueError, match="classed by 'weather'"):
            Representation(ontology).evaluate(_flat().classed_by("weather"))

    def test_outliers_aggregate_by_class_follows_the_axis(self):
        """It reads class_labels and index2label, so a pivot reaches it with no change."""
        from dataeval.quality import OutliersOutput

        pivot = _flat().classed_by("weather")
        frame = pl.DataFrame([
            {"item_index": index, "target_index": None, "metric_name": "contrast", "metric_value": 1.0}
            for index in range(8)
        ])
        summary = OutliersOutput(frame).aggregate_by_class(pivot)
        assert set(summary["class_name"].to_list()) <= {"clear", "rain", "Total"}
