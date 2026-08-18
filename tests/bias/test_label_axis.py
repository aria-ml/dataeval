"""Bias analysis at every level, and the label axis that makes it reachable.

``class_labels`` is defined only at :attr:`~dataeval.Metadata.label_level` and raises
above it, by design — a frame, a track or a sequence has no single class label. Every
bias evaluator reads it, so before ``label=`` existed none of them ran at any coarser
view, and the level machinery was unreachable from the evaluators. These pin both
halves: that the default still refuses, and that naming a factor gets through.
"""

import random

import numpy as np
import pytest

from dataeval import Metadata
from dataeval._helpers import factors_excluding, resolve_label_axis
from dataeval.bias import Balance, Diversity, Parity

EVALUATORS = (Balance, Diversity, Parity)


def _metadata(get_od_dataset, items: int = 60, targets: int = 3) -> Metadata:
    random.seed(0)
    metadata = [
        {"weather": random.choice(["rain", "clear"]), "alt": random.choice([10.0, 20.0, 30.0])} for _ in range(items)
    ]
    return Metadata(get_od_dataset(items, targets_per_image=targets, metadata=metadata))


class _BareMetadata:
    """The four-member protocol and nothing else — no levels, no rows_at."""

    factor_names = ["a", "b"]
    factor_data = np.array([[0, 1], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0]])
    class_labels = np.array([0, 1, 0, 1, 0, 1])
    is_binned = [False, False]


class _SingleFactorMetadata(_BareMetadata):
    """One factor and nothing else, so conditioning on it leaves nothing to measure."""

    factor_names = ["a"]
    factor_data = np.array([[0], [1], [0], [1], [0], [1]])
    is_binned = [False]


@pytest.mark.required
class TestLabelDefaultsToClassLabels:
    """label=None must behave exactly as before it existed."""

    @pytest.mark.parametrize("evaluator", EVALUATORS)
    def test_runs_at_the_label_level(self, evaluator, get_od_dataset):
        metadata = _metadata(get_od_dataset)
        assert evaluator().evaluate(metadata.at(metadata.label_level)) is not None

    @pytest.mark.parametrize("evaluator", EVALUATORS)
    def test_still_refuses_above_the_label_level(self, evaluator, get_od_dataset):
        # Refusing is correct: there is no single class label per unit row, and the
        # alternative is handing an evaluator labels that do not match its factor rows.
        with pytest.raises(ValueError, match="class_labels is defined at the 'instance' level"):
            evaluator().evaluate(_metadata(get_od_dataset).at("unit"))


@pytest.mark.required
class TestLabelReachesEveryLevel:
    """Naming a factor is what makes a coarser view analysable at all."""

    @pytest.mark.parametrize("evaluator", EVALUATORS)
    @pytest.mark.parametrize("level", ["unit", "instance"])
    def test_named_factor_runs_at_every_level(self, evaluator, level, get_od_dataset):
        metadata = _metadata(get_od_dataset).at(level)
        result = evaluator(label="weather").evaluate(metadata)
        assert result is not None

    def test_row_counts_follow_the_view(self, get_od_dataset):
        metadata = _metadata(get_od_dataset)
        assert metadata.at("unit").factor_data.shape[0] == metadata.level_counts["unit"]
        assert metadata.at("instance").factor_data.shape[0] == metadata.level_counts["instance"]

    def test_the_axis_factor_is_excluded_from_the_factors(self, get_od_dataset):
        """A factor left in place would report perfect correlation with itself."""
        metadata = _metadata(get_od_dataset).at("unit")
        axis = resolve_label_axis(metadata, "weather")
        _, names, _ = factors_excluding(metadata, axis.excluded)
        assert "weather" not in names
        assert set(names) == set(metadata.factor_names) - {"weather"}


@pytest.mark.required
class TestLabelAxisResolution:
    def test_none_is_the_class_labels(self, get_od_dataset):
        metadata = _metadata(get_od_dataset)
        axis = resolve_label_axis(metadata, None)
        assert np.array_equal(axis.values, metadata.class_labels)
        assert axis.label == "class_label"
        assert axis.excluded == ()

    def test_a_named_factor_is_densely_numbered(self, get_od_dataset):
        axis = resolve_label_axis(_metadata(get_od_dataset).at("unit"), "alt")
        assert set(axis.values.tolist()) == set(range(len(set(axis.values.tolist()))))

    def test_groups_are_named_by_their_own_values(self, get_od_dataset):
        axis = resolve_label_axis(_metadata(get_od_dataset).at("unit"), "weather")
        assert sorted(axis.names.values()) == ["clear", "rain"]

    def test_several_factors_combine_into_one_axis(self, get_od_dataset):
        axis = resolve_label_axis(_metadata(get_od_dataset).at("unit"), ["weather", "alt"])
        assert axis.label == "weather × alt"
        assert len(axis.excluded) == 2
        assert all(" × " in name for name in axis.names.values())

    def test_a_bare_metadata_like_falls_back_to_bin_indices(self):
        # No rows_at, so the pre-binning values are unreachable and the codes are the
        # only honest answer. The protocol stays four members.
        axis = resolve_label_axis(_BareMetadata(), "a")
        assert sorted(axis.names.values()) == ["0", "1"]

    def test_a_bare_metadata_like_can_still_be_evaluated(self):
        assert Balance(label="a").evaluate(_BareMetadata()) is not None

    def test_unknown_factor_names_are_rejected(self, get_od_dataset):
        with pytest.raises(ValueError, match="are not among this metadata's factors"):
            resolve_label_axis(_metadata(get_od_dataset), "nope")

    def test_an_empty_sequence_is_rejected(self, get_od_dataset):
        with pytest.raises(ValueError, match="empty sequence names none"):
            resolve_label_axis(_metadata(get_od_dataset), [])

    def test_a_continuous_factors_groups_are_named_by_their_span(self):
        """A bin covers a range, so naming it after one member would read as an exact value."""
        rng = np.random.default_rng(0)
        metadata = Metadata.from_factors({"alt": rng.normal(50, 10, 200)}, class_labels=rng.integers(0, 3, 200))
        assert metadata.is_discrete[metadata.factor_names.index("alt")] is False
        axis = resolve_label_axis(metadata, "alt")
        assert all(name.startswith("[") and ", " in name for name in axis.names.values())

    def test_an_axis_naming_every_factor_is_rejected(self):
        # Every evaluator checks that the metadata has factors before the axis is
        # resolved, so an axis that consumes all of them would otherwise reach the
        # statistics as an empty matrix and report on nothing.
        with pytest.raises(ValueError, match="names every factor this metadata has"):
            Balance(label="a").evaluate(_SingleFactorMetadata())


@pytest.mark.required
class TestLabelAxisIsConfigurable:
    """label rides the same Config/state machinery as every other setting."""

    @pytest.mark.parametrize("evaluator", EVALUATORS)
    def test_label_is_accepted_via_config(self, evaluator):
        assert evaluator(config=evaluator.Config(label="weather")).label == "weather"

    @pytest.mark.parametrize("evaluator", EVALUATORS)
    def test_label_defaults_to_none(self, evaluator):
        assert evaluator().label is None
