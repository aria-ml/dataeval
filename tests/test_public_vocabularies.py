"""The vocabularies a caller needs to enumerate, and the array helpers downstreams use.

A caller can construct an :class:`~dataeval.types.Aggregator` but, until these were
exported, could not ask what ``how`` accepts — so a config layer wanting to reject a
misspelled reduction at declaration time had to hardcode the eighteen names and reach into
a private module to check them. The same held for ``every`` and ``epoch``.

These pin the export, and that each name is one the thing it configures actually takes.
"""

import pytest

from dataeval.types import (
    DATETIME_GRANULARITIES,
    EPOCH_UNITS,
    REDUCTION_NAMES,
    Aggregator,
    ParseDateTime,
)


@pytest.mark.required
class TestTheVocabulariesAreReachable:
    def test_reduction_names_are_exported(self):
        assert len(REDUCTION_NAMES) == 18

    def test_datetime_granularities_are_exported(self):
        assert len(DATETIME_GRANULARITIES) == 9

    def test_epoch_units_are_exported(self):
        assert len(EPOCH_UNITS) == 4

    def test_they_are_tuples_a_caller_can_hold(self):
        """A caller pins these into a schema, so they must not be a live view of a mapping."""
        for vocabulary in (REDUCTION_NAMES, DATETIME_GRANULARITIES, EPOCH_UNITS):
            assert isinstance(vocabulary, tuple)


@pytest.mark.required
class TestTheNamesAreTheOnesTheyConfigure:
    """An exported list that drifts from what the code accepts is worse than none: it makes
    a caller confident about a value the library then refuses."""

    def test_every_reduction_name_resolves_in_the_registry(self):
        from dataeval._metadata._reductions import lookup

        for how in REDUCTION_NAMES:
            lookup(how)

    def test_the_registry_holds_nothing_the_names_omit(self):
        from dataeval._metadata._reductions import REDUCTIONS

        assert set(REDUCTIONS) == set(REDUCTION_NAMES)

    def test_every_granularity_is_one_parse_datetime_takes(self):
        for period in DATETIME_GRANULARITIES:
            ParseDateTime("f", every=period)

    def test_every_epoch_unit_is_one_parse_datetime_takes(self):
        for unit in EPOCH_UNITS:
            ParseDateTime("f", epoch=unit)

    def test_an_aggregator_accepts_every_reduction_name(self):
        for how in REDUCTION_NAMES:
            assert Aggregator(how=how, source="unit", target="sequence").how == how


@pytest.mark.required
class TestTheArrayHelpersAreReachable:
    """Three helpers downstream projects were importing from ``utils._internal``, which
    moved to ``utils._array`` and broke them. Exported so the next move cannot."""

    def test_they_are_exported_from_utils(self):
        from dataeval.utils import as_numpy, flatten_samples, to_numpy

        assert all(callable(fn) for fn in (as_numpy, flatten_samples, to_numpy))

    def test_as_numpy_reads_a_sequence(self):
        import numpy as np

        from dataeval.utils import as_numpy

        np.testing.assert_array_equal(as_numpy([1, 2, 3]), np.array([1, 2, 3]))

    def test_flatten_samples_keeps_the_leading_axis(self):
        import numpy as np

        from dataeval.utils import flatten_samples

        assert flatten_samples(np.zeros((4, 2, 3))).shape == (4, 6)
