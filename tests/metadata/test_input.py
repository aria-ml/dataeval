"""Unpacking a whole statistics result into factors, placement and declarations."""

from __future__ import annotations

import numpy as np
import pytest

from dataeval._metadata._input import _is_stats_result, unpack_stats_result
from dataeval.types import Aggregator, SourceIndex


@pytest.mark.required
class TestIsStatsResult:
    def test_recognizes_a_result_without_a_source_index(self):
        # Keyed producers place by level and key, so they carry no addresses.
        assert _is_stats_result({"stats": {"pan_speed": [1.0]}})

    def test_recognizes_a_result_with_a_source_index(self):
        result = {"stats": {"brightness": [1.0]}, "source_index": [SourceIndex(0, None, None)]}
        assert _is_stats_result(result)

    def test_rejects_a_plain_factor_mapping(self):
        assert not _is_stats_result({"brightness": np.array([1.0])})

    def test_rejects_a_factor_literally_named_stats(self):
        # A factor mapping may hold a column called `stats`; its values are not a mapping.
        assert not _is_stats_result({"stats": np.array([1.0, 2.0])})

    def test_rejects_a_malformed_source_index(self):
        assert not _is_stats_result({"stats": {"a": [1]}, "source_index": "not-a-sequence"})


@pytest.mark.required
class TestUnpackStatsResult:
    def test_returns_declarations_when_present(self):
        declared = (Aggregator("mean", "unit", "sequence", ("pan_speed",)),)
        factors, index, aggregations = unpack_stats_result(
            {"stats": {"pan_speed": [1.0]}, "aggregations": declared}, None
        )
        assert factors == {"pan_speed": [1.0]}
        assert index is None
        assert aggregations == declared

    def test_returns_an_empty_tuple_when_absent(self):
        _, _, aggregations = unpack_stats_result({"stats": {"pan_speed": [1.0]}}, None)
        assert aggregations == ()

    def test_a_plain_mapping_passes_through_with_no_declarations(self):
        factors = {"brightness": np.array([1.0])}
        got, index, aggregations = unpack_stats_result(factors, None)
        assert got is factors
        assert index is None
        assert aggregations == ()

    def test_bookkeeping_keys_are_not_factors(self):
        factors, _, _ = unpack_stats_result(
            {"stats": {"brightness": [1.0]}, "image_count": 1, "object_count": [2]}, None
        )
        assert set(factors) == {"brightness"}

    def test_level_alongside_a_source_index_is_refused(self):
        result = {"stats": {"a": [1]}, "source_index": [SourceIndex(0, None, None)]}
        with pytest.raises(ValueError, match="mutually exclusive"):
            unpack_stats_result(result, None, level="unit")

    def test_level_is_allowed_when_the_result_carries_no_addresses(self):
        # A keyed producer needs the caller to name the level; that is not a contradiction.
        factors, index, _ = unpack_stats_result({"stats": {"a": [1]}}, None, level="unit")
        assert factors == {"a": [1]}
        assert index is None
