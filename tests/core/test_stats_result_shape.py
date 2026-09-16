"""The shape every statistics producer returns."""

from __future__ import annotations

from typing import Any, get_type_hints

import pytest

from dataeval.core import FactorResult, StatsResult
from dataeval.types import Aggregator


@pytest.mark.required
class TestFactorResultShape:
    """The base every producer returns: measurements, and optionally how they roll up."""

    def test_is_generic(self):
        # Parameterizing must not raise; this is what per-producer factor types need.
        assert FactorResult[dict] is not None

    def test_declares_stats_and_optional_aggregations(self):
        hints = get_type_hints(FactorResult, include_extras=True)
        assert "stats" in hints
        assert "aggregations" in hints

    def test_bare_use_is_still_valid(self):
        # Existing annotations spelled without a parameter keep working.
        result: FactorResult[Any] = {"stats": {"brightness": [1.0]}}
        assert result["stats"]["brightness"] == [1.0]

    def test_carries_declarations(self):
        declared = (Aggregator("mean", "unit", "sequence", ("brightness",)),)
        result: FactorResult[Any] = {"stats": {"brightness": [1.0]}, "aggregations": declared}
        assert result["aggregations"][0].how == "mean"

    def test_stats_alone_is_enough(self):
        """A keyed producer declares no addresses, so nothing beyond `stats` is required."""
        assert FactorResult.__required_keys__ == frozenset({"stats"})


@pytest.mark.required
class TestStatsResultShape:
    """The addressed subtype, which guarantees every value can be placed and combined."""

    def test_extends_the_base(self):
        """Everything the base promises, the addressed subtype promises too."""
        assert FactorResult.__required_keys__ <= StatsResult.__required_keys__
        assert "stats" in get_type_hints(StatsResult, include_extras=True)

    def test_requires_its_address_keys(self):
        """`compute_stats` and `compute_ratios` always set these, and everything that
        offsets or concatenates results reads them -- so the type promises them rather
        than making each consumer re-check.
        """
        assert StatsResult.__required_keys__ == frozenset({
            "stats",
            "source_index",
            "object_count",
            "invalid_box_count",
            "image_count",
        })

    def test_still_carries_declarations(self):
        assert "aggregations" in StatsResult.__optional_keys__
