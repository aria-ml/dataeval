"""Tests for dataeval.types module."""

from dataeval.types import SourceIndex


class TestSourceIndex:
    """Tests for SourceIndex class."""

    def test_source_index_repr_item_only(self):
        """Test __repr__ with only item."""
        si = SourceIndex(item=5)
        assert repr(si) == "SourceIndex(5)"

    def test_source_index_repr_with_target(self):
        """Test __repr__ with item and target."""
        si = SourceIndex(item=5, target=2)
        assert repr(si) == "SourceIndex(5, 2)"

    def test_source_index_repr_with_channel_no_target(self):
        """Test __repr__ with item, channel but no target (line 86-87)."""
        si = SourceIndex(item=5, channel=1)
        assert repr(si) == "SourceIndex(5, None, 1)"

    def test_source_index_repr_with_target_and_channel(self):
        """Test __repr__ with item, target, and channel."""
        si = SourceIndex(item=5, target=2, channel=1)
        assert repr(si) == "SourceIndex(5, 2, 1)"

    def test_source_index_str_item_only(self):
        """Test __str__ with only item."""
        si = SourceIndex(item=5)
        assert str(si) == "5"

    def test_source_index_str_with_target(self):
        """Test __str__ with item and target."""
        si = SourceIndex(item=5, target=2)
        assert str(si) == "5/2"

    def test_source_index_str_with_channel_no_target(self):
        """Test __str__ with item, channel but no target (line 97-98)."""
        si = SourceIndex(item=5, channel=1)
        assert str(si) == "5/-/1"

    def test_source_index_str_with_target_and_channel(self):
        """Test __str__ with item, target, and channel."""
        si = SourceIndex(item=5, target=2, channel=1)
        assert str(si) == "5/2/1"

    def test_source_index_equality(self):
        """Test equality comparison (line 129-140)."""
        si1 = SourceIndex(item=5, target=2, channel=1)
        si2 = SourceIndex(item=5, target=2, channel=1)
        si3 = SourceIndex(item=5, target=2)
        si4 = SourceIndex(item=6, target=2, channel=1)

        # Test equality
        assert si1 == si2
        assert si1 != si3  # Different channel
        assert si1 != si4  # Different item

        # Test with non-SourceIndex
        assert si1 != "5/2/1"
        assert si1 != 5
