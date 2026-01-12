"""Tests for dataeval.types module."""

import pytest

from dataeval.types import MappingOutput, SequenceOutput, SourceIndex


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

    def test_from_string_item_only(self):
        """Test from_string with only item."""
        si = SourceIndex.from_string("0")
        assert si == SourceIndex(0)

    def test_from_string_with_target(self):
        """Test from_string with item and target."""
        si = SourceIndex.from_string("0/3")
        assert si == SourceIndex(0, 3)

    def test_from_string_with_none_target(self):
        """Test from_string with item, None target, and channel (line 134)."""
        si = SourceIndex.from_string("0/-/1")
        assert si == SourceIndex(0, None, 1)

    def test_from_string_with_all_fields(self):
        """Test from_string with item, target, and channel."""
        si = SourceIndex.from_string("0/3/1")
        assert si == SourceIndex(0, 3, 1)

    def test_from_string_with_none_channel(self):
        """Test from_string with item, target, and None channel (line 136)."""
        si = SourceIndex.from_string("0/3/-")
        assert si == SourceIndex(0, 3, None)

    def test_from_string_invalid_too_many_parts(self):
        """Test from_string with too many parts (line 137-138)."""
        with pytest.raises(ValueError, match="Invalid SourceIndex string format"):
            SourceIndex.from_string("0/1/2/3")


class TestMappingOutput:
    """Tests for MappingOutput class."""

    def test_getitem(self):
        """Test __getitem__ method (line 247)."""
        data = {"a": 1, "b": 2, "c": 3}
        output = MappingOutput(data)
        assert output["a"] == 1
        assert output["b"] == 2
        assert output["c"] == 3

    def test_iter(self):
        """Test __iter__ method (line 250)."""
        data = {"a": 1, "b": 2, "c": 3}
        output = MappingOutput(data)
        keys = list(output)
        assert keys == ["a", "b", "c"]


class TestSequenceOutput:
    """Tests for SequenceOutput class."""

    def test_getitem_int(self):
        """Test __getitem__ with int index (line 263)."""
        data = [10, 20, 30, 40]
        output = SequenceOutput(data)
        assert output[0] == 10
        assert output[2] == 30
        assert output[-1] == 40

    def test_getitem_slice(self):
        """Test __getitem__ with slice (line 263)."""
        data = [10, 20, 30, 40]
        output = SequenceOutput(data)
        assert output[1:3] == [20, 30]
        assert output[:2] == [10, 20]

    def test_iter(self):
        """Test __iter__ method (line 266)."""
        data = [10, 20, 30, 40]
        output = SequenceOutput(data)
        result = list(output)
        assert result == [10, 20, 30, 40]
