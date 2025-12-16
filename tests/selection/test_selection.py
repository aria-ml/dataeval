from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.selection._classbalance import ClassBalance
from dataeval.selection._classfilter import ClassFilter
from dataeval.selection._indices import Indices
from dataeval.selection._limit import Limit
from dataeval.selection._reverse import Reverse
from dataeval.selection._select import Select
from dataeval.selection._shuffle import Shuffle
from dataeval.types import SourceIndex


def one_hot(label: int):
    oh = np.zeros(3)
    oh[label] = 1
    return oh


@pytest.fixture(scope="module")
def mock_dataset():
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 10
    mock_dataset.__getitem__.side_effect = lambda idx: (idx, one_hot(idx % 3), {"id": idx})
    return mock_dataset


@pytest.mark.required
class TestSelectionClasses:
    def test_classfilter(self, mock_dataset):
        # Test ClassFilter classes
        class_filter = ClassFilter(classes=(0, 1))
        select = Select(mock_dataset, selections=[class_filter])
        assert len(select) == 7
        counts = {0: 0, 1: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] = counts[label] + 1
        assert counts == {0: 4, 1: 3}
        assert "ClassFilter(classes=(0, 1)" in str(select)

    def test_classbalance(self, mock_dataset):
        # Test ClassBalance with interclass method
        class_balance = ClassBalance(method="interclass")
        select = Select(mock_dataset, selections=[class_balance])
        # Dataset has 10 images (classes 0,1,2,0,1,2,0,1,2,0)
        # interclass should balance them as 4,3,3 or similar
        assert len(select) == 10
        counts = {0: 0, 1: 0, 2: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] = counts[label] + 1
        # Check that all classes are represented
        assert all(count > 0 for count in counts.values())
        assert "ClassBalance(" in str(select)

    def test_classfilter_and_balance(self, mock_dataset):
        # Test ClassFilter balance
        class_filter = ClassFilter(classes=[0, 1])
        class_balance = ClassBalance(method="interclass")
        select = Select(mock_dataset, selections=[class_filter, class_balance])
        # After filtering and balancing, check that we get results
        assert len(select) > 0
        counts = {0: 0, 1: 0, 2: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] += 1
        # Check that classes 0 and 1 are present
        assert counts[0] > 0 and counts[1] > 0
        assert "ClassFilter(classes=[0, 1]" in str(select)
        assert "ClassBalance(" in str(select)

    def test_classfilter_with_unsupported_target(self):
        class MockTarget:
            def __init__(self, label: int):
                self.label = label

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = lambda idx: (f"data_{idx}", MockTarget(idx), {"id": idx})

        class_filter = ClassFilter(classes=[0])
        with pytest.raises(TypeError):
            Select(mock_dataset, selections=[class_filter])

    def test_classbalance_with_unsupported_target(self):
        class MockTarget:
            def __init__(self, label: int):
                self.label = label

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = lambda idx: (f"data_{idx}", MockTarget(idx), {"id": idx})

        class_balance = ClassBalance(method="interclass")
        # Unsupported target types are treated as empty images (no error)
        select = Select(mock_dataset, selections=[class_balance])
        assert len(select) == 0  # All images are treated as empty, no classes to balance

    def test_classfilter_with_nothing(self, mock_dataset):
        # Test ClassFilter with no params
        class_filter = ClassFilter([])
        select = Select(mock_dataset, selections=class_filter)
        assert len(select) == 10

    def test_classfilter_and_balance_with_limit(self, mock_dataset):
        # Test ClassFilter balance with limit
        class_filter = ClassFilter(classes=[0, 1])
        class_balance = ClassBalance(method="interclass")
        limit = Limit(size=5)
        select = Select(mock_dataset, selections=[limit, class_filter, class_balance])
        # After limit, filter, and balance, check we get results
        assert len(select) > 0 and len(select) <= 5
        counts = {0: 0, 1: 0, 2: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] += 1
        # Check that at least one of classes 0 or 1 is present
        assert counts[0] > 0 or counts[1] > 0
        assert "ClassBalance(" in str(select)
        assert "ClassFilter(classes=[0, 1]" in str(select)
        assert "Limit(size=5)" in str(select)

    def test_limit(self, mock_dataset):
        # Test Limit
        limit = Limit(size=5)
        select = Select(mock_dataset, selections=[limit])
        assert len(select) == 5
        assert "Limit(size=5)" in str(select)

    def test_reverse(self, mock_dataset):
        # Test Reverse
        reverse = Reverse()
        select = Select(mock_dataset, selections=[reverse])
        expected_order = list(range(9, -1, -1))
        for i, (data, _, _) in enumerate(select):
            assert data == expected_order[i]
        assert "Reverse()" in str(select)

    def test_shuffle(self, mock_dataset):
        # Test Shuffle
        shuffle = Shuffle(seed=0)
        select = Select(mock_dataset, selections=[shuffle])
        # Since shuffle is random, we just check if the length is correct
        assert len(select) == 10
        # Check if the shuffled order is not the same as the original order
        original_order = [f"data_{i}" for i in range(10)]
        shuffled_order = [data for data, _, _ in select]
        assert original_order != shuffled_order
        assert "Shuffle(seed=0)" in str(select)

    def test_indices(self, mock_dataset):
        indices = Indices([12, 10, 8, 6, 4, 2, 0])
        select = Select(mock_dataset, indices)
        assert len(select) == 5
        assert select._selection == [8, 6, 4, 2, 0]
        assert "Indices(indices=[12, 10, 8, 6, 4, 2, 0])" in str(select)

    def test_indices_repeats(self, mock_dataset):
        indices = Indices([12, 12, 4, 4, 12, 12, 0])
        select = Select(mock_dataset, indices)
        assert len(select) == 3
        assert select._selection == [4, 4, 0]
        assert "Indices(indices=[12, 12, 4, 4, 12, 12, 0])" in str(select)

    def test_indices_with_classfilter(self, mock_dataset):
        class_filter = ClassFilter(classes=[0, 1])
        indices = Indices([12, 10, 8, 6, 4, 2, 0])
        select = Select(mock_dataset, [indices, class_filter])
        assert len(select) == 3
        assert select._selection == [6, 4, 0]
        assert "ClassFilter(classes=[0, 1]" in str(select)
        assert "Indices(indices=[12, 10, 8, 6, 4, 2, 0])" in str(select)

    def test_indices_with_classfilter_layered(self, mock_dataset):
        class_filter = ClassFilter(classes=[0, 1])
        select_cf = Select(mock_dataset, class_filter)
        assert len(select_cf) == 7
        indices = Indices([12, 10, 8, 6, 4, 2, 0])
        select = Select(select_cf, indices)
        assert len(select) == 4
        assert select._selection == [6, 4, 2, 0]
        assert "ClassFilter(classes=[0, 1]" in str(select_cf)
        assert "Indices(indices=[12, 10, 8, 6, 4, 2, 0])" in str(select)


@pytest.mark.required
class TestResolveIndices:
    """Test suite for the resolve_indices method with new SourceIndex functionality."""

    def test_resolve_indices_none_returns_all_selections(self, mock_dataset):
        """Test that passing None returns all selected indices (original behavior)."""
        select = Select(mock_dataset)
        resolved = select.resolve_indices(None)
        assert resolved == list(range(10))
        # Ensure we get a copy, not the original list
        assert resolved is not select._selection

    def test_resolve_indices_no_args_returns_all_selections(self, mock_dataset):
        """Test that calling without arguments returns all selected indices."""
        select = Select(mock_dataset)
        resolved = select.resolve_indices()
        assert resolved == list(range(10))

    def test_resolve_indices_with_single_int(self, mock_dataset):
        """Test resolving a single integer index."""
        select = Select(mock_dataset)
        resolved = select.resolve_indices(5)
        assert resolved == [5]

    def test_resolve_indices_with_single_sourceindex(self, mock_dataset):
        """Test resolving a single SourceIndex."""
        select = Select(mock_dataset)
        source_idx = SourceIndex(item=3, target=None, channel=None)
        resolved = select.resolve_indices(source_idx)
        assert resolved == [3]

    def test_resolve_indices_with_sourceindex_with_box_and_channel(self, mock_dataset):
        """Test that SourceIndex with target and channel uses only the item index."""
        select = Select(mock_dataset)
        source_idx = SourceIndex(item=7, target=2, channel=1)
        resolved = select.resolve_indices(source_idx)
        assert resolved == [7]

    def test_resolve_indices_with_sequence_of_ints(self, mock_dataset):
        """Test resolving a sequence of integer indices."""
        select = Select(mock_dataset)
        resolved = select.resolve_indices([1, 3, 5, 7])
        assert resolved == [1, 3, 5, 7]

    def test_resolve_indices_with_sequence_of_sourceindices(self, mock_dataset):
        """Test resolving a sequence of SourceIndex objects."""
        select = Select(mock_dataset)
        source_indices = [
            SourceIndex(item=0, target=None, channel=None),
            SourceIndex(item=2, target=1, channel=None),
            SourceIndex(item=4, target=None, channel=2),
            SourceIndex(item=6, target=3, channel=1),
        ]
        resolved = select.resolve_indices(source_indices)
        assert resolved == [0, 2, 4, 6]

    def test_resolve_indices_with_mixed_sequence(self, mock_dataset):
        """Test resolving a sequence with both ints and SourceIndex objects."""
        select = Select(mock_dataset)
        mixed_indices = [
            1,
            SourceIndex(item=3, target=None, channel=None),
            5,
            SourceIndex(item=7, target=2, channel=1),
        ]
        resolved = select.resolve_indices(mixed_indices)
        assert resolved == [1, 3, 5, 7]

    def test_resolve_indices_with_selections_applied(self, mock_dataset):
        """Test that resolve_indices respects selections applied to the dataset."""
        # Apply a limit selection
        limit = Limit(size=5)
        select = Select(mock_dataset, selections=[limit])

        # Resolving without args should return the limited selection
        resolved = select.resolve_indices()
        assert resolved == [0, 1, 2, 3, 4]
        assert len(resolved) == 5

    def test_resolve_indices_with_reverse_selection(self, mock_dataset):
        """Test resolve_indices with a reverse selection applied."""
        reverse = Reverse()
        select = Select(mock_dataset, selections=[reverse])

        # The internal selection should be reversed
        resolved = select.resolve_indices()
        assert resolved == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    def test_resolve_indices_with_classfilter_selection(self, mock_dataset):
        """Test resolve_indices with a class filter selection applied."""
        class_filter = ClassFilter(classes=[0, 1])
        select = Select(mock_dataset, selections=[class_filter])

        # Should only include indices where class is 0 or 1
        resolved = select.resolve_indices()
        assert len(resolved) == 7
        # Classes: 0,1,2,0,1,2,0,1,2,0 -> indices 0,1,3,4,6,7,9
        assert resolved == [0, 1, 3, 4, 6, 7, 9]

    def test_resolve_indices_after_selections_single_int(self, mock_dataset):
        """Test resolving single int after selections have been applied."""
        limit = Limit(size=5)
        select = Select(mock_dataset, selections=[limit])

        # Index 2 in the selected dataset maps to index 2 in original
        resolved = select.resolve_indices(2)
        assert resolved == [2]

    def test_resolve_indices_after_selections_sequence(self, mock_dataset):
        """Test resolving sequence of indices after selections have been applied."""
        limit = Limit(size=5)
        select = Select(mock_dataset, selections=[limit])

        # Indices in selected dataset map to same indices in original (for this case)
        resolved = select.resolve_indices([0, 2, 4])
        assert resolved == [0, 2, 4]

    def test_resolve_indices_out_of_range_negative(self, mock_dataset):
        """Test that negative indices raise IndexError."""
        select = Select(mock_dataset)

        with pytest.raises(IndexError, match="Index -1 out of range"):
            select.resolve_indices(-1)

    def test_resolve_indices_out_of_range_too_large(self, mock_dataset):
        """Test that indices beyond dataset size raise IndexError."""
        select = Select(mock_dataset)

        with pytest.raises(IndexError, match="Index 10 out of range"):
            select.resolve_indices(10)

    def test_resolve_indices_sourceindex_out_of_range(self, mock_dataset):
        """Test that SourceIndex with out-of-range item raises IndexError."""
        select = Select(mock_dataset)
        source_idx = SourceIndex(item=15, target=None, channel=None)

        with pytest.raises(IndexError, match="Index 15 out of range"):
            select.resolve_indices(source_idx)

    def test_resolve_indices_sequence_with_invalid_index(self, mock_dataset):
        """Test that sequence with one invalid index raises IndexError."""
        select = Select(mock_dataset)

        with pytest.raises(IndexError, match="out of range"):
            select.resolve_indices([1, 3, 20, 5])

    def test_resolve_indices_empty_sequence(self, mock_dataset):
        """Test resolving an empty sequence returns an empty list."""
        select = Select(mock_dataset)
        resolved = select.resolve_indices([])
        assert resolved == []

    def test_resolve_indices_duplicate_indices(self, mock_dataset):
        """Test that duplicate indices in input are preserved in output."""
        select = Select(mock_dataset)
        resolved = select.resolve_indices([1, 1, 3, 3, 1])
        assert resolved == [1, 1, 3, 3, 1]

    def test_resolve_indices_duplicate_sourceindices(self, mock_dataset):
        """Test that duplicate SourceIndices are preserved in output."""
        select = Select(mock_dataset)
        source_indices = [
            SourceIndex(item=2, target=None, channel=None),
            SourceIndex(item=2, target=1, channel=None),
            SourceIndex(item=5, target=None, channel=None),
            SourceIndex(item=2, target=None, channel=2),
        ]
        resolved = select.resolve_indices(source_indices)
        assert resolved == [2, 2, 5, 2]

    def test_resolve_indices_with_limit_mixed_valid_invalid(self, mock_dataset):
        """Test resolve_indices with Limit where some indices are valid and some invalid."""
        limit = Limit(size=5)
        select = Select(mock_dataset, selections=[limit])

        # With limit=5, only indices 0-4 are valid in the selection
        # Index 6 from original dataset is now out of range
        resolved_valid = select.resolve_indices([0, 3])
        assert resolved_valid == [0, 3]

        # Index 6 is out of range after applying Limit(5)
        with pytest.raises(IndexError, match="Index 6 out of range"):
            select.resolve_indices([0, 3, 6])

    def test_resolve_indices_empty_dataset(self):
        """Test resolve_indices with an empty dataset."""
        empty_dataset = MagicMock()
        empty_dataset.__len__.return_value = 0
        empty_dataset.__getitem__.side_effect = lambda idx: (idx, one_hot(idx % 3), {"id": idx})

        select = Select(empty_dataset)

        # Resolving without arguments should return empty list
        resolved = select.resolve_indices()
        assert resolved == []
        assert len(resolved) == 0

        # Any index should raise IndexError
        with pytest.raises(IndexError, match="Index 0 out of range"):
            select.resolve_indices(0)

        # Empty sequence should return empty list
        resolved_empty = select.resolve_indices([])
        assert resolved_empty == []
