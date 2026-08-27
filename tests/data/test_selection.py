from typing import NamedTuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.data._classbalance import ClassBalance
from dataeval.data._classfilter import ClassFilter
from dataeval.data._indices import Indices
from dataeval.data._limit import Limit
from dataeval.data._reverse import Reverse
from dataeval.data._shuffle import Shuffle
from dataeval.data._view import View
from dataeval.types import SourceIndex


class MockTarget(NamedTuple):
    label: int


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
        select = View(mock_dataset, operations=[class_filter])
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
        select = View(mock_dataset, operations=[class_balance])
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
        select = View(mock_dataset, operations=[class_filter, class_balance])
        # After filtering and balancing, check that we get results
        assert len(select) > 0
        counts = {0: 0, 1: 0, 2: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] += 1
        # Check that classes 0 and 1 are present
        assert counts[0] > 0
        assert counts[1] > 0
        assert "ClassFilter(classes=[0, 1]" in str(select)
        assert "ClassBalance(" in str(select)

    def test_classfilter_with_unsupported_target(self):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = lambda idx: (f"data_{idx}", MockTarget(idx), {"id": idx})

        class_filter = ClassFilter(classes=[0])
        with pytest.raises(TypeError):
            View(mock_dataset, operations=[class_filter])

    def test_classbalance_with_unsupported_target(self):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = lambda idx: (f"data_{idx}", MockTarget(idx), {"id": idx})

        class_balance = ClassBalance(method="interclass")
        # MAITE-shape validation now fails fast on unsupported targets rather
        # than silently producing an empty selection (see MaiteShapeError).
        with pytest.raises(TypeError):
            View(mock_dataset, operations=[class_balance])

    def test_classfilter_with_nothing(self, mock_dataset):
        # Test ClassFilter with no params
        class_filter = ClassFilter([])
        select = View(mock_dataset, operations=class_filter)
        assert len(select) == 10

    def test_classfilter_and_balance_with_limit(self, mock_dataset):
        # Test ClassFilter balance with limit
        class_filter = ClassFilter(classes=[0, 1])
        class_balance = ClassBalance(method="interclass")
        limit = Limit(size=5)
        select = View(mock_dataset, operations=[limit, class_filter, class_balance])
        # After limit, filter, and balance, check we get results
        assert len(select) > 0
        assert len(select) <= 5
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
        select = View(mock_dataset, operations=[limit])
        assert len(select) == 5
        assert "Limit(size=5)" in str(select)

    def test_limit_shuffle_limit_composes_in_order(self, mock_dataset):
        # Selectors run in the given order, so Limit can appear more than once and
        # truncate an intermediate window. This pipeline — cap to 8, shuffle those,
        # cap to 3 — keeps a random 3 of the first 8, which the removed stage-based
        # reordering could not express.
        windowed = View(mock_dataset, operations=[Limit(8), Limit(3)])
        assert windowed.resolve_indices() == [0, 1, 2]  # in-order: cap 8, then cap 3

        resolved = View(mock_dataset, operations=[Limit(8), Shuffle(seed=0), Limit(3)]).resolve_indices()
        assert len(resolved) == 3
        assert all(idx < 8 for idx in resolved)  # every survivor came from the first-8 window
        assert resolved != [0, 1, 2]  # the middle Shuffle took effect before the final Limit

    def test_reverse(self, mock_dataset):
        # Test Reverse
        reverse = Reverse()
        select = View(mock_dataset, operations=[reverse])
        expected_order = list(range(9, -1, -1))
        for i, (data, _, _) in enumerate(select):
            assert data == expected_order[i]
        assert "Reverse()" in str(select)

    def test_shuffle(self, mock_dataset):
        # Test Shuffle
        shuffle = Shuffle(seed=0)
        select = View(mock_dataset, operations=[shuffle])
        # Since shuffle is random, we just check if the length is correct
        assert len(select) == 10
        # Check if the shuffled order is not the same as the original order
        original_order = [f"data_{i}" for i in range(10)]
        shuffled_order = [data for data, _, _ in select]
        assert original_order != shuffled_order
        assert "Shuffle(seed=0)" in str(select)

    def test_indices(self, mock_dataset):
        indices = Indices([12, 10, 8, 6, 4, 2, 0])
        select = View(mock_dataset, indices)
        assert len(select) == 5
        assert select.selection == [8, 6, 4, 2, 0]
        assert "Indices(indices=[12, 10, 8, 6, 4, 2, 0])" in str(select)

    def test_indices_repeats(self, mock_dataset):
        indices = Indices([12, 12, 4, 4, 12, 12, 0])
        select = View(mock_dataset, indices)
        assert len(select) == 3
        assert select.selection == [4, 4, 0]
        assert "Indices(indices=[12, 12, 4, 4, 12, 12, 0])" in str(select)

    def test_indices_with_classfilter(self, mock_dataset):
        class_filter = ClassFilter(classes=[0, 1])
        indices = Indices([12, 10, 8, 6, 4, 2, 0])
        select = View(mock_dataset, [indices, class_filter])
        assert len(select) == 3
        assert select.selection == [6, 4, 0]
        assert "ClassFilter(classes=[0, 1]" in str(select)
        assert "Indices(indices=[12, 10, 8, 6, 4, 2, 0])" in str(select)

    def test_indices_with_classfilter_layered(self, mock_dataset):
        class_filter = ClassFilter(classes=[0, 1])
        select_cf = View(mock_dataset, class_filter)
        assert len(select_cf) == 7
        indices = Indices([12, 10, 8, 6, 4, 2, 0])
        select = View(select_cf, indices)
        assert len(select) == 4
        assert select.selection == [6, 4, 2, 0]
        assert "ClassFilter(classes=[0, 1]" in str(select_cf)
        assert "Indices(indices=[12, 10, 8, 6, 4, 2, 0])" in str(select)


@pytest.mark.required
class TestResolveIndices:
    """Test suite for the resolve_indices method with new SourceIndex functionality."""

    def test_resolve_indices_none_returns_all_selections(self, mock_dataset):
        """Test that passing None returns all selected indices (original behavior)."""
        select = View(mock_dataset)
        resolved = select.resolve_indices(None)
        assert resolved == list(range(10))
        # Ensure we get a copy, not the original list
        assert resolved is not select.selection

    def test_resolve_indices_no_args_returns_all_selections(self, mock_dataset):
        """Test that calling without arguments returns all selected indices."""
        select = View(mock_dataset)
        resolved = select.resolve_indices()
        assert resolved == list(range(10))

    def test_resolve_indices_with_single_int(self, mock_dataset):
        """Test resolving a single integer index."""
        select = View(mock_dataset)
        resolved = select.resolve_indices(5)
        assert resolved == [5]

    def test_resolve_indices_with_single_sourceindex(self, mock_dataset):
        """Test resolving a single SourceIndex."""
        select = View(mock_dataset)
        source_idx = SourceIndex(item=3, key=None)
        resolved = select.resolve_indices(source_idx)
        assert resolved == [3]

    def test_resolve_indices_with_sourceindex_with_box(self, mock_dataset):
        """Test that SourceIndex with a target uses only the item index."""
        select = View(mock_dataset)
        source_idx = SourceIndex(item=7, key=2)
        resolved = select.resolve_indices(source_idx)
        assert resolved == [7]

    def test_resolve_indices_with_sequence_of_ints(self, mock_dataset):
        """Test resolving a sequence of integer indices."""
        select = View(mock_dataset)
        resolved = select.resolve_indices([1, 3, 5, 7])
        assert resolved == [1, 3, 5, 7]

    def test_resolve_indices_with_sequence_of_sourceindices(self, mock_dataset):
        """Test resolving a sequence of SourceIndex objects."""
        select = View(mock_dataset)
        source_indices = [
            SourceIndex(item=0, key=None),
            SourceIndex(item=2, key=1),
            SourceIndex(item=4, key=None),
            SourceIndex(item=6, key=3),
        ]
        resolved = select.resolve_indices(source_indices)
        assert resolved == [0, 2, 4, 6]

    def test_resolve_indices_with_mixed_sequence(self, mock_dataset):
        """Test resolving a sequence with both ints and SourceIndex objects."""
        select = View(mock_dataset)
        mixed_indices = [
            1,
            SourceIndex(item=3, key=None),
            5,
            SourceIndex(item=7, key=2),
        ]
        resolved = select.resolve_indices(mixed_indices)
        assert resolved == [1, 3, 5, 7]

    def test_resolve_indices_with_selections_applied(self, mock_dataset):
        """Test that resolve_indices respects selections applied to the dataset."""
        # Apply a limit selection
        limit = Limit(size=5)
        select = View(mock_dataset, operations=[limit])

        # Resolving without args should return the limited selection
        resolved = select.resolve_indices()
        assert resolved == [0, 1, 2, 3, 4]
        assert len(resolved) == 5

    def test_resolve_indices_with_reverse_selection(self, mock_dataset):
        """Test resolve_indices with a reverse selection applied."""
        reverse = Reverse()
        select = View(mock_dataset, operations=[reverse])

        # The internal selection should be reversed
        resolved = select.resolve_indices()
        assert resolved == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    def test_resolve_indices_with_classfilter_selection(self, mock_dataset):
        """Test resolve_indices with a class filter selection applied."""
        class_filter = ClassFilter(classes=[0, 1])
        select = View(mock_dataset, operations=[class_filter])

        # Should only include indices where class is 0 or 1
        resolved = select.resolve_indices()
        assert len(resolved) == 7
        # Classes: 0,1,2,0,1,2,0,1,2,0 -> indices 0,1,3,4,6,7,9
        assert resolved == [0, 1, 3, 4, 6, 7, 9]

    def test_resolve_indices_after_selections_single_int(self, mock_dataset):
        """Test resolving single int after selections have been applied."""
        limit = Limit(size=5)
        select = View(mock_dataset, operations=[limit])

        # Index 2 in the selected dataset maps to index 2 in original
        resolved = select.resolve_indices(2)
        assert resolved == [2]

    def test_resolve_indices_after_selections_sequence(self, mock_dataset):
        """Test resolving sequence of indices after selections have been applied."""
        limit = Limit(size=5)
        select = View(mock_dataset, operations=[limit])

        # Indices in selected dataset map to same indices in original (for this case)
        resolved = select.resolve_indices([0, 2, 4])
        assert resolved == [0, 2, 4]

    def test_resolve_indices_out_of_range_negative(self, mock_dataset):
        """Test that negative indices raise IndexError."""
        select = View(mock_dataset)

        with pytest.raises(IndexError, match="Index -1 out of range"):
            select.resolve_indices(-1)

    def test_resolve_indices_out_of_range_too_large(self, mock_dataset):
        """Test that indices beyond dataset size raise IndexError."""
        select = View(mock_dataset)

        with pytest.raises(IndexError, match="Index 10 out of range"):
            select.resolve_indices(10)

    def test_resolve_indices_sourceindex_out_of_range(self, mock_dataset):
        """Test that SourceIndex with out-of-range item raises IndexError."""
        select = View(mock_dataset)
        source_idx = SourceIndex(item=15, key=None)

        with pytest.raises(IndexError, match="Index 15 out of range"):
            select.resolve_indices(source_idx)

    def test_resolve_indices_sequence_with_invalid_index(self, mock_dataset):
        """Test that sequence with one invalid index raises IndexError."""
        select = View(mock_dataset)

        with pytest.raises(IndexError, match="out of range"):
            select.resolve_indices([1, 3, 20, 5])

    def test_resolve_indices_empty_sequence(self, mock_dataset):
        """Test resolving an empty sequence returns an empty list."""
        select = View(mock_dataset)
        resolved = select.resolve_indices([])
        assert resolved == []

    def test_resolve_indices_duplicate_indices(self, mock_dataset):
        """Test that duplicate indices in input are preserved in output."""
        select = View(mock_dataset)
        resolved = select.resolve_indices([1, 1, 3, 3, 1])
        assert resolved == [1, 1, 3, 3, 1]

    def test_resolve_indices_duplicate_sourceindices(self, mock_dataset):
        """Test that duplicate SourceIndices are preserved in output."""
        select = View(mock_dataset)
        source_indices = [
            SourceIndex(item=2, key=None),
            SourceIndex(item=2, key=1),
            SourceIndex(item=5, key=None),
            SourceIndex(item=2, key=None),
        ]
        resolved = select.resolve_indices(source_indices)
        assert resolved == [2, 2, 5, 2]

    def test_resolve_indices_with_limit_mixed_valid_invalid(self, mock_dataset):
        """Test resolve_indices with Limit where some indices are valid and some invalid."""
        limit = Limit(size=5)
        select = View(mock_dataset, operations=[limit])

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

        select = View(empty_dataset)

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
