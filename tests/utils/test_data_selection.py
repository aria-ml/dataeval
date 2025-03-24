from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.utils.data._selection import Select
from dataeval.utils.data.selections._classfilter import ClassFilter
from dataeval.utils.data.selections._indices import Indices
from dataeval.utils.data.selections._limit import Limit
from dataeval.utils.data.selections._reverse import Reverse
from dataeval.utils.data.selections._shuffle import Shuffle


def one_hot(label: int):
    oh = np.zeros(3)
    oh[label] = 1
    return oh


@pytest.fixture(scope="module")
def mock_dataset():
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 10
    mock_dataset.__getitem__.side_effect = lambda idx: (f"data_{idx}", one_hot(idx % 3), {"id": idx})
    return mock_dataset


@pytest.mark.required
class TestSelectionClasses:
    def test_classfilter_classes_only(self, mock_dataset):
        # Test ClassFilter classes
        class_filter = ClassFilter(classes=(0, 1))
        select = Select(mock_dataset, selections=[class_filter])
        assert len(select) == 7
        counts = {0: 0, 1: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] = counts[label] + 1
        assert counts == {0: 4, 1: 3}
        assert "ClassFilter(classes=(0, 1), balance=False)" in str(select)

    def test_classfilter_balance_only(self, mock_dataset):
        # Test ClassFilter balance
        class_filter = ClassFilter(balance=True)
        select = Select(mock_dataset, selections=[class_filter])
        assert len(select) == 9
        counts = {0: 0, 1: 0, 2: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] = counts[label] + 1
        assert counts == {0: 3, 1: 3, 2: 3}
        assert "ClassFilter(classes=None, balance=True)" in str(select)

    def test_classfilter_classes_and_balance(self, mock_dataset):
        # Test ClassFilter balance
        class_filter = ClassFilter(classes=[0, 1], balance=True)
        select = Select(mock_dataset, selections=[class_filter])
        assert len(select) == 6
        counts = {0: 0, 1: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] = counts[label] + 1
        assert counts == {0: 3, 1: 3}
        assert "ClassFilter(classes=[0, 1], balance=True)" in str(select)

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

    def test_classfilter_with_nothing(self, mock_dataset):
        # Test ClassFilter with no params
        class_filter = ClassFilter()
        select = Select(mock_dataset, selections=class_filter)
        assert len(select) == 10

    def test_classfilter_with_limit(self, mock_dataset):
        # Test ClassFilter balance
        class_filter = ClassFilter(classes=[0, 1], balance=True)
        limit = Limit(size=5)
        select = Select(mock_dataset, selections=[limit, class_filter])
        assert len(select) == 4
        counts = {0: 0, 1: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] = counts[label] + 1
        assert counts == {0: 2, 1: 2}
        assert "ClassFilter(classes=[0, 1], balance=True)" in str(select)
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
            assert data == f"data_{expected_order[i]}"
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
        class_filter = ClassFilter(classes=[0, 1], balance=False)
        indices = Indices([12, 10, 8, 6, 4, 2, 0])
        select = Select(mock_dataset, [indices, class_filter])
        assert len(select) == 3
        assert select._selection == [6, 4, 0]
        assert "ClassFilter(classes=[0, 1], balance=False)" in str(select)
        assert "Indices(indices=[12, 10, 8, 6, 4, 2, 0])" in str(select)
