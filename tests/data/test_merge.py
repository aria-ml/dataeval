import pytest

from dataeval.data import merge_datasets
from dataeval.protocols import DatasetMetadata


class _LabeledDataset:
    """Minimal AnnotatedDataset: integer-tagged datums + an index2label."""

    def __init__(self, tag: str, n: int, index2label: dict[int, str]) -> None:
        self._items = [(tag, i) for i in range(n)]
        self.metadata = DatasetMetadata(id=tag, index2label=dict(index2label))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        return self._items[index]


@pytest.mark.required
class TestMergeDatasets:
    def test_concatenates_in_order(self):
        i2l = {0: "cat", 1: "dog"}
        merged = merge_datasets(_LabeledDataset("a", 2, i2l), _LabeledDataset("b", 3, i2l))
        assert len(merged) == 5
        assert list(iter(merged)) == [("a", 0), ("a", 1), ("b", 0), ("b", 1), ("b", 2)]
        assert merged.metadata.get("index2label") == i2l

    def test_getitem_routes_and_supports_negative_index(self):
        i2l = {0: "cat"}
        merged = merge_datasets(_LabeledDataset("a", 2, i2l), _LabeledDataset("b", 2, i2l))
        assert merged[2] == ("b", 0)
        assert merged[-1] == ("b", 1)
        with pytest.raises(IndexError):
            _ = merged[4]

    def test_requires_matching_index2label(self):
        with pytest.raises(ValueError, match="index2label"):
            merge_datasets(_LabeledDataset("a", 1, {0: "cat"}), _LabeledDataset("b", 1, {0: "dog"}))

    def test_requires_at_least_one_dataset(self):
        with pytest.raises(ValueError, match="at least one"):
            merge_datasets()

    def test_single_dataset_is_a_view(self):
        i2l = {0: "cat", 1: "dog"}
        merged = merge_datasets(_LabeledDataset("a", 3, i2l))
        assert len(merged) == 3
        assert merged.metadata.get("index2label") == i2l
