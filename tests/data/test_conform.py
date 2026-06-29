from collections.abc import Mapping

import pytest

from dataeval.data import Conform, Conformer
from dataeval.protocols import DatasetMetadata
from dataeval.utils._validate import aggregate_required_kind


class SimpleDataset:
    """A dataset whose datums are plain ints, to exercise Conform generically."""

    def __init__(self, n: int, metadata: dict | None = None) -> None:
        self._data = list(range(n))
        self.metadata = metadata if metadata is not None else {"id": "simple"}

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> int:
        return self._data[index]


class KeepEven(Conformer[int]):
    def keeps(self, datum: int) -> bool:
        return datum % 2 == 0


class AddOne(Conformer[int]):
    def conform_datum(self, datum: int) -> int:
        return datum + 1


class SetVocab(Conformer[int]):
    def __init__(self, index2label: Mapping[int, str]) -> None:
        self.index2label = index2label

    def conform_metadata(self, metadata: DatasetMetadata) -> DatasetMetadata:
        return {**metadata, "index2label": dict(self.index2label)}  # type: ignore[typeddict-item]


@pytest.mark.required
class TestConform:
    def test_identity_when_no_conformers(self):
        conformed = Conform(SimpleDataset(5))
        assert len(conformed) == 5
        assert list(conformed) == [0, 1, 2, 3, 4]

    def test_default_metadata_id(self):
        conformed = Conform(SimpleDataset(3, metadata={}))
        assert conformed.metadata["id"] == "SimpleDataset"

    def test_keeps_filters_rows(self):
        conformed = Conform(SimpleDataset(5), [KeepEven()])
        assert len(conformed) == 3
        assert list(conformed) == [0, 2, 4]

    def test_conform_datum_transforms(self):
        conformed = Conform(SimpleDataset(3), [AddOne()])
        assert list(conformed) == [1, 2, 3]

    def test_conformers_applied_in_order(self):
        # KeepEven drops odds first, then AddOne transforms survivors
        conformed = Conform(SimpleDataset(5), [KeepEven(), AddOne()])
        assert list(conformed) == [1, 3, 5]

    def test_conform_metadata_folded_once(self):
        conformed = Conform(SimpleDataset(2), [SetVocab({0: "a", 1: "b"})])
        assert "index2label" in conformed.metadata
        assert conformed.metadata["index2label"] == {0: "a", 1: "b"}

    def test_single_conformer_accepted(self):
        conformed = Conform(SimpleDataset(3), AddOne())
        assert list(conformed) == [1, 2, 3]

    def test_getitem_and_len_consistent(self):
        conformed = Conform(SimpleDataset(5), [KeepEven()])
        assert [conformed[i] for i in range(len(conformed))] == list(conformed)

    def test_aggregate_required_kind(self):
        assert aggregate_required_kind([None]) is None
        assert aggregate_required_kind(["any_target", "object_detection"]) == "object_detection"  # specific wins
        assert aggregate_required_kind(["any_target"]) == "any_target"
