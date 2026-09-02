from typing import Any, NamedTuple

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


class _IterableDataset:
    """Dataset that offers __iter__ as a convenience and carries no metadata.

    Both traits matter: a dataset like this is structurally indistinguishable from a
    collection of datasets, so it guards against unpacking exploding it into its datums.
    """

    def __init__(self, tag: str, n: int) -> None:
        self._items = [(tag, i) for i in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        return self._items[index]

    def __iter__(self):
        return iter(self._items)


class _Datum(NamedTuple):
    """A MAITE triple spelled as a NamedTuple, which callers read by field name."""

    input: str
    target: int
    metadata: dict[str, Any]


class _TripleDataset:
    """Datums as MAITE ``(input, target, metadata)`` triples carrying their own id."""

    def __init__(self, tag: str, n: int, index2label: dict[int, str], datum_type=tuple) -> None:
        self._items = [
            datum_type((tag, 0, {"id": i})) if datum_type is tuple else datum_type(tag, 0, {"id": i}) for i in range(n)
        ]
        self.metadata = DatasetMetadata(id=tag, index2label=dict(index2label))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        return self._items[index]


@pytest.mark.required
class TestMergedDatumIds:
    """Two sources can both name an item ``0``; the merged view keeps them apart."""

    i2l = {0: "cat"}

    def test_ids_are_namespaced_by_source_position(self):
        merged = merge_datasets(_TripleDataset("a", 2, self.i2l), _TripleDataset("b", 2, self.i2l))
        assert [datum[2]["id"] for datum in merged] == ["0:0", "0:1", "1:0", "1:1"]

    def test_the_source_datum_is_not_mutated(self):
        source = _TripleDataset("a", 1, self.i2l)
        _ = merge_datasets(source, _TripleDataset("b", 1, self.i2l))[0]
        assert source[0][2]["id"] == 0

    def test_a_namedtuple_datum_stays_a_namedtuple(self):
        """Rebuilding it as a bare tuple would turn ``datum.metadata`` into an AttributeError."""
        merged = merge_datasets(
            _TripleDataset("a", 1, self.i2l, _Datum),
            _TripleDataset("b", 1, self.i2l, _Datum),
        )
        datum = merged[1]
        assert isinstance(datum, _Datum)
        assert datum.metadata["id"] == "1:0"

    def test_a_datum_without_an_id_is_untouched(self):
        merged = merge_datasets(_LabeledDataset("a", 1, self.i2l), _LabeledDataset("b", 1, self.i2l))
        assert merged[1] == ("b", 0)


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

    def test_repr_reports_dataset_count_and_length(self):
        i2l = {0: "cat", 1: "dog"}
        merged = merge_datasets(_LabeledDataset("a", 2, i2l), _LabeledDataset("b", 3, i2l))
        assert repr(merged) == "merge_datasets(2 datasets, len=5)"


@pytest.mark.required
class TestMergeDatasetsInputForms:
    """Datasets may be passed one per argument or packed in a sequence/iterable."""

    I2L = {0: "cat", 1: "dog"}

    def _expected(self):
        return [("a", 0), ("a", 1), ("b", 0), ("b", 1), ("b", 2)]

    def _datasets(self):
        return _LabeledDataset("a", 2, self.I2L), _LabeledDataset("b", 3, self.I2L)

    @pytest.mark.parametrize("pack", [list, tuple, iter, lambda ds: (d for d in ds)])
    def test_packed_matches_unpacked(self, pack):
        """A list, tuple, iterator or generator of datasets merges like separate arguments."""
        merged = merge_datasets(pack(self._datasets()))
        assert len(merged) == 5
        assert list(iter(merged)) == self._expected()
        assert merged.metadata.get("index2label") == self.I2L
        assert repr(merged) == "merge_datasets(2 datasets, len=5)"

    def test_mixed_packed_and_unpacked(self):
        """Packed and unpacked arguments can be combined, and order is preserved."""
        a, b = self._datasets()
        c = _LabeledDataset("c", 1, self.I2L)
        merged = merge_datasets(a, [b, c])
        assert list(iter(merged)) == [*self._expected(), ("c", 0)]

    def test_nested_collections_are_flattened(self):
        """Nesting the collections does not change the result.

        Only one level of packing is part of the declared signature, hence the untyped
        input; deeper nesting is tolerated rather than advertised.
        """
        a, b = self._datasets()
        nested: list[Any] = [[a], (b,)]
        assert list(iter(merge_datasets(nested))) == self._expected()

    def test_single_packed_dataset_is_a_view(self):
        merged = merge_datasets([_LabeledDataset("a", 3, self.I2L)])
        assert len(merged) == 3
        assert merged.metadata.get("index2label") == self.I2L

    def test_empty_collection_requires_at_least_one_dataset(self):
        with pytest.raises(ValueError, match="at least one"):
            merge_datasets([])

    @pytest.mark.parametrize("bad", [1, None, "dataset"])
    def test_non_dataset_argument_raises(self, bad):
        """Anything that is neither a dataset nor a collection of them is rejected."""
        with pytest.raises(TypeError, match="merge_datasets accepts datasets"):
            merge_datasets(_LabeledDataset("a", 1, self.I2L), bad)

    def test_iterable_dataset_without_metadata_is_not_unpacked(self):
        """A dataset with __iter__ and no metadata merges as one dataset, not as a container.

        The declared signature asks for an :class:`~dataeval.protocols.AnnotatedDataset`,
        hence the untyped inputs; a missing vocabulary is tolerated at runtime (it only
        warns), so unpacking must not mistake such a dataset for a collection.
        """
        a: Any = _IterableDataset("a", 2)
        b: Any = _IterableDataset("b", 3)
        with pytest.warns(UserWarning, match="index2label"):
            merged = merge_datasets(a, b)
        assert len(merged) == 5
        assert list(iter(merged)) == self._expected()

    def test_non_dataset_nested_in_a_collection_raises(self):
        """A bad element is rejected wherever it sits, not only at the top level."""
        a, b = self._datasets()
        packed: list[Any] = [a, [b, None]]
        with pytest.raises(TypeError, match="merge_datasets accepts datasets"):
            merge_datasets(packed)

    def test_self_referential_collection_raises_type_error(self):
        """A container holding itself reports the cycle instead of overflowing the stack."""
        packed: list[Any] = [_LabeledDataset("a", 1, self.I2L)]
        packed.append(packed)
        with pytest.raises(TypeError, match="contains itself"):
            merge_datasets(packed)

    def test_mapping_reports_that_it_yields_keys(self):
        """Iterating a mapping yields keys, so the error names the mapping, not 'str'."""
        a, b = self._datasets()
        by_name: Any = {"a": a, "b": b}
        with pytest.raises(TypeError, match="pass the values"):
            merge_datasets(by_name)

    def test_packed_input_still_validates_vocabularies(self):
        """Validation applies to packed datasets exactly as to unpacked ones."""
        with pytest.raises(ValueError, match="index2label"):
            merge_datasets([_LabeledDataset("a", 1, {0: "cat"}), _LabeledDataset("b", 1, {0: "dog"})])
