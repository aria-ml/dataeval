from dataeval.data._view import Operation, View
from dataeval.protocols import DatasetMetadata


class MockDataset:
    metadata = {"id": "MockDataset"}

    def __init__(self, size=5):
        self._size = size

    def __getitem__(self, index: int):
        return (f"data_{index}", [1 if i == index % 3 else 0 for i in range(3)], {"id": index})

    def __len__(self):
        return self._size


class SimpleDataset:
    """A dataset whose datums are plain ints, to exercise transform/filter ops generically."""

    def __init__(self, n: int, metadata: dict | None = None) -> None:
        self._data = list(range(n))
        self.metadata = metadata if metadata is not None else {"id": "simple"}
        self.fetches = 0

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> int:
        self.fetches += 1
        return self._data[index]


class FilterEven(Operation):
    """Cardinality op: keep source indices whose datum value is even."""

    def apply(self, view: View) -> None:
        view.selection = [i for i in view.selection if view.read(i) % 2 == 0]


class AddOne(Operation):
    """Content op: register a per-datum transform, touch no data at build."""

    def apply(self, view: View) -> None:
        view.map(lambda datum: datum + 1)


class SetVocab(Operation):
    """Metadata-only op: rewrite index2label, no cardinality/content change."""

    def __init__(self, index2label: dict) -> None:
        self.index2label = index2label

    def apply_metadata(self, metadata) -> DatasetMetadata:
        return {**metadata, "index2label": dict(self.index2label)}

    def apply(self, view: View) -> None:
        pass


class TestViewCore:
    def test_init_no_metadata(self):
        v = View(MockDataset())  # type: ignore
        assert v.metadata["id"] == "MockDataset"

    def test_init_with_metadata(self):
        m = MockDataset()
        m.metadata = {"id": "ManualMockDataset"}
        v = View(m)  # type: ignore
        assert v.metadata["id"] == "ManualMockDataset"

    def test_normalize_empty(self):
        assert View._normalize([]) == []

    def test_normalize_none(self):
        assert View._normalize(None) == []

    def test_normalize_single_operation(self):
        op = AddOne()
        assert View._normalize(op) == [op]

    def test_passthrough_data(self):
        v = View(MockDataset())  # type: ignore
        assert len(v) == 5
        for i in range(len(v)):
            assert v[i][0] == f"data_{i}"
            assert v[i][2]["id"] == i

    def test_operation_filters(self):
        v = View(SimpleDataset(5), [FilterEven()])
        assert len(v) == 3
        assert list(v) == [0, 2, 4]

    def test_operation_transforms(self):
        v = View(SimpleDataset(3), [AddOne()])
        assert list(v) == [1, 2, 3]

    def test_operations_run_in_given_order(self):
        order = []

        class Recorder(Operation):
            def __init__(self, tag):
                self.tag = tag

            def apply(self, view):
                order.append(self.tag)

        View(MockDataset(), [Recorder("C"), Recorder("A"), Recorder("B")])  # type: ignore
        assert order == ["C", "A", "B"]

    def test_filter_then_transform_composes(self):
        # Keep evens (reads raw), then +1 on survivors.
        v = View(SimpleDataset(5), [FilterEven(), AddOne()])
        assert list(v) == [1, 3, 5]

    def test_identity_when_no_operations(self):
        v = View(SimpleDataset(5))
        assert len(v) == 5
        assert list(v) == [0, 1, 2, 3, 4]

    def test_single_operation_accepted(self):
        v = View(SimpleDataset(3), AddOne())
        assert list(v) == [1, 2, 3]

    def test_metadata_folded_once(self):
        v = View(SimpleDataset(2), [SetVocab({0: "a", 1: "b"})])
        assert "index2label" in v.metadata
        assert v.metadata["index2label"] == {0: "a", 1: "b"}

    def test_default_metadata_id(self):
        v = View(SimpleDataset(3, metadata={}))
        assert v.metadata["id"] == "SimpleDataset"

    def test_pure_transform_touches_no_data_at_build(self):
        ds = SimpleDataset(5)
        View(ds, [AddOne()])  # no cardinality op => no build-time scan
        assert ds.fetches == 0

    def test_filter_scans_at_build(self):
        ds = SimpleDataset(5)
        View(ds, [FilterEven()])
        assert ds.fetches > 0

    def test_map_where_transforms_subset(self):
        class SubAt2(Operation):
            def apply(self, view):
                view.map(lambda d: ("sub_" + d[0], d[1], d[2]), where={2})

        v = View(MockDataset(), [SubAt2()])  # type: ignore
        for i in range(len(v)):
            data, labels, meta = v[i]
            if i == 2:
                assert data.startswith("sub_")
                assert meta == {"id": 2}
                assert labels == [0, 0, 1]
            else:
                assert data.startswith("data_")

    def test_resolve_indices_returns_selection(self):
        v = View(MockDataset())  # type: ignore
        assert v.resolve_indices() == list(range(len(v)))

    def test_resolve_indices_after_filter(self):
        v = View(SimpleDataset(5), [FilterEven()])
        assert v.resolve_indices() == [0, 2, 4]

    def test_repr_and_str(self):
        v = View(SimpleDataset(5), [FilterEven(), AddOne()])
        assert repr(v).startswith("View(")
        assert "operations=[" in repr(v)
        text = str(v)
        assert "View Dataset" in text
        assert "Operations: [" in text
        assert "Selected Size: 3" in text


class TestRootAndOperationGroups:
    """``root`` and ``operation_groups`` walk a nested View chain for metadata."""

    @staticmethod
    def _noop() -> Operation:
        return type("_Noop", (Operation,), {"apply": lambda self_, view: None})()

    def test_root_on_unwrapped_view(self):
        base = MockDataset()
        v = View(base)  # type: ignore
        assert v.root is base

    def test_root_walks_through_nesting(self):
        base = MockDataset()
        chain = View(View(View(base)))  # type: ignore
        assert chain.root is base

    def test_operation_groups_empty_when_no_operations_anywhere(self):
        base = MockDataset()
        assert View(base).operation_groups == []  # type: ignore
        assert View(View(base)).operation_groups == []  # type: ignore

    def test_operation_groups_single_construction(self):
        base = MockDataset()
        a, b = self._noop(), self._noop()
        v = View(base, [a, b])  # type: ignore
        assert v.operation_groups == [[a, b]]

    def test_operation_groups_preserves_nesting_innermost_first(self):
        base = MockDataset()
        a, b, c = self._noop(), self._noop(), self._noop()
        outer = View(View(base, [a, b]), [c])  # type: ignore
        assert outer.operation_groups == [[a, b], [c]]

    def test_operation_groups_skips_empty_wrappers(self):
        base = MockDataset()
        a = self._noop()
        outer = View(View(base), [a])  # type: ignore
        assert outer.operation_groups == [[a]]

    def test_nested_metadata_propagates_from_base(self):
        base = MockDataset()
        base.metadata = {"id": "RealBase"}
        outer = View(View(base))  # type: ignore
        assert outer.metadata["id"] == "RealBase"


class LabeledDataset:
    """A dataset that mirrors its label vocabulary as an instance attribute.

    The convention a real source dataset follows -- ``index2label`` reachable both as
    ``ds.index2label`` and through ``ds.metadata`` -- and the shape of the conflict the
    ``source`` property exists to prevent.
    """

    def __init__(self) -> None:
        self.index2label = {0: "cat", 1: "dog"}
        self.metadata = {"id": "labeled", "index2label": dict(self.index2label)}
        self._cache = ["private", "state"]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        return index


class CollapseVocabulary(Operation):
    """Metadata op: rewrite the class vocabulary to a single merged class."""

    def apply_metadata(self, metadata: DatasetMetadata) -> DatasetMetadata:
        return {**metadata, "index2label": {0: "animal"}}  # type: ignore[return-value]

    def apply(self, view: View) -> None:
        return None


class TestSource:
    """``source`` names the immediate wrapped dataset; nothing else crosses the boundary."""

    def test_source_is_the_immediate_wrapped_dataset(self):
        base = MockDataset()
        assert View(base).source is base  # type: ignore

    def test_source_of_a_nested_view_is_the_inner_view(self):
        base = MockDataset()
        inner = View(base)  # type: ignore
        outer = View(inner)
        assert outer.source is inner
        assert inner.source is base

    def test_source_and_root_differ_on_a_nested_chain(self):
        base = MockDataset()
        inner = View(base)  # type: ignore
        outer = View(inner)
        assert outer.source is inner
        assert outer.root is base

    def test_source_reaches_the_original_vocabulary_explicitly(self):
        view = View(LabeledDataset(), [CollapseVocabulary()])  # type: ignore
        source = view.source
        assert isinstance(source, LabeledDataset)
        assert source.index2label == {0: "cat", 1: "dog"}

    def test_source_public_attributes_do_not_leak_onto_the_view(self):
        view = View(LabeledDataset())  # type: ignore
        assert not hasattr(view, "index2label")

    def test_source_private_state_does_not_leak_onto_the_view(self):
        view = View(LabeledDataset())  # type: ignore
        assert "_cache" not in view.__dict__

    def test_stale_source_vocabulary_is_unreachable_when_an_operation_rewrites_it(self):
        # The conflict: the view's metadata is relabeled, so a leaked ``index2label``
        # attribute would contradict it. Only the rewritten vocabulary is reachable.
        view = View(LabeledDataset(), [CollapseVocabulary()])  # type: ignore
        assert "index2label" in view.metadata
        assert view.metadata["index2label"] == {0: "animal"}
        assert not hasattr(view, "index2label")
