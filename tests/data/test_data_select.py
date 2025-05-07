from dataeval.data._selection import Select


class MockDataset:
    def __getitem__(self, index: int):
        return index

    def __len__(self):
        return 1


class TestSelect:
    def test_init_no_metadata(self):
        s = Select(MockDataset())  # type: ignore
        assert s._metadata["id"] == "MockDataset"

    def test_init_with_metadata(self):
        m = MockDataset()
        setattr(m, "metadata", {"id": "ManualMockDataset"})
        s = Select(m)  # type: ignore
        assert s._metadata["id"] == "ManualMockDataset"

    def test_sort_selections_empty(self):
        s = Select(MockDataset())  # type: ignore
        assert s._sort_selections([]) == []

    def test_sort_selections_none(self):
        s = Select(MockDataset())  # type: ignore
        assert s._sort_selections(None) == []
