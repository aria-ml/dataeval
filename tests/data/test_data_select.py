from dataeval.data._selection import Select, Selection, SelectionStage, Subselection


class MockDataset:
    def __init__(self, size=5):
        self._size = size

    def __getitem__(self, index: int):
        return (f"data_{index}", [1 if i == index % 3 else 0 for i in range(3)], {"id": index})

    def __len__(self):
        return self._size


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

    def test_passthrough_data(self):
        s = Select(MockDataset())  # type: ignore
        assert len(s) == 5
        for i in range(len(s)):
            assert s[i][0] == f"data_{i}"
            assert s[i][2]["id"] == i

    def test_label_vectors(self):
        ds = MockDataset()
        expected_labels = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
        for i in range(len(ds)):
            _, labels, _ = ds[i]
            assert labels == expected_labels[i]

    def test_selection_applied(self):
        class FilterEven(Selection):
            stage = SelectionStage.FILTER

            def __call__(self, dataset):
                dataset._selection = [i for i in dataset._selection if i % 2 == 0]

        s = Select(MockDataset(), selections=[FilterEven()])  # type: ignore

        expected_data = ["data_0", "data_2", "data_4"]
        assert len(s) == len(expected_data)
        assert [s[i][0] for i in range(len(s))] == expected_data

    def test_selection_stage_ordering(self):
        order = []

        class A(Selection):
            stage = SelectionStage.STATE

            def __call__(self, dataset):
                order.append("A")

        class B(Selection):
            stage = SelectionStage.FILTER

            def __call__(self, dataset):
                order.append("B")

        class C(Selection):
            stage = SelectionStage.ORDER

            def __call__(self, dataset):
                order.append("C")

        Select(MockDataset(), selections=[C(), A(), B()])  # type: ignore
        assert order == ["A", "B", "C"]

    def test_subselection_applied(self):
        class Sub(Selection):
            stage = SelectionStage.FILTER

            def __call__(self, dataset):
                class InlineSubselection(Subselection):
                    def __call__(self, datum):
                        return ("sub_" + datum[0], datum[1], datum[2])

                dataset._subselections.append((InlineSubselection(), {2}))

        s = Select(MockDataset(), selections=[Sub()])  # type: ignore

        # Confirm only index 2 is transformed
        for i in range(len(s)):
            data, _, _ = s[i]
            if i == 2:
                data, labels, meta = s[2]
                assert data.startswith("sub_")
                assert meta == {"id": 2}
                assert labels == [0, 0, 1]
            else:
                assert data.startswith("data_")
