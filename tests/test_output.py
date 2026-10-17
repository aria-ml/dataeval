from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from dataeval.types import DictOutput, ExecutionMetadata, MappingOutput, SequenceOutput, set_metadata


@dataclass
class MockOutput(DictOutput):
    test1: int
    test2: bool
    test3: str


@set_metadata
def mock_metric(arg1: int, arg2: bool, arg3: str) -> MockOutput:
    return MockOutput(arg1, arg2, arg3)


class MockMetric:
    state1: int = 1
    state2: float = 1.5
    state3: list = ["a", "very", "long", "input", "list"]

    @set_metadata(state=["state1", "state2", "state3"])
    def evaluate(self, arg1: int, arg2: bool, arg3: str = "mock_default") -> MockOutput:
        return MockOutput(arg1, arg2, arg3)


class MockMappingOutput(MappingOutput[str, float]): ...


class MockSequenceOutput(SequenceOutput[float]): ...


@pytest.mark.required
class TestOutputMetadata:
    def test_output_metadata_str(self):
        output = mock_metric(1, True, "value")
        assert "{'test1': 1" in output.__str__()

    def test_output_metadata_repr(self):
        output = mock_metric(1, True, "value")
        assert "MockOutput(test1=1" in output.__repr__()

    def test_output_metadata_data(self):
        output = mock_metric(1, True, "value")
        assert output.test1 == 1
        assert output.test2
        assert output.test3 == "value"

    def test_output_metadata_dict(self):
        output_dict = mock_metric(1, True, "value").data()
        assert output_dict == {"test1": 1, "test2": True, "test3": "value"}

    def test_output_metadata_meta(self):
        output_meta = mock_metric(1, True, "value").meta()
        assert output_meta.name == "tests.test_output.mock_metric"
        assert output_meta.execution_time
        assert output_meta.execution_duration > 0
        assert set(output_meta.arguments) == {"arg1", "arg2", "arg3"}
        assert output_meta.state == {}
        assert output_meta.version

    def test_output_default_args_kwargs(self):
        output = MockMetric().evaluate(1, True)
        output_dict = output.data()
        assert output_dict == {"test1": 1, "test2": True, "test3": "mock_default"}
        output_meta = output.meta()
        assert output_meta.name == "tests.test_output.MockMetric.evaluate"
        assert output_meta.execution_time
        assert output_meta.execution_duration > 0
        assert output_meta.arguments == {"arg1": 1, "arg2": True, "arg3": "mock_default"}
        assert output_meta.state == {"state1": 1, "state2": 1.5, "state3": "list: len=5"}
        assert output_meta.version

    def test_output_metadata_text(self):
        @set_metadata
        def mock_metric(a: np.ndarray, s: list, d: dict, i: Iterable, t: tuple, z: bytes, n: MockMetric) -> MockOutput:
            return MockOutput(1, True, "hello")

        result = mock_metric(
            np.array([[1, 2], [3, 4], [5, 6]]),
            [1, 2, 3],
            {1: 1, 2: 2},
            range(5),
            (1, 2, 3, 4),
            b"bytes",
            MockMetric(),
        )

        meta = result.meta()
        assert meta.arguments == {
            "a": "ndarray: shape=(3, 2)",
            "s": "list: len=3",
            "d": "dict: len=2",
            "i": "range: len=5",
            "t": "tuple: len=4",
            "z": b"bytes",
            "n": "MockMetric",
        }


@pytest.mark.required
class TestStateIsReadFromTheInstance:
    """``state`` names attributes to record alongside a result.

    Read after the call rather than before, because an attribute that describes what the
    call ran *against* does not exist until it has run.
    """

    def test_a_state_name_the_class_does_not_carry_is_an_error(self):
        """A misspelled name is a typo in the decorator, not a null worth recording.

        Recording it as ``None`` writes a plausible-looking null into every result the
        method returns, and the typo is then invisible in the output it was meant to
        attribute.
        """

        class _Typo:
            state1: int = 1

            @set_metadata(state=["state1", "state1_typo"])
            def evaluate(self) -> MockOutput:
                return MockOutput(1, True, "value")

        with pytest.raises(AttributeError, match="state1_typo"):
            _Typo().evaluate()

    def test_state_is_read_after_the_call(self):
        """An attribute the call itself sets is recorded with the value it ended on."""

        class _Late:
            computed: str = "before"

            @set_metadata(state=["computed"])
            def evaluate(self) -> MockOutput:
                self.computed = "after"
                return MockOutput(1, True, "value")

        assert _Late().evaluate().meta().state == {"computed": "after"}


@pytest.mark.required
class TestCollectionOutput:
    dict_data = {"mock_key": 1.0}
    list_data = [1.0]

    def test_mapping_output(self):
        t = MockMappingOutput(self.dict_data)

        assert t.meta() == ExecutionMetadata._empty()
        assert len(t) == 1
        assert t.data() == self.dict_data
        assert repr(t)

    def test_mapping_output_meta(self):
        @set_metadata
        def mock_output():
            return MockMappingOutput(self.dict_data)

        t = mock_output()
        assert t.meta() != {}
        assert len(t) == 1
        assert t.data() == self.dict_data
        assert repr(t)

    def test_sequence_output(self):
        t = MockSequenceOutput(self.list_data)

        assert t.meta() == ExecutionMetadata._empty()
        assert len(t) == 1
        assert t.data() == self.list_data
        assert str(t)
        assert repr(t)

    def test_sequence_output_meta(self):
        @set_metadata
        def mock_output():
            return MockSequenceOutput(self.list_data)

        t = mock_output()
        assert t.meta() != {}
        assert len(t) == 1
        assert t.data() == self.list_data


@pytest.mark.required
class TestOutputSource:
    def test_defaults_to_none(self):
        out = MockOutput(1, True, "a")
        assert out.dataset is None
        assert out.metadata is None

    def test_attach_sets_both_and_dataset_falls_back_to_metadata(self):
        ds: Any = object()
        md: Any = type("Md", (), {"dataset": "from-metadata"})()
        assert MockOutput(1, True, "a")._attach(dataset=ds, metadata=md).dataset is ds
        assert MockOutput(1, True, "a")._attach(metadata=md).dataset == "from-metadata"

    def test_source_is_not_data_and_is_not_pickled(self):
        import copy
        import pickle

        md: Any = type("Md", (), {"dataset": None})()
        out = MockOutput(1, True, "a")._attach(dataset=cast(Any, [1, 2]), metadata=md)
        assert set(out.data()) == {"test1", "test2", "test3"}
        for clone in (pickle.loads(pickle.dumps(out)), copy.deepcopy(out)):
            assert clone.dataset is None
            assert clone.metadata is None
            assert clone.test1 == 1


class _Annotated:
    """The smallest thing `AnnotatedDataset` accepts."""

    metadata = {"id": "d"}

    def __len__(self) -> int:
        return 0

    def __getitem__(self, i: int):
        raise IndexError(i)


class _Evaluator:
    @set_metadata
    def evaluate(self, *inputs: Any, **named: Any) -> MockOutput:
        return MockOutput(1, True, "a")


@pytest.mark.required
class TestSetMetadataRecordsSource:
    def test_the_one_annotated_dataset_handed_in_is_attached(self):
        ds = _Annotated()
        out = _Evaluator().evaluate(ds, np.zeros(3), [1, 2])
        assert out.dataset is ds
        assert out.metadata is None

    def test_two_datasets_are_ambiguous_and_attach_nothing(self):
        assert _Evaluator().evaluate(_Annotated(), _Annotated()).dataset is None

    def test_a_metadata_is_attached_and_names_its_dataset(self, get_ic_dataset):
        from dataeval import Metadata

        ds = get_ic_dataset(5)
        md = Metadata(ds)
        out = _Evaluator().evaluate(data=md)
        assert out.metadata is md
        assert out.dataset is ds

    def test_what_the_evaluator_attached_is_kept(self):
        class Keeps:
            @set_metadata
            def evaluate(self, data: Any) -> MockOutput:
                return MockOutput(1, True, "a")._attach(dataset=cast(Any, "mine"))

        assert Keeps().evaluate(_Annotated()).dataset == "mine"


@pytest.mark.required
class TestEvaluatorsRecordSource:
    @pytest.mark.parametrize("name", ["Balance", "Diversity", "Parity", "Coverage", "Duplicates"])
    def test_a_dataset_rides_on_the_output_of_each_evaluator(self, name, get_ic_dataset):
        from dataeval import Metadata
        from dataeval.bias import Balance, Diversity, Parity
        from dataeval.quality import Duplicates
        from dataeval.scope import Coverage

        rng = np.random.default_rng(0)
        ds = get_ic_dataset(
            list(rng.random((30, 3, 8, 8))), None, metadata=[{"a": i % 3, "b": i % 5} for i in range(30)]
        )
        evaluators = {
            "Balance": lambda: Balance().evaluate(ds),
            "Diversity": lambda: Diversity().evaluate(ds),
            "Parity": lambda: Parity().evaluate(ds),
            "Coverage": lambda: Coverage().evaluate(ds, embeddings=rng.random((30, 4))),
            "Duplicates": lambda: Duplicates().evaluate(ds),
        }
        out = evaluators[name]()
        assert out.dataset is ds
        assert isinstance(out.metadata, Metadata) is (name != "Duplicates")
