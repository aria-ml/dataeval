from dataclasses import dataclass
from typing import Iterable

import numpy as np

from dataeval.output import OutputMetadata, set_metadata


@dataclass
class MockOutput(OutputMetadata):
    test1: int
    test2: bool
    test3: str


@set_metadata()
def mock_metric(arg1: int, arg2: bool, arg3: str) -> MockOutput:
    return MockOutput(arg1, arg2, arg3)


class MockMetric:
    state1: int = 1
    state2: float = 1.5
    state3: list = ["a", "very", "long", "input", "list"]

    @set_metadata(["state1", "state2", "state3"])
    def evaluate(self, arg1: int, arg2: bool, arg3: str = "mock_default") -> MockOutput:
        return MockOutput(arg1, arg2, arg3)


class TestOutputMetadata:
    def test_output_metadata_data(self):
        output = mock_metric(1, True, "value")
        assert output.test1 == 1
        assert output.test2
        assert output.test3 == "value"

    def test_output_metadata_dict(self):
        output_dict = mock_metric(1, True, "value").dict()
        assert output_dict == {"test1": 1, "test2": True, "test3": "value"}

    def test_output_metadata_meta(self):
        output_meta = mock_metric(1, True, "value").meta()
        assert output_meta["name"] == "tests.test_output_metadata.mock_metric"
        assert output_meta["execution_time"]
        assert output_meta["execution_duration"] > 0
        assert set(output_meta["arguments"]) == {"arg1", "arg2", "arg3"}
        assert output_meta["state"] == {}
        assert output_meta["version"]

    def test_output_default_args_kwargs(self):
        output = MockMetric().evaluate(1, True)
        output_dict = output.dict()
        assert output_dict == {"test1": 1, "test2": True, "test3": "mock_default"}
        output_meta = output.meta()
        assert output_meta["name"] == "tests.test_output_metadata.MockMetric.evaluate"
        assert output_meta["execution_time"]
        assert output_meta["execution_duration"] > 0
        assert output_meta["arguments"] == {"arg1": 1, "arg2": True, "arg3": "mock_default"}
        assert output_meta["state"] == {"state1": 1, "state2": 1.5, "state3": "list: len=5"}
        assert output_meta["version"]

    def test_output_metadata_text(self):
        @set_metadata()
        def mock_metric(a: np.ndarray, s: list, d: dict, i: Iterable, t: tuple, z: bytes, n: MockMetric) -> MockOutput:
            return MockOutput(1, True, "hello")

        result = mock_metric(
            np.array([[1, 2], [3, 4], [5, 6]]), [1, 2, 3], {1: 1, 2: 2}, range(5), (1, 2, 3, 4), b"bytes", MockMetric()
        )

        meta = result.meta()
        assert meta["arguments"] == {
            "a": "ndarray: shape=(3, 2)",
            "s": "list: len=3",
            "d": "dict: len=2",
            "i": "range: len=5",
            "t": "tuple: len=4",
            "z": b"bytes",
            "n": "MockMetric",
        }
