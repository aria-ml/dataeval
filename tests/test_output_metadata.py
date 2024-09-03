from dataclasses import dataclass

from dataeval._internal.output import OutputMetadata, set_metadata


@dataclass
class TestOutput(OutputMetadata):
    test1: int
    test2: bool
    test3: str


@set_metadata("dataeval.test.mock_metric")
def mock_metric(arg1: int, arg2: bool, arg3: str) -> TestOutput:
    return TestOutput(arg1, arg2, arg3)


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
        assert output_meta["name"] == "dataeval.test.mock_metric"
        assert output_meta["execution_time"]
        assert output_meta["execution_duration"] > 0
        assert set(output_meta["arguments"]) == {"arg1", "arg2", "arg3"}
        assert output_meta["version"]
