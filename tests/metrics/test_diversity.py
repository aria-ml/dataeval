from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.data._metadata import Metadata
from dataeval.outputs._base import ExecutionMetadata

try:
    from matplotlib.figure import Figure
except ImportError:
    Figure = type(None)

from dataeval.metrics.bias import diversity
from tests.conftest import preprocess


@pytest.fixture(scope="module")
def metadata_results():
    str_vals = ["a", "a", "a", "a", "b", "a", "a", "a", "b", "b"]
    cnt_vals = [0.63784, -0.86422, -0.1017, -1.95131, -0.08494, -1.02940, 0.07908, -0.31724, -1.45562, 1.03368]
    class_labels = ["dog", "dog", "dog", "cat", "dog", "cat", "dog", "dog", "dog", "cat"]
    md = {"var_cat": str_vals, "var_cnt": cnt_vals}
    return preprocess(md, class_labels, {"var_cnt": 3})


@pytest.mark.required
class TestDiversityUnit:
    @pytest.mark.parametrize("met", ["Simpson", "ShANnOn"])
    def test_invalid_method(self, metadata_results, met):
        with pytest.raises(ValueError):
            diversity(metadata_results, method=met)

    @pytest.mark.parametrize("met", ["simpson", "shannon"])
    def test_range_of_values(self, metadata_results, met):
        result = diversity(metadata_results, method=met)
        for div in (result.diversity_index, result.classwise):
            assert np.issubdtype(div.dtype, np.double)
            assert np.logical_and((div >= 0).all(), (div <= 1).all())

    @pytest.mark.parametrize("met", ["simpson", "shannon"])
    def test_output_dtypes(self, metadata_results, met):
        result = diversity(metadata_results, method=met)
        assert np.issubdtype(result.diversity_index.dtype, np.double)
        assert np.issubdtype(result.classwise.dtype, np.double)
        assert isinstance(result.factor_names[0], str)
        assert isinstance(result.meta(), ExecutionMetadata)

    def test_empty_metadata(self):
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.discrete_factor_names = []
        mock_metadata.continuous_factor_names = []
        with pytest.raises(ValueError):
            diversity(mock_metadata)


@pytest.mark.requires_all
class TestDiversityPlot:
    def test_base_plotting(self, metadata_results):
        result = diversity(metadata_results, method="simpson")
        output = result.plot()
        assert isinstance(output, Figure)
        classwise_output = result.plot(plot_classwise=True)
        assert isinstance(classwise_output, Figure)

    def test_plotting_vars(self, metadata_results):
        result = diversity(metadata_results, method="shannon")
        row_labels = np.arange(len(result.class_names))
        col_labels = np.arange(len(result.factor_names))
        classwise_output = result.plot(row_labels, col_labels, plot_classwise=True)
        assert isinstance(classwise_output, Figure)


@pytest.mark.optional
class TestDiversityFunctional:
    @pytest.mark.parametrize(
        "metadata, expected_result",
        [
            (
                preprocess({"factor1": [5, 5, 5, 6, 6, 6]}, [0, 0, 0, 1, 1, 1], {}),
                (np.array([1, 1]), np.array([[0], [0]])),
            ),
            (
                preprocess({"factor1": [5.1, 4.9, 4.9, 6.1, 6.2, 6.1]}, [0, 0, 0, 1, 1, 1], {"factor1": 2}),
                (np.array([1, 1]), np.array([[0], [0]])),
            ),
            (
                preprocess({"factor1": [5, 5, 5, 5, 5, 5]}, [0, 0, 0, 1, 1, 1], {}),
                (np.array([1, 0]), np.array([[0], [0]])),
            ),
            (
                preprocess({"factor1": [5, 5, 5, 6, 6, 6]}, [0, 0, 0, 0, 0, 0], {}),
                (np.array([0, 1]), np.array([[1]])),
            ),
            (
                preprocess({"factor1": [5, 6, 5, 6, 5, 6], "factor2": [0, 0, 5, 0, 5, 5]}, [1, 1, 1, 2, 2, 2], {}),
                (np.array([1, 1, 1]), np.array([[0.8, 0.8], [0.8, 0.8]])),
            ),
        ],
    )
    def test_simpson(self, metadata, expected_result):
        result = diversity(metadata, method="simpson")
        np.testing.assert_array_almost_equal(result.diversity_index, expected_result[0])
        np.testing.assert_array_almost_equal(result.classwise, expected_result[1])

    @pytest.mark.parametrize(
        "metadata, expected_result",
        [
            (
                preprocess({"factor1": [5, 5, 5, 6, 6, 6]}, [0, 0, 0, 1, 1, 1], {}),
                (np.array([1, 1]), np.array([[0], [0]])),
            ),
            (
                preprocess({"factor1": [5.1, 4.9, 4.9, 6.1, 6.2, 6.1]}, [0, 0, 0, 1, 1, 1], {"factor1": 2}),
                (np.array([1, 1]), np.array([[0], [0]])),
            ),
            (
                preprocess({"factor1": [5, 5, 5, 5, 5, 5]}, [0, 0, 0, 1, 1, 1], {}),
                (np.array([1, 0]), np.array([[0], [0]])),
            ),
            (
                preprocess({"factor1": [5, 5, 5, 6, 6, 6]}, [0, 0, 0, 0, 0, 0], {}),
                (np.array([0, 1]), np.array([[1]])),
            ),
            (
                preprocess({"factor1": [5, 6, 5, 6, 5, 6], "factor2": [0, 0, 5, 0, 5, 5]}, [1, 1, 1, 2, 2, 2], {}),
                (np.array([1, 1, 1]), np.array([[0.91829583, 0.91829583], [0.91829583, 0.91829583]])),
            ),
        ],
    )
    def test_shannon(self, metadata, expected_result):
        result = diversity(metadata, method="shannon")
        np.testing.assert_array_almost_equal(result.diversity_index, expected_result[0])
        np.testing.assert_array_almost_equal(result.classwise, expected_result[1])
