import numpy as np
import pytest
from matplotlib.figure import Figure

from dataeval.metrics.bias import diversity
from dataeval.metrics.bias.metadata import entropy


@pytest.fixture
def metadata():
    # vals = ["a", "b"]
    str_vals = np.array(["a", "a", "a", "a", "b", "a", "a", "a", "b", "b"])
    cnt_vals = np.array(
        [0.63784, -0.86422, -0.1017, -1.95131, -0.08494, -1.02940, 0.07908, -0.31724, -1.45562, 1.03368]
    )
    md = {"var_cat": str_vals, "var_cnt": cnt_vals}
    return md


@pytest.fixture
def class_labels_int():
    return np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 1])


@pytest.fixture
def class_labels():
    return np.array(["dog", "dog", "dog", "cat", "dog", "cat", "dog", "dog", "dog", "cat"])


@pytest.fixture
def entropy_test_vars():
    ent = {}
    ent["continuous_factor_bincounts"] = [{"data": 5}]
    ent["data"] = np.stack([np.ones(10, dtype=np.intp), np.array([1, 5, 3, 5, 8, 9, 0, 2, 4, 7])]).T
    ent["names"] = ["a", "b"]
    return ent


@pytest.mark.parametrize("norm", [True, False])
def test_entropy_normalization(norm, entropy_test_vars):
    ent = entropy(
        entropy_test_vars["data"],
        entropy_test_vars["names"],
        entropy_test_vars["continuous_factor_bincounts"],
        normalized=norm,
    )
    assert ent[0] == 0


class TestDiversityUnit:
    @pytest.mark.parametrize("met", ["Simpson", "ShANnOn"])
    def test_invalid_method(self, met):
        with pytest.raises(ValueError):
            diversity([], [], method=met)  # type: ignore

    @pytest.mark.parametrize("met", ["simpson", "shannon"])
    def test_range_of_values(self, met, metadata, class_labels):
        result = diversity(class_labels, metadata, method=met)
        for div in (result.diversity_index, result.classwise):
            assert div.dtype == np.float64
            assert np.logical_and(div >= 0, div <= 1).all()

    @pytest.mark.parametrize("met", ["simpson", "shannon"])
    def test_output_dtypes(self, met, metadata, class_labels_int):
        result = diversity(class_labels_int, metadata, method=met)
        assert result.class_list.dtype is np.dtype(np.int64)
        assert type(result.metadata_names[0]) is str
        assert type(result.meta()["arguments"]["method"]) is str

    def test_base_plotting(self, class_labels_int, metadata):
        result = diversity(class_labels_int, metadata, method="simpson")
        output = result.plot()
        assert isinstance(output, Figure)
        classwise_output = result.plot(plot_classwise=True)
        assert isinstance(classwise_output, Figure)

    def test_plotting_vars(self, class_labels, metadata):
        result = diversity(class_labels, metadata, method="shannon")
        row_labels = np.arange(result.class_list.size)
        col_labels = np.arange(len(result.metadata_names))
        classwise_output = result.plot(row_labels, col_labels, True)
        assert isinstance(classwise_output, Figure)


class TestDiversityFunctional:
    def test_simple_input_simpson(self):
        metadata = {"factor1": [5, 5, 6, 6]}
        class_labels = [0, 0, 1, 1]
        continuous_factor_bincounts = {}
        result = diversity(class_labels, metadata, continuous_factor_bincounts, method="simpson")
        expected_index = np.array([1, 1])
        expected_classwise = np.array([[0], [0]])
        np.testing.assert_array_almost_equal(result.diversity_index, expected_index)
        np.testing.assert_array_almost_equal(result.classwise, expected_classwise)

    def test_binned_input_simpson(self):
        metadata = {"factor1": [5.1, 4.9, 6.1, 6.2]}
        class_labels = [0, 0, 1, 1]
        continuous_factor_bincounts = {"factor1": 2}
        result = diversity(class_labels, metadata, continuous_factor_bincounts, method="simpson")
        expected_index = np.array([1, 1])
        expected_classwise = np.array([[0], [0]])
        np.testing.assert_array_almost_equal(result.diversity_index, expected_index)
        np.testing.assert_array_almost_equal(result.classwise, expected_classwise)

    def test_homog_md_input_simpson(self):
        metadata = {"factor1": [5, 5, 5, 5]}
        class_labels = [0, 0, 1, 1]
        continuous_factor_bincounts = {}
        result = diversity(class_labels, metadata, continuous_factor_bincounts, method="simpson")
        expected_index = np.array([1, 0])
        expected_classwise = np.array([[0], [0]])
        np.testing.assert_array_almost_equal(result.diversity_index, expected_index)
        np.testing.assert_array_almost_equal(result.classwise, expected_classwise)

    def test_homog_cls_input_simpson(self):
        metadata = {"factor1": [5, 5, 6, 6]}
        class_labels = [0, 0, 0, 0]
        continuous_factor_bincounts = {}
        result = diversity(class_labels, metadata, continuous_factor_bincounts, method="simpson")
        expected_index = np.array([0, 1])
        expected_classwise = np.array([[1]])
        np.testing.assert_array_almost_equal(result.diversity_index, expected_index)
        np.testing.assert_array_almost_equal(result.classwise, expected_classwise)

    def test_diverse_input_simpson(self):
        metadata = {"factor1": [5, 6, 5, 6, 5, 6], "factor2": [0, 0, 5, 0, 5, 5]}
        class_labels = [1, 1, 1, 2, 2, 2]
        continuous_factor_bincounts = {}
        result = diversity(class_labels, metadata, continuous_factor_bincounts, method="simpson")
        expected_index = np.array([1, 1, 1])
        expected_classwise = np.array([[0.8, 0.8], [0.8, 0.8]])
        np.testing.assert_array_almost_equal(result.diversity_index, expected_index)
        np.testing.assert_array_almost_equal(result.classwise, expected_classwise)

    def test_simple_input_shannon(self):
        metadata = {"factor1": [5, 5, 5, 6, 6, 6]}
        class_labels = [0, 0, 0, 1, 1, 1]
        continuous_factor_bincounts = {}
        result = diversity(class_labels, metadata, continuous_factor_bincounts, method="shannon")
        expected_index = np.array([1, 1])
        expected_classwise = np.array([[0], [0]])
        np.testing.assert_array_almost_equal(result.diversity_index, expected_index)
        np.testing.assert_array_almost_equal(result.classwise, expected_classwise)

    def test_homog_md_input_shannon(self):
        metadata = {"factor1": [5, 5, 5, 5, 5, 5]}
        class_labels = [0, 0, 0, 1, 1, 1]
        continuous_factor_bincounts = {}
        result = diversity(class_labels, metadata, continuous_factor_bincounts, method="shannon")
        expected_index = np.array([1, 0])
        expected_classwise = np.array([[0], [0]])
        np.testing.assert_array_almost_equal(result.diversity_index, expected_index)
        np.testing.assert_array_almost_equal(result.classwise, expected_classwise)

    def test_homog_cls_input_shannon(self):
        metadata = {"factor1": [5, 5, 5, 6, 6, 6]}
        class_labels = [0, 0, 0, 0, 0, 0]
        continuous_factor_bincounts = {}
        result = diversity(class_labels, metadata, continuous_factor_bincounts, method="shannon")
        expected_index = np.array([0, 1])
        expected_classwise = np.array([[1]])
        np.testing.assert_array_almost_equal(result.diversity_index, expected_index)
        np.testing.assert_array_almost_equal(result.classwise, expected_classwise)

    def test_diverse_input_shannon(self):
        metadata = {"factor1": [5, 6, 5, 6, 5, 6], "factor2": [0, 0, 5, 0, 5, 5]}
        class_labels = [1, 1, 1, 2, 2, 2]
        continuous_factor_bincounts = {}
        result = diversity(class_labels, metadata, continuous_factor_bincounts, method="shannon")
        expected_index = np.array([1, 1, 1])
        expected_classwise = np.array([[0.8, 0.8], [0.8, 0.8]])
        np.testing.assert_array_almost_equal(result.diversity_index, expected_index)
        np.testing.assert_array_less(expected_classwise, result.classwise)
