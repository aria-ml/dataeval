import re
from itertools import compress

import numpy as np
import numpy.testing as npt
import pytest

from dataeval.metadata._ood import (
    OODOutput,
    _calc_median_deviations,
    _combine_metadata,
    _validate_factors_and_data,
    _validate_keys,
    most_deviated_factors,
)
from dataeval.utils.metadata import Metadata
from src.dataeval.utils.metadata import merge, preprocess

BIG_SAMPLES_COUNT = 1000
BIG_FEATURE_NAMES = ["DJIA", "temperature", "uptime"]


class MockMetadata(Metadata):
    def __init__(
        self,
        discrete_factor_names=None,
        discrete_data=None,
        continuous_factor_names=None,
        continuous_data=None,
        total_num_factors=0,
    ):
        discrete_factor_names = [] if discrete_factor_names is None else discrete_factor_names
        discrete_data = np.array([]) if discrete_data is None else discrete_data
        continuous_factor_names = [] if continuous_factor_names is None else continuous_factor_names
        continuous_data = np.array([]) if continuous_data is None else continuous_data

        super().__init__(
            discrete_factor_names=discrete_factor_names,
            discrete_data=discrete_data,
            continuous_factor_names=continuous_factor_names,
            continuous_data=continuous_data,
            class_names=np.array([]),
            class_labels=np.array([]),
            total_num_factors=total_num_factors,
            image_indices=np.array([]),
        )


@pytest.fixture
def expected_deviations_small() -> list[tuple[str, float]]:
    """
    Expected returns for meta_small data

    ```
    [("time", 2.0), ("time", 2.590909), ("time", 3.509091)]
    ```
    """
    return [("time", 2.0), ("time", 2.590909), ("time", 3.509091)]


@pytest.fixture
def metadata_ref_small() -> Metadata:
    """
    Creates a reference Metadata output class from raw metadata using merge and preprocess

    Currently cannot handle numpy arrays.
    Discrete data format incompatible with metadata function. Only continuous is used

    ```
    meta: dict[str, list] = {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]}
    ```
    """
    meta: dict[str, list] = {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]}

    samples = len(meta["time"])

    metadata = preprocess(merge([meta]), range(samples), continuous_factor_bins={k: len(v) for k, v in meta.items()})

    return MockMetadata(
        continuous_factor_names=metadata.continuous_factor_names,
        continuous_data=metadata.continuous_data,
        total_num_factors=metadata.total_num_factors,
    )


@pytest.fixture
def metadata_tst_small() -> Metadata:
    """
    Creates a test Metadata output class from raw metadata using merge and preprocess

    Currently cannot handle numpy arrays.
    Discrete data format incompatible with metadata function. Only continuous is used

    ```
    meta: dict[str, list] = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]}
    ```
    """
    meta: dict[str, list] = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, -2111]}

    samples = len(meta["time"])

    metadata = preprocess(merge([meta]), range(samples), continuous_factor_bins={k: len(v) for k, v in meta.items()})

    return MockMetadata(
        continuous_factor_names=metadata.continuous_factor_names,
        continuous_data=metadata.continuous_data,
        total_num_factors=metadata.total_num_factors,
    )


@pytest.fixture
def metadata_tst_scalar() -> Metadata:
    """
    Creates a test Metadata output class with a single value from raw metadata using merge and preprocess

    Currently cannot handle numpy arrays.
    Discrete data format incompatible with metadata function. Only continuous is used

    ```
    meta: dict[str, list] = {"time": [42], "altitude": [0]}
    ```
    """
    meta: dict[str, list] = {"time": [42], "altitude": [0]}

    metadata = preprocess(merge([meta]), [0], continuous_factor_bins={k: len(v) for k, v in meta.items()})

    return MockMetadata(
        continuous_factor_names=metadata.continuous_factor_names,
        continuous_data=metadata.continuous_data,
        total_num_factors=metadata.total_num_factors,
    )


@pytest.fixture()
def metadata_ref_big(RNG):
    """
    Higher count Metadata values with a normal distribution
    """
    shape = (BIG_SAMPLES_COUNT, len(BIG_FEATURE_NAMES))

    # Generate data samples
    X = RNG.normal(size=shape)

    # Each key has 1000 samples of feature_i
    metadata = {k: X[:, i].tolist() for i, k in enumerate(BIG_FEATURE_NAMES)}

    MD = preprocess(
        merge([metadata]),
        class_labels=range(1000),
        continuous_factor_bins={k: len(v) for k, v in metadata.items()},
    )

    return MockMetadata(
        continuous_factor_names=MD.continuous_factor_names,
        continuous_data=MD.continuous_data,
        total_num_factors=MD.total_num_factors,
    )


@pytest.fixture
def metadata_tst_big(RNG):
    """
    Higher count Metadata values with two added random normal
    distributions in the first half and second half respectively
    """
    bigdata_size = (BIG_SAMPLES_COUNT, len(BIG_FEATURE_NAMES))

    # Generate data samples
    X = RNG.normal(size=bigdata_size)

    # Add noise
    half = BIG_SAMPLES_COUNT // 2
    X[:half, 0] -= RNG.normal(loc=5000, scale=200, size=half)
    X[half:, 1] += RNG.normal(loc=100, scale=10, size=half)

    # Each key has BIG_SAMPLES_COUNT of feature_i
    metadata_tst = {k: X[:, i].tolist() for i, k in enumerate(BIG_FEATURE_NAMES)}

    MD = preprocess(
        merge([metadata_tst]),
        class_labels=range(1000),
        continuous_factor_bins={k: len(v) for k, v in metadata_tst.items()},
    )

    return MockMetadata(
        continuous_factor_names=MD.continuous_factor_names,
        continuous_data=MD.continuous_data,
        total_num_factors=MD.total_num_factors,
    )


@pytest.mark.required
class TestMetadataValidation:
    """
    Group functions used for validation of Metadata properties

    Methods
    -------
    test_sufficient_samples
        Tests all combinations of mask lengths and True values
    test_is_identical
        Tests if two list of strings are identical
    """

    @pytest.mark.parametrize(
        "metadata_fixture",
        (
            "metadata_tst_scalar",
            "metadata_tst_small",
        ),
    )
    def test_insufficient_samples(self, request, metadata_tst_scalar, metadata_fixture):
        """
        Tests reference set has more than 3 samples regardless of test set
        """

        tst = request.getfixturevalue(metadata_fixture)

        ood = OODOutput(
            is_ood=np.array([True, True, True]),
            instance_score=np.array([]),
            feature_score=None,
        )

        # Neither have enough samples
        with pytest.warns(UserWarning, match="At least 3 reference metadata samples are needed, got 1"):
            res = most_deviated_factors(
                metadata_1=metadata_tst_scalar,
                metadata_2=tst,
                ood=ood,
            )
            assert res == []

    @pytest.mark.parametrize(
        "lst",
        (
            ["a", "b", "c"],
            ["a", "b", "c", "e"],
            ["a", "b", "d", "c"],
            [],
        ),
    )
    def test_validate_different_keys(self, lst):
        """
        Test lists that differ from the reference list are found to be not identical
        """
        reference: list[str] = ["a", "b", "c", "d"]

        error_msg = f"Metadata keys must be identical, got {reference} and {lst}"

        with pytest.raises(ValueError) as exec_info:
            _validate_keys(reference, lst)

        assert str(exec_info.value) == error_msg

    @pytest.mark.parametrize("lst", (["a", "b"], ["a", "a"], []))
    def test_validate_identical_keys(self, lst):
        """Tests known positive cases"""

        _validate_keys(lst, lst)

    @pytest.mark.parametrize(
        "ood_mask, error_msg",
        (
            (
                np.array([True, True, True, True]),
                "ood and test metadata must have the same length, got 4 and 3 respectively.",
            ),
            (
                np.array([True, True]),
                "ood and test metadata must have the same length, got 2 and 3 respectively.",
            ),
        ),
    )
    def test_mismatched_ood_test_lengths(self, ood_mask, error_msg, metadata_ref_small, metadata_tst_small):
        """Tests the case where the ood mask differs from the length of the data lengths"""

        ood = OODOutput(
            is_ood=ood_mask,
            instance_score=np.array([]),
            feature_score=None,
        )

        with pytest.raises(
            ValueError,
            match=error_msg,
        ):
            most_deviated_factors(metadata_ref_small, metadata_tst_small, ood)

    def test_mismatched_discrete_continuous(self):
        """
        Tests that discrete and continuous data with differing number of factors
        is not allowed as deviations are calculated over the entire data.

        Should raise error during hstack
        """
        SAMPLES = 3
        ood = OODOutput(
            is_ood=np.array([True] * SAMPLES),
            instance_score=np.array([]),
            feature_score=None,
        )

        # If discrete and continuous should have separate factors,
        # this can be updated in the future
        md_tst = MockMetadata(
            discrete_factor_names=["a", "b"],
            discrete_data=np.ones((SAMPLES, 2)),
            continuous_factor_names=["a", "b", "c"],
            continuous_data=np.ones((SAMPLES, 3)),
            total_num_factors=5,
        )

        # with pytest.raises(ValueError):
        most_deviated_factors(md_tst, md_tst, ood)


@pytest.mark.required
class TestDeviatedFactors:
    @pytest.mark.parametrize("is_ood_0", (False, True))
    @pytest.mark.parametrize("is_ood_1", (False, True))
    @pytest.mark.parametrize("is_ood_2", (False, True))
    def test_output_values(
        self, metadata_ref_small, metadata_tst_small, is_ood_0, is_ood_1, is_ood_2, expected_deviations_small
    ):
        """Tests all combinations of ood flags correctly select most deviated metadata factor"""

        # Creates
        ood_mask = np.array([is_ood_0, is_ood_1, is_ood_2], dtype=np.bool)

        ood = OODOutput(
            is_ood=ood_mask,
            instance_score=np.array([]),
            feature_score=None,
        )

        result = most_deviated_factors(metadata_ref_small, metadata_tst_small, ood)

        # Built-in list masking function
        expected = compress(expected_deviations_small, ood_mask)

        for ex, res in zip(expected, result):
            assert ex[0] == res[0]
            npt.assert_allclose(res[1], ex[1])

    def test_scalar_metadata(self, metadata_ref_small, metadata_tst_scalar):
        """Tests single test value runs with correct output shape and value"""

        ood = OODOutput(
            is_ood=np.array([True]),
            instance_score=np.array([]),
            feature_score=None,
        )
        result = most_deviated_factors(metadata_ref_small, metadata_tst_scalar, ood=ood)

        assert len(result) == 1
        assert result[0] == ("time", 17.545454545454547)

    # With a more realistic number of samples, make sure that
    @pytest.mark.optional
    def test_big_data_with_noise(self, metadata_ref_big, metadata_tst_big):
        """Tests larger matrix with multiple samples and factors"""

        samples = BIG_SAMPLES_COUNT  # * 2 TODO: Fix when discrete data added to Metadata

        is_ood = np.array([True] * samples)
        ood = OODOutput(is_ood=is_ood, instance_score=np.array([]), feature_score=np.array([]))
        output = most_deviated_factors(metadata_ref_big, metadata_tst_big, ood)

        half = samples // 2

        assert {out[0] for out in output[:half]} == {BIG_FEATURE_NAMES[0]}
        assert {out[0] for out in output[half:]} == {BIG_FEATURE_NAMES[1]}


@pytest.mark.parametrize("samples_ref", (1, 5, 10, 100))
@pytest.mark.parametrize("samples_tst", (1, 5, 10, 100))
@pytest.mark.parametrize("factors", (1, 5, 10))
def test_calc_median_deviations(samples_ref, samples_tst, factors):
    """Tests all combinations of 1-D and 2-D inputs including (1, F), (S, 1), and (1, 1)"""

    # (S_ref, F)
    r = np.arange(samples_ref * factors).reshape(samples_ref, factors)
    # (S_tst, F)
    t = np.arange(samples_tst * factors).reshape(samples_tst, factors)

    res = _calc_median_deviations(r, t)

    assert res.shape == (samples_tst, factors)
    assert not np.any(np.isnan(res))


@pytest.mark.parametrize("samples", (1, 3, 5))
@pytest.mark.parametrize("factors", (0, 1, 3))
def test_valid_factors_and_data(samples, factors):
    """Tests only number of factors determines validity"""

    f = ["a"] * factors
    d = np.ones(shape=(samples, factors))

    _validate_factors_and_data(f, d)


def test_invalid_factors_and_data():
    """Tests inequality of factors and data columns"""
    f = ["a", "b", "c"]
    d = np.ones((3, 1))

    error_msg = "Factors and data have mismatched lengths. Got 3 and 1"

    with pytest.raises(ValueError, match=error_msg):
        _validate_factors_and_data(f, d)


def test_differing_num_factors(metadata_ref_small):
    """Tests the quick check of total num factors equality"""

    error_msg = re.escape("Number of factors differs between metadata_1 (5) and metadata_2 (0)")

    with pytest.raises(ValueError, match=error_msg):
        _combine_metadata(metadata_ref_small, MockMetadata())
