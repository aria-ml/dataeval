from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.shift._drift._thresholds import ConstantThreshold, StandardDeviationThreshold, Threshold


@pytest.mark.parametrize("lower, upper", [(0.0, 1.0), (0, 1), (-1, 1), (None, 1.0), (0.1, None), (None, None)])
def test_constant_threshold_init_sets_instance_attributes(lower, upper):
    sut = ConstantThreshold(lower, upper)

    assert sut.lower == lower
    assert sut.upper == upper


def test_constant_threshold_init_sets_default_instance_attributes():
    sut = ConstantThreshold()

    assert sut.lower is None
    assert sut.upper is None


@pytest.mark.parametrize("lower, upper", [(1.0, 0.0), (0.0, -1.0), (2.1, 2.1)])
def test_constant_threshold_init_raises_threshold_exception_when_breaking_lower_upper_strict_order(lower, upper):
    with pytest.raises(ValueError, match=f"lower threshold {lower} must be less than upper threshold {upper}"):
        _ = ConstantThreshold(lower, upper)


@pytest.mark.parametrize(
    "lower, upper, param, param_type",
    [
        ("0.0", 1.0, "lower", "str"),
        (0.0, "1.0", "upper", "str"),
        (True, 1.0, "lower", "bool"),
        (0.0, True, "upper", "bool"),
        (0.0, {}, "upper", "dict"),
    ],
)
def test_constant_threshold_init_raises_invalid_arguments_exception_when_given_wrongly_typed_arguments(
    upper, lower, param, param_type
):
    with pytest.raises(
        ValueError,
        match=f"expected type of '{param}' to be 'float', 'int' or None but got '{param_type}'",
    ):
        _ = ConstantThreshold(lower, upper)


@pytest.mark.parametrize("lower, upper", [(0.0, 1.0), (0, 1), (-1, 1), (None, 1.0), (0.1, None), (None, None)])
def test_constant_threshold_returns_correct_threshold_values(lower, upper):
    t = ConstantThreshold(lower, upper)
    lt, ut = t.thresholds(np.ndarray(range(10)))

    assert lt == lower
    assert ut == upper


@pytest.mark.parametrize(
    "lower_multiplier, upper_multiplier, offset_from",
    [(1, 1, np.median), (1, None, np.median), (None, 1, np.median), (None, None, np.median)],
)
def test_standard_deviation_threshold_init_sets_instance_attributes(lower_multiplier, upper_multiplier, offset_from):
    sut = StandardDeviationThreshold(lower_multiplier, upper_multiplier, offset_from)

    assert sut.std_lower_multiplier == lower_multiplier
    assert sut.std_upper_multiplier == upper_multiplier
    assert sut.offset_from == offset_from


def test_standard_deviation_threshold_init_sets_default_instance_attributes():
    sut = StandardDeviationThreshold()

    assert sut.std_lower_multiplier == 3
    assert sut.std_upper_multiplier == 3
    assert sut.offset_from == np.nanmean


@pytest.mark.parametrize(
    "lower_multiplier, upper_multiplier, param, param_type",
    [
        ("0.0", 1.0, "std_lower_multiplier", "str"),
        (0.0, "1.0", "std_upper_multiplier", "str"),
        (True, 1.0, "std_lower_multiplier", "bool"),
        (0.0, True, "std_upper_multiplier", "bool"),
        (0.0, {}, "std_upper_multiplier", "dict"),
    ],
)
def test_standard_deviation_threshold_init_raises_invalid_arguments_exception_when_given_wrongly_typed_arguments(
    lower_multiplier, upper_multiplier, param, param_type
):
    with pytest.raises(
        ValueError,
        match=f"expected type of '{param}' to be 'float', 'int' or None but got '{param_type}'",
    ):
        _ = StandardDeviationThreshold(std_lower_multiplier=lower_multiplier, std_upper_multiplier=upper_multiplier)


@pytest.mark.parametrize("offset_from, expected", [(np.min, -1), (np.max, 1), (np.median, 0), (np.mean, 0)])
def test_standard_deviation_threshold_applies_offset_from(offset_from, expected):
    t = StandardDeviationThreshold(std_lower_multiplier=0, std_upper_multiplier=0, offset_from=offset_from)

    lt, ut = t.thresholds(np.asarray([-1, -0.5, 0, 0.5, 1]))

    assert lt == expected
    assert ut == expected


@pytest.mark.parametrize(
    "std_lower_multiplier, expected_threshold", [(1, -1.8660254037844386), (0, -1), (2, -2.732050807568877)]
)
def test_standard_deviation_threshold_correctly_applies_std_lower_multiplier(std_lower_multiplier, expected_threshold):
    t = StandardDeviationThreshold(std_lower_multiplier=std_lower_multiplier, offset_from=np.min)
    lt, _ = t.thresholds(np.asarray([-1, 1, 1, 1]))
    assert lt == expected_threshold


@pytest.mark.parametrize(
    "std_lower_multiplier, std_upper_multiplier, exp_lower_threshold, exp_upper_threshold",
    [(None, 0, None, -1.0), (0, None, -1.0, None), (None, None, None, None)],
)
def test_standard_deviation_threshold_treats_none_multiplier_as_no_threshold(
    std_lower_multiplier, std_upper_multiplier, exp_lower_threshold, exp_upper_threshold
):
    t = StandardDeviationThreshold(std_lower_multiplier, std_upper_multiplier, offset_from=np.min)
    lt, ut = t.thresholds(np.asarray([-1, 1, 1, 1]))

    assert lt == exp_lower_threshold
    assert ut == exp_upper_threshold


@pytest.mark.parametrize(
    "low_mult, up_mult, offset_from, exp_low_threshold, exp_up_threshold",
    [
        (1.4, 2, np.median, 2.382381972241136, 31.81088289679838),
        (0.3, 3.1, np.min, -2.5966324345197567, 26.83186849003749),
    ],
)
def test_standard_deviation_threshold_correctly_returns_thresholds(
    low_mult, up_mult, offset_from, exp_low_threshold, exp_up_threshold
):
    t = StandardDeviationThreshold(low_mult, up_mult, offset_from)
    lt, ut = t.thresholds(np.asarray(range(30)))

    assert lt == exp_low_threshold
    assert ut == exp_up_threshold


def test_standard_deviation_threshold_raises_threshold_exception_when_negative_lower_multiplier_given():
    with pytest.raises(ValueError, match="'std_lower_multiplier' should be greater than 0 but got value -1"):
        StandardDeviationThreshold(-1, 0)


def test_standard_deviation_threshold_raises_threshold_exception_when_negative_upper_multiplier_given():
    with pytest.raises(ValueError, match="'std_upper_multiplier' should be greater than 0 but got value -1"):
        StandardDeviationThreshold(0, -1)


def test_standard_deviation_threshold_deals_with_nan_values():
    t = StandardDeviationThreshold()
    upper, lower = t.thresholds(np.asarray([-1, 1, np.nan, 1, np.nan]))
    assert upper is not None
    assert lower is not None
    assert not np.isnan(upper)
    assert not np.isnan(lower)


@pytest.mark.parametrize(
    "threshold, obj_dict",
    [
        (
            ConstantThreshold(0.5, 0.7),
            {"type": "constant", "lower": 0.5, "upper": 0.7},
        ),
        (
            StandardDeviationThreshold(1, 2),
            {"type": "standard_deviation", "std_lower_multiplier": 1, "std_upper_multiplier": 2},
        ),
    ],
)
def test_parse_object(threshold, obj_dict):
    parsed = Threshold.parse_object(obj_dict)
    assert threshold == parsed


class MockThreshold(Threshold, threshold_type="mock"):
    def __init__(self):
        self.t = (0.2, 0.8)

    def thresholds(self, data: np.ndarray) -> tuple[float, float]:
        return self.t


def test_threshold_str():
    t = MockThreshold()
    assert str(t) == "MockThreshold({'t': (0.2, 0.8)})"


def test_threshold_repr():
    t = MockThreshold()
    assert repr(t) == "MockThreshold({'t': (0.2, 0.8)})"


def test_parse_object_raises_exception_when_threshold_type_is_not_supported():
    with pytest.raises(ValueError, match="Expected one of"):
        Threshold.parse_object({"type": "unknown"})


@pytest.mark.parametrize(
    "lower, upper, expected, override, logger",
    [
        (0.3, None, (0.3, 0.8), False, False),
        (None, 0.7, (0.2, 0.7), False, False),
        (0.3, 0.7, (0.3, 0.7), False, True),
        (0.3, None, (None, 0.8), True, False),
        (None, 0.7, (0.2, None), True, False),
        (0.3, 0.7, (None, None), True, True),
    ],
)
def test_calculate_lower_limit(lower, upper, expected, override, logger):
    t = MockThreshold()
    logger = MagicMock() if logger else None
    thresholds = t.calculate(
        np.array([]), lower_limit=lower, upper_limit=upper, override_using_none=override, logger=logger
    )
    assert thresholds == expected
    expected_call_count = (lower is not None) + (upper is not None)
    assert logger is None or logger.warning.call_count == expected_call_count


def test_threshold_equality():
    """Test that Threshold equality works correctly."""
    t1 = ConstantThreshold(0.1, 0.9)
    t2 = ConstantThreshold(0.1, 0.9)
    t3 = ConstantThreshold(0.2, 0.9)

    assert t1 == t2
    assert t1 != t3
    assert t1 != "not a threshold"


def test_threshold_registry():
    """Test that threshold subclasses are properly registered."""
    assert "constant" in Threshold._registry
    assert "standard_deviation" in Threshold._registry
    assert Threshold._registry["constant"] == ConstantThreshold
    assert Threshold._registry["standard_deviation"] == StandardDeviationThreshold


def test_standard_deviation_threshold_with_empty_array():
    """Test StandardDeviationThreshold with an empty array."""
    t = StandardDeviationThreshold()
    lt, ut = t.thresholds(np.array([]))
    assert lt is not None
    assert ut is not None
    assert np.isnan(lt)
    assert np.isnan(ut)


def test_standard_deviation_threshold_with_single_value():
    """Test StandardDeviationThreshold with a single value (std=0)."""
    t = StandardDeviationThreshold(std_lower_multiplier=2, std_upper_multiplier=2)
    lt, ut = t.thresholds(np.array([5.0]))
    # With only one value, std is 0, so thresholds should equal the value
    assert lt == 5.0
    assert ut == 5.0


def test_standard_deviation_threshold_with_all_same_values():
    """Test StandardDeviationThreshold when all values are identical (std=0)."""
    t = StandardDeviationThreshold(std_lower_multiplier=3, std_upper_multiplier=3)
    lt, ut = t.thresholds(np.array([10.0, 10.0, 10.0, 10.0]))
    # With identical values, std is 0, so thresholds should equal the mean
    assert lt == 10.0
    assert ut == 10.0


def test_standard_deviation_threshold_upper_multiplier_only():
    """Test StandardDeviationThreshold correctly applies upper multiplier only."""
    t = StandardDeviationThreshold(std_lower_multiplier=None, std_upper_multiplier=2, offset_from=np.mean)
    lt, ut = t.thresholds(np.array([0, 2, 4, 6, 8]))

    assert lt is None
    expected_mean = 4.0
    expected_std = np.std([0, 2, 4, 6, 8], ddof=0)
    assert ut is not None
    assert ut == pytest.approx(expected_mean + 2 * expected_std)


def test_constant_threshold_with_zero_values():
    """Test ConstantThreshold with zero as one of the thresholds."""
    t = ConstantThreshold(lower=-0.5, upper=0.0)
    lt, ut = t.thresholds(np.array([]))
    assert lt == -0.5
    assert ut == 0.0


def test_constant_threshold_with_large_values():
    """Test ConstantThreshold with very large values."""
    t = ConstantThreshold(lower=1e10, upper=1e15)
    lt, ut = t.thresholds(np.array([]))
    assert lt == 1e10
    assert ut == 1e15


def test_constant_threshold_with_negative_values():
    """Test ConstantThreshold with negative values."""
    t = ConstantThreshold(lower=-100.5, upper=-10.2)
    lt, ut = t.thresholds(np.array([]))
    assert lt == -100.5
    assert ut == -10.2


def test_calculate_with_no_limits():
    """Test calculate method without any limits."""
    t = MockThreshold()
    lt, ut = t.calculate(np.array([1, 2, 3]))
    assert lt == 0.2
    assert ut == 0.8


def test_calculate_with_lower_limit_not_exceeded():
    """Test calculate when lower limit is not exceeded."""
    t = MockThreshold()
    # t.thresholds returns (0.2, 0.8), lower_limit=0.1 won't trigger override
    lt, ut = t.calculate(np.array([]), lower_limit=0.1)
    assert lt == 0.2
    assert ut == 0.8


def test_calculate_with_upper_limit_not_exceeded():
    """Test calculate when upper limit is not exceeded."""
    t = MockThreshold()
    # t.thresholds returns (0.2, 0.8), upper_limit=0.9 won't trigger override
    lt, ut = t.calculate(np.array([]), upper_limit=0.9)
    assert lt == 0.2
    assert ut == 0.8


def test_parse_object_preserves_dict_mutation():
    """Test that parse_object mutates the input dict by popping 'type'."""
    obj_dict = {
        "type": "constant",
        "lower": 0.3,
        "upper": 0.7,
    }
    threshold = Threshold.parse_object(obj_dict)
    assert isinstance(threshold, ConstantThreshold)
    assert threshold.lower == 0.3
    assert threshold.upper == 0.7
    # Verify 'type' was removed from the dict
    assert "type" not in obj_dict


def test_standard_deviation_threshold_with_custom_offset_function():
    """Test StandardDeviationThreshold with a custom offset function."""

    def custom_offset(data):
        return np.percentile(data, 75)

    t = StandardDeviationThreshold(std_lower_multiplier=1, std_upper_multiplier=1, offset_from=custom_offset)
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lt, ut = t.thresholds(data)

    expected_offset = np.percentile(data, 75)
    expected_std = np.nanstd(data)
    assert lt is not None
    assert ut is not None
    assert lt == pytest.approx(expected_offset - expected_std)
    assert ut == pytest.approx(expected_offset + expected_std)
