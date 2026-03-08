import numpy as np
import pytest

from dataeval.utils.thresholds import (
    AdaptiveThreshold,
    ConstantThreshold,
    IQRThreshold,
    ModifiedZScoreThreshold,
    ZScoreThreshold,
    _Threshold,
    resolve_threshold,
)


@pytest.mark.parametrize(("lower", "upper"), [(0.0, 1.0), (0, 1), (-1, 1), (None, 1.0), (0.1, None), (None, None)])
def test_constant_threshold_init_sets_instance_attributes(lower, upper):
    sut = ConstantThreshold(lower, upper)

    assert sut.lower == lower
    assert sut.upper == upper


def test_constant_threshold_init_sets_default_instance_attributes():
    sut = ConstantThreshold()

    assert sut.lower is None
    assert sut.upper is None


@pytest.mark.parametrize(("lower", "upper"), [(1.0, 0.0), (0.0, -1.0), (2.1, 2.1)])
def test_constant_threshold_init_raises_threshold_exception_when_breaking_lower_upper_strict_order(lower, upper):
    with pytest.raises(ValueError, match=f"lower threshold {lower} must be less than upper threshold {upper}"):
        _ = ConstantThreshold(lower, upper)


@pytest.mark.parametrize(
    ("lower", "upper", "param", "param_type"),
    [
        ("0.0", 1.0, "lower", "str"),
        (0.0, "1.0", "upper", "str"),
        (True, 1.0, "lower", "bool"),
        (0.0, True, "upper", "bool"),
        (0.0, {}, "upper", "dict"),
    ],
)
def test_constant_threshold_init_raises_invalid_arguments_exception_when_given_wrongly_typed_arguments(
    upper,
    lower,
    param,
    param_type,
):
    with pytest.raises(
        ValueError,
        match=f"expected type of '{param}' to be 'float', 'int' or None but got '{param_type}'",
    ):
        _ = ConstantThreshold(lower, upper)


@pytest.mark.parametrize(("lower", "upper"), [(0.0, 1.0), (0, 1), (-1, 1), (None, 1.0), (0.1, None), (None, None)])
def test_constant_threshold_returns_correct_threshold_values(lower, upper):
    t = ConstantThreshold(lower, upper)
    lt, ut = t._derive(np.ndarray(range(10)))

    assert lt == lower
    assert ut == upper


@pytest.mark.parametrize(
    ("threshold", "obj_dict"),
    [
        (
            ConstantThreshold(0.5, 0.7),
            {"type": "constant", "lower": 0.5, "upper": 0.7},
        ),
    ],
)
def test_parse_object(threshold, obj_dict):
    parsed = _Threshold.parse_object(obj_dict)
    assert threshold == parsed


class MockThreshold(_Threshold, threshold_type="mock"):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = (0.2, 0.8)

    def _derive(self, data: np.ndarray) -> tuple[float, float]:
        return self.t


def test_threshold_str():
    t = MockThreshold()
    assert str(t) == "MockThreshold({'t': (0.2, 0.8)})"


def test_threshold_str_with_limits():
    t = MockThreshold(lower_limit=0.3)
    assert str(t) == "MockThreshold({'lower_limit': 0.3, 't': (0.2, 0.8)})"


def test_threshold_repr():
    t = MockThreshold()
    assert repr(t) == "MockThreshold({'t': (0.2, 0.8)})"


def test_parse_object_raises_exception_when_threshold_type_is_not_supported():
    with pytest.raises(ValueError, match="Expected one of"):
        _Threshold.parse_object({"type": "unknown"})


@pytest.mark.parametrize(
    ("lower_limit", "upper_limit", "expected"),
    [
        (None, None, (0.2, 0.8)),
        (0.3, None, (0.3, 0.8)),  # lower clamped up from 0.2 to 0.3
        (None, 0.7, (0.2, 0.7)),  # upper clamped down from 0.8 to 0.7
        (0.3, 0.7, (0.3, 0.7)),  # both clamped
        (0.1, None, (0.2, 0.8)),  # lower_limit below threshold → no effect
        (None, 0.9, (0.2, 0.8)),  # upper_limit above threshold → no effect
    ],
)
def test_calculate_clamps_with_limits(lower_limit, upper_limit, expected):
    t = MockThreshold(lower_limit=lower_limit, upper_limit=upper_limit)
    result = t(np.array([]))
    assert result == expected


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
    assert "constant" in _Threshold._registry
    assert "zscore" in _Threshold._registry
    assert _Threshold._registry["constant"] == ConstantThreshold
    assert _Threshold._registry["zscore"] == ZScoreThreshold


def test_constant_threshold_with_zero_values():
    """Test ConstantThreshold with zero as one of the thresholds."""
    t = ConstantThreshold(lower=-0.5, upper=0.0)
    lt, ut = t._derive(np.array([]))
    assert lt == -0.5
    assert ut == 0.0


def test_constant_threshold_with_large_values():
    """Test ConstantThreshold with very large values."""
    t = ConstantThreshold(lower=1e10, upper=1e15)
    lt, ut = t._derive(np.array([]))
    assert lt == 1e10
    assert ut == 1e15


def test_constant_threshold_with_negative_values():
    """Test ConstantThreshold with negative values."""
    t = ConstantThreshold(lower=-100.5, upper=-10.2)
    lt, ut = t._derive(np.array([]))
    assert lt == -100.5
    assert ut == -10.2


def test_calculate_with_no_limits():
    """Test calculate method without any limits."""
    t = MockThreshold()
    lt, ut = t(np.array([1, 2, 3]))
    assert lt == 0.2
    assert ut == 0.8


def test_calculate_limits_stored_on_init():
    """Test that limits are configured at construction, not per-call."""
    t = MockThreshold(lower_limit=0.3, upper_limit=0.7)
    assert t.lower_limit == 0.3
    assert t.upper_limit == 0.7


def test_parse_object_preserves_dict_mutation():
    """Test that parse_object mutates the input dict by popping 'type'."""
    obj_dict = {
        "type": "constant",
        "lower": 0.3,
        "upper": 0.7,
    }
    threshold = _Threshold.parse_object(obj_dict)
    assert isinstance(threshold, ConstantThreshold)
    assert threshold.lower == 0.3
    assert threshold.upper == 0.7
    # Verify 'type' was removed from the dict
    assert "type" not in obj_dict


# --- ZScoreThreshold tests ---


class TestZScoreThreshold:
    def test_defaults(self):
        t = ZScoreThreshold()
        assert t.lower_multiplier == 3.0
        assert t.upper_multiplier == 3.0

    def test_custom_multipliers(self):
        t = ZScoreThreshold(lower_multiplier=2.0, upper_multiplier=4.0)
        assert t.lower_multiplier == 2.0
        assert t.upper_multiplier == 4.0

    def test_thresholds_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t = ZScoreThreshold(lower_multiplier=1.0, upper_multiplier=1.0)
        lt, ut = t._derive(data)
        mean = np.mean(data)
        std = np.std(data)
        assert lt == pytest.approx(mean - std)
        assert ut == pytest.approx(mean + std)

    def test_thresholds_none_lower(self):
        t = ZScoreThreshold(lower_multiplier=None, upper_multiplier=2.0)
        lt, ut = t._derive(np.array([1.0, 2.0, 3.0]))
        assert lt is None
        assert ut is not None

    def test_thresholds_none_upper(self):
        t = ZScoreThreshold(lower_multiplier=2.0, upper_multiplier=None)
        lt, ut = t._derive(np.array([1.0, 2.0, 3.0]))
        assert lt is not None
        assert ut is None

    def test_zero_variance_returns_none(self):
        t = ZScoreThreshold()
        lt, ut = t._derive(np.array([5.0, 5.0, 5.0]))
        assert lt is None
        assert ut is None

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            ZScoreThreshold(lower_multiplier=-1.0)

    def test_registry(self):
        assert "zscore" in _Threshold._registry
        assert _Threshold._registry["zscore"] == ZScoreThreshold


# --- ModifiedZScoreThreshold tests ---


class TestModifiedZScoreThreshold:
    def test_defaults(self):
        t = ModifiedZScoreThreshold()
        assert t.lower_multiplier == 3.5
        assert t.upper_multiplier == 3.5

    def test_thresholds_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t = ModifiedZScoreThreshold(lower_multiplier=1.0, upper_multiplier=1.0)
        lt, ut = t._derive(data)
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        scale = mad / 0.6745
        assert lt == pytest.approx(median_val - scale)
        assert ut == pytest.approx(median_val + scale)

    def test_thresholds_none_lower(self):
        t = ModifiedZScoreThreshold(lower_multiplier=None)
        lt, ut = t._derive(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert lt is None
        assert ut is not None

    def test_zero_mad_falls_back_to_mean(self):
        """When MAD is 0 (all values equal except one), falls back to mean of abs deviations."""
        data = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
        t = ModifiedZScoreThreshold(lower_multiplier=2.0, upper_multiplier=2.0)
        lt, ut = t._derive(data)
        # MAD would be 0 (median of |x - 1.0|), so falls back to mean
        assert lt is not None
        assert ut is not None

    def test_all_equal_returns_none(self):
        t = ModifiedZScoreThreshold()
        lt, ut = t._derive(np.array([3.0, 3.0, 3.0]))
        assert lt is None
        assert ut is None

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            ModifiedZScoreThreshold(upper_multiplier=-0.5)

    def test_registry(self):
        assert "modzscore" in _Threshold._registry
        assert _Threshold._registry["modzscore"] == ModifiedZScoreThreshold


# --- IQRThreshold tests ---


class TestIQRThreshold:
    def test_defaults(self):
        t = IQRThreshold()
        assert t.lower_multiplier == 1.5
        assert t.upper_multiplier == 1.5

    def test_thresholds_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        t = IQRThreshold(lower_multiplier=1.5, upper_multiplier=1.5)
        lt, ut = t._derive(data)
        q1, q3 = np.percentile(data, [25, 75], method="midpoint")
        iqr = q3 - q1
        assert lt == pytest.approx(q1 - 1.5 * iqr)
        assert ut == pytest.approx(q3 + 1.5 * iqr)

    def test_thresholds_none_lower(self):
        t = IQRThreshold(lower_multiplier=None, upper_multiplier=2.0)
        lt, ut = t._derive(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert lt is None
        assert ut is not None

    def test_thresholds_none_upper(self):
        t = IQRThreshold(lower_multiplier=2.0, upper_multiplier=None)
        lt, ut = t._derive(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert lt is not None
        assert ut is None

    def test_zero_iqr_returns_none(self):
        t = IQRThreshold()
        lt, ut = t._derive(np.array([5.0, 5.0, 5.0, 5.0]))
        assert lt is None
        assert ut is None

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            IQRThreshold(lower_multiplier=-2.0)

    def test_registry(self):
        assert "iqr" in _Threshold._registry
        assert _Threshold._registry["iqr"] == IQRThreshold

    def test_asymmetric_multipliers(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        t = IQRThreshold(lower_multiplier=1.0, upper_multiplier=3.0)
        lt, ut = t._derive(data)
        q1, q3 = np.percentile(data, [25, 75], method="midpoint")
        iqr = q3 - q1
        assert lt == pytest.approx(q1 - 1.0 * iqr)
        assert ut == pytest.approx(q3 + 3.0 * iqr)


# --- resolve_threshold tests ---


class TestResolveThreshold:
    def test_none_returns_default_adaptive(self):
        t = resolve_threshold(None)
        assert isinstance(t, AdaptiveThreshold)
        assert t.lower_multiplier == 3.5
        assert t.upper_multiplier == 3.5

    def test_str_named_threshold(self):
        t = resolve_threshold("zscore")
        assert isinstance(t, ZScoreThreshold)
        assert t.lower_multiplier == 3.0

    def test_float_symmetric_default(self):
        t = resolve_threshold(2.5)
        assert isinstance(t, AdaptiveThreshold)
        assert t.lower_multiplier == 2.5
        assert t.upper_multiplier == 2.5

    def test_int_symmetric_default(self):
        t = resolve_threshold(3)
        assert isinstance(t, AdaptiveThreshold)
        assert t.lower_multiplier == 3
        assert t.upper_multiplier == 3

    def test_tuple_asymmetric_default(self):
        t = resolve_threshold((1.0, 2.0))
        assert isinstance(t, AdaptiveThreshold)
        assert t.lower_multiplier == 1.0
        assert t.upper_multiplier == 2.0

    def test_named_tuple_float(self):
        t = resolve_threshold(("zscore", 2.5))
        assert isinstance(t, ZScoreThreshold)
        assert t.lower_multiplier == 2.5
        assert t.upper_multiplier == 2.5

    def test_named_tuple_asymmetric(self):
        t = resolve_threshold(("iqr", (1.0, 2.0)))
        assert isinstance(t, IQRThreshold)
        assert t.lower_multiplier == 1.0
        assert t.upper_multiplier == 2.0

    def test_named_tuple_constant(self):
        t = resolve_threshold(("constant", (0.0, 1.0)))
        assert isinstance(t, ConstantThreshold)
        assert t.lower == 0.0
        assert t.upper == 1.0

    def test_named_tuple_one_sided(self):
        t = resolve_threshold(("modzscore", (None, 5.0)))
        assert isinstance(t, ModifiedZScoreThreshold)
        assert t.lower_multiplier is None
        assert t.upper_multiplier == 5.0

    def test_str_invalid_raises(self):
        with pytest.raises(ValueError, match="Expected one of"):
            resolve_threshold("unknown")

    def test_threshold_passthrough(self):
        original = ConstantThreshold(0.1, 0.9)
        t = resolve_threshold(original)
        assert t is original

    def test_threshold_passthrough_preserves_type(self):
        original = IQRThreshold(lower_multiplier=1.0, upper_multiplier=2.0)
        t = resolve_threshold(original)
        assert t is original
        assert isinstance(t, IQRThreshold)

    def test_3tuple_symmetric_multiplier_with_limits(self):
        t = resolve_threshold(("zscore", 3.0, (0.0, 1.0)))
        assert isinstance(t, ZScoreThreshold)
        assert t.lower_multiplier == 3.0
        assert t.upper_multiplier == 3.0
        assert t.lower_limit == 0.0
        assert t.upper_limit == 1.0

    def test_3tuple_asymmetric_multiplier_with_limits(self):
        t = resolve_threshold(("zscore", (1.0, 3.5), (0.0, 1.0)))
        assert isinstance(t, ZScoreThreshold)
        assert t.lower_multiplier == 1.0
        assert t.upper_multiplier == 3.5
        assert t.lower_limit == 0.0
        assert t.upper_limit == 1.0

    def test_3tuple_none_bounds_uses_defaults(self):
        t = resolve_threshold(("zscore", None, (0.0, 1.0)))
        assert isinstance(t, ZScoreThreshold)
        assert t.lower_multiplier == 3.0
        assert t.upper_multiplier == 3.0
        assert t.lower_limit == 0.0
        assert t.upper_limit == 1.0

    def test_3tuple_partial_limits(self):
        t = resolve_threshold(("iqr", (1.0, 2.0), (None, 0.9)))
        assert isinstance(t, IQRThreshold)
        assert t.lower_multiplier == 1.0
        assert t.upper_multiplier == 2.0
        assert t.lower_limit is None
        assert t.upper_limit == 0.9

    def test_3tuple_modzscore_with_limits(self):
        t = resolve_threshold(("modzscore", 2.0, (0.1, 0.9)))
        assert isinstance(t, ModifiedZScoreThreshold)
        assert t.lower_multiplier == 2.0
        assert t.upper_multiplier == 2.0
        assert t.lower_limit == 0.1
        assert t.upper_limit == 0.9

    def test_3tuple_limits_clamp_computed_bounds(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t = resolve_threshold(("zscore", 1.0, (2.5, 3.5)))
        lower, upper = t(data)
        # limits should clamp the computed thresholds
        assert lower is not None
        assert lower >= 2.5
        assert upper is not None
        assert upper <= 3.5

    def test_2tuple_none_bounds_with_limits(self):
        t = resolve_threshold((None, (0.0, 1.0)))
        assert isinstance(t, AdaptiveThreshold)
        assert t.lower_multiplier == 3.5
        assert t.upper_multiplier == 3.5
        assert t.lower_limit == 0.0
        assert t.upper_limit == 1.0

    def test_2tuple_float_bounds_with_limits(self):
        t = resolve_threshold((2.5, (0.0, 1.0)))
        assert isinstance(t, AdaptiveThreshold)
        assert t.lower_multiplier == 2.5
        assert t.upper_multiplier == 2.5
        assert t.lower_limit == 0.0
        assert t.upper_limit == 1.0

    def test_2tuple_partial_limits_shorthand(self):
        t = resolve_threshold((None, (None, 0.9)))
        assert isinstance(t, AdaptiveThreshold)
        assert t.lower_multiplier == 3.5
        assert t.upper_multiplier == 3.5
        assert t.lower_limit is None
        assert t.upper_limit == 0.9

    def test_2tuple_asymmetric_bounds_with_limits(self):
        t = resolve_threshold(((1.5, 1.5), (0.0, 1.0)))
        assert isinstance(t, AdaptiveThreshold)
        assert t.lower_multiplier == 1.5
        assert t.upper_multiplier == 1.5
        assert t.lower_limit == 0.0
        assert t.upper_limit == 1.0

    def test_2tuple_asymmetric_bounds_different_multipliers_with_limits(self):
        t = resolve_threshold(((1.0, 3.5), (0.0, 1.0)))
        assert isinstance(t, AdaptiveThreshold)
        assert t.lower_multiplier == 1.0
        assert t.upper_multiplier == 3.5
        assert t.lower_limit == 0.0
        assert t.upper_limit == 1.0

    def test_2tuple_lower_limit_only(self):
        t = resolve_threshold((2.5, (0.3, None)))
        assert isinstance(t, AdaptiveThreshold)
        assert t.lower_multiplier == 2.5
        assert t.upper_multiplier == 2.5
        assert t.lower_limit == 0.3
        assert t.upper_limit is None

    def test_2tuple_limits_clamp_computed_bounds(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t = resolve_threshold((1.0, (2.5, 3.5)))
        lower, upper = t(data)
        assert lower is not None
        assert lower >= 2.5
        assert upper is not None
        assert upper <= 3.5

    def test_3tuple_lower_limit_only(self):
        t = resolve_threshold(("iqr", 1.5, (0.2, None)))
        assert isinstance(t, IQRThreshold)
        assert t.lower_multiplier == 1.5
        assert t.upper_multiplier == 1.5
        assert t.lower_limit == 0.2
        assert t.upper_limit is None


# --- AdaptiveThreshold tests ---


class TestAdaptiveThreshold:
    def test_defaults(self):
        t = AdaptiveThreshold()
        assert t.lower_multiplier == 3.5
        assert t.upper_multiplier == 3.5

    def test_custom_multipliers(self):
        t = AdaptiveThreshold(lower_multiplier=2.0, upper_multiplier=4.0)
        assert t.lower_multiplier == 2.0
        assert t.upper_multiplier == 4.0

    def test_registry(self):
        assert "adaptive" in _Threshold._registry
        assert _Threshold._registry["adaptive"] == AdaptiveThreshold

    def test_resolve_string(self):
        t = resolve_threshold("adaptive")
        assert isinstance(t, AdaptiveThreshold)

    def test_resolve_tuple(self):
        t = resolve_threshold(("adaptive", 2.5))
        assert isinstance(t, AdaptiveThreshold)
        assert t.lower_multiplier == 2.5
        assert t.upper_multiplier == 2.5

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            AdaptiveThreshold(lower_multiplier=-1.0)

    def test_zero_variance_returns_none(self):
        t = AdaptiveThreshold()
        lt, ut = t(np.array([5.0, 5.0, 5.0, 5.0]))
        assert lt is None
        assert ut is None

    def test_too_few_samples_returns_none(self):
        t = AdaptiveThreshold()
        lt, ut = t(np.array([1.0, 2.0]))
        assert lt is None
        assert ut is None

    def test_normal_data_approximately_symmetric(self):
        """On normal data, the lower and upper bounds should be approximately
        symmetric around the median, and wider than plain ModifiedZScore due
        to tail-weight adjustment."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=50.0, scale=5.0, size=1000)
        mult = 3.0

        adaptive = AdaptiveThreshold(mult)
        a_lower, a_upper = adaptive(data)

        assert a_lower is not None
        assert a_upper is not None

        median_val = np.median(data)
        lower_extent = median_val - a_lower
        upper_extent = a_upper - median_val

        # Bounds should be approximately symmetric for normal data
        assert lower_extent == pytest.approx(upper_extent, rel=0.3)

        # Should be at least as wide as plain ModifiedZScore
        modz = ModifiedZScoreThreshold(mult)
        m_lower, m_upper = modz(data)

        assert m_lower is not None
        assert m_upper is not None

        assert a_lower <= m_lower
        assert a_upper >= m_upper

    def test_skewed_data_asymmetric_bounds(self):
        """On right-skewed data, the upper bound should be wider than the lower bound
        relative to the median, reflecting the asymmetric spread."""
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=10.0, size=5000)
        mult = 3.0

        adaptive = AdaptiveThreshold(mult)
        a_lower, a_upper = adaptive(data)

        assert a_lower is not None
        assert a_upper is not None

        median_val = np.median(data)

        # Upper bound should extend further from median than lower bound
        upper_extent = a_upper - median_val
        lower_extent = median_val - a_lower
        assert upper_extent > lower_extent, (
            f"Right-skewed data should have wider upper extent ({upper_extent:.2f}) "
            f"than lower extent ({lower_extent:.2f})"
        )

    def test_symmetric_catches_outliers_at_both_tails(self):
        """On symmetric data with outliers, adaptive still catches them."""
        data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        low_outlier = 0.0
        high_outlier = 30.0
        full = np.concatenate([[low_outlier], data, [high_outlier]])

        t = AdaptiveThreshold(2.0)
        lower, upper = t(full)

        assert lower is not None
        assert upper is not None
        assert low_outlier < lower, f"Low outlier {low_outlier} should be below lower bound {lower}"
        assert high_outlier > upper, f"High outlier {high_outlier} should be above upper bound {upper}"
        assert all(lower <= x <= upper for x in data), "Core data should be within bounds"

    def test_skewed_catches_extreme_outlier(self):
        """On right-skewed data, adaptive catches extreme outliers without flagging bulk."""
        bulk = np.array([1.0, 1.2, 1.1, 1.3, 1.0, 1.4, 1.2, 1.1, 1.3, 1.0, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        outlier = np.array([50.0])
        full = np.concatenate([bulk, outlier])

        t = AdaptiveThreshold(3.0)
        lower, upper = t(full)

        assert upper is not None
        assert upper < 50.0, "Extreme outlier should exceed upper bound"
        flagged_bulk = sum(1 for x in bulk if (lower is not None and x < lower) or (upper is not None and x > upper))
        assert flagged_bulk == 0, f"No bulk points should be flagged, got {flagged_bulk}"

    def test_upper_only(self):
        """Upper-only adaptive threshold returns None for lower."""
        data = np.concatenate([np.ones(20), np.array([2.0, 3.0, 4.0]), np.array([100.0])])
        t = AdaptiveThreshold(upper_multiplier=2.0, lower_multiplier=None)
        lower, upper = t(data)
        assert lower is None
        assert upper is not None
        assert upper < 100.0

    def test_nan_handling(self):
        """NaN values in input should be ignored."""
        data = np.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0, np.nan])
        t = AdaptiveThreshold(2.0)
        lower, upper = t(data)
        assert lower is not None
        assert upper is not None

    def test_zero_inflated_produces_bounds(self):
        """Zero-inflated data: global MAD fallback kicks in and produces valid bounds."""
        rng = np.random.default_rng(42)
        data = np.zeros(500)
        nonzero_idx = rng.choice(500, size=100, replace=False)
        data[nonzero_idx] = rng.exponential(scale=0.05, size=100)

        mult = 3.5
        adaptive = AdaptiveThreshold(mult)

        a_lo, a_hi = adaptive(data)

        assert a_lo is not None
        assert a_hi is not None

    def test_heavy_tailed_reasonable_flag_rate(self):
        """Heavy-tailed distributions should not flag an unreasonable proportion of data."""
        rng = np.random.default_rng(42)
        n = 10000
        mult = 3.5
        adaptive = AdaptiveThreshold(mult)

        heavy = rng.standard_t(df=3, size=n)

        a_lo, a_hi = adaptive(heavy)

        assert a_lo is not None
        assert a_hi is not None

        rate = np.sum((heavy < a_lo) | (heavy > a_hi)) / n

        # With 3.5 multiplier, flag rate should be well under 5%
        assert rate < 0.05, f"Flag rate {rate:.4f} is too high for 3.5× multiplier"

    def test_mad_zero_falls_back_to_global(self):
        """When half-MAD is zero (>50% identical values), falls back to global MAD."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 10.0])
        t = AdaptiveThreshold(2.0)
        lower, upper = t(data)

        # Left half is all 5.0 so left MAD=0, but global fallback provides bounds
        assert lower is not None
        assert upper is not None
