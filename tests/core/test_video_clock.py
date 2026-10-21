import numpy as np
import pytest

from dataeval.core._video_runner import video_clock

pytestmark = pytest.mark.required
F = 1 / 30


def test_constant_rate_keeps_time_s_as_the_order():
    t = [k * F for k in range(10)]
    clock = video_clock(t)
    assert clock.order is not None
    assert clock.order.tolist() == t
    assert clock.dt == pytest.approx(F)
    assert clock.duration == pytest.approx(10 * F)


def test_irregular_rate_uses_the_median_step():
    clock = video_clock([0.0, 0.1, 0.3, 0.4])
    assert clock.dt == pytest.approx(0.1)
    assert clock.duration == pytest.approx(0.5)


def test_a_stitched_stream_counts_the_restart_as_one_interval():
    clock = video_clock([0.0, F, 0.0, F])
    assert clock.order is not None
    assert clock.order.tolist() == pytest.approx([0.0, F, 2 * F, 3 * F])
    assert clock.duration == pytest.approx(4 * F)


def test_a_repeated_timestamp_advances_by_one_interval():
    clock = video_clock([0.0, F, F, 2 * F])
    assert clock.order is not None
    assert clock.order.tolist() == pytest.approx([0.0, F, 2 * F, 3 * F])


@pytest.mark.parametrize("t", [[0.0, None, 2 * F], [0.0, 0.0, 0.0], [], [0.0]])
def test_undefined_time_reads_nan(t):
    clock = video_clock(t)
    assert np.isnan(clock.duration)
    assert np.isnan(clock.dt)


def test_a_missing_timestamp_leaves_no_order():
    assert video_clock([0.0, None, 2 * F]).order is None


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_timestamp_reads_as_missing(bad):
    clock = video_clock([0.0, bad, 2 * F])
    assert clock.order is None
    assert np.isnan(clock.duration)


def test_a_single_frame_orders_by_its_own_time():
    clock = video_clock([2.5])
    assert clock.order is not None
    assert clock.order.tolist() == [2.5]
    assert np.isnan(clock.duration)
