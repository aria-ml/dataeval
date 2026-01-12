"""
Tests for EvaluationSchedule protocol and implementations.

These tests verify that schedule strategies correctly determine
evaluation points for sufficiency analysis.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dataeval.performance.schedules import GeometricSchedule, ManualSchedule
from dataeval.protocols import EvaluationSchedule


class TestEvaluationScheduleProtocol:
    """Test that schedule implementations conform to protocol."""

    def test_geometric_schedule_conforms_to_protocol(self):
        """Verify GeometricSchedule implements protocol."""
        schedule = GeometricSchedule(substeps=5)
        assert isinstance(schedule, EvaluationSchedule)

    def test_custom_schedule_conforms_to_protocol(self):
        """Verify CustomSchedule implements protocol."""
        schedule = ManualSchedule([10, 20, 30])
        assert isinstance(schedule, EvaluationSchedule)


class TestGeometricSchedule:
    """Test geometric spacing schedule."""

    def test_creates_geometric_spacing(self):
        """Verify geometric spacing from 1% to 100% of dataset."""
        schedule = GeometricSchedule(substeps=5)
        steps = schedule.get_steps(dataset_length=100)

        # Should have 5 steps
        assert len(steps) == 5

        # First step should be ~1% (1)
        assert steps[0] == 1

        # Last step should be 100%
        assert steps[-1] == 100

        # Should be monotonically increasing
        assert all(steps[i] < steps[i + 1] for i in range(len(steps) - 1))

    def test_geometric_spacing_with_different_substeps(self):
        """Verify substeps parameter controls number of points."""
        schedule = GeometricSchedule(substeps=3)
        steps = schedule.get_steps(dataset_length=1000)

        assert len(steps) == 3
        assert steps[0] == 10  # 1% of 1000
        assert steps[-1] == 1000

    def test_returns_intp_array(self):
        """Verify output is uint32 array for indexing."""
        schedule = GeometricSchedule(substeps=5)
        steps = schedule.get_steps(dataset_length=100)

        assert isinstance(steps, np.ndarray)
        assert steps.dtype == np.intp


class TestManualSchedule:
    """Test custom evaluation point schedule."""

    def test_accepts_single_int(self):
        """Verify ManualSchedule handles single integer."""
        schedule = ManualSchedule(50)
        steps = schedule.get_steps(dataset_length=100)

        assert len(steps) == 1
        assert steps[0] == 50

    def test_accepts_list(self):
        """Verify ManualSchedule handles list of ints."""
        schedule = ManualSchedule([10, 20, 50, 100])
        steps = schedule.get_steps(dataset_length=100)

        assert len(steps) == 4
        assert_array_equal(steps, [10, 20, 50, 100])

    def test_accepts_numpy_array(self):
        """Verify ManualSchedule handles numpy array."""
        schedule = ManualSchedule(np.array([5, 15, 25]))
        steps = schedule.get_steps(dataset_length=100)

        assert len(steps) == 3
        assert_array_equal(steps, [5, 15, 25])

    def test_accepts_iterable(self):
        """Verify ManualSchedule handles any iterable."""
        schedule = ManualSchedule(range(10, 101, 10))
        steps = schedule.get_steps(dataset_length=100)

        assert len(steps) == 10
        assert steps[0] == 10
        assert steps[-1] == 100

    def test_rejects_non_numeric(self):
        """Verify ManualSchedule validates input types."""

        # Mainly verifies call to `to_numpy`
        with pytest.raises(ValueError, match="invalid literal"):
            ManualSchedule(["a", "b", "c"])  # pyright: ignore[reportArgumentType] --> testing incorrect type

    def test_returns_intp_array(self):
        """Verify output is uint32 array for indexing."""
        schedule = ManualSchedule([10, 20, 30])
        steps = schedule.get_steps(dataset_length=100)

        assert isinstance(steps, np.ndarray)
        assert steps.dtype == np.intp
