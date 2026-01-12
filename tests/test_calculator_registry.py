"""Tests for CalculatorRegistry."""

from enum import Flag, auto
from typing import Any

from dataeval.core._calculators._base import Calculator
from dataeval.core._calculators._registry import CalculatorRegistry


# Test flag types
class MockFlagA(Flag):
    """Mock flag type A."""

    FLAG_1 = auto()
    FLAG_2 = auto()
    FLAG_3 = auto()
    ALL = FLAG_1 | FLAG_2 | FLAG_3


class MockFlagB(Flag):
    """Mock flag type B."""

    FLAG_X = auto()
    FLAG_Y = auto()
    ALL = FLAG_X | FLAG_Y


# Test calculators
class MockCalculatorA1(Calculator[MockFlagA]):
    """Mock calculator that handles FLAG_1."""

    def get_applicable_flags(self) -> MockFlagA:
        return MockFlagA.FLAG_1

    def get_handlers(self) -> dict[MockFlagA, tuple[str, Any]]:
        return {MockFlagA.FLAG_1: ("stat1", lambda: [1.0])}


class MockCalculatorA2(Calculator[MockFlagA]):
    """Mock calculator that handles FLAG_2 and FLAG_3."""

    def get_applicable_flags(self) -> MockFlagA:
        return MockFlagA.FLAG_2 | MockFlagA.FLAG_3

    def get_handlers(self) -> dict[MockFlagA, tuple[str, Any]]:
        return {
            MockFlagA.FLAG_2: ("stat2", lambda: [2.0]),
            MockFlagA.FLAG_3: ("stat3", lambda: [3.0]),
        }


class MockCalculatorB(Calculator[MockFlagB]):
    """Mock calculator that handles FLAG_X."""

    def get_applicable_flags(self) -> MockFlagB:
        return MockFlagB.FLAG_X

    def get_handlers(self) -> dict[MockFlagB, tuple[str, Any]]:
        return {MockFlagB.FLAG_X: ("statX", lambda: ["x"])}


class TestCalculatorRegistry:
    """Tests for CalculatorRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        CalculatorRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        CalculatorRegistry.clear()

    def test_register_single_calculator(self):
        """Test registering a single calculator."""
        decorator = CalculatorRegistry.register(MockFlagA)
        result = decorator(MockCalculatorA1)

        # Decorator should return the class unchanged
        assert result is MockCalculatorA1

        # Calculator should be in registry
        calculators = CalculatorRegistry.get_all_calculators(MockFlagA)
        assert len(calculators) == 1
        assert calculators[0] is MockCalculatorA1

    def test_register_multiple_calculators_same_flag_type(self):
        """Test registering multiple calculators for the same flag type."""
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA1)
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA2)

        calculators = CalculatorRegistry.get_all_calculators(MockFlagA)
        assert len(calculators) == 2
        assert MockCalculatorA1 in calculators
        assert MockCalculatorA2 in calculators

    def test_register_different_flag_types(self):
        """Test registering calculators for different flag types."""
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA1)
        CalculatorRegistry.register(MockFlagB)(MockCalculatorB)

        calculators_a = CalculatorRegistry.get_all_calculators(MockFlagA)
        calculators_b = CalculatorRegistry.get_all_calculators(MockFlagB)

        assert len(calculators_a) == 1
        assert calculators_a[0] is MockCalculatorA1

        assert len(calculators_b) == 1
        assert calculators_b[0] is MockCalculatorB

    def test_get_calculators_single_flag(self):
        """Test getting calculators for a single flag."""
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA1)
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA2)

        # Request FLAG_1
        result = CalculatorRegistry.get_calculators(MockFlagA.FLAG_1)
        assert len(result) == 1
        calc_class, flags = result[0]
        assert calc_class is MockCalculatorA1
        assert flags == MockFlagA.FLAG_1

    def test_get_calculators_multiple_flags(self):
        """Test getting calculators for multiple flags."""
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA1)
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA2)

        # Request FLAG_1 | FLAG_2
        result = CalculatorRegistry.get_calculators(MockFlagA.FLAG_1 | MockFlagA.FLAG_2)
        assert len(result) == 2

        # Convert to dict for easier checking
        result_dict = dict(result)

        assert MockCalculatorA1 in result_dict
        assert result_dict[MockCalculatorA1] == MockFlagA.FLAG_1

        assert MockCalculatorA2 in result_dict
        assert result_dict[MockCalculatorA2] == MockFlagA.FLAG_2

    def test_get_calculators_all_flags(self):
        """Test getting calculators for all flags."""
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA1)
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA2)

        # Request all flags
        result = CalculatorRegistry.get_calculators(MockFlagA.ALL)
        assert len(result) == 2

        result_dict = dict(result)

        assert MockCalculatorA1 in result_dict
        assert result_dict[MockCalculatorA1] == MockFlagA.FLAG_1

        assert MockCalculatorA2 in result_dict
        assert result_dict[MockCalculatorA2] == (MockFlagA.FLAG_2 | MockFlagA.FLAG_3)

    def test_get_calculators_no_match(self):
        """Test getting calculators when no flags match."""
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA1)

        # Request FLAG_2, but only MockCalculatorA1 is registered (handles FLAG_1)
        result = CalculatorRegistry.get_calculators(MockFlagA.FLAG_2)
        assert len(result) == 0

    def test_get_calculators_empty_registry(self):
        """Test getting calculators from empty registry."""
        result = CalculatorRegistry.get_calculators(MockFlagA.FLAG_1)
        assert len(result) == 0

    def test_get_all_calculators_empty(self):
        """Test get_all_calculators on unregistered flag type."""
        calculators = CalculatorRegistry.get_all_calculators(MockFlagA)
        assert len(calculators) == 0

    def test_clear_registry(self):
        """Test clearing the registry."""
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA1)
        CalculatorRegistry.register(MockFlagB)(MockCalculatorB)

        # Verify calculators are registered
        assert len(CalculatorRegistry.get_all_calculators(MockFlagA)) == 1
        assert len(CalculatorRegistry.get_all_calculators(MockFlagB)) == 1

        # Clear registry
        CalculatorRegistry.clear()

        # Verify registry is empty
        assert len(CalculatorRegistry.get_all_calculators(MockFlagA)) == 0
        assert len(CalculatorRegistry.get_all_calculators(MockFlagB)) == 0

    def test_register_as_decorator(self):
        """Test using register as a decorator."""

        @CalculatorRegistry.register(MockFlagA)
        class DecoratedCalculator(Calculator[MockFlagA]):
            def get_applicable_flags(self) -> MockFlagA:
                return MockFlagA.FLAG_1

            def get_handlers(self) -> dict[MockFlagA, tuple[str, Any]]:
                return {}

        calculators = CalculatorRegistry.get_all_calculators(MockFlagA)
        assert len(calculators) == 1
        assert calculators[0] is DecoratedCalculator

    def test_get_calculators_preserves_order(self):
        """Test that get_calculators returns unique calculator-flag pairs."""
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA1)
        CalculatorRegistry.register(MockFlagA)(MockCalculatorA2)

        # Request multiple times - should get consistent results
        result1 = CalculatorRegistry.get_calculators(MockFlagA.ALL)
        result2 = CalculatorRegistry.get_calculators(MockFlagA.ALL)

        assert len(result1) == len(result2)
        # Results should be consistent (dict keys maintain order in Python 3.7+)
        assert result1 == result2
