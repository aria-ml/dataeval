__all__ = []

from enum import Flag
from typing import Any

from dataeval.core._calculators._base import Calculator


class CalculatorRegistry:
    """
    Registry for mapping flag types to their stat calculators.

    This registry enables auto-registration of stat calculators and supports
    multiple flag enum types (ImageStats, TextStats, etc.) in a single system.
    """

    _registry: dict[type[Flag], list[type[Calculator[Any]]]] = {}

    @classmethod
    def register(cls, flag_type: type[Flag]) -> Any:
        """
        Decorator to register a calculator with a flag type.

        Parameters
        ----------
        flag_type : type[Flag]
            The flag enum type this calculator handles (e.g., ImageStats).

        Returns
        -------
        Callable
            Decorator function that registers the calculator class.
        """

        def wrapper(calculator_class: type[Calculator[Any]]) -> type[Calculator[Any]]:
            cls._registry.setdefault(flag_type, []).append(calculator_class)
            return calculator_class

        return wrapper

    @classmethod
    def get_calculators(cls, flags: Flag) -> list[tuple[type[Calculator[Any]], Flag]]:
        """
        Get relevant calculators for the given flags.

        This method finds all registered calculators that can handle any of the
        requested flags, and returns them paired with the subset of flags they
        should process.

        Parameters
        ----------
        flags : Flag
            The requested flags to compute statistics for.

        Returns
        -------
        list[tuple[type[Calculator], Flag]]
            List of (calculator_class, applicable_flags) tuples.
        """
        flag_type = type(flags)

        # dict == poor man's ordered set
        result: dict[tuple[type[Calculator[Any]], Flag], None] = {}

        for calculator_class in cls._registry.get(flag_type, []):
            # Create a temporary instance to query applicable flags
            # We use __new__ to avoid needing constructor args
            temp_instance = calculator_class.__new__(calculator_class)
            applicable_flags = temp_instance.get_applicable_flags()

            # Check if this calculator handles any of the requested flags
            intersection = flags & applicable_flags
            if intersection:
                result[(calculator_class, intersection)] = None

        return list(result.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Primarily for testing."""
        cls._registry.clear()

    @classmethod
    def get_all_calculators(cls, flag_type: type[Flag]) -> list[type[Calculator[Any]]]:
        """
        Get all registered calculators for a specific flag type.

        Parameters
        ----------
        flag_type : type[Flag]
            The flag enum type to query.

        Returns
        -------
        list[type[Calculator]]
            List of all calculator classes registered for this flag type.
        """
        return cls._registry.get(flag_type, [])
