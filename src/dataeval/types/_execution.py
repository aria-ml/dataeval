"""Execution provenance records stamped onto evaluator outputs."""

__all__ = [
    "ExecutionMetadata",
]

from datetime import datetime
from typing import Any

from pydantic.dataclasses import dataclass
from typing_extensions import Self

try:
    from dataeval._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"
__version__ = str(__version__)


@dataclass(frozen=True)
class ExecutionMetadata:
    """
    Metadata about the execution of the function or method for the Output class.

    Attributes
    ----------
    name: str
        Name of the function or method
    execution_time: datetime
        Time of execution
    execution_duration: float
        Duration of execution in seconds
    arguments: dict[str, Any]
        Arguments passed to the function or method
    state: dict[str, Any]
        State attributes of the executing class
    version: str
        Version of DataEval
    """

    name: str
    execution_time: datetime
    execution_duration: float
    arguments: dict[str, Any]
    state: dict[str, Any]
    version: str

    def __repr__(self) -> str:
        """Return a detailed string representation of the execution metadata."""
        return (
            f"ExecutionMetadata(name={self.name!r}, "
            f"execution_time={self.execution_time.isoformat()}, "
            f"execution_duration={self.execution_duration:.4f}s, "
            f"version={self.version!r})"
        )

    def __str__(self) -> str:
        """Return a string representation showing the name and duration."""
        return f"{self.name} ({self.execution_duration:.4f}s)"

    @classmethod
    def _empty(cls) -> Self:
        return cls(
            name="",
            execution_time=datetime.min,
            execution_duration=0.0,
            arguments={},
            state={},
            version=__version__,
        )
