__all__ = []

from collections.abc import Callable


class LogMessage:
    """
    Deferred message callback for logging expensive messages.
    """

    def __init__(self, fn: Callable[..., str]) -> None:
        self._fn = fn
        self._str = None

    def __str__(self) -> str:
        if self._str is None:
            self._str = self._fn()
        return self._str
