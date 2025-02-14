from __future__ import annotations

import sys
from typing import Callable, TypeVar

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def get_method(method_map: dict[str, Callable[P, R]], method: str) -> Callable[P, R]:
    if method not in method_map:
        raise ValueError(f"Specified method {method} is not a valid method: {method_map}.")
    return method_map[method]
