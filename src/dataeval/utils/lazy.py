from __future__ import annotations

from functools import cached_property
from importlib import import_module
from typing import Any


class LazyModule:
    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, key: str) -> Any:
        return getattr(self._module, key)

    @cached_property
    def _module(self):
        return import_module(self._name)


LAZY_MODULES: dict[str, LazyModule] = {}


def lazyload(name: str) -> LazyModule:
    if name not in LAZY_MODULES:
        LAZY_MODULES[name] = LazyModule(name)
    return LAZY_MODULES[name]
