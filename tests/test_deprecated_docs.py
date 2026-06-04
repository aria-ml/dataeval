"""Enforce that every ``@deprecated`` public API object documents it in its docstring.

``sphinx-autoapi`` parses source statically and never imports the package, so the
``@deprecated`` decorator cannot inject a deprecation note at build time (its runtime
``__doc__`` change is invisible to autoapi). The note must therefore live in the source
docstring as a ``.. deprecated::`` directive. This test walks the exposed API, finds
everything the decorator marked with ``__deprecated__``, and fails if the directive is
missing.
"""

import importlib
import inspect
import pkgutil

import pytest

import dataeval

DIRECTIVE = ".. deprecated::"


def _iter_modules():
    """Yield ``dataeval`` and every importable submodule."""
    yield dataeval
    for info in pkgutil.walk_packages(dataeval.__path__, "dataeval."):
        try:
            yield importlib.import_module(info.name)
        except Exception:  # pragma: no cover - skip modules with unmet optional deps
            continue


def _is_dataeval(obj) -> bool:
    return getattr(obj, "__module__", "").split(".", 1)[0] == "dataeval"


def _collect_deprecated():
    """Return ``[(qualified_name, obj), ...]`` for every API object marked deprecated.

    Detection uses ``__deprecated__`` in the object's *own* ``__dict__`` so a
    non-deprecated subclass of a deprecated base is not flagged by inheritance.
    """
    found = {}
    for module in _iter_modules():
        for name, obj in inspect.getmembers(module):
            if not (inspect.isclass(obj) or inspect.isfunction(obj)) or not _is_dataeval(obj):
                continue
            candidates = [(f"{obj.__module__}.{getattr(obj, '__qualname__', name)}", obj)]
            if inspect.isclass(obj):
                for mname, meth in inspect.getmembers(obj, inspect.isfunction):
                    if _is_dataeval(meth):
                        candidates.append((f"{obj.__module__}.{obj.__qualname__}.{mname}", meth))
            for qual, target in candidates:
                if "__deprecated__" in getattr(target, "__dict__", {}):
                    found.setdefault(qual, target)
    return sorted(found.items())


DEPRECATED = _collect_deprecated()


@pytest.mark.required
def test_walker_discovers_known_deprecation():
    """Guard against the walker silently finding nothing (e.g. import/marker regressions)."""
    names = [q for q, _ in DEPRECATED]
    assert any(q.endswith("ClassifierUncertaintyExtractor") for q in names), names


@pytest.mark.required
@pytest.mark.parametrize(("qual", "obj"), DEPRECATED, ids=[q for q, _ in DEPRECATED] or None)
def test_deprecated_object_has_directive(qual, obj):
    doc = obj.__doc__ or ""
    assert DIRECTIVE in doc, (
        f"{qual} is @deprecated but its docstring has no '{DIRECTIVE}' directive. "
        "autoapi won't render a deprecation note without it - add a `.. deprecated:: <version>` "
        "block to the source docstring."
    )
