"""The signature convention dataeval-flow reads: evaluators name precomputed inputs `from_<kind>` and return an
output class, and operations state the dataset kind they need."""

import inspect
from collections.abc import Iterator
from types import ModuleType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import pytest

import dataeval.bias
import dataeval.data
import dataeval.quality
import dataeval.scope
import dataeval.shift
from dataeval.data import Operation
from dataeval.types import Output

# The values of dataeval-flow's InputKind. A new kind of precomputed input is added here and in Flow together.
INPUT_KINDS = frozenset({"stats", "clusters", "metadata", "labels", "embeddings"})
EVALUATOR_PACKAGES = (dataeval.bias, dataeval.quality, dataeval.scope, dataeval.shift)


def _public_classes(package: ModuleType) -> list[type]:
    return [obj for name in package.__all__ if inspect.isclass(obj := getattr(package, name))]


def _is_output(annotation: Any) -> bool:
    """Whether `annotation` is an Output subclass, or a parametrized one such as ``OutliersOutput[...]``."""
    cls = get_origin(annotation) or annotation
    return inspect.isclass(cls) and issubclass(cls, Output)


def _entry_points() -> Iterator[tuple[str, Any]]:
    """Every public `from_*` method, read from the class dictionaries so no other attribute is evaluated."""
    for package in EVALUATOR_PACKAGES:
        for cls in _public_classes(package):
            names = {name for klass in cls.__mro__ for name in vars(klass) if name.startswith("from_")}
            for name in sorted(names):
                yield f"{cls.__name__}.{name}", getattr(cls, name)


@pytest.mark.required
class TestSignatureConvention:
    def test_there_are_entry_points_to_check(self):
        assert list(_entry_points())

    def test_each_entry_point_names_an_input_kind(self):
        wrong = [name for name, _ in _entry_points() if name.rsplit(".from_", 1)[1] not in INPUT_KINDS]
        assert wrong == []

    def test_each_entry_point_is_fully_annotated(self):
        missing: list[str] = []
        for name, member in _entry_points():
            signature = inspect.signature(member)
            if signature.return_annotation is inspect.Signature.empty:
                missing.append(f"{name} -> ?")
            missing += [
                f"{name}({parameter.name})"
                for parameter in signature.parameters.values()
                if parameter.name not in ("self", "cls") and parameter.annotation is inspect.Parameter.empty
            ]
        assert missing == []

    def test_each_entry_point_returns_an_output_class(self):
        wrong: list[str] = []
        for name, member in _entry_points():
            returned = get_type_hints(member).get("return")
            arms = get_args(returned) if get_origin(returned) in (Union, UnionType) else (returned,)
            wrong += [f"{name} -> {arm}" for arm in arms if not _is_output(arm)]
        assert wrong == []

    def test_each_operation_states_the_dataset_kind_it_needs(self):
        operations = [
            cls for cls in _public_classes(dataeval.data) if issubclass(cls, Operation) and cls is not Operation
        ]
        assert operations
        assert [cls.__name__ for cls in operations if "requires" not in cls.__dict__] == []
