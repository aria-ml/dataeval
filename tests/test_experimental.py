"""Tests for the experimental and deprecated marker infrastructure."""

import sys
import warnings

import pytest

from dataeval._experimental import _lazy_import_with_warning, _warn_on_access, deprecated, experimental
from dataeval.exceptions import DeprecatedWarning, ExperimentalWarning


class TestExperimentalDecorator:
    """Test @experimental decorator on functions and classes."""

    def test_function_warns_on_call(self):
        @experimental
        def my_func():
            return 42

        with pytest.warns(ExperimentalWarning, match="my_func"):
            result = my_func()
        assert result == 42

    def test_function_with_args_warns(self):
        @experimental
        def add(a, b):
            return a + b

        with pytest.warns(ExperimentalWarning, match="add"):
            result = add(1, 2)
        assert result == 3

    def test_function_preserves_metadata(self):
        @experimental
        def documented_func():
            """My docstring."""

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ is not None
        assert "My docstring." in documented_func.__doc__
        assert ".. warning::" in documented_func.__doc__

    def test_function_with_alternative(self):
        @experimental(alternative="stable_func")
        def my_func():
            return 1

        with pytest.warns(ExperimentalWarning, match="stable_func"):
            my_func()

    def test_function_with_details(self):
        @experimental(details="Use at your own risk.")
        def my_func():
            return 1

        with pytest.warns(ExperimentalWarning, match="Use at your own risk"):
            my_func()

    def test_class_warns_on_init(self):
        @experimental
        class MyClass:
            def __init__(self, x):
                self.x = x

        with pytest.warns(ExperimentalWarning, match="MyClass"):
            obj = MyClass(10)
        assert obj.x == 10

    def test_class_preserves_identity(self):
        @experimental
        class MyClass:
            pass

        assert MyClass.__name__ == "MyClass"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            obj = MyClass()
        assert isinstance(obj, MyClass)

    def test_class_with_alternative(self):
        @experimental(alternative="StableClass")
        class MyClass:
            pass

        with pytest.warns(ExperimentalWarning, match="StableClass"):
            MyClass()

    def test_function_warns_only_once(self):
        @experimental
        def my_func():
            return 1

        with pytest.warns(ExperimentalWarning):
            my_func()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ExperimentalWarning)
            my_func()
        assert len(w) == 0

    def test_class_warns_only_once(self):
        @experimental
        class MyClass:
            pass

        with pytest.warns(ExperimentalWarning):
            MyClass()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ExperimentalWarning)
            MyClass()
        assert len(w) == 0


class TestDeprecatedDecorator:
    """Test @deprecated decorator on functions and classes."""

    def test_function_warns_on_call(self):
        @deprecated
        def old_func():
            return 99

        with pytest.warns(DeprecatedWarning, match="old_func.*deprecated"):
            result = old_func()
        assert result == 99

    def test_function_with_since(self):
        @deprecated(since="1.0")
        def old_func():
            return 1

        with pytest.warns(DeprecatedWarning, match="since version 1.0"):
            old_func()

    def test_function_with_removal(self):
        @deprecated(since="1.0", removal="2.0")
        def old_func():
            return 1

        with pytest.warns(DeprecatedWarning, match="removed in version 2.0"):
            old_func()

    def test_function_with_alternative(self):
        @deprecated(alternative="new_func")
        def old_func():
            return 1

        with pytest.warns(DeprecatedWarning, match="new_func"):
            old_func()

    def test_function_with_all_options(self):
        @deprecated(since="1.0", removal="2.0", alternative="new_func", details="See migration guide.")
        def old_func():
            return 1

        with pytest.warns(DeprecatedWarning, match="old_func"):
            old_func()

    def test_function_preserves_metadata(self):
        @deprecated
        def documented_func():
            """My docstring."""

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ is not None
        assert "My docstring." in documented_func.__doc__
        assert ".. warning::" in documented_func.__doc__

    def test_class_warns_on_init(self):
        @deprecated(since="0.9")
        class OldClass:
            pass

        with pytest.warns(DeprecatedWarning, match="OldClass"):
            OldClass()

    def test_class_preserves_identity(self):
        @deprecated
        class OldClass:
            pass

        assert OldClass.__name__ == "OldClass"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecatedWarning)
            obj = OldClass()
        assert isinstance(obj, OldClass)

    def test_function_warns_only_once(self):
        @deprecated
        def old_func():
            return 1

        with pytest.warns(DeprecatedWarning):
            old_func()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecatedWarning)
            old_func()
        assert len(w) == 0

    def test_class_warns_only_once(self):
        @deprecated
        class OldClass:
            pass

        with pytest.warns(DeprecatedWarning):
            OldClass()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecatedWarning)
            OldClass()
        assert len(w) == 0


class TestWarningClasses:
    """Test that warning classes have correct inheritance."""

    def test_experimental_is_future_warning(self):
        assert issubclass(ExperimentalWarning, FutureWarning)

    def test_deprecated_is_future_warning(self):
        assert issubclass(DeprecatedWarning, FutureWarning)

    def test_warnings_are_distinct(self):
        assert not issubclass(ExperimentalWarning, DeprecatedWarning)
        assert not issubclass(DeprecatedWarning, ExperimentalWarning)


class TestLazyImportWithWarning:
    """Test _lazy_import_with_warning and _warn_on_access helpers."""

    def test_lazy_import_experimental_warns(self):
        """Importing a real stdlib attr via the helper emits ExperimentalWarning."""
        with pytest.warns(ExperimentalWarning, match="fake.module.pi"):
            result = _lazy_import_with_warning("math", "pi", "fake.module.pi", "experimental")
        import math

        assert result == math.pi

    def test_lazy_import_deprecated_warns(self):
        """Importing via the helper with kind='deprecated' emits DeprecatedWarning."""
        with pytest.warns(DeprecatedWarning, match="fake.module.e"):
            result = _lazy_import_with_warning("math", "e", "fake.module.e", "deprecated", since="1.0")
        import math

        assert result == math.e

    def test_lazy_import_returns_correct_object(self):
        """The helper returns the actual attribute from the target module."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            result = _lazy_import_with_warning("os.path", "join", "fake.os.path.join", "experimental")
        from os.path import join

        assert result is join

    def test_warn_on_access_experimental(self):
        with pytest.warns(ExperimentalWarning, match="my_module.MyThing"):
            _warn_on_access("my_module.MyThing", "experimental")

    def test_warn_on_access_deprecated_with_metadata(self):
        with pytest.warns(DeprecatedWarning, match="removed in version 3.0"):
            _warn_on_access("my_module.OldThing", "deprecated", since="2.0", removal="3.0")

    def test_getattr_for_missing_name_raises(self):
        """Modules using __getattr__ raise ImportError for unknown names."""
        import types as t

        mod = t.ModuleType("fake_pkg")
        mod.__package__ = "fake_pkg"

        def __getattr__(name: str):
            raise AttributeError(f"module 'fake_pkg' has no attribute {name!r}")

        mod.__getattr__ = __getattr__  # type: ignore[attr-defined]
        sys.modules["fake_pkg"] = mod
        try:
            with pytest.raises(ImportError):
                from fake_pkg import DoesNotExist  # type: ignore  # noqa: F401
        finally:
            del sys.modules["fake_pkg"]


class TestExceptionsModule:
    """Test that warning classes are accessible from the exceptions module."""

    def test_experimental_warning_from_exceptions(self):
        from dataeval.exceptions import ExperimentalWarning as EW

        assert EW is ExperimentalWarning

    def test_deprecated_warning_from_exceptions(self):
        from dataeval.exceptions import DeprecatedWarning as DW

        assert DW is DeprecatedWarning

    def test_exceptions_module_accessible(self):
        from dataeval import exceptions

        assert hasattr(exceptions, "ExperimentalWarning")
        assert hasattr(exceptions, "DeprecatedWarning")

    def test_can_filter_experimental_warnings(self):
        """Verify users can suppress ExperimentalWarning."""

        @experimental
        def noisy_func():
            return 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", ExperimentalWarning)
            noisy_func()
        assert len(w) == 0
