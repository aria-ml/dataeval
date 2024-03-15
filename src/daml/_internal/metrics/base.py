from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, List, TypeVar

TOutput = TypeVar("TOutput", bound=dict)
TMethods = TypeVar("TMethods")
TCallable = TypeVar("TCallable", bound=Callable)


class EvaluateMixin(ABC, Generic[TOutput]):
    @abstractmethod
    def evaluate(self) -> TOutput:
        """Abstract method to calculate metric based off of constructor parameters"""


class MethodsMixin(ABC, Generic[TMethods, TCallable]):
    """
    Use this mixin to define a mapping of functions to method names which
    can be queried by the user and called internally with the appropriate
    method name as the key.

    Explicitly defining the Callable generic helps with type safety and
    hinting for function signatures and recommended but optional.

    e.g.:

    def _mult(x: float, y: float) -> float:
        return x * y

    class MyMetric(MethodsMixin[Callable[float, float], float]):

        def _methods(cls) -> Dict[str, Callable[float, float], float]:
            return {
                "ADD": lambda x, y: x + y,
                "MULT":  _mult,
                ...
            }

    Then during evaluate, you can call the method specified with the getter.

    e.g.:

        def evaluate(self):
            return self._method(x, y)

    The resulting class can be used like so.

    m = MyMetric(1.0, 2.0, "ADD")
    m.evaluate()       #  returns 3.0
    m.method           #  returns "ADD"
    MyMetric.methods() #  returns "['ADD', 'MULT']
    m.method = "MULT"
    m.evaluate()       #  returns 2.0
    """

    @classmethod
    @abstractmethod
    def _methods(cls) -> Dict[str, TCallable]:
        """Abstract method returning available method functions for class"""

    @property
    def _method(self) -> TCallable:
        return self._methods()[self.method]

    @classmethod
    def methods(cls) -> List[str]:
        return list(cls._methods().keys())

    @property
    def method(self) -> str:
        return self._method_key

    @method.setter
    def method(self, value: TMethods):
        self._set_method(value)

    def _set_method(self, value: TMethods):
        """This setter is to fix pyright incorrect detection of
        incorrectly overriding the 'method' property"""
        if value not in self.methods():
            raise KeyError(
                f"Specified method not available for class ({self.methods()})."
            )
        self._method_key = value
