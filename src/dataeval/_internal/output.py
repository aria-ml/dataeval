import inspect
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, Tuple

from dataeval import __version__


class OutputMetadata:
    _name: str
    _execution_time: str
    _execution_duration: float
    _arguments: Dict
    _version: str

    def dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def meta(self) -> Dict:
        return {k.removeprefix("_"): v for k, v in self.__dict__.items() if k.startswith("_")}


def set_metadata(name: str):
    def decorator(fn: Callable[..., OutputMetadata]):
        def wrapper(*args, **kwargs):
            def fmt(m: Iterable[Tuple]):
                return {k: f"{v.__class__.__name__}: {getattr(v, 'shape', v)}" for k, v in m}

            time = datetime.now(timezone.utc)
            result = fn(*args, **kwargs)
            duration = (datetime.now(timezone.utc) - time).total_seconds()
            fn_params = inspect.signature(fn).parameters
            # set all params with defaults then update params with mapped arguments and explicit keyword args
            arguments = {k: None if v.default is inspect.Parameter.empty else v.default for k, v in fn_params.items()}
            arguments.update(**fmt(zip(fn_params, args)), **fmt(kwargs.items()))
            metadata = {
                "_name": name,
                "_execution_time": time,
                "_execution_duration": duration,
                "_arguments": arguments,
                "_version": __version__,
            }
            for k, v in metadata.items():
                object.__setattr__(result, k, v)
            return result

        return wrapper

    return decorator
