import inspect
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from dataeval import __version__


class OutputMetadata:
    _name: str
    _execution_time: str
    _execution_duration: float
    _arguments: Dict[str, str]
    _state: Dict[str, str]
    _version: str

    def dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def meta(self) -> Dict:
        return {k.removeprefix("_"): v for k, v in self.__dict__.items() if k.startswith("_")}


def set_metadata(name: str, state_attr: Optional[List[str]] = None):
    def decorator(fn: Callable[..., OutputMetadata]):
        def wrapper(*args, **kwargs):
            def fmt(v):
                return f"{v.__class__.__name__}: {getattr(v, 'shape', v)}"

            time = datetime.now(timezone.utc)
            result = fn(*args, **kwargs)
            duration = (datetime.now(timezone.utc) - time).total_seconds()
            fn_params = inspect.signature(fn).parameters
            # set all params with defaults then update params with mapped arguments and explicit keyword args
            arguments = {k: None if v.default is inspect.Parameter.empty else v.default for k, v in fn_params.items()}
            arguments.update(zip(fn_params, args))
            arguments.update(kwargs)
            arguments = {k: fmt(v) for k, v in arguments.items()}
            state = (
                {k: fmt(getattr(args[0], k)) for k in state_attr if "self" in arguments}
                if "self" in arguments and state_attr
                else {}
            )
            metadata = {
                "_name": name,
                "_execution_time": time,
                "_execution_duration": duration,
                "_arguments": {k: v for k, v in arguments.items() if k != "self"},
                "_state": state,
                "_version": __version__,
            }
            for k, v in metadata.items():
                object.__setattr__(result, k, v)
            return result

        return wrapper

    return decorator
