from importlib.util import find_spec
from typing import List

__all__: List[str] = []

if find_spec("torch") is not None:  # pragma: no cover
    from dataeval._internal.utils import read_dataset

    __all__ += ["read_dataset"]
