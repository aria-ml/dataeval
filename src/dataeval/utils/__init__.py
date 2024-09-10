from dataeval import _IS_TORCH_AVAILABLE

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from dataeval._internal.utils import read_dataset

    __all__ = ["read_dataset"]
