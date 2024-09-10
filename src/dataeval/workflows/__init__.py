from dataeval import _IS_TORCH_AVAILABLE

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from dataeval._internal.workflows.sufficiency import Sufficiency

    __all__ = ["Sufficiency"]
