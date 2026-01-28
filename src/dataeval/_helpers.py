__all__ = []

from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel

from dataeval.protocols import Metadata

IGNORE_KEYS = {"self", "config", "__class__"}


def get_overrides(local_vars: dict[str, Any], exclude: set[str] | None = None) -> dict[str, Any]:
    """
    Extracts explicit arguments from locals() to create a config override dictionary.
    Removes 'self', 'config', and any variable that is None.
    """
    # 1. Standard things to always ignore in __init__
    exclude = IGNORE_KEYS | (exclude or set())
    return {key: value for key, value in local_vars.items() if key not in exclude and value is not None}


def apply_config(obj: Any, config: BaseModel, exclude: set[str] | None = None) -> None:
    """
    Applies attributes onto obj from config, excluding specified keys.
    """
    exclude = IGNORE_KEYS | (exclude or set())
    setattr(obj, "config", config)
    for key, value in config.model_dump().items():
        if key not in exclude:
            setattr(obj, key, value)


def _get_item_indices(metadata: Metadata) -> Sequence[int]:
    """Get item indices from metadata, generating default if not available."""
    item_indices = getattr(metadata, "item_indices", None)
    if item_indices is not None:
        return item_indices
    return list(range(len(metadata.class_labels)))


def _get_index2label(metadata: Metadata) -> dict[int, str]:
    """Get index2label mapping, generating default if not available."""
    index2label = getattr(metadata, "index2label", None)
    if index2label:
        return dict(index2label)
    return {int(i): str(i) for i in np.unique(metadata.class_labels)}
