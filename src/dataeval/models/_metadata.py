"""Parse and validate the opinionated model-metadata.json (IR-3.1)."""

__all__ = ["ModelIOSpec", "read_model_metadata"]

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_TASKS = ("IMAGE_CLASSIFICATION", "IMAGE_OBJECT_DETECTION")


@dataclass(frozen=True)
class ModelIOSpec:
    """
    Typed, immutable view of a model-metadata.json input/output contract.

    Produced by :func:`~dataeval.models.read_model_metadata` and consumed by
    :func:`~dataeval.models.build_model_input` and the opinionated predictors to
    shape and normalize images to the contract a model expects. Instances are
    frozen (hashable, read-only).

    Attributes
    ----------
    task : {"IMAGE_CLASSIFICATION", "IMAGE_OBJECT_DETECTION"}
        Declared model interface (``io.interface`` in the metadata file).
    channels : {"RGB", "GRAYSCALE"}
        Expected input color layout. ``"RGB"`` is 3-channel, ``"GRAYSCALE"`` is
        1-channel; input images are converted to match.
    height : int
        Expected input height in pixels. ``-1`` marks a variable dimension that
        callers must override explicitly.
    width : int
        Expected input width in pixels. ``-1`` marks a variable dimension that
        callers must override explicitly.
    batch_size : int
        Declared batch size (``io.batchSize``). ``-1`` marks a dynamic batch.
    n_classes : int
        Number of output classes (``io.output.nClasses``).
    n_boxes : int or None, default None
        Number of detection boxes (``io.output.nBoxes``); ``None`` for
        classification models.

    See Also
    --------
    ~dataeval.models.read_model_metadata : Parse a metadata file into a :class:`~dataeval.models.ModelIOSpec`.
    ~dataeval.models.build_model_input : Shape images to this contract.

    Examples
    --------
    >>> spec = ModelIOSpec(
    ...     task="IMAGE_CLASSIFICATION",
    ...     channels="RGB",
    ...     height=8,
    ...     width=8,
    ...     batch_size=-1,
    ...     n_classes=4,
    ... )
    >>> spec.channels
    'RGB'
    >>> spec.n_boxes is None
    True
    """

    task: Literal["IMAGE_CLASSIFICATION", "IMAGE_OBJECT_DETECTION"]
    channels: Literal["RGB", "GRAYSCALE"]
    height: int
    width: int
    batch_size: int
    n_classes: int
    n_boxes: int | None = None


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"model-metadata.json missing required field '{key}' in {where}")
    return mapping[key]


def read_model_metadata(path: str | Path) -> ModelIOSpec:
    """
    Read and validate an opinionated model-metadata.json file.

    Parses the ``io`` block of the metadata file and returns its input/output
    contract as a :class:`~dataeval.models.ModelIOSpec`. The interface, color layout, and required
    shape fields are validated; detection metadata additionally reads the optional
    box count.

    Parameters
    ----------
    path : str or Path
        Path to the model-metadata.json file describing the model contract.

    Returns
    -------
    ModelIOSpec
        Validated, immutable view of the model's input/output contract.

    Raises
    ------
    ValueError
        If a required field is missing, or ``io.interface`` / ``io.input.channels``
        holds an unsupported value.
    FileNotFoundError
        If ``path`` does not exist.

    See Also
    --------
    ~dataeval.models.ModelIOSpec : The returned contract.

    Examples
    --------
    >>> spec = read_model_metadata(classifier_metadata_path)
    >>> spec.task
    'IMAGE_CLASSIFICATION'
    >>> (spec.channels, spec.height, spec.width)
    ('RGB', 8, 8)
    >>> spec.n_classes
    4
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    io = _require(data, "io", "<root>")
    task = _require(io, "interface", "io")
    if task not in _TASKS:
        raise ValueError(f"model-metadata.json io.interface must be one of {_TASKS}; got {task!r}")
    inp = _require(io, "input", "io")
    out = _require(io, "output", "io")
    channels = _require(inp, "channels", "io.input")
    if channels not in ("RGB", "GRAYSCALE"):
        raise ValueError(f"io.input.channels must be 'RGB' or 'GRAYSCALE'; got {channels!r}")
    n_boxes = out.get("nBoxes") if task == "IMAGE_OBJECT_DETECTION" else None
    return ModelIOSpec(
        task=task,
        channels=channels,
        height=int(_require(inp, "height", "io.input")),
        width=int(_require(inp, "width", "io.input")),
        batch_size=int(_require(io, "batchSize", "io")),
        n_classes=int(_require(out, "nClasses", "io.output")),
        n_boxes=None if n_boxes is None else int(n_boxes),
    )
