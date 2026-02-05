"""
PyTorch-based feature extractor.
"""

__all__ = []

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from dataeval.config import get_device
from dataeval.protocols import Array, DeviceLike, Transform

_logger = logging.getLogger(__name__)


class TorchExtractor:
    """
    Extracts embeddings from a PyTorch model, with optional intermediate layer hooking.

    Encapsulates all PyTorch-specific logic for feature extraction:

    - Model management (torch.nn.Module)
    - Device handling
    - Transform pipeline
    - Layer hooking for intermediate layer extraction

    Implements the :class:`~dataeval.protocols.FeatureExtractor` protocol.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model for feature extraction.
    transforms : Transform or Sequence[Transform] or None, default None
        Preprocessing transforms to apply before encoding. When None, uses raw images.
    device : DeviceLike or None, default None
        Device for computation. When None, uses DataEval's configured device.
    layer_name : str or None, default None
        Layer to extract embeddings from. When None, uses model output.
    use_output : bool, default True
        If True, captures layer output; if False, captures layer input.
        Only used when layer_name is specified.

    Example
    -------
    Basic usage with a model:

    >>> import torch.nn as nn
    >>> from dataeval import Embeddings
    >>> from dataeval.extractors import TorchExtractor
    >>>
    >>> model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128))
    >>> extractor = TorchExtractor(model, device="cpu")
    >>> embeddings = Embeddings(dataset, extractor=extractor, batch_size=32)

    Extracting from an intermediate layer:

    >>> extractor = TorchExtractor(
    ...     model,
    ...     layer_name="0",  # Extract from Flatten layer
    ...     use_output=True,
    ... )
    """

    device: torch.device

    def __init__(
        self,
        model: torch.nn.Module,
        transforms: Transform[torch.Tensor] | Iterable[Transform[torch.Tensor]] | None = None,
        device: DeviceLike | None = None,
        layer_name: str | None = None,
        use_output: bool = True,
    ) -> None:
        self.device = get_device(device)
        self._transforms = self._normalize_transforms(transforms)
        self._layer_name = layer_name
        self._use_output = use_output

        # Setup model
        self._model = model.to(self.device).eval()

        # Setup hook for intermediate layer extraction
        self._captured_output: Any = None
        if layer_name is not None:
            target_layer = self._get_valid_layer(layer_name, model)
            target_layer.register_forward_hook(self._hook_fn)
            _logger.debug(f"Capturing {'output' if use_output else 'input'} data from layer {layer_name}.")

    @property
    def layer_name(self) -> str | None:
        """Return the layer name for intermediate extraction, if set."""
        return self._layer_name

    @property
    def use_output(self) -> bool:
        """Return whether output (True) or input (False) is captured from the layer."""
        return self._use_output

    def _normalize_transforms(
        self, transforms: Transform[torch.Tensor] | Iterable[Transform[torch.Tensor]] | None
    ) -> list[Transform[torch.Tensor]]:
        """Normalize transforms to a list."""
        if transforms is None:
            return []
        if isinstance(transforms, Transform):
            return [transforms]
        return list(transforms)

    def _hook_fn(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """Forward hook to capture layer input or output."""
        if self._use_output:
            self._captured_output = output.detach().clone()
        else:
            self._captured_output = inputs[0].detach().clone()

    def _get_valid_layer(self, layer_name: str, model: torch.nn.Module) -> torch.nn.Module:
        """Validate and return the target layer for hook registration."""
        modules_dict = dict(model.named_modules())

        if layer_name not in modules_dict:
            formatted_layers = "\n".join(f"  {layer}" for layer in modules_dict)
            raise ValueError(f"Invalid layer '{layer_name}'. Available layers are:\n{formatted_layers}")

        return modules_dict[layer_name]

    def __call__(self, data: Any) -> Array:
        """
        Extract features from a batch of images.

        Parameters
        ----------
        data : Any
            Iterable of images to extract features from. Each image should be
            array-like and convertible to a torch tensor.

        Returns
        -------
        Array
            Embeddings array of shape (n_images, embedding_dim).
        """
        images: list[torch.Tensor] = []
        for img in data:
            tensor = torch.as_tensor(img)
            for transform in self._transforms:
                tensor = transform(tensor)
            images.append(tensor)

        if not images:
            return np.empty((0,), dtype=np.float32)

        input_tensor = torch.stack(images).to(self.device)

        with torch.no_grad():
            if self._layer_name:
                _ = self._model(input_tensor)  # Triggers hook
                output = self._captured_output
            else:
                output = self._model(input_tensor)

        return output.cpu().numpy()

    def __repr__(self) -> str:
        layer_info = f", layer_name={self._layer_name!r}" if self._layer_name else ""
        return f"TorchExtractor(device={self.device}{layer_info})"
