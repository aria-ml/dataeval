"""PyTorch-based feature extractor."""

__all__ = []

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from dataeval.config import get_device
from dataeval.protocols import Array, DeviceLike, Transform
from dataeval.types import ReprMixin
from dataeval.utils._internal import as_numpy, iter_images
from dataeval.utils.training import PostprocessFn

_logger = logging.getLogger(__name__)


class TorchExtractor(ReprMixin):
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
    flatten : bool, default True
        If True, flattens outputs with more than 2 dimensions to (N, D) shape.
        If False, preserves the original output shape.
    batch_size : int or None, default None
        Forward-pass (compute) batch size: how many images go through the model
        at once. ``None`` runs a single forward pass over all inputs. When this
        extractor is wrapped by ``Embeddings``, ``Embeddings`` loads images in its
        own (I/O) chunks and this extractor sub-batches each chunk by this value,
        so the smaller of the two bounds the forward pass.
    postprocess_fn : PostprocessFn or None, default None
        Batch-level decode applied to each minibatch's full raw model output
        (passed as-is, including a tuple output), e.g. to turn a detection head's
        raw output into a ``(n_detections, n_classes)`` score tensor. Must return
        a 2D tensor per batch (or a tuple whose element 0 is one). Mutually
        exclusive with ``layer_name``. When set, ``flatten`` is bypassed (decoded
        output is used as scores as-is). When not set, a tuple model output is
        reduced to its element 0.

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
        flatten: bool = True,
        batch_size: int | None = None,
        postprocess_fn: PostprocessFn | None = None,
    ) -> None:
        self.device = get_device(device)
        self._transforms = self._normalize_transforms(transforms)
        self._layer_name = layer_name
        self._use_output = use_output
        self._flatten = flatten
        self._batch_size = batch_size
        if postprocess_fn is not None and layer_name is not None:
            raise ValueError(
                "postprocess_fn and layer_name are mutually exclusive: layer_name hooks an "
                "intermediate layer, while postprocess_fn decodes the final model output."
            )
        self._postprocess_fn = postprocess_fn

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

    @property
    def flatten(self) -> bool:
        """Return whether outputs are flattened to 2D."""
        return self._flatten

    @property
    def batch_size(self) -> int | None:
        """Return the default batch size for inference, if set."""
        return self._batch_size

    def _normalize_transforms(
        self,
        transforms: Transform[torch.Tensor] | Iterable[Transform[torch.Tensor]] | None,
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

    def __call__(self, data: Any) -> Array:  # noqa: C901
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
        for img in iter_images(data):
            tensor = torch.as_tensor(as_numpy(img))
            for transform in self._transforms:
                tensor = transform(tensor)
            images.append(tensor)

        if not images:
            return np.empty((0,), dtype=np.float32)

        # batch_size is None -> a single forward pass over all images.
        bs = self._batch_size if self._batch_size is not None else len(images)

        outputs: list[torch.Tensor] = []
        for start in range(0, len(images), bs):
            input_tensor = torch.stack(images[start : start + bs]).to(self.device)
            with torch.no_grad():
                if self._layer_name:
                    _ = self._model(input_tensor)  # Triggers hook
                    output = self._captured_output
                else:
                    output = self._model(input_tensor)
            postprocess_fn = self._postprocess_fn
            if postprocess_fn is not None:
                # postprocess_fn decodes the full raw model output (which may be a tuple).
                decoded = postprocess_fn(output)
                # It may itself return a tuple of tensors; the scores are element 0.
                output = decoded[0] if isinstance(decoded, tuple) else decoded
            elif isinstance(output, tuple):
                # No decoder: take the primary output (e.g. a VAE returns (recon, mu, logvar)).
                output = output[0]
            outputs.append(output.detach().cpu())

        result = torch.cat(outputs).numpy()

        if self._postprocess_fn is not None:
            # Decoded detection scores: collapse any leading dims to (n, n_classes).
            if result.ndim > 2:
                result = result.reshape(-1, result.shape[-1])
        elif self._flatten and result.ndim > 2:
            # Embeddings: flatten spatial dims (N, C, H, W) -> (N, C*H*W).
            result = result.reshape(result.shape[0], -1)

        return result

    def __repr__(self) -> str:
        layer_info = f", layer_name={self._layer_name!r}" if self._layer_name else ""
        flatten_info = "" if self._flatten else ", flatten=False"
        return f"TorchExtractor(device={self.device}{layer_info}{flatten_info})"
