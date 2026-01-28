"""
PyTorch-based embedding encoder.
"""

__all__ = []

import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal, overload

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Subset

from dataeval.config import get_batch_size, get_device
from dataeval.protocols import ArrayLike, Dataset, DeviceLike, Transform

_logger = logging.getLogger(__name__)


class TorchEmbeddingEncoder:
    """
    PyTorch-based embedding encoder.

    Encapsulates all PyTorch-specific logic for embedding extraction:
    - Model management (torch.nn.Module)
    - Device handling
    - Transform pipeline
    - Batch processing via DataLoader
    - Layer hooking for intermediate layer extraction

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model for embedding extraction.
    batch_size : int or None, default None
        Number of samples per batch. When None, uses DataEval's configured batch size.
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
    >>> from dataeval.encoders import TorchEmbeddingEncoder
    >>> from dataeval import Embeddings
    >>>
    >>> model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128))
    >>> encoder = TorchEmbeddingEncoder(model, batch_size=32, device="cpu")
    >>> embeddings = Embeddings(dataset, encoder=encoder)

    Extracting from an intermediate layer:

    >>> encoder = TorchEmbeddingEncoder(
    ...     model,
    ...     batch_size=32,
    ...     layer_name="0",  # Extract from Flatten layer
    ...     use_output=True,
    ... )
    """

    device: torch.device

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int | None = None,
        transforms: Transform[torch.Tensor] | Iterable[Transform[torch.Tensor]] | None = None,
        device: DeviceLike | None = None,
        layer_name: str | None = None,
        use_output: bool = True,
    ) -> None:
        self.device = get_device(device)
        self._batch_size = get_batch_size(batch_size)
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
    def batch_size(self) -> int:
        """Return the batch size used for encoding."""
        return self._batch_size

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

    def _encode_batch(self, images: list[torch.Tensor]) -> NDArray[Any]:
        """Encode a batch of images to embeddings using the model."""
        input_tensor = torch.stack(images).to(self.device)

        with torch.no_grad():
            if self._layer_name:
                _ = self._model(input_tensor)  # Triggers hook
                output = self._captured_output
            else:
                output = self._model(input_tensor)

        return output.cpu().numpy()

    class _DatasetWrapper(torch.utils.data.Dataset[torch.Tensor]):
        """Wrapper for dataset to convert to PyTorch tensors and apply transforms."""

        def __init__(
            self,
            dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
            transforms: list[Transform[torch.Tensor]],
        ) -> None:
            self._dataset = dataset
            self._transforms = transforms

        def __getitem__(self, index: int) -> torch.Tensor:
            item = self._dataset[index]
            image = torch.as_tensor(item[0] if isinstance(item, tuple) else item)
            for transform in self._transforms:
                image = transform(image)
            return image

        def __len__(self) -> int:
            return len(self._dataset)

    @overload
    def encode(
        self,
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        indices: Sequence[int],
        stream: Literal[True],
    ) -> Iterator[tuple[Sequence[int], NDArray[Any]]]: ...

    @overload
    def encode(
        self,
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        indices: Sequence[int],
        stream: Literal[False] = ...,
    ) -> NDArray[Any]: ...

    def encode(
        self,
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        indices: Sequence[int],
        stream: bool = False,
    ) -> Iterator[tuple[Sequence[int], NDArray[Any]]] | NDArray[Any]:
        """
        Encode images at specified indices to embeddings.

        Parameters
        ----------
        dataset : Dataset
            Dataset providing images to encode.
        indices : Sequence[int]
            Indices of images to encode from the dataset.
        stream : bool, default False
            If True, yields (batch_indices, batch_embeddings) tuples.
            If False, returns all embeddings as a single array.

        Returns
        -------
        NDArray[Any] or Iterator[tuple[Sequence[int], NDArray[Any]]]
            Embeddings array or iterator of batches.

        Raises
        ------
        IndexError
            If any indices are out of range for the dataset.
        """

        def _generate() -> Iterator[tuple[Sequence[int], NDArray[Any]]]:
            # Wrap dataset for PyTorch
            wrapped = self._DatasetWrapper(dataset, self._transforms)

            # Process in batches
            for batch_start in range(0, len(indices), self._batch_size):
                batch_idx = list(indices[batch_start : batch_start + self._batch_size])

                # Use DataLoader with custom collate function to encode the batch
                loader = DataLoader(
                    Subset(wrapped, batch_idx),
                    batch_size=len(batch_idx),
                    collate_fn=self._encode_batch,
                )

                for embeddings in loader:
                    yield batch_idx, embeddings

        if not indices:
            if stream:
                return iter([])
            return np.empty((0,), dtype=np.float32)

        # Validate indices
        out_of_range = set(indices) - set(range(len(dataset)))
        if out_of_range:
            raise IndexError(f"Indices {sorted(out_of_range)} are out of range for dataset of size {len(dataset)}")

        if stream:
            return _generate()

        return np.vstack([emb for _, emb in _generate()])

    def __repr__(self) -> str:
        layer_info = f", layer_name={self._layer_name!r}" if self._layer_name else ""
        return f"TorchEmbeddingEncoder(batch_size={self._batch_size}, device={self.device}{layer_info})"
