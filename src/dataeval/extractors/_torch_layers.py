"""Several layers of one torch model, read in one forward pass."""

__all__ = []

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch

from dataeval._experimental import experimental
from dataeval.extractors._torch import TorchExtractor, get_valid_layer
from dataeval.protocols import Array, DeviceLike, Transform


@experimental
class TorchLayers:
    """
    Read several layers of one PyTorch model in one forward pass.

    Each layer gives what a :class:`~dataeval.extractors.TorchExtractor` with that `layer_name` gives on its own, but
    the model runs once per batch for all of them, not once for each. It has no call of its own, so it is not a
    feature extractor: give it to :meth:`~dataeval.Embeddings.compute_multi`, which stores each layer's embeddings as
    an :class:`~dataeval.Embeddings` of its own.

    Parameters
    ----------
    model : torch.nn.Module
        The model to read.
    layer_names : Sequence[str]
        The layers to read, as :meth:`torch.nn.Module.named_modules` names them. Each layer's output is captured.
    transforms : Transform or Iterable[Transform] or None, default None
        Transforms applied to each image before the model, as for :class:`~dataeval.extractors.TorchExtractor`.
    device : DeviceLike or None, default None
        Device the model runs on. None uses the configured default, which is the CPU unless set.
    flatten : bool, default True
        Whether outputs with more than 2 dimensions are flattened to (N, D).
    batch_size : int or None, default None
        Images per forward pass. None runs one forward pass over all the images it is given.

    Raises
    ------
    ValueError
        When `layer_names` is empty, names a layer twice, or names a layer the model does not have.

    Notes
    -----
    Each layer is hooked only while the model runs, so nothing is left registered on the model. Each layer's result is
    tested equal to its own extractor's on the CPU; a GPU is untested. The layer outputs of one batch are all held in
    memory at once. Input capture is not offered: a layer's output is what is read.

    This raises :class:`~dataeval.exceptions.ExperimentalWarning` the first time one is constructed in a process.

    .. versionadded:: 1.3

    Examples
    --------
    >>> from dataeval import Embeddings
    >>> from dataeval.extractors import TorchLayers
    >>>
    >>> layers = TorchLayers(my_model, ["2", "4"])
    >>> children = Embeddings.compute_multi(train_dataset, {"model": layers}, batch_size=32)
    >>> list(children)
    ['model/2', 'model/4']
    >>> children["model/2"].shape, children["model/4"].shape
    ((40, 64), (40, 32))
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: Sequence[str],
        *,
        transforms: Transform[torch.Tensor] | Iterable[Transform[torch.Tensor]] | None = None,
        device: DeviceLike | None = None,
        flatten: bool = True,
        batch_size: int | None = None,
    ) -> None:
        if isinstance(layer_names, str):
            raise TypeError(
                f"layer_names must be a sequence of layer names, such as [{layer_names!r}], not one string."
            )
        names = tuple(layer_names)
        if not names:
            raise ValueError("TorchLayers needs at least one layer name.")
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Layer {', '.join(repr(name) for name in repeated)} is named more than once.")
        self._base = TorchExtractor(model, transforms=transforms, device=device, flatten=flatten, batch_size=batch_size)
        self._layers = {name: get_valid_layer(name, model) for name in names}

    @property
    def layer_names(self) -> tuple[str, ...]:
        """The layers read, in the order they were given."""
        return tuple(self._layers)

    @property
    def batch_size(self) -> int | None:
        """Images per forward pass, if set."""
        return self._base.batch_size

    def extract(self, data: Any) -> dict[str, Array]:
        """
        Read every layer for a batch of images.

        Parameters
        ----------
        data : Any
            The images, as :class:`~dataeval.extractors.TorchExtractor` accepts them.

        Returns
        -------
        dict[str, Array]
            One array of shape (n_images, ...) for each layer, in the order of `layer_names`.
        """
        images = self._base._to_tensors(data)
        if not images:
            return {name: np.empty((0,), dtype=np.float32) for name in self._layers}
        size = self._base.batch_size if self._base.batch_size is not None else len(images)
        outputs = self._base._forward_layers(images, size, self._layers)
        return {name: self._base._postprocess_outputs(rows) for name, rows in outputs.items()}

    def solo(self, layer_name: str) -> TorchExtractor:
        """
        Return a :class:`~dataeval.extractors.TorchExtractor` that reads one of the layers on its own.

        Parameters
        ----------
        layer_name : str
            One of `layer_names`.

        Returns
        -------
        TorchExtractor
            An extractor with the same model, transforms, device, flattening and batch size.

        Raises
        ------
        ValueError
            When `layer_name` is not one of `layer_names`.
        """
        if layer_name not in self._layers:
            raise ValueError(f"{layer_name!r} is not one of the layers read: {', '.join(self._layers)}.")
        return TorchExtractor(
            self._base._model,
            # The base extractor's list, not the argument, which may have been a one-shot iterable it consumed.
            transforms=self._base._transforms,
            device=self._base.device,
            layer_name=layer_name,
            flatten=self._base.flatten,
            batch_size=self._base.batch_size,
        )

    def __repr__(self) -> str:
        return f"TorchLayers(device={self._base.device}, layer_names={self.layer_names!r})"
