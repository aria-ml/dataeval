"""Escape hatch: run a torchvision v2 transform over a dataset view."""

from __future__ import annotations

__all__ = []

import warnings
from typing import Any

import numpy as np
import xxhash as xxh
from numpy.typing import NDArray

from dataeval.data._view import Operation, View
from dataeval.flags import ImageStats
from dataeval.protocols import ObjectDetectionTarget, SegmentationTarget
from dataeval.utils._internal import MaskedTarget, as_numpy, mask_metadata
from dataeval.utils.preprocessing import normalize_image_shape

#: Sample stride for the missing-``id`` content-hash fallback: enough pixels to separate
#: datums, few enough that the hash is not the cost of the transform.
_CONTENT_HASH_STRIDE = 7


def _install_hint() -> str:
    """Name the torchvision install that matches the torch build already in the environment."""
    try:
        import torch
    except ImportError:
        return (
            "Install both from one wheel index, e.g. `pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu` (or cu118/cu128)."
        )
    # A wheel-index build tags its local version with the index name ("2.13.0+cu128"); a
    # PyPI build has no tag, and anything else is too unusual to turn into a URL.
    _, _, build = torch.__version__.partition("+")
    if not build:
        return "Your torch came from PyPI, so `pip install torchvision` will match it."
    if not build.isalnum():
        return (
            f"Your torch is a '{build}' build, so install the torchvision published alongside it "
            "rather than the one on PyPI."
        )
    return (
        f"Your torch is a '{build}' build, so install torchvision from that same index: "
        f"`pip install torchvision --index-url https://download.pytorch.org/whl/{build}`."
    )


def _import_torchvision() -> tuple[Any, Any]:
    """Import torchvision lazily, so ``import dataeval.data`` stays torchvision-free."""
    try:
        import torch
        from torchvision import tv_tensors
    except ImportError as error:
        raise ImportError(
            f"TorchvisionTransform requires torchvision, which is not installed. {_install_hint()} "
            "torchvision's compiled ops are built against one specific torch build, so a mismatched "
            "pair installs cleanly and then fails on import."
        ) from error
    return torch, tv_tensors


def _datum_seed(metadata: Any, image: NDArray[Any]) -> tuple[int, bool]:
    """Derive a stable per-datum seed; report whether the ``id`` fallback was taken.

    ``xxhash`` rather than the builtin ``hash()``: ``hash()`` on strings is randomized per
    process by ``PYTHONHASHSEED``, so the same script would produce different augmentations
    across runs.
    """
    datum_id = metadata.get("id") if isinstance(metadata, dict) else None
    if datum_id is not None:
        return xxh.xxh64_intdigest(str(datum_id)), False
    subsample = image[..., ::_CONTENT_HASH_STRIDE, ::_CONTENT_HASH_STRIDE]
    return xxh.xxh64_intdigest(np.ascontiguousarray(subsample).tobytes()), True


class TorchvisionTransform(Operation):
    """
    Apply a torchvision v2 transform to every datum, keeping targets consistent.

    The curated operations -- :class:`~dataeval.data.Resize`,
    :class:`~dataeval.data.Crop`, :class:`~dataeval.data.SelectChannels` -- cover
    dataset-*defining* manipulation and are the documented path. This is the escape hatch,
    for an existing torchvision pipeline or for augmentation and corruption applied
    dataset-wide to probe robustness.

    torchvision v2 is the contract because it already carries geometry across the whole
    sample: boxes and masks are transformed with the image rather than silently left
    behind. A bare ``Callable[[image], image]`` is not accepted for exactly that reason --
    a geometric transform would corrupt every bounding box with no error and no warning.

    Parameters
    ----------
    transform : Callable
        A torchvision v2 transform (or ``v2.Compose`` of them). It is handed a sample dict
        with ``"image"``, and for object-detection data ``"boxes"`` and ``"labels"``.
    seed : int, default 0
        Base seed for random transforms. Mixed with a per-operation salt derived from the
        transform's ``repr``, then with each datum's ``id``. Two *identically*-configured
        operations chained in one view need distinct ``seed=`` values to decorrelate.
    invalidates : ImageStats, default ImageStats.ALL
        Statistics this operation declares it invalidates. Defaults to everything, because
        an arbitrary transform can move anything. Narrow it when you know what your
        transform does -- ``ImageStats.NONE`` for a no-op, ``ImageStats.DIMENSION`` for a
        pure geometric one.

    Raises
    ------
    ImportError
        At first read if torchvision is not installed.
    NotImplementedError
        At first read for a segmentation target.

    Warns
    -----
    UserWarning
        Once, if a datum's metadata omits the protocol-required ``id`` (the content-hash
        fallback is used), or if the transform looks like model preprocessing.

    See Also
    --------
    :doc:`/notebooks/h2_place_transforms` : choosing between a view operation and an extractor transform

    Notes
    -----
    **Determinism.** A random transform would otherwise make ``view[i]`` differ between the
    statistics pass, the embedding pass, and the duplicates pass -- breaking the stability
    that ``resolve_indices``, embeddings caching, and duplicate detection depend on. Each
    datum is therefore seeded from its own ``id``, which makes an augmented view a fixed
    object you can run evaluators against and compare against the source.

    The seed comes from the datum's ``id``, never its position. ``View.read`` receives an
    index into the *immediate* source, so seeding on the index would mean that inserting a
    :class:`~dataeval.data.Shuffle` upstream silently re-rolled every augmentation. An
    ``id`` survives reordering. For a source that omits it, a hash of a strided subsample
    of the image is used instead, and a warning is issued once.

    **A view built through this operation may not be reconstructable from its sidecar.**
    Provenance records an operation's ``repr``; a curated operation reprs to something you
    can paste back, while ``v2.Compose([...])`` reprs multi-line and a lambda inside one
    records nothing recoverable. That is a further reason the curated operations are the
    documented path and this is the escape hatch.

    Examples
    --------
    Apply a fixed corruption dataset-wide and evaluate what it does:

    >>> from torchvision.transforms import v2
    >>> from dataeval.data import TorchvisionTransform, View
    >>> view = View(dataset, [TorchvisionTransform(v2.GaussianBlur(kernel_size=3), seed=0)])
    >>> view[0][0].shape
    (3, 64, 64)
    """

    def __init__(self, transform: Any, *, seed: int = 0, invalidates: ImageStats = ImageStats.ALL) -> None:
        self.transform = transform
        self.seed = seed
        self._invalidates = invalidates
        # Salt per operation: two operations deriving the same seed from the same id would
        # produce correlated draws — two RandomHorizontalFlips that always agree.
        self._salt = seed ^ xxh.xxh64_intdigest(repr(transform))
        self._warned_missing_id = False
        self._checked_preprocessing = False

    @property
    def invalidates(self) -> ImageStats:
        """Statistics this transform may make describe itself rather than the data."""
        return self._invalidates

    def apply(self, view: View[Any]) -> None:
        view.map(self._transform)

    def _transform(self, datum: Any) -> Any:
        is_tuple = isinstance(datum, tuple) and len(datum) == 3
        target = datum[1] if is_tuple else None
        metadata = datum[2] if is_tuple else {}
        if isinstance(target, SegmentationTarget):
            raise NotImplementedError(
                "TorchvisionTransform does not support segmentation targets yet: the mask would "
                "have to be carried through the sample as a tv_tensors.Mask."
            )

        image = normalize_image_shape(as_numpy(datum[0] if is_tuple else datum))
        output = self._run(image, target, metadata)
        new_image = as_numpy(output["image"])
        self._check_preprocessing(image, new_image)

        if not is_tuple:
            return new_image
        if not isinstance(target, ObjectDetectionTarget):
            return new_image, target, metadata
        return (new_image, *self._rewrite_target(target, metadata, output, len(np.asarray(target.labels))))

    def _run(self, image: NDArray[Any], target: Any, metadata: Any) -> dict[str, Any]:
        """Build the v2 sample, run the transform under a forked, per-datum-seeded RNG."""
        torch, tv_tensors = _import_torchvision()
        height, width = image.shape[-2], image.shape[-1]
        sample: dict[str, Any] = {"image": tv_tensors.Image(torch.as_tensor(np.ascontiguousarray(image)))}

        if isinstance(target, ObjectDetectionTarget):
            boxes = as_numpy(target.boxes).reshape(-1, 4)
            labels = as_numpy(target.labels).reshape(-1)
            sample["boxes"] = tv_tensors.BoundingBoxes(
                torch.as_tensor(boxes, dtype=torch.float32), format="XYXY", canvas_size=(height, width)
            )
            # Labels are packed as (N, 2) of [label, source position]. Transforms that drop
            # detections (SanitizeBoundingBoxes, RandomIoUCrop) mask the labels alongside
            # the boxes, so column 1 comes back as exactly the surviving positions — which
            # is the mask the datum's per-detection metadata has to be filtered by.
            positions = torch.arange(len(labels), dtype=torch.int64)
            sample["labels"] = torch.stack([torch.as_tensor(labels, dtype=torch.int64), positions], dim=1)

        seed, used_fallback = _datum_seed(metadata, image)
        self._warn_missing_id(used_fallback)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._salt ^ seed)
            return self.transform(sample)

    @staticmethod
    def _rewrite_target(target: Any, metadata: Any, output: dict[str, Any], count: int) -> tuple[Any, Any]:
        """Rebuild the target and metadata from the transformed sample."""
        labels = as_numpy(output["labels"]).reshape(-1, 2)
        mask = np.zeros(count, dtype=np.bool_)
        mask[labels[:, 1]] = True
        overrides = {"boxes": as_numpy(output["boxes"]).reshape(-1, 4), "labels": labels[:, 0]}
        return MaskedTarget(target, mask, overrides), mask_metadata(metadata, mask)

    def _warn_missing_id(self, used_fallback: bool) -> None:
        if not used_fallback or self._warned_missing_id:
            return
        self._warned_missing_id = True
        warnings.warn(
            f"{type(self).__name__}: datum metadata has no 'id', so augmentations are seeded from a "
            "hash of the image content instead. Results stay reproducible, but two byte-identical "
            "images will receive the same augmentation.",
            UserWarning,
            stacklevel=2,
        )

    def _check_preprocessing(self, source: NDArray[Any], result: NDArray[Any]) -> None:
        """Warn once if the transform looks like model preprocessing rather than data.

        ``Normalize`` and ``ToDtype(scale=True)`` end essentially every real torchvision
        pipeline, and a pasted-in pipeline puts model preprocessing at the view level, where
        it corrupts every pixel and visual statistic. Both are detectable: an integer image
        that came back floating point, or values that left the source's range.
        """
        if self._checked_preprocessing:
            return
        self._checked_preprocessing = True
        if not np.issubdtype(result.dtype, np.floating) or result.size == 0 or source.size == 0:
            return
        rescaled = np.issubdtype(source.dtype, np.integer)
        shifted = bool(result.min() < source.min() or result.max() > source.max())
        if not (rescaled or shifted):
            return
        warnings.warn(
            f"{type(self).__name__}: {self.transform!r} rescaled or re-centered pixel values. If this "
            "is model preprocessing (Normalize, ToDtype(scale=True)), it belongs on the extractor's "
            "transforms= rather than on the view, where it makes every pixel and visual statistic "
            "describe the preprocessing instead of the data.",
            UserWarning,
            stacklevel=2,
        )
