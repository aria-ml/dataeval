"""Present an object-detection dataset's boxes as an image-classification dataset."""

from __future__ import annotations

__all__ = ["DetectionCrops"]

import logging
from collections.abc import Iterator, Mapping
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import (
    AnnotatedDataset,
    DatasetMetadata,
    DatumMetadata,
    ObjectDetectionDataset,
)
from dataeval.utils._internal import as_numpy
from dataeval.utils._validate import requires_maite_dataset
from dataeval.utils.preprocessing import clip_box, crop_with_fill, normalize_image_shape

_logger = logging.getLogger(__name__)

RegionType = Literal["object", "context", "surround"]
SquareType = Literal["off", "expand", "pad"]
FillType = Literal["mean", "zero"]


def _validate_params(region: str, padding: float, min_size: int, square: str, fill: str) -> None:  # noqa: C901
    """Validate the constructor's policy parameters."""
    if region not in ("object", "context", "surround"):
        raise ValueError(f"region must be 'object', 'context', or 'surround'; got {region!r}.")
    if square not in ("off", "expand", "pad"):
        raise ValueError(f"square must be 'off', 'expand', or 'pad'; got {square!r}.")
    if fill not in ("mean", "zero"):
        raise ValueError(f"fill must be 'mean' or 'zero'; got {fill!r}.")
    if padding < 0:
        raise ValueError(f"padding must be >= 0; got {padding}.")
    if min_size < 0:
        raise ValueError(f"min_size must be >= 0; got {min_size}.")
    if region == "surround" and padding <= 0:
        raise ValueError("region='surround' requires padding > 0, otherwise the masked crop is empty.")


class _CropDatumMetadata(DatumMetadata):
    """Internal typing aid for a crop's metadata dict; see :class:`DetectionCrops`.

    Not part of the public API and not meant to be instantiated or subclassed — each
    crop's metadata is a plain ``dict`` at runtime. This subclass only lets the type
    checker verify the literal built in :meth:`DetectionCrops.__getitem__`. The keys are
    documented for users on :class:`DetectionCrops`.
    """

    source_id: int | str
    target: int
    box: list[float]


DetectionCropDatum: TypeAlias = tuple[NDArray[Any], NDArray[np.float32], DatumMetadata]


class DetectionCrops(AnnotatedDataset[DetectionCropDatum]):
    """Present an object-detection dataset's ground-truth boxes as an image-classification dataset.

    Each kept detection becomes one classification datum — a crop derived from the
    detection's box, labeled (one-hot) with the detection's class. The view satisfies the
    :obj:`~dataeval.protocols.ImageClassificationDataset` shape, so it drops into
    :class:`~dataeval.Embeddings`, :class:`~dataeval.scope.Coverage`,
    :func:`~dataeval.core.ber_mst`, and :class:`~dataeval.bias.Balance` — every
    per-(image, label) tool — unchanged, with crops aligned 1:1 to labels by construction.

    This makes object-detection feasibility (the bounding-box-classification reduction
    behind :term:`Upper-Bound Average Precision (UAP)`) and embedding-space coverage
    available to object-detection datasets without computing detection-level embeddings
    by hand. Crops are produced lazily on access, so an extractor's transforms still
    handle resize/normalize and :class:`~dataeval.Embeddings` still batches and caches.

    Parameters
    ----------
    dataset : ObjectDetectionDataset
        The source object-detection dataset. Each datum is a MAITE
        ``(image, ObjectDetectionTarget, metadata)`` 3-tuple; images are read in
        ``(C, H, W)`` layout and boxes in absolute-pixel ``[x0, y0, x1, y1]`` format.
    region : {"object", "context", "surround"}, default "object"
        Which pixels each crop retains. ``"object"`` and ``"context"`` both return the box
        widened by ``padding`` (``"context"`` is the conventional name when ``padding`` is
        large enough to bring in surroundings); ``"surround"`` returns the widened box with
        the original box masked to ``fill``, leaving only the background ring — a probe for
        whether the surroundings alone predict the class. ``"surround"`` requires
        ``padding > 0``. Only ``"object"`` is exercised by the shipped tutorials.
    padding : float, default 0.0
        Context margin added to each box, as a fraction of that box dimension: each side is
        extended by ``padding`` times the box's width (left/right) or height (top/bottom).
        ``0.1`` grows a 100x200 box to 120x240, centered. Must be ``>= 0``.
    min_size : int, default 1
        Drop detections whose box's shorter side is below this many pixels (degenerate or
        zero-area boxes are always dropped). The number dropped is logged and exposed as
        :attr:`n_dropped`; because dropping shrinks the view, ``len(crops)`` may be less
        than the source's total detection count.
    square : {"off", "expand", "pad"}, default "expand"
        How a non-square crop is reconciled with a square model input. ``"expand"`` squares
        the crop by extending the shorter side into real image pixels (shifting the window
        inward at image edges, falling back to ``fill`` only for unavoidable overflow) — no
        synthetic fill for interior boxes, but it brings in real background, which for
        extreme aspect ratios can dilute thin objects. ``"pad"`` squares by padding the
        shorter side with synthetic ``fill``, keeping the embedding object-focused (prefer
        this for strict feasibility / BER). ``"off"`` leaves crops rectangular for the
        extractor's resize to stretch (the prior default behavior).
    fill : {"mean", "zero"}, default "mean"
        Value for invented pixels — used by ``square="pad"``, by edge overflow in
        ``square="expand"``, and to mask the object in ``region="surround"``. ``"mean"``
        uses the per-crop, per-channel mean (normalization-agnostic, minimal contrast at
        the fill boundary); ``"zero"`` uses 0 (set this to your normalization mean if you
        need strict post-normalization neutrality).

    Attributes
    ----------
    n_dropped : int
        Number of detections dropped by ``min_size`` (or for being degenerate).
    index2label : dict[int, str]
        Mapping from class index to name, inherited from the source dataset.

    Notes
    -----
    Each datum's third element is its metadata — a plain ``dict`` at runtime, conforming to
    :obj:`~dataeval.protocols.DatumMetadata` — with the protocol-required ``id`` plus three
    keys added by this view that trace a crop back to its source detection:

    - ``id`` (``int``) — the crop's own identifier: its position in this view, ``0`` to
      ``len(crops) - 1``, aligned 1:1 with the labels and embeddings.
    - ``source_id`` (``int | str``) — the source datum's own ``DatumMetadata`` ``id`` (not a
      positional index), so a crop flagged downstream (e.g. as low-dispersion or uncovered)
      still resolves to the correct image after the source has been filtered, sorted, or
      otherwise re-indexed by a view such as :class:`~dataeval.data.Select` (which renumbers
      positions but passes each datum's ``id`` through unchanged). Falls back to the
      positional index for source data that omits the protocol-required ``id``.
    - ``target`` (``int``) — the detection's index within its source image (its position in
      that image's target arrays).
    - ``box`` (``list[float]``) — the detection's absolute-pixel ``[x0, y0, x1, y1]`` in the
      source image.

    Examples
    --------
    Wrap an object-detection dataset and run the classification-only tools on it:

    >>> from dataeval.data import DetectionCrops
    >>> crops = DetectionCrops(od_dataset)  # doctest: +SKIP
    >>> emb = Embeddings(crops, extractor=extractor, batch_size=64)  # doctest: +SKIP
    >>> Coverage().evaluate(crops, embeddings=emb)  # per-class dispersion over OD classes  # doctest: +SKIP
    """

    @requires_maite_dataset("dataset", expected="object_detection")
    def __init__(
        self,
        dataset: ObjectDetectionDataset,
        *,
        region: RegionType = "object",
        padding: float = 0.0,
        min_size: int = 1,
        square: SquareType = "expand",
        fill: FillType = "mean",
    ) -> None:
        _validate_params(region, padding, min_size, square, fill)

        self._dataset = dataset
        self._region: RegionType = region
        self._padding = float(padding)
        self._min_size = int(min_size)
        self._square: SquareType = square
        self._fill: FillType = fill

        # One pass to flatten detections into (item, target, label, box) rows, in image then
        # detection order — matching how Metadata flattens OD targets — applying the
        # min_size filter so crops and labels stay aligned by construction.
        index_map: list[tuple[int, int, int, NDArray[np.float64]]] = []
        source_ids: list[int | str] = []
        observed: set[int] = set()
        n_dropped = 0
        for item_index in range(len(dataset)):
            _, target, datum_metadata = dataset[item_index]
            # Stable provenance: the source datum's own id survives filtering/sorting/relabeling
            # views (which renumber positions but pass each datum's id through); fall back to the
            # positional index only for source data missing the protocol-required id.
            source_ids.append(datum_metadata.get("id", item_index))
            labels = as_numpy(target.labels).reshape(-1)
            boxes = as_numpy(target.boxes).reshape(-1, 4).astype(np.float64)
            for target_index, (label, box) in enumerate(zip(labels, boxes, strict=True)):
                width, height = box[2] - box[0], box[3] - box[1]
                if width <= 0 or height <= 0 or min(width, height) < self._min_size:
                    n_dropped += 1
                    continue
                index_map.append((item_index, target_index, int(label), box))
                observed.add(int(label))

        self._index_map = index_map
        self._source_ids = source_ids
        self.n_dropped: int = n_dropped
        self.index2label: Mapping[int, str] = self._resolve_index2label(observed)
        self._n_classes = (max(self.index2label) + 1) if self.index2label else 0
        source_id = str(dataset.metadata.get("id", "dataset"))
        self._metadata = DatasetMetadata({"id": f"{source_id}-crops", "index2label": dict(self.index2label)})

        # Single-slot cache: index_map is in image order, so consecutive detections from the
        # same image reuse one read instead of re-reading the image per detection.
        self._cache_index: int | None = None
        self._cache_image: NDArray[Any] | None = None

        _logger.debug(
            "DetectionCrops: %d crops over %d source images (%d dropped by min_size=%d)",
            len(index_map),
            len(dataset),
            n_dropped,
            self._min_size,
        )

    def _resolve_index2label(self, observed: set[int]) -> dict[int, str]:
        """Inherit the source mapping, adding fallbacks for any unmapped observed labels."""
        provided = self._dataset.metadata.get("index2label", None)
        if provided is not None:
            index2label = {int(k): str(v) for k, v in provided.items()}
            for label in observed:
                index2label.setdefault(label, f"UNDEFINED_CLASS_{label}")
            return index2label
        return {label: str(label) for label in sorted(observed)}

    @property
    def metadata(self) -> DatasetMetadata:
        """MAITE dataset metadata for the crop view (id and inherited index2label)."""
        return self._metadata

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, index: int) -> DetectionCropDatum:
        item_index, target_index, label, box = self._index_map[index]
        crop = self._crop(self._read_image(item_index), box)
        onehot = np.zeros(self._n_classes, dtype=np.float32)
        onehot[label] = 1.0
        meta: _CropDatumMetadata = {
            "id": index,
            "source_id": self._source_ids[item_index],
            "target": target_index,
            "box": box.tolist(),
        }
        return crop, onehot, meta

    def __iter__(self) -> Iterator[DetectionCropDatum]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return (
            f"DetectionCrops(dataset={self._dataset!r}, region={self._region!r}, padding={self._padding}, "
            f"min_size={self._min_size}, square={self._square!r}, fill={self._fill!r}, len={len(self)})"
        )

    def __str__(self) -> str:
        title = "DetectionCrops Dataset"
        sep = "-" * len(title)
        return (
            f"{title}\n{sep}\n    region: {self._region}\n    square: {self._square}\n"
            f"    crops: {len(self)} ({self.n_dropped} dropped)\n    classes: {len(self.index2label)}\n\n"
            f"{self._dataset}"
        )

    def _read_image(self, item_index: int) -> NDArray[Any]:
        """Read (and briefly cache) the source image at ``item_index`` as ``(C, H, W)``."""
        if self._cache_index == item_index and self._cache_image is not None:
            return self._cache_image
        image = normalize_image_shape(as_numpy(self._dataset[item_index][0]))
        self._cache_index, self._cache_image = item_index, image
        return image

    def _region_to_pixels(
        self, image_shape: tuple[int, ...], rx0: float, ry0: float, rx1: float, ry1: float
    ) -> tuple[int, int, int, int]:
        """Clip a float region to integer pixel bounds via the shared ``clip_box``, never empty."""
        ix0, iy0, ix1, iy1 = clip_box(image_shape, (rx0, ry0, rx1, ry1))
        return ix0, iy0, max(ix1, ix0 + 1), max(iy1, iy0 + 1)

    def _crop(self, image: NDArray[Any], box: NDArray[np.float64]) -> NDArray[Any]:
        """Crop ``box`` from ``image`` honoring the region / padding / square / fill policy."""
        x0, y0, x1, y1 = (float(v) for v in box)

        # Box widened by `padding` on each side (the region for all crops).
        pad_x, pad_y = self._padding * (x1 - x0), self._padding * (y1 - y0)
        rx0, ry0, rx1, ry1 = x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y

        if self._square == "off":
            ix0, iy0, ix1, iy1 = self._region_to_pixels(image.shape, rx0, ry0, rx1, ry1)
            crop, origin = image[..., iy0:iy1, ix0:ix1].copy(), (ix0, iy0)
        else:
            crop, origin = self._square_crop(image, rx0, ry0, rx1, ry1)

        if self._region == "surround":
            self._mask_box(crop, origin, x0, y0, x1, y1)
        return crop

    def _square_crop(
        self, image: NDArray[Any], rx0: float, ry0: float, rx1: float, ry1: float
    ) -> tuple[NDArray[Any], tuple[int, int]]:
        """Return a square crop and its (x, y) origin, per the ``square`` strategy.

        ``"expand"`` squares the region by extending the shorter side into real image
        pixels (shifting the window inward at edges, filling only unavoidable overflow).
        ``"pad"`` crops only the region's real pixels and centers them in a square ``fill``
        canvas, so the squaring adds no real background.
        """
        if self._square == "pad":
            return self._pad_crop(image, rx0, ry0, rx1, ry1)

        height, width, channels = image.shape[-2], image.shape[-1], image.shape[0]
        side = max(int(round(max(rx1 - rx0, ry1 - ry0))), 1)

        # Window top-left centered on the region, then shifted inward to capture real pixels.
        wx0, wy0 = (rx0 + rx1) / 2 - side / 2, (ry0 + ry1) / 2 - side / 2
        if side <= width:
            wx0 = min(max(wx0, 0.0), width - side)
        if side <= height:
            wy0 = min(max(wy0, 0.0), height - side)
        ox, oy = int(round(wx0)), int(round(wy0))

        # Paste the square window's real pixels into a same-size canvas, filling overflow per policy.
        # Pin the output to the image dtype so the mean fill is cast down (matching the other crop modes).
        return crop_with_fill(
            image, (ox, oy, ox + side, oy + side), fill=lambda px: self._fill_values(px, channels), dtype=image.dtype
        )

    def _pad_crop(
        self, image: NDArray[Any], rx0: float, ry0: float, rx1: float, ry1: float
    ) -> tuple[NDArray[Any], tuple[int, int]]:
        """Crop the region's real pixels and center them in a square ``fill`` canvas."""
        channels = image.shape[0]
        ix0, iy0, ix1, iy1 = self._region_to_pixels(image.shape, rx0, ry0, rx1, ry1)
        region = image[..., iy0:iy1, ix0:ix1]
        region_h, region_w = region.shape[-2], region.shape[-1]
        side = max(region_h, region_w)

        offx, offy = (side - region_w) // 2, (side - region_h) // 2
        crop = np.empty((channels, side, side), dtype=image.dtype)
        crop[:] = self._fill_values(region, channels).reshape(channels, 1, 1)
        crop[..., offy : offy + region_h, offx : offx + region_w] = region
        # Origin maps image coords to canvas coords: image (ix0, iy0) sits at canvas (offx, offy).
        return crop, (ix0 - offx, iy0 - offy)

    def _mask_box(
        self, crop: NDArray[Any], origin: tuple[int, int], x0: float, y0: float, x1: float, y1: float
    ) -> None:
        """Mask the original box (in crop coordinates) to the fill value, in place."""
        ox, oy = origin
        mx0, my0 = max(int(round(x0)) - ox, 0), max(int(round(y0)) - oy, 0)
        mx1, my1 = min(int(round(x1)) - ox, crop.shape[-1]), min(int(round(y1)) - oy, crop.shape[-2])
        if mx1 <= mx0 or my1 <= my0:
            return
        if self._fill == "zero":
            crop[..., my0:my1, mx0:mx1] = 0
            return
        # Mean over the surrounding ring (the kept pixels), not the box being masked.
        mask = np.ones(crop.shape[-2:], dtype=bool)
        mask[my0:my1, mx0:mx1] = False
        ring = crop[..., mask]
        fill = ring.mean(axis=-1) if ring.size else np.zeros(crop.shape[0], dtype=np.float64)
        crop[..., my0:my1, mx0:mx1] = fill.reshape(crop.shape[0], 1, 1).astype(crop.dtype)

    def _fill_values(self, pixels: NDArray[Any], channels: int) -> NDArray[np.float64]:
        """Per-channel fill, computed from the real pixels available for this crop."""
        if self._fill == "zero" or pixels.size == 0:
            return np.zeros(channels, dtype=np.float64)
        return pixels.reshape(channels, -1).mean(axis=1)
