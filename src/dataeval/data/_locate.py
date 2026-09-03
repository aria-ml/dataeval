"""Retrieve what a :class:`~dataeval.types.SourceIndex` names, from the data it was measured over."""

# Empty, as every module under this package is: the names are public at
# ``dataeval.data.SourceItem`` and autoapi skips a module that exports none of its own, so
# they are not also rendered at the private ``dataeval.data._locate`` path.
__all__ = []

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.data._crops import CropPolicy, FillType, RegionType, SquareType
from dataeval.data._tracks import build_tracks
from dataeval.data._view import View
from dataeval.protocols import (
    DatumMetadata,
    MultiobjectTrackingTarget,
    ObjectDetectionTarget,
    VideoFrame,
    _is_protocol_instance,
)
from dataeval.types import FactorLevel, FactorLevelSchema, SourceIndex, Track
from dataeval.types._target import detection_score
from dataeval.utils._array import as_numpy
from dataeval.utils.preprocessing import normalize_image_shape

# Which levels each task's addresses can name, keyed by the task's item level. An address
# at a level absent from its dataset is rejected with this list rather than followed into
# an attribute the target does not have.
_LEVELS_BY_ITEM_LEVEL: dict[FactorLevel, tuple[FactorLevel, ...]] = {
    "unit": ("unit", "instance"),
    "sequence": ("sequence", "unit", "track", "instance"),
}


@dataclass(frozen=True)
class SourceItem:
    """What one :class:`~dataeval.types.SourceIndex` names, retrieved from its dataset.

    Returned by :class:`SourceLocator`; not constructed directly. Every accessor reads
    through the locator, so building one costs nothing and a caller holding a thousand
    addresses pays only for the ones it looks at.

    What an item carries depends on the level its address resolved to:

    ==================  ==========================  ======================================
    Level               ``pixels``                  Also carries
    ==================  ==========================  ======================================
    ``sequence``        — (a video is no raster)    ``stream``
    ``unit`` (image)    the image                   —
    ``unit`` (frame)    the frame's pixels          ``frame``
    ``track``           — (a track spans frames)    ``track``
    ``instance`` (box)  the image or frame it       ``box``, ``label``, ``score``, and on
                        sits in                     video ``frame`` and ``track``
    ==================  ==========================  ======================================

    An accessor a level does not carry **raises** :class:`TypeError` rather than answering
    ``None``, so ``found.box`` is a box wherever it answers at all and a mistake is a
    message rather than a ``None`` propagating into a plot. Branch on :attr:`level` where a
    batch of findings holds more than one kind.

    ``None`` is kept for what is genuinely absent rather than out of place: a detection no
    tracker linked has no :attr:`track`, and a target carrying no confidences has no
    :attr:`score`. Both are real answers about a real detection.

    An instance's `pixels` is the whole image rather than the detection, because the
    surrounding scene is most of what a caller inspecting an outlier wants to see. Use
    :meth:`crop` for the cut-out.

    Attributes
    ----------
    address : SourceIndex
        The address this was retrieved for, exactly as it was passed in — including
        whether it stated a level.
    level : FactorLevel
        The level `address` resolved to against this dataset's task, via
        :meth:`~dataeval.types.SourceIndex.resolve`. Always stated, where `address` may
        not be.
    item_index : int
        Which item of the object the locator was built over. This is `address.item`; it
        is repeated here because it is the one field that means something different
        depending on what that object was — see :attr:`source_item_index`.

    Examples
    --------
    >>> locator = SourceLocator(dataset)
    >>> found = locator[SourceIndex(1, 0)]
    >>> found.level
    'instance'
    >>> found.box.tolist()
    [7.0, 22.0, 20.0, 33.0]
    >>> found.pixels.shape
    (3, 64, 64)
    >>> found.crop().shape
    (3, 11, 13)
    """

    address: SourceIndex
    level: FactorLevel
    item_index: int
    # Compared (by identity, which is what SourceLocator has) so that one address retrieved
    # over two different datasets gives two different items — a batch of findings and the
    # same batch read through a view must not collapse into one another in a set. Excluded
    # from the repr so that printing an item does not print a whole dataset.
    locator: "SourceLocator" = field(repr=False)

    @property
    def key(self) -> int | None:
        """Which row at :attr:`level`, within the item — the address's key.

        ``None`` for an item's own row, which its item names outright. Otherwise the value
        of whichever column names rows at this level: ``target_index`` for a detection,
        ``unit_index`` for a video frame, ``track_id`` for a track.
        """
        return self.address.key

    @property
    def datum(self) -> tuple[Any, Any, DatumMetadata]:
        """The whole source datum this address sits in, as the dataset yields it.

        The escape hatch: whatever this class does not surface for a task, the datum has.

        Returns
        -------
        tuple
            ``(image, target, metadata)`` for an image task, ``(stream, target,
            metadata)`` for a tracking one.
        """
        return self.locator.datum(self.item_index)

    @property
    def datum_metadata(self) -> DatumMetadata:
        """The source datum's own metadata, carrying the protocol-required ``id``."""
        return self.datum[2]

    @property
    def source_item_index(self) -> int:
        """Which item of the *underlying* dataset this is, with view operations undone.

        Equal to :attr:`item_index` unless the locator was built over a
        :class:`~dataeval.data.View`, in which case operations such as
        :class:`~dataeval.data.Shuffle` and :class:`~dataeval.data.ClassFilter` have
        renumbered positions and this is the position before any of them did. Use it to
        trace a finding back to a file on disk; use :attr:`item_index` to index the object
        you measured.

        This is a position in :attr:`SourceLocator.source`, which for stacked views is the
        bottom of the chain — **not** what a single
        :meth:`~dataeval.data.View.resolve_indices` call reports, since that steps one link.
        The two agree for the one-view case, which is almost every case.
        """
        return self.locator.source_item_index(self.item_index)

    @property
    def source_datum(self) -> tuple[Any, Any, DatumMetadata]:
        """The same datum as :attr:`datum`, read with **no view transform applied**.

        :attr:`datum` is what was measured — resized, channel-selected, cropped, whatever
        the view does. This is what sits underneath: the datum at the bottom of the view
        chain, or the locator's `source` where one was named. Equal to :attr:`datum` when
        there is no view.

        Use it to compare a finding against the original — an outlier that is only an
        outlier after a transform is a finding about the transform.

        Returns
        -------
        tuple
            ``(image, target, metadata)``, or ``(stream, target, metadata)`` on video.
        """
        return self.locator.source_datum(self.item_index)

    @property
    def source_pixels(self) -> NDArray[Any]:
        """The item's untransformed pixels as ``(C, H, W)``.

        Raises
        ------
        TypeError
            On video, whose source datum is a stream rather than a raster; reach its frames
            through :attr:`source_datum`.
        """
        return self.locator.source_pixels(self.item_index)

    def at(self, level: FactorLevel) -> "SourceItem":
        """Return the row at `level` that this one sits inside — one step or many, up the graph.

        The way from a finding to its context: a flagged detection to the frame it was
        seen in, to the track it belongs to, to the video the whole thing came from.

        Parameters
        ----------
        level : FactorLevel
            An ancestor of this item's level, or its own level. The levels sit in a
            diamond, not a chain — an ``instance`` has both ``unit`` and ``track`` above
            it — so "ancestor" is by any path.

        Returns
        -------
        SourceItem
            The containing row, addressed canonically at `level`.

        Raises
        ------
        ValueError
            When `level` is not above this one — including a *sibling*, which ``unit`` and
            ``track`` are to each other. Go through ``instance`` for those: a frame and a
            track meet in the detections they share, not in one containing the other.
            Also when the containing row does not exist, which happens in exactly one
            place: a detection no tracker linked belongs to no track.

        See Also
        --------
        within : the other direction, one row to the many inside it.

        Examples
        --------
        >>> locator = SourceLocator(dataset)
        >>> flagged = locator[SourceIndex(1, 0)]
        >>> flagged.at("unit").address
        SourceIndex(1)

        On video the same climb reaches both parents, ``flagged.at("track")`` and
        ``flagged.at("unit")``, and ``flagged.at("sequence")`` reaches the video itself.
        """
        return self.locator[self.locator.address_at(self, level)]

    def within(self, level: FactorLevel) -> "list[SourceItem]":
        """Every row at `level` that sits inside this one, in address order.

        The inverse of :meth:`at`. A track's detections, a frame's detections, a video's
        frames — the fan-out an address cannot express, since one address names one row.

        Parameters
        ----------
        level : FactorLevel
            A descendant of this item's level. Its own level is not a descendant of
            itself; asking for it is a mistake worth naming rather than answering with a
            list of one.

        Returns
        -------
        list of SourceItem
            Possibly empty — a video frame holding no detections is a real row with
            nothing inside it.

        Raises
        ------
        ValueError
            When `level` is not below this one, siblings included.

        Notes
        -----
        A track's rows are all inside its own item: a ``track_id`` is unique within a
        sequence and two videos reusing one id hold two unrelated tracks, which is why
        this never crosses an item boundary and why there is no collection of *datums* to
        return — there is one, and it is :attr:`datum`.

        Examples
        --------
        >>> locator = SourceLocator(dataset)
        >>> scene = locator[SourceIndex(1)]
        >>> [found.address for found in scene.within("instance")]
        [SourceIndex(1, 0), SourceIndex(1, 1), SourceIndex(1, 2)]

        A track's rows are reached the same way — ``track.within("instance")`` — and the
        frames it appears in through those, since ``unit`` is its sibling rather than its
        descendant.
        """
        return [self.locator[address] for address in self.locator.addresses_within(self, level)]

    @property
    def pixels(self) -> NDArray[Any]:
        """The pixels this address names, or the pixels it sits in, as ``(C, H, W)``.

        Raises
        ------
        TypeError
            At the two levels that name no single raster — a ``sequence``, which is a whole
            video, and a ``track``, which spans frames. Reach for :attr:`stream` and
            :attr:`track` there.
        """
        return self.locator.pixels(self)

    @property
    def stream(self) -> Any:
        """The video stream this address sits in.

        Raises
        ------
        TypeError
            On an image task, whose items are rasters rather than streams; their pixels are
            :attr:`pixels`.
        """
        if self.locator.item_level == "unit":
            raise TypeError(
                f"{self.address!r} is on an image task, whose items are images rather than video "
                "streams. Read its pixels with `pixels`.",
            )
        return self.datum[0]

    @property
    def frame(self) -> VideoFrame:
        """The decoded video frame this address names or sits in.

        Carried at ``unit`` level on video, and on a video ``instance`` — a detection was
        seen in exactly one frame, and this is it.

        Raises
        ------
        TypeError
            On an image task, where the item *is* the frame and :attr:`pixels` is what to
            read, and at ``sequence`` and ``track`` level, which span frames.

        Notes
        -----
        Located by scanning the stream for the frame whose ``frame_index`` matches, since
        :obj:`~dataeval.protocols.VideoStream` is only an iterable and a frame's index is
        not obliged to equal its position. A stream that cannot be re-iterated — a bare
        generator — can therefore be read only once.
        """
        return self.locator.frame(self)

    @property
    def box(self) -> NDArray[np.float64]:
        """The detection's box as ``[x0, y0, x1, y1]``.

        Raises
        ------
        TypeError
            Above ``instance`` level, where an address names no single box.
        """
        return self.locator.detection(self)[0]

    @property
    def label(self) -> int:
        """The detection's integer class label.

        Raises
        ------
        TypeError
            Above ``instance`` level, where an address names no single detection.
        """
        return self.locator.detection(self)[1]

    @property
    def score(self) -> float | None:
        """The detection's confidence, or ``None`` where its target carries none.

        A ground-truth target scores ``1.0``. Where a target carries per-class scores
        rather than one per box, this is the score of the box's own class — the same
        number the metadata frame's ``score`` column holds for this detection, read the
        same way, so retrieving a row and reading it agree.

        Raises
        ------
        TypeError
            Above ``instance`` level, where an address names no single detection.
        """
        return self.locator.detection(self)[2]

    @property
    def track(self) -> Track | None:
        """The track this address names, or the one its detection belongs to.

        ``None`` for a detection no tracker linked — those carry ``track_id`` ``-1`` and
        belong to no track, which is also why :class:`~dataeval.Metadata` gives them no
        ``track`` row. That is a real answer about a real detection, where a level that
        names no track at all raises.

        Raises
        ------
        TypeError
            On an image task, which has no tracks, and at ``sequence`` or ``unit`` level,
            neither of which names one.
        """
        return self.locator.track(self)

    def crop(
        self,
        *,
        policy: "CropPolicy | None" = None,
        region: RegionType = "object",
        padding: float = 0.0,
        square: SquareType = "off",
        fill: FillType = "mean",
    ) -> NDArray[Any]:
        """Cut this detection out of the pixels it sits in.

        Shares its implementation with :class:`~dataeval.data.DetectionCrops`, so a crop
        eyeballed here and a crop an embedding was computed from are the same pixels under
        the same policy — but **the defaults here are not that view's defaults**. This one
        cuts the box at its own aspect ratio, which is what a crop being looked at wants;
        :class:`.DetectionCrops` squares by ``"expand"``, which is what a crop feeding a
        model wants. To reproduce a particular view's crops exactly, hand over its
        :attr:`~dataeval.data.DetectionCrops.policy` rather than restating four arguments.

        Parameters
        ----------
        policy : CropPolicy or None, default None
            A policy to cut with, as :attr:`.DetectionCrops.policy` reports one. Given, it
            supplies all four parameters below and passing any of them as well is an error
            — two policies for one crop have no resolution between them.
        region : {"object", "context", "surround"}, default "object"
            Which pixels to keep. ``"object"`` is the box, ``"context"`` the box widened
            by `padding`, and ``"surround"`` the widened region with the box masked out —
            what the object was seen against.
        padding : float, default 0.0
            Fraction of the box's width and height added on each side.
        square : {"off", "expand", "pad"}, default "off"
            Whether to square the crop. Defaults to ``"off"`` because a crop being looked
            at wants the detection's own aspect ratio; :class:`.DetectionCrops`, whose
            crops feed a model, defaults to ``"expand"``.
        fill : {"mean", "zero"}, default "mean"
            What to put where there are no real pixels, and what to mask with under
            ``region="surround"``.

        Returns
        -------
        NDArray
            The crop, as ``(C, H, W)``.

        Raises
        ------
        TypeError
            When this address is not at ``instance`` level, and so names no box, or when
            `policy` is given alongside any of the four parameters it already supplies.
        ValueError
            When the parameters do not describe a crop, or when the box lies wholly outside
            the raster and there is nothing to cut.
        """
        if self.level != "instance":
            raise TypeError(
                f"{self.address!r} is at {self.level!r} level and names no box, so there is nothing to "
                "crop. Only an instance-level address does.",
            )
        if policy is None:
            policy = CropPolicy(region, padding, square, fill)
        elif (region, padding, square, fill) != ("object", 0.0, "off", "mean"):
            raise TypeError(
                "pass policy= or the individual crop parameters, not both; a policy already supplies "
                "region, padding, square and fill.",
            )
        crop = policy.crop(self.pixels, self.box)
        # A box wholly off the raster clips to nothing. Caught here rather than handed back,
        # because an empty array reaches a plot as "Invalid shape (3, 0, 0)" — a message
        # about matplotlib rather than about the detection that produced it.
        if crop.size == 0:
            raise ValueError(
                f"{self.address!r} names the box {self.box.tolist()}, which lies outside its "
                f"{self.pixels.shape[-1]}x{self.pixels.shape[-2]} raster, so the crop is empty. A box "
                "in another coordinate space — source coordinates read through a resizing view, or a "
                "tracker's predicted box — is the usual cause.",
            )
        return crop


class SourceLocator:
    """Retrieve the item, frame, track or detection a :class:`~dataeval.types.SourceIndex` names.

    An address is the coordinate an evaluator hands back — :class:`~dataeval.quality.Outliers`
    and :class:`~dataeval.quality.Duplicates` key their findings on one — and this is the
    way from that coordinate back to the data it was measured over.

    **Give it the same object you computed the statistics over.** An address's ``item`` is
    a position in whatever :func:`~dataeval.core.compute_stats` walked, so a locator over
    a different object resolves it to different data. Where that object is a
    :class:`View`, :attr:`SourceItem.source_item_index` reports the position in the
    dataset underneath.

    Parameters
    ----------
    dataset : Dataset or View
        The dataset or view the addresses were produced over. Its task is read from the
        first target it yields, which decides what levels its addresses may name: an
        image task has ``unit`` and ``instance``, a tracking one has all four.
    source : Dataset or None, default None
        The untransformed data behind `dataset`, reached through
        :attr:`SourceItem.source_datum`. Defaults to the bottom of `dataset`'s view chain
        — :attr:`~dataeval.data.View.root` — which is the right answer whenever the view
        was built over the original dataset.

        A named `source` is read at :attr:`SourceItem.source_item_index`, the position the
        view chain resolves to, so it has to be **numbered like the bottom of that chain**:
        the original of a re-ordered or filtered view, not some other dataset that happens
        to hold the same pictures. A dataset that is not a view resolves to the index given,
        so a `source` named there is read at the same position `dataset` is.

    See Also
    --------
    dataeval.types.SourceIndex : The address itself, and what its three fields name.
    dataeval.data.View.resolve_indices : Item indices alone, with view operations undone.

    Notes
    -----
    One datum is held at a time, so addresses *read* in item order — which is the order
    :func:`~dataeval.core.compute_stats` emits and evaluators preserve — read each item
    once, while a shuffled batch re-reads. Where a batch is not already in that order,
    ``sorted(addresses, key=lambda address: address.sort_key)`` puts it there; the frames
    and tracks of one video are held alongside its datum, so the saving compounds.

    Examples
    --------
    Bind a dataset, then follow the addresses an evaluator handed back — the keys of
    :attr:`~dataeval.quality.OutliersOutput.outliers` are addresses of exactly this shape:

    >>> locator = SourceLocator(dataset)
    >>> locator.item_level
    'unit'
    >>> locator.levels
    ('unit', 'instance')
    >>> found = locator[SourceIndex(1, 0)]
    >>> found.level, found.item_index
    ('instance', 1)
    """

    def __init__(self, dataset: Any, *, source: Any | None = None) -> None:
        self._dataset = dataset
        self._view = dataset if isinstance(dataset, View) else None
        # A view's root, not its `source`: `resolve_indices` steps one link, and a chain of
        # views has to be walked the whole way down for "no transforms applied" to mean it.
        self._source = source if source is not None else (self._view.root if self._view else dataset)
        self._item_level: FactorLevel | None = None
        self._detections: bool = True
        self._schema: FactorLevelSchema | None = None
        # Single slot, as DetectionCrops keeps: consecutive addresses into one item are the
        # common case, and holding more than one decoded video would be a memory policy
        # this class has no business setting.
        self._datum_index: int | None = None
        self._datum: tuple[Any, Any, DatumMetadata] | None = None
        self._frames_index: int | None = None
        self._frames: Any | None = None
        self._frame_iter: Any | None = None
        self._raster_key: tuple[int, int | None] | None = None
        self._raster_value: NDArray[Any] | None = None
        self._tracks_index: int | None = None
        self._tracks: dict[int, Track] | None = None
        self._instances_index: int | None = None
        self._instances: list[tuple[int, int, int]] | None = None
        self._numbers_index: int | None = None
        self._numbers: list[int] | None = None
        self._source_index: int | None = None
        self._source_datum: tuple[Any, Any, DatumMetadata] | None = None

    def __repr__(self) -> str:
        return f"SourceLocator(dataset={self._dataset!r})"

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def item_level(self) -> FactorLevel:
        """The level one item of this dataset sits at, read from its first target.

        ``unit`` for an image task, ``sequence`` for tracking. What an unkeyed address
        resolves to, and what :attr:`dataeval.Metadata.item_level` reports for the same data.
        """
        if self._item_level is None:
            self._item_level = self._read_item_level()
        return self._item_level

    @property
    def levels(self) -> tuple[FactorLevel, ...]:
        """Which levels this dataset's addresses may name, coarsest first."""
        return _LEVELS_BY_ITEM_LEVEL[self.item_level]

    @property
    def source(self) -> Any:
        """The untransformed data behind this locator's dataset."""
        return self._source

    @property
    def schema(self) -> FactorLevelSchema:
        """How this dataset's levels sit relative to each other, as a graph.

        The same graph :class:`~dataeval.Metadata` reports for the same data, which is
        what makes :meth:`~dataeval.data.SourceItem.at` and
        :meth:`~dataeval.data.SourceItem.within` agree with which factors propagate where.
        """
        if self._schema is None:
            self._schema = FactorLevelSchema.of(*self.levels)
        return self._schema

    def _read_item_level(self) -> FactorLevel:
        """Infer the task's item level from the first datum's target.

        An empty dataset is answered ``unit``, the level of every task but tracking:
        nothing can be retrieved from it either way, and the alternative is refusing to
        construct over a dataset a caller may only mean to check the length of.

        The same inspection records whether the target carries boxes at all, so that a
        classification dataset — which has ``instance`` rows but no detections — refuses an
        instance accessor with a message rather than walking into a missing attribute.
        """
        if len(self._dataset) == 0:
            self._detections = False
            return "unit"
        datum = self.datum(0)
        if not isinstance(datum, tuple) or len(datum) != 3:
            raise TypeError(
                f"{type(self._dataset).__name__} yields {type(datum).__name__} rather than a MAITE "
                "(image, target, metadata) datum, so there is no target to read a task from and no "
                "metadata to trace a finding by. `compute_stats` accepts a bare iterable of arrays; "
                "retrieving what its addresses name needs the dataset those arrays came from.",
            )
        target = datum[1]
        if _is_protocol_instance(target, MultiobjectTrackingTarget):
            self._detections = True
            return "sequence"
        self._detections = _is_protocol_instance(target, ObjectDetectionTarget)
        return "unit"

    def _reject_detectionless(self, item: SourceItem) -> None:
        """Refuse a detection on a task whose targets carry no boxes."""
        # Reading the level is what forces the task inspection that records `_detections`.
        _ = self.item_level
        if not self._detections:
            raise TypeError(
                f"{item.address!r} reaches instance-level data, but this dataset's targets carry no boxes "
                "— it is a classification task, whose label rows have no box, score or track to retrieve. "
                "Its images are reachable at 'unit' level.",
            )

    def __getitem__(self, address: SourceIndex | int | np.integer[Any]) -> SourceItem:
        """Retrieve what one address names.

        Parameters
        ----------
        address : SourceIndex or int
            The address to retrieve. A plain integer — including a NumPy one, which is what
            indexing an array of positions yields — is an item index, which is what an
            evaluator's findings are keyed on where nothing below the item was measured.

        Returns
        -------
        SourceItem
            A handle on the located data. Nothing is read until one of its accessors is.

        Raises
        ------
        IndexError
            When `address` names an item this dataset does not have.
        ValueError
            When `address` states a level this dataset's task does not have, or carries a
            key at the item level, whose one row per item a key cannot pick out.
        """
        if isinstance(address, (int, np.integer)):
            address = SourceIndex(int(address))
        if not 0 <= address.item < len(self._dataset):
            raise IndexError(
                f"{address!r} names item {address.item}, but the dataset has {len(self._dataset)} items.",
            )
        level = address.resolve(self.item_level, "instance")
        if level not in self.levels:
            raise ValueError(
                f"{address!r} names {level!r}-level data, but this dataset's levels are "
                f"{', '.join(repr(name) for name in self.levels)}. An address at a level a dataset does "
                "not have is a statistic computed over different data.",
            )
        if level == self.item_level and address.key is not None:
            raise ValueError(
                f"{address!r} carries a key at {level!r} level, which is this dataset's item level — an "
                "item holds exactly one such row and names it outright, so there is nothing for a key to "
                f"pick out. Address it as SourceIndex({address.item}).",
            )
        return SourceItem(address, level, address.item, self)

    def gather(self, addresses: Iterable[SourceIndex | int | np.integer[Any]]) -> list[SourceItem]:
        """Retrieve a batch of addresses as handles, in the order given.

        The batch form of ``locator[address]``: it takes the mapping an evaluator returns
        as well as a plain sequence, and turns each key — address or bare integer — into a
        handle. Nothing is read here, so a thousand findings cost a thousand small objects
        and no dataset access at all.

        Parameters
        ----------
        addresses : iterable of SourceIndex or int
            The addresses to retrieve. Duplicates are kept, as
            :meth:`~dataeval.data.View.resolve_indices` keeps them.

        Returns
        -------
        list of SourceItem
            One per address, in the order given. Nothing is read until an accessor is.

        Notes
        -----
        One datum is held at a time, so *reading* the batch is cheapest in address order —
        ``sorted(found, key=lambda item: item.address.sort_key)`` — which for video is the
        difference between one decode per address and one per item. The list itself is
        returned in the order given, since that is the order a caller's own findings are in.

        Examples
        --------
        >>> locator = SourceLocator(dataset)
        >>> found = locator.gather([SourceIndex(1, 0), SourceIndex(2)])
        >>> [item.level for item in found]
        ['instance', 'unit']
        """
        return [self[address] for address in addresses]

    def datum(self, item_index: int) -> tuple[Any, Any, DatumMetadata]:
        """Read one datum, holding it so that the next address into the same item is free.

        Parameters
        ----------
        item_index : int
            A position in this locator's dataset.

        Returns
        -------
        tuple
            ``(image, target, metadata)`` for an image task, ``(stream, target, metadata)``
            for a tracking one — whatever the dataset yields, unchanged.

        Notes
        -----
        One datum is held at a time. Reading a different item displaces it, along with the
        frames, tracks and instance table derived from it.
        """
        if self._datum_index == item_index and self._datum is not None:
            return self._datum
        datum = self._dataset[item_index]
        self._datum_index, self._datum = item_index, datum
        # The frames and tracks held alongside belong to the item just displaced.
        self._frames_index, self._frames, self._frame_iter = None, None, None
        self._raster_key, self._raster_value = None, None
        self._tracks_index, self._tracks = None, None
        self._instances_index, self._instances = None, None
        self._numbers_index, self._numbers = None, None
        return datum

    def source_item_index(self, item_index: int) -> int:
        """Undo every view's renumbering, all the way down, or answer with the index given.

        Walks the whole chain rather than calling
        :meth:`~dataeval.data.View.resolve_indices`, which steps one link: a view over a
        view renumbers twice, and only the bottom index names a row of :attr:`source`.

        Parameters
        ----------
        item_index : int
            A position in this locator's dataset.

        Returns
        -------
        int
            The position in :attr:`source` it resolves to, equal to `item_index` where the
            dataset is not a view.

        Raises
        ------
        IndexError
            When `item_index` is not a position this dataset has. Checked at every link,
            as :meth:`~dataeval.data.View.resolve_indices` checks its one, so a negative
            index reports the mistake rather than resolving from the end of a selection.
        """
        current: Any = self._dataset
        index = item_index
        while isinstance(current, View):
            if not 0 <= index < len(current.selection):
                raise IndexError(
                    f"Item {item_index} does not resolve to a row of the source: {index} is out of range "
                    f"for a view of {len(current.selection)} items.",
                )
            index = current.selection[index]
            current = current.source
        return index

    def source_datum(self, item_index: int) -> tuple[Any, Any, DatumMetadata]:
        """Read one datum from :attr:`source`, with no view transform applied.

        Parameters
        ----------
        item_index : int
            A position in this locator's dataset, not in :attr:`source`; it is resolved
            through :meth:`source_item_index`.

        Returns
        -------
        tuple
            ``(image, target, metadata)``, or ``(stream, target, metadata)`` on video.
            Identical to :meth:`datum` where there is no view.

        Raises
        ------
        IndexError
            When `item_index` resolves to a position :attr:`source` does not have, which
            means a named `source` is numbered unlike the bottom of the view chain.
        """
        if self._source is self._dataset:
            return self.datum(item_index)
        index = self.source_item_index(item_index)
        if self._source_index == index and self._source_datum is not None:
            return self._source_datum
        if not 0 <= index < len(self._source):
            raise IndexError(
                f"Item {item_index} resolves to position {index}, which the source does not have — it "
                f"holds {len(self._source)} items. A named `source` has to be numbered like the bottom "
                "of this locator's view chain.",
            )
        datum = self._source[index]
        self._source_index, self._source_datum = index, datum
        return datum

    def source_pixels(self, item_index: int) -> NDArray[Any]:
        """Return the untransformed raster of one item, refusing a video, which has none.

        Parameters
        ----------
        item_index : int
            A position in this locator's dataset.

        Returns
        -------
        NDArray
            The item's pixels as they sit in :attr:`source`, normalized to ``(C, H, W)``.

        Raises
        ------
        TypeError
            On video, whose source datum is a stream rather than a raster; reach its
            frames through :meth:`source_datum`.
        IndexError
            When `item_index` resolves to a position :attr:`source` does not have.
        """
        if self.item_level != "unit":
            raise TypeError(
                f"Item {item_index} is a video, whose source datum is a stream rather than a raster. "
                "Reach its frames through `source_datum`.",
            )
        return _readonly(normalize_image_shape(as_numpy(self.source_datum(item_index)[0])))

    def address_at(self, item: SourceItem, level: FactorLevel) -> SourceIndex:
        """Address the row at `level` that `item` sits inside. See :meth:`SourceItem.at`.

        Parameters
        ----------
        item : SourceItem
            The row to climb from.
        level : FactorLevel
            An ancestor of `item`'s level, or its own.

        Returns
        -------
        SourceIndex
            The containing row's address, canonically spelled at `level`.

        Raises
        ------
        ValueError
            When `level` is not a level of this dataset, is not above `item`'s level
            (siblings included), or names a row that does not exist — a detection no
            tracker linked belongs to no track.
        """
        if level == item.level:
            return self._canonical(item.item_index, item.address.key, level)
        self._reject_unrelated(item, level, above=True)
        if level == self.item_level:
            return SourceIndex(item.item_index)
        # Only an instance has levels between it and its item, and it has two of them.
        return self._instance_ancestor(item, level)

    def _canonical(self, item_index: int, key: int | None, level: FactorLevel) -> SourceIndex:
        """Spell one row the way a producer spells it: a level stated only where it must be.

        Two spellings of one row are not ``==`` and do not hash alike, so an address handed
        back from :meth:`address_at` has to be the same one an evaluator would have emitted
        — otherwise a finding and the row it climbs to land in different buckets of a set.
        """
        if level == self.item_level:
            return SourceIndex(item_index)
        if level == "instance":
            return SourceIndex(item_index, key)
        return SourceIndex(item_index, key, level)

    def _instance_ancestor(self, item: SourceItem, level: FactorLevel) -> SourceIndex:
        """Address the frame or the track one video detection sits in."""
        if level == "unit":
            position, _ = self._instance_position(item)
            return SourceIndex(item.item_index, self._frame_number_at(item.item_index, position), "unit")
        track = self.track(item)
        if track is None:
            raise ValueError(
                f"{item.address!r} is a detection no tracker linked — it carries track_id -1 and "
                "belongs to no track, which is also why Metadata gives it no track row.",
            )
        return SourceIndex(item.item_index, int(track.track_id), "track")

    def addresses_within(self, item: SourceItem, level: FactorLevel) -> list[SourceIndex]:
        """Address every row at `level` inside `item`. See :meth:`SourceItem.within`.

        Parameters
        ----------
        item : SourceItem
            The row to descend from.
        level : FactorLevel
            A descendant of `item`'s level.

        Returns
        -------
        list of SourceIndex
            One address per row, in address order. Possibly empty.

        Raises
        ------
        ValueError
            When `level` is not a level of this dataset, or does not sit below `item`'s
            level — its own level and its siblings included.
        TypeError
            When the descent reaches detections on a task whose targets carry no boxes.
        IndexError
            When `item` names a frame or track this dataset's item does not hold.
        """
        self._reject_unrelated(item, level, above=False)
        if self.item_level == "unit":
            # An image task's only descent is item -> its detections.
            self._reject_detectionless(item)
            count = len(as_numpy(self.datum(item.item_index)[1].boxes).reshape(-1, 4))
            return [SourceIndex(item.item_index, key) for key in range(count)]
        if item.level == "sequence":
            return self._sequence_holds(item, level)
        # From here `level` can only be `instance`: a frame and a track have nothing else
        # below them, and neither is above the other.
        return self._instances_under(item)

    def _sequence_holds(self, item: SourceItem, level: FactorLevel) -> list[SourceIndex]:
        """Address every frame, track or detection of one video item."""
        if level == "unit":
            # Sorted, not stream order: both docstrings promise address order, the track and
            # instance branches give it, and a stream is allowed to number its frames out of
            # order — which is the one case where the two disagree.
            return [
                SourceIndex(item.item_index, number, "unit")
                for number in sorted(self._frame_numbers_of(item.item_index))
            ]
        if level == "track":
            return [
                SourceIndex(item.item_index, track_id, "track")
                for track_id in sorted(self._tracks_for(item.item_index))
            ]
        return [SourceIndex(item.item_index, key) for key in range(len(self._instances_of(item.item_index)))]

    def _instances_under(self, item: SourceItem) -> list[SourceIndex]:
        """Address the detections held by one frame or belonging to one track."""
        instances = self._instances_of(item.item_index)
        if item.level == "unit":
            wanted = self._position_of_frame(item.item_index, _keyed(item))
            keys = [key for key, (position, _, _) in enumerate(instances) if position == wanted]
        else:
            wanted = _keyed(item)
            self._require_tracks(item, wanted)
            keys = [key for key, (_, _, track_id) in enumerate(instances) if track_id == wanted]
        return [SourceIndex(item.item_index, key) for key in keys]

    def _reject_unrelated(self, item: SourceItem, level: FactorLevel, *, above: bool) -> None:
        """Refuse a level this dataset lacks, or one not on the asked-for side of `item`."""
        if level not in self.levels:
            raise ValueError(
                f"{level!r} is not a level of this dataset, whose levels are "
                f"{', '.join(repr(name) for name in self.levels)}.",
            )
        related = self.schema.is_ancestor(level, item.level) if above else self.schema.is_ancestor(item.level, level)
        if related:
            return
        side, contains = ("above", f"no single {level} row contains") if above else ("below", "nothing at it is inside")
        raise ValueError(
            f"{level!r} does not sit {side} {item.level!r}, so {contains} {item.address!r}. "
            f"{self._route(item, level, above=above)}",
        )

    def _route(self, item: SourceItem, level: FactorLevel, *, above: bool) -> str:
        """Say where to go instead, naming the sibling case for what it is."""
        if {level, item.level} == {"unit", "track"}:
            return (
                "A frame and a track are siblings — each holds detections the other also holds, and "
                "neither contains the other. Reach one from the other through 'instance'."
                if above
                else "A frame and a track are siblings — neither sits inside the other. Ask each for its "
                "'instance' rows and intersect those."
            )
        names = self.schema.ancestors(item.level) if above else self.schema.descendants(item.level)
        return f"Levels {'above' if above else 'below'} {item.level!r} are {', '.join(map(repr, names)) or 'none'}."

    def pixels(self, item: SourceItem) -> NDArray[Any]:
        """Return the raster an item names or sits in, normalized to ``(C, H, W)``.

        Parameters
        ----------
        item : SourceItem
            The row whose pixels to read.

        Returns
        -------
        NDArray
            The image on an image task, and the frame the row names or sits in on video.

        Raises
        ------
        TypeError
            At ``sequence`` and ``track`` level, neither of which names one raster.
        IndexError
            When the address names a frame the stream does not hold.
        """
        if item.level in ("sequence", "track"):
            spans = "a whole video" if item.level == "sequence" else "frames"
            reach = "stream" if item.level == "sequence" else "track"
            raise TypeError(
                f"{item.address!r} is at {item.level!r} level and names no single raster — it is "
                f"{spans}. Reach it through `{reach}`, or descend to a level that has pixels with "
                "`within()`.",
            )
        if self.item_level == "unit":
            return self._raster(item.item_index, None, lambda: self.datum(item.item_index)[0])
        # ``getattr``, because a stream element that is not a VideoFrame *is* its raster:
        # dispatch duck-types the target, so a stream of bare arrays is a real input and
        # the array is the only pixels it has.
        position = self._frame_position(item)
        return self._raster(item.item_index, position, lambda: getattr(self.frame(item), "pixels", self.frame(item)))

    def _raster(self, item_index: int, position: int | None, read: "Callable[[], Any]") -> NDArray[Any]:
        """Normalize one raster to ``(C, H, W)``, holding the result for repeated reads.

        `as_numpy` on a device tensor is a copy back to the host, and a caller cropping
        several detections out of one image asks for that image once per crop, so the
        normalized array is held exactly as :class:`~dataeval.data.DetectionCrops` holds
        the image it cuts from.
        """
        key = (item_index, position)
        if self._raster_key == key and self._raster_value is not None:
            return self._raster_value
        value = _readonly(normalize_image_shape(as_numpy(read())))
        self._raster_key, self._raster_value = key, value
        return value

    def _frame_position(self, item: SourceItem) -> int:
        """Which position in the stream the frame an item names or sits in occupies."""
        if item.level == "unit":
            return self._position_of_frame(item.item_index, _keyed(item))
        return self._instance_position(item)[0]

    def frame(self, item: SourceItem) -> VideoFrame:
        """Return the frame an item names or sits in, refusing a level that names none.

        Parameters
        ----------
        item : SourceItem
            A ``unit``- or ``instance``-level row of a tracking dataset.

        Returns
        -------
        VideoFrame
            The frame, as the stream yields it.

        Raises
        ------
        TypeError
            On an image task, where the item *is* the frame, and at ``sequence`` and
            ``track`` level, which span frames rather than naming one.
        IndexError
            When the address names a frame number or position the stream does not hold.
        """
        if self.item_level == "unit":
            raise TypeError(
                f"{item.address!r} is on an image task, where the item is the frame. Read its pixels with `pixels`.",
            )
        if item.level in ("sequence", "track"):
            raise TypeError(
                f"{item.address!r} is at {item.level!r} level, which spans frames rather than naming "
                "one. Use `within('unit')` for a sequence, or `within('instance')` then `at('unit')` "
                "for a track.",
            )
        return self._frame_at(item.item_index, self._frame_position(item))

    def track(self, item: SourceItem) -> Track | None:
        """Return the track an item names, or the one its detection belongs to.

        Parameters
        ----------
        item : SourceItem
            A ``track``- or ``instance``-level row of a tracking dataset.

        Returns
        -------
        Track or None
            ``None`` only for a detection no tracker linked, which carries ``track_id``
            ``-1`` and belongs to no track — the same rule
            :func:`~dataeval.data.build_tracks` applies.

        Raises
        ------
        TypeError
            On an image task, and at ``sequence`` or ``unit`` level, none of which names
            a track.
        IndexError
            When the address names a track id this item does not hold.
        """
        self._reject_trackless(item)
        if item.level == "track":
            # A track address names a row outright, so a sentinel id names a row that is not
            # there — `within('instance')` refuses the same address, and Metadata's instance
            # rows carry -1, so a caller building an address from that column has to be told.
            track_id = _keyed(item)
            return self._require_tracks(item, track_id)[track_id]
        # -1 is the marker for a detection no tracker linked. It belongs to no track, and
        # gathering those under the sentinel would produce a bag of unrelated objects —
        # the same call build_tracks makes.
        track_id = self._instance_row(item)[2]
        if track_id < 0:
            return None
        return self._require_tracks(item, track_id)[track_id]

    def _require_tracks(self, item: SourceItem, track_id: int) -> dict[int, Track]:
        """Return one item's tracks, refusing an id it does not hold."""
        tracks = self._tracks_for(item.item_index)
        if track_id not in tracks:
            raise IndexError(
                f"{item.address!r} names track {track_id}, but item {item.item_index} has tracks {sorted(tracks)}.",
            )
        return tracks

    def _reject_trackless(self, item: SourceItem) -> None:
        """Refuse to name a track for a level, or a task, that has none."""
        if self.item_level == "unit":
            raise TypeError(f"{item.address!r} is on an image task, which has no tracks.")
        if item.level in ("sequence", "unit"):
            raise TypeError(
                f"{item.address!r} is at {item.level!r} level and names no track. Use `within('track')` "
                "for a sequence's tracks, or reach a frame's through its detections.",
            )

    def detection(self, item: SourceItem) -> tuple[NDArray[np.float64], int, float | None]:
        """Return the box, label and score an instance-level address names.

        Parameters
        ----------
        item : SourceItem
            An ``instance``-level row.

        Returns
        -------
        tuple
            ``(box, label, score)`` — the box as ``[x0, y0, x1, y1]``, the label as an
            integer, and the score as a float or ``None`` where the target carries none.

        Raises
        ------
        TypeError
            Above ``instance`` level, and on a task whose targets carry no boxes.
        IndexError
            When the address names a detection the target does not hold.
        """
        if item.level != "instance":
            raise TypeError(
                f"{item.address!r} is at {item.level!r} level and names no single detection. Use "
                "`within('instance')` for the detections inside it.",
            )
        self._reject_detectionless(item)
        target = self.datum(item.item_index)[1]
        if self.item_level == "sequence":
            frame_position, within = self._instance_position(item)
            target = target.frame_tracks[frame_position]
            index = within
        else:
            index = _keyed(item)
        boxes = as_numpy(target.boxes).reshape(-1, 4).astype(np.float64)
        labels = as_numpy(target.labels).reshape(-1)
        if not 0 <= index < len(boxes):
            raise IndexError(
                f"{item.address!r} names detection {index}, but there are {len(boxes)} to name.",
            )
        return boxes[index], int(labels[index]), detection_score(target, index, int(labels[index]))

    def _tracks_for(self, item_index: int) -> dict[int, Track]:
        """Build (and hold) one sequence's tracks, keyed as a track address keys them."""
        if self._tracks_index == item_index and self._tracks is not None:
            return self._tracks
        tracks = dict(build_tracks(self.datum(item_index)[1]))
        self._tracks_index, self._tracks = item_index, tracks
        return tracks

    def _instances_of(self, item_index: int) -> list[tuple[int, int, int]]:
        """One row per detection of a video item: (frame position, index in frame, track id).

        Indexed by ``target_index``, which counts detections across the whole sequence in
        frame order — how :class:`~dataeval.Metadata` numbers instance rows on every task.
        ``instance_index``, which restarts each frame, names several rows on a video and is
        why it is not the key; this table is what converts between the two.

        Built in one walk over ``frame_tracks`` and held for the item, since every descent
        into a video's detections needs it and none of them needs pixels. A frame holds as
        many detections as it has labels, which is the count the structuring walk behind
        :class:`~dataeval.Metadata` uses; reading it off ``track_ids`` instead would number
        the rows differently wherever the two disagree.
        """
        if self._instances_index == item_index and self._instances is not None:
            return self._instances
        rows: list[tuple[int, int, int]] = []
        for frame_position, frame_target in enumerate(self.datum(item_index)[1].frame_tracks):
            for within, track_id in enumerate(_track_ids_of(frame_target)):
                rows.append((frame_position, within, track_id))
        self._instances_index, self._instances = item_index, rows
        return rows

    def _instance_row(self, item: SourceItem) -> tuple[int, int, int]:
        """One video detection's (frame position, index within frame, track id), checked."""
        target_index = _keyed(item)
        rows = self._instances_of(item.item_index)
        if not 0 <= target_index < len(rows):
            raise IndexError(
                f"{item.address!r} names detection {target_index}, but item {item.item_index} holds "
                f"{len(rows)} across all of its frames.",
            )
        return rows[target_index]

    def _instance_position(self, item: SourceItem) -> tuple[int, int]:
        """Where a video item's ``target_index`` sits: (frame position, index within frame)."""
        frame_position, within, _ = self._instance_row(item)
        return frame_position, within

    def _frames_of(self, item_index: int) -> Any:
        """Return one item's frames, indexing the stream where it can be indexed.

        A stream that supports ``[]`` is used where it lies, so nothing is copied and any
        position costs one lookup. A stream that is only iterable — which is all MAITE
        requires, and what a lazy decoder is — is walked *forward only*, keeping the
        frames it passes in :attr:`_frames`. Walking it more than once is either expensive
        (a decoder re-reads the file) or wrong (a one-shot iterator resumes where it left
        off, so the second walk answers with a different frame), and both are worse than
        holding the frames of the one item already held.
        """
        if self._frames_index == item_index and self._frames is not None:
            return self._frames
        stream = self.datum(item_index)[0]
        indexable = hasattr(stream, "__getitem__") and hasattr(stream, "__len__")
        self._frame_iter = None if indexable else iter(stream)
        held: Any = stream if indexable else []
        self._frames_index, self._frames = item_index, held
        return held

    def _walk_frames(self, item_index: int, upto: int | None) -> Any:
        """Advance a non-indexable stream until `upto` frames are held, or to its end."""
        frames = self._frames_of(item_index)
        if self._frame_iter is None:
            return frames
        if upto is None:
            frames.extend(self._frame_iter)
            self._frame_iter = None
            return frames
        frames.extend(islice(self._frame_iter, max(upto - len(frames), 0)))
        if len(frames) < upto:
            # Short of what was asked for means the stream ended; nothing is left to walk.
            self._frame_iter = None
        return frames

    def _frame_numbers_of(self, item_index: int) -> list[int]:
        """Return every frame number in one item's stream, in stream order.

        Held per item, since every descent into a video's frames asks for them again.
        """
        if self._numbers_index == item_index and self._numbers is not None:
            return self._numbers
        numbers = [_frame_number(frame, position) for position, frame in enumerate(self._walk_frames(item_index, None))]
        self._numbers_index, self._numbers = item_index, numbers
        return numbers

    def _frame_number_at(self, item_index: int, position: int) -> int:
        """Return the number carried by the frame at `position` in the stream.

        Bounds-checked, because ``frame_tracks`` and the stream are two independent
        sequences: a target declaring more frames than its stream yields is a dataset bug,
        which the structuring walk behind :class:`~dataeval.Metadata` raises on rather than
        absorbing, and which would otherwise surface here as a bare ``IndexError``.
        """
        numbers = self._frame_numbers_of(item_index)
        if not 0 <= position < len(numbers):
            raise IndexError(
                f"Item {item_index} declares a detection in frame {position}, but its stream yielded only "
                f"{len(numbers)} frame(s). `frame_tracks` must hold exactly one target per frame.",
            )
        return numbers[position]

    def _position_of_frame(self, item_index: int, frame_number: int) -> int:
        """Return which position in the stream carries `frame_number`.

        Looked up rather than assumed: a frame's number is its position in the yielded
        stream for a conforming one, but a stream is only obliged to be iterable and
        nothing here relies on the two agreeing.
        """
        try:
            return self._frame_numbers_of(item_index).index(frame_number)
        except ValueError:
            raise IndexError(f"Item {item_index} has no frame numbered {frame_number}.") from None

    def _frame_at(self, item_index: int, position: int) -> VideoFrame:
        """Return the frame at `position` in the stream, which is what ``frame_tracks`` indexes."""
        frames = self._frames_of(item_index) if position < 0 else self._walk_frames(item_index, position + 1)
        if not 0 <= position < len(frames):
            raise IndexError(f"Item {item_index} has no frame at position {position}.")
        return frames[position]


def _readonly(array: NDArray[Any]) -> NDArray[Any]:
    """Return a non-writeable view of `array`, sharing its memory.

    ``as_numpy`` does not copy and ``normalize_image_shape`` returns a ``(C, H, W)`` array
    unchanged, so a raster handed back here is the dataset's own buffer — and, since it is
    held, the buffer every later reader of this item gets too. A caller annotating a
    finding in place would edit the data the statistics were computed over, silently. A
    view costs nothing to make and turns that into an error at the write; anything that
    needs a mutable array can ``.copy()``, as :meth:`SourceItem.crop` already returns one.
    """
    view = array.view()
    view.flags.writeable = False
    return view


def _frame_number(frame: Any, position: int) -> int:
    """Return a frame's own number, or its decode order where it declares none.

    The same fallback the structuring walk behind :class:`~dataeval.Metadata` applies, and
    it has to be: ``unit_index`` is what a ``unit``-level address keys on, so a locator
    numbering a bare stream differently would answer a Metadata-shaped address with the
    wrong frame. MAITE declares ``frame_index`` but dispatch duck-types the target rather
    than requiring the whole protocol, so a stream of bare arrays is a real input.
    """
    return int(getattr(frame, "frame_index", position))


def _track_ids_of(frame_target: Any) -> list[int]:
    """One frame's track ids, one per detection, ``-1`` where the frame declares none.

    Counted off ``labels`` rather than ``track_ids`` so the numbering matches the
    structuring walk behind :class:`~dataeval.Metadata`, which reads a frame's detection
    count from its labels and only then reads track ids — and skips the attribute
    entirely for a detection-free frame, which is therefore allowed not to carry it.
    """
    count = len(as_numpy(frame_target.labels).reshape(-1))
    if not count:
        return []
    ids = getattr(frame_target, "track_ids", None)
    if ids is None:
        return [-1] * count
    values = [int(value) for value in as_numpy(ids).reshape(-1).tolist()]
    return values[:count] if len(values) >= count else values + [-1] * (count - len(values))


def _keyed(item: SourceItem) -> int:
    """Return the address's key, where the level it resolved to guarantees there is one."""
    key = item.address.key
    if key is None:
        raise ValueError(
            f"{item.address!r} resolved to {item.level!r} level, whose rows are named by a key, but it "
            "states none. Address one of them as SourceIndex(item, key, level).",
        )
    return key
