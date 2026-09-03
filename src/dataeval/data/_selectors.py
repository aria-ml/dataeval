"""Deciding which frames of a video sequence take part, and what to record about each.

Frame selection is not one thing. A stride is decided from a frame number; a target frame rate
needs each frame's timestamp; collapsing redundancy needs the pixels; a medoid-per-shot key frame
needs *every* frame's descriptor before it can choose any of them. A single ``sample: int | float``
argument covers the first two and forecloses the rest, so selection is a declared object instead,
in the shape :class:`~dataeval.data.Operation` already establishes for :class:`~dataeval.data.View`.

What a selector decides is consumed by :class:`~dataeval.data.SequenceFrames`.
"""

__all__ = []

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.flags import ImageStats
from dataeval.protocols import DatumMetadata, SingleFrameObjectTrackingTarget
from dataeval.types import ReprMixin
from dataeval.utils._array import as_numpy
from dataeval.utils.preprocessing import normalize_image_shape

_logger = get_logger(__name__)


class FrameInput(Flag):
    """What a :class:`FrameSelector` reads in order to decide.

    Declared rather than discovered, so :class:`~dataeval.data.SequenceFrames` knows whether a
    decision needs pixels before it materializes any. A :obj:`~dataeval.protocols.VideoStream`
    that decodes lazily can then hand back frames whose pixels are never realized.

    Attributes
    ----------
    STRUCTURE
        Frame count and position only. Nothing needs decoding to decide.
    TIMING
        Each frame's ``time_s`` / ``pts``. The stream is walked, but pixels are never touched.
    PIXELS
        Each frame's image data.
    """

    STRUCTURE = auto()
    TIMING = auto()
    PIXELS = auto()


@dataclass(frozen=True)
class SequenceInfo:
    """What is known about a sequence before any of it is decoded.

    Attributes
    ----------
    index : int
        Position of the sequence in the source dataset.
    source_id : int or str
        The sequence datum's own :obj:`~dataeval.protocols.DatumMetadata` ``id``, which survives
        filtering and reordering views as a positional index does not.
    n_frames : int
        How many frames the sequence holds, read from its target's ``frame_tracks`` and so known
        without decoding anything. See :class:`~dataeval.data.SequenceFrames`.
    metadata : DatumMetadata
        The sequence's own datum metadata: ``height``, ``width``, ``time_base``, ``size``.
    """

    index: int
    source_id: int | str
    n_frames: int
    metadata: DatumMetadata


@dataclass(frozen=True, eq=False)
class FrameCandidate:
    """One frame offered to a selector for a keep-or-drop decision.

    Attributes
    ----------
    sequence : SequenceInfo
        The sequence this frame belongs to.
    position : int
        Zero-based position within the sequence's stream, counting every frame whether kept or
        not. This is the coordinate a :class:`FrameVerdict` reports.
    frame_index : int
        The frame's own ``frame_index``, or ``position`` for a stream that declares none.
    time_s : float or None
        Seconds from the start of the sequence, where the frame declares it.
    pts : int or None
        Presentation timestamp, where the frame declares it.
    target : SingleFrameObjectTrackingTarget
        This frame's detections: ``boxes``, ``labels``, ``scores``, ``track_ids``.

    Notes
    -----
    :attr:`pixels` is a property rather than a field, and reading it is what costs. A selector
    that declares :attr:`FrameInput.STRUCTURE` or :attr:`FrameInput.TIMING` and reads it anyway
    raises, rather than quietly paying for a decode it said it did not need.
    """

    sequence: SequenceInfo
    position: int
    frame_index: int
    time_s: float | None
    pts: int | None
    target: SingleFrameObjectTrackingTarget
    _frame: Any = field(repr=False, compare=False, default=None)
    _allow_pixels: bool = field(repr=False, compare=False, default=True)
    _cache: list[NDArray[Any]] = field(repr=False, compare=False, default_factory=list)

    @property
    def pixels(self) -> NDArray[Any]:
        """The frame's image data in ``(C, H, W)``, materialized on first read and then cached.

        Raises
        ------
        AttributeError
            When the selector declared it does not read pixels. Declaring
            :attr:`FrameInput.PIXELS` is what makes this available.
        """
        if not self._allow_pixels:
            raise AttributeError(
                "FrameCandidate.pixels was read by a selector that does not declare "
                "FrameInput.PIXELS. Add PIXELS to the selector's `needs` so SequenceFrames "
                "knows the decision costs a decode."
            )
        if not self._cache:
            self._cache.append(normalize_image_shape(as_numpy(self._frame.pixels)))
        return self._cache[0]


@dataclass(frozen=True)
class FrameVerdict:
    """A selector's decision to keep one frame, and what it wants recorded about it.

    Attributes
    ----------
    position : int
        The :attr:`FrameCandidate.position` being kept.
    weight : float or None, default None
        How many source frames this kept frame stands for. ``None`` -- the default -- lets
        :class:`~dataeval.data.SequenceFrames` compute it as the gap to the next kept frame, which
        is correct for any selector whose representatives are contiguous and makes the per-sequence
        weights sum to the frame count by construction. Override it only when what a frame stands
        for is *not* contiguous, such as a medoid standing for a scattered cluster; the sum still
        has to come out right.
    factors : Mapping[str, Any], default empty
        Anything the selector derived while deciding -- a shot index, a novelty score. Each becomes
        a factor on the frame's datum metadata, and so a ``unit``-level factor in
        :class:`~dataeval.Metadata`. Names reserved by
        :class:`~dataeval.data.SequenceFrames` are rejected rather than silently overwritten.
    """

    position: int
    weight: float | None = None
    factors: Mapping[str, Any] = field(default_factory=dict)


class FrameSelector(ReprMixin, ABC):
    """Decides which frames of a sequence survive, and what to record about each.

    Subclass this to control which frames :class:`~dataeval.data.SequenceFrames` presents. A
    selector implements :meth:`select`, and may additionally implement :meth:`plan` when its
    decision is a function of structure alone.

    Attributes
    ----------
    needs : FrameInput
        What the selector reads. See :class:`FrameInput`.
    two_pass : bool
        Whether verdicts can only be produced after the whole sequence has been seen. A selector
        that buffers -- clustering a sequence's descriptors before choosing medoids, say -- sets
        this, and :class:`~dataeval.data.SequenceFrames` then walks each sequence twice: once to
        drive :meth:`select` to exhaustion, once to emit. A buffering selector must buffer
        *descriptors and positions, never pixels*: one sequence of 1080p frames is tens of
        gigabytes.
    invalidates : ImageStats
        Statistics this selector makes describe itself rather than the data. Selection alone
        invalidates nothing -- a kept frame is an unmodified image -- so this is
        :attr:`~dataeval.flags.ImageStats.NONE` unless a subclass also rewrites content.

    Notes
    -----
    **A selector that implements** :meth:`plan` **is authoritative through it.** Its positions are
    used for sizing *and* for selection, and :meth:`select` is not consulted, so such a selector
    contributes no :attr:`FrameVerdict.factors`. That is the trade for a view whose length is known
    without decoding anything. A selector needing to record derived values leaves :meth:`plan`
    alone and answers through :meth:`select`.

    :meth:`select` is abstract and :meth:`plan` is not, deliberately: every selector must be able
    to run streaming, and planning is a shortcut rather than an alternative. A selector that could
    only plan would break on the first sequence whose frame count it cannot learn.

    Examples
    --------
    A selector keeping every third frame, decided from position alone:

    >>> import numpy as np
    >>> from dataeval.data import FrameSelector, FrameVerdict
    >>>
    >>> class EveryThird(FrameSelector):
    ...     def plan(self, info):
    ...         return np.arange(0, info.n_frames, 3)
    ...
    ...     def select(self, frames):
    ...         return (FrameVerdict(f.position) for f in frames if f.position % 3 == 0)
    """

    needs: FrameInput = FrameInput.STRUCTURE
    two_pass: bool = False
    invalidates: ImageStats = ImageStats.NONE

    def plan(self, info: SequenceInfo) -> NDArray[np.intp] | None:  # noqa: ARG002
        """Return the positions to keep from structure alone, or None when the frames decide.

        Parameters
        ----------
        info : SequenceInfo
            What is known about the sequence without decoding it.

        Returns
        -------
        NDArray[np.intp] or None
            Ascending positions within the sequence, or None to defer to :meth:`select`.
        """
        return None

    @abstractmethod
    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Yield one verdict per kept frame, in non-decreasing position order.

        Parameters
        ----------
        frames : Iterator[FrameCandidate]
            The sequence's frames, in order.

        Yields
        ------
        FrameVerdict
            One per kept frame. Unless :attr:`two_pass` is set, a verdict must be yielded before
            the walk moves past the frame it names; :class:`~dataeval.data.SequenceFrames` holds
            exactly one frame and raises if a verdict arrives for one it has already released.
        """
        ...


class AllFrames(FrameSelector):
    """Keep every frame. The default.

    Examples
    --------
    >>> from dataeval.data import AllFrames, SequenceFrames
    >>> frames = SequenceFrames(mot_dataset, AllFrames())  # doctest: +SKIP
    """

    needs: FrameInput = FrameInput.STRUCTURE

    def plan(self, info: SequenceInfo) -> NDArray[np.intp]:
        """Return every position in the sequence."""
        return np.arange(info.n_frames, dtype=np.intp)

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Keep each frame as it arrives."""
        return (FrameVerdict(frame.position) for frame in frames)


class Stride(FrameSelector):
    """Keep every ``n``-th frame.

    The cheapest way to thin a sequence, and decided from position alone, so a view built on it
    knows its own length without decoding anything.

    Parameters
    ----------
    step : int
        Keep positions ``0, step, 2 * step, ...``. Must be at least 1.

    Raises
    ------
    ValueError
        If ``step`` is less than 1.

    See Also
    --------
    :class:`FrameRate` : Thin to a target rate using real timestamps rather than frame counts

    Examples
    --------
    >>> from dataeval.data import SequenceFrames, Stride
    >>> frames = SequenceFrames(mot_dataset, Stride(5))  # doctest: +SKIP
    """

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self, step: int) -> None:
        if step < 1:
            raise ValueError(f"Stride: step must be at least 1; got {step}.")
        self.step: int = int(step)

    def plan(self, info: SequenceInfo) -> NDArray[np.intp]:
        """Return every ``step``-th position."""
        return np.arange(0, info.n_frames, self.step, dtype=np.intp)

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Keep each frame whose position is a multiple of ``step``."""
        return (FrameVerdict(frame.position) for frame in frames if frame.position % self.step == 0)


class FrameIndices(FrameSelector):
    """Keep an explicitly named set of positions per sequence.

    Replays a selection some other pass produced -- a key-frame set computed offline, or one a
    previous :class:`~dataeval.data.SequenceFrames` recorded -- with no decoding needed to plan it.
    That is what makes a selection reproducible and reviewable.

    Parameters
    ----------
    positions : Mapping[int, Sequence[int]]
        Positions to keep, keyed by the sequence's index in the source dataset. A sequence absent
        from the mapping contributes no frames.

    Raises
    ------
    ValueError
        If any position is negative.

    Examples
    --------
    >>> from dataeval.data import FrameIndices, SequenceFrames
    >>> frames = SequenceFrames(mot_dataset, FrameIndices({0: [0, 30, 60], 1: [12]}))  # doctest: +SKIP
    """

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self, positions: Mapping[int, Sequence[int]]) -> None:
        self.positions: dict[int, NDArray[np.intp]] = {
            int(key): np.asarray(value, dtype=np.intp) for key, value in positions.items()
        }
        for key, value in self.positions.items():
            if value.size and int(value.min()) < 0:
                raise ValueError(f"FrameIndices: positions for sequence {key} must be non-negative.")

    def plan(self, info: SequenceInfo) -> NDArray[np.intp]:
        """Return the named positions for this sequence, clipped to the frames it has."""
        wanted = self.positions.get(info.index, np.empty(0, dtype=np.intp))
        kept = np.unique(wanted[wanted < info.n_frames])
        if len(kept) != len(wanted):
            _logger.info(
                "FrameIndices: sequence %d has %d frame(s); %d named position(s) were out of range "
                "or repeated and are dropped.",
                info.index,
                info.n_frames,
                len(wanted) - len(kept),
            )
        return kept.astype(np.intp)

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Keep each frame whose position was named for its sequence."""
        wanted: set[int] | None = None
        for frame in frames:
            if wanted is None:
                # Resolved once per sequence rather than per frame: `in` over an array is a scan,
                # which over a sequence of frames is quadratic.
                named = self.positions.get(frame.sequence.index)
                wanted = set() if named is None else set(named.tolist())
            if frame.position in wanted:
                yield FrameVerdict(frame.position)


class Redundancy(FrameSelector):
    """Drop frames that carry nothing new over the last frame kept.

    The first content-dependent selector, and the shape a key-frame extractor takes: it reads the
    pixels, keeps state across the walk, and decides each frame as it arrives -- one pass, no
    buffering. Consecutive frames of a static camera or a stalled feed differ by a handful of
    bits, so a sequence that never changes collapses to a single frame.

    Parameters
    ----------
    radius : int, default 4
        Maximum Hamming distance, in bits, from the last kept frame for a frame to be dropped.
        ``0`` drops only frames identical to the last kept one. For the 64-bit hashes DataEval
        computes, ``1-5`` is very similar; the default is deliberately tighter than the radius
        used to *match* frames across videos, because "carries no new information" is a stronger
        claim than "is a copy of".
    method : {"phash", "dhash", "phash_d4", "dhash_d4", "xxhash"}, default "phash"
        Which hash to compare frames by. The ``_d4`` variants are invariant to rotation and
        mirroring, which for consecutive frames of one video is rarely what is wanted and costs
        eight times as much. ``xxhash`` drops only byte-identical frames.

    Raises
    ------
    ValueError
        If ``radius`` is negative or ``method`` is not one of the named hashes.

    See Also
    --------
    :func:`~dataeval.core.redundant_runs` : Measure redundancy rather than select against it

    Notes
    -----
    Anchored on the last **kept** frame, not on the predecessor. That is what makes it a selection
    rule: under a pairwise anchor a slow pan is a series of short runs and nothing is ever
    dropped, while anchoring on what was kept lets a drift accumulate until it is worth recording.
    :func:`~dataeval.core.redundant_runs` takes the pairwise view instead, because *measuring* how
    much a sequence repeats itself is a different question from choosing what to keep.

    Because the decision depends on the frames, the view cannot know its own length without
    walking them -- see :class:`~dataeval.data.SequenceFrames`.

    Examples
    --------
    >>> from dataeval.data import Redundancy, SequenceFrames
    >>> frames = SequenceFrames(mot_dataset, Redundancy(radius=4))  # doctest: +SKIP
    """

    needs: FrameInput = FrameInput.PIXELS

    _METHODS = ("phash", "dhash", "phash_d4", "dhash_d4", "xxhash")

    def __init__(self, radius: int = 4, method: str = "phash") -> None:
        if radius < 0:
            raise ValueError(f"Redundancy: radius must be non-negative; got {radius}.")
        if method not in self._METHODS:
            raise ValueError(f"Redundancy: method must be one of {self._METHODS}; got {method!r}.")
        self.radius: int = int(radius)
        self.method: str = method

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Keep a frame when its hash is more than ``radius`` bits from the last frame kept."""
        from dataeval.core import _hash as hashes
        from dataeval.core import hamming_distance

        digest = getattr(hashes, self.method)
        kept: str | None = None
        for frame in frames:
            current = digest(frame.pixels)
            if not current:
                # No digest is no evidence that nothing changed, so the frame is kept rather than
                # dropped on an assumption.
                yield FrameVerdict(frame.position)
                continue
            if kept is None or hamming_distance(kept, current) > self.radius:
                kept = current
                yield FrameVerdict(frame.position)


class FrameRate(FrameSelector):
    """Thin a sequence to approximately ``fps`` frames per second, using real timestamps.

    Keeps the first frame, then the next frame at least ``1 / fps`` seconds after the last one
    kept. Because it reads timestamps rather than counting frames, it thins sequences captured at
    different rates to a common rate -- which frame-count striding cannot do.

    Parameters
    ----------
    fps : float
        Target frames per second. Must be positive.

    Raises
    ------
    ValueError
        If ``fps`` is not positive.

    See Also
    --------
    :class:`Stride` : Thin by frame count, with no timestamps needed

    Notes
    -----
    :meth:`plan` returns None on purpose. Nothing in the multi-object-tracking protocol declares a
    frame rate -- a video's :obj:`~dataeval.protocols.DatumMetadata` carries ``height``, ``width``,
    ``time_base`` and ``size``, not a duration or an fps -- so a target rate can only be honoured
    against each frame's own ``time_s``, which requires the walk. Guessing a rate would make every
    derived timing quietly wrong.

    A sequence whose frames declare no ``time_s`` is kept in full, with a log line saying so.
    Silently thinning on an assumed frame rate is the one outcome worth ruling out.

    Examples
    --------
    >>> from dataeval.data import FrameRate, SequenceFrames
    >>> frames = SequenceFrames(mot_dataset, FrameRate(2.0))  # doctest: +SKIP
    """

    needs: FrameInput = FrameInput.TIMING

    def __init__(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError(f"FrameRate: fps must be positive; got {fps}.")
        self.fps: float = float(fps)

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Keep the first frame, then each frame at least ``1 / fps`` seconds after the last kept."""
        interval = 1.0 / self.fps
        last_kept: float | None = None
        untimed = 0
        sequence: SequenceInfo | None = None
        for frame in frames:
            sequence = frame.sequence
            if frame.time_s is None:
                # No timestamp is no basis for thinning, so the frame is kept rather than
                # dropped on an assumption. Reported once per sequence, below.
                untimed += 1
                yield FrameVerdict(frame.position)
                continue
            if last_kept is None or frame.time_s - last_kept >= interval:
                last_kept = frame.time_s
                yield FrameVerdict(frame.position)
        if untimed and sequence is not None:
            _logger.info(
                "FrameRate: %d of %d frame(s) in sequence %d declare no time_s and were kept in "
                "full; a target rate cannot be honoured without timestamps.",
                untimed,
                sequence.n_frames,
                sequence.index,
            )
