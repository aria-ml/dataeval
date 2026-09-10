"""Deciding which frames of a video sequence take part, and what to record about each.

Frame selection is not one thing. A stride is decided from a frame number; a target frame rate
needs each frame's timestamp; collapsing redundancy needs the pixels; a medoid-per-shot key frame
needs *every* frame's descriptor before it can choose any of them. A single ``sample: int | float``
argument covers the first two and forecloses the rest, so selection is a declared object instead,
in the shape :class:`~dataeval.data.Operation` already establishes for :class:`~dataeval.data.View`.

What a selector decides is consumed by :class:`~dataeval.data.SequenceFrames`.
"""

__all__ = []

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any

import numpy as np
import xxhash as xxh
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.config import get_seed, resolve_batch_size
from dataeval.flags import ImageStats
from dataeval.protocols import DatumMetadata, FeatureExtractor, SingleFrameObjectTrackingTarget
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


def _bin_picks(edges: NDArray[np.intp], rng: np.random.Generator | None) -> NDArray[np.intp]:
    """One position per ``[start, end)`` bin: its start, or a uniform draw when jittering."""
    starts, ends = edges[:-1], edges[1:]
    if rng is None:
        return starts.astype(np.intp)
    return rng.integers(starts, ends).astype(np.intp)


class _Binned(FrameSelector, ABC):
    """A selector that cuts a sequence into bins and keeps one frame from each.

    The two rules for sizing those bins -- a fixed bin *size* (:class:`Stride`) and a fixed bin
    *count* (:class:`EvenlySpaced`) -- are the only difference between the subclasses, so the
    jitter, seeding and replay behaviour they share lives here.

    Because bins are contiguous and each kept frame stands for its own bin, the default
    :attr:`FrameVerdict.weight` -- the gap to the next kept frame -- is already right, and no
    subclass declares one.
    """

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self, jitter: bool, seed: int | None) -> None:
        self.jitter: bool = bool(jitter)
        # The global fallback is resolved at selection time, not here: a selector is declared
        # before the configuration it will eventually run under.
        self.seed: int | None = None if seed is None else int(seed)

    @abstractmethod
    def _edges(self, n_frames: int) -> NDArray[np.intp]:
        """Bin edges spanning ``[0, n_frames]``, ascending and strictly increasing."""
        ...

    def _rng(self, info: SequenceInfo) -> np.random.Generator | None:
        """Return a generator keyed to this sequence, or None when the selector does not jitter.

        Keyed on :attr:`SequenceInfo.source_id` rather than :attr:`SequenceInfo.index`, so a
        sequence draws the same frames however the dataset around it is filtered or reordered. A
        generator drawn once for the whole dataset would instead make a video's key frames depend
        on how many videos happened to precede it.
        """
        if not self.jitter:
            return None
        seed = get_seed() if self.seed is None else self.seed
        if seed is None:
            return np.random.default_rng()
        source_id = info.source_id
        # Anything that is not already valid seed entropy -- a string, or a negative integer id --
        # is folded through a digest, which is stable across processes as `hash` is not.
        if isinstance(source_id, int) and source_id >= 0:
            key = source_id
        else:
            key = xxh.xxh64_intdigest(str(source_id).encode())
        return np.random.default_rng([seed, key])

    def plan(self, info: SequenceInfo) -> NDArray[np.intp]:
        """Return one position per bin, decided from the sequence's frame count alone."""
        return _bin_picks(self._edges(info.n_frames), self._rng(info))

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Keep the positions :meth:`plan` names, resolved once per sequence."""
        wanted: set[int] | None = None
        for frame in frames:
            if wanted is None:
                # Resolved once per sequence rather than per frame, as FrameIndices does: `in`
                # over an array is a scan, which over a sequence of frames is quadratic.
                wanted = set(self.plan(frame.sequence).tolist())
            if frame.position in wanted:
                yield FrameVerdict(frame.position)


class Stride(_Binned):
    """Keep one frame out of every ``step``.

    The cheapest way to thin a sequence, and decided from position alone, so a view built on it
    knows its own length without decoding anything.

    Left unjittered this keeps positions ``0, step, 2 * step, ...``, which is a frame rate
    reduction: what survives is still perfectly periodic, so a frame remains predictable from its
    neighbours. ``jitter=True`` keeps a *random* frame from each block of ``step`` instead, which
    breaks that periodicity while leaving the spacing about the same -- the difference between
    thinning a video and sampling one.

    Parameters
    ----------
    step : int
        Width of each block, in frames. Must be at least 1.
    jitter : bool, default False
        Draw a uniformly random position from each block rather than always taking its first.
    seed : int or None, default None
        Seed for the jitter. ``None`` falls back to :func:`~dataeval.config.get_seed`, and
        selection is not reproducible when that is unset too. Ignored when ``jitter`` is False.

    Raises
    ------
    ValueError
        If ``step`` is less than 1.

    See Also
    --------
    :class:`EvenlySpaced` : Keep a fixed number of frames per sequence, whatever its length
    :class:`FrameRate` : Thin to a target rate using real timestamps rather than frame counts

    Notes
    -----
    Each sequence is jittered from a generator keyed on its ``source_id``, so a sequence draws the
    same frames however the dataset around it is filtered or reordered -- see :class:`_Binned`.

    Examples
    --------
    >>> from dataeval.data import SequenceFrames, Stride
    >>> frames = SequenceFrames(mot_dataset, Stride(5))  # doctest: +SKIP

    One random frame per 5, rather than every 5th:

    >>> frames = SequenceFrames(mot_dataset, Stride(5, jitter=True, seed=0))  # doctest: +SKIP
    """

    def __init__(self, step: int, jitter: bool = False, seed: int | None = None) -> None:
        if step < 1:
            raise ValueError(f"Stride: step must be at least 1; got {step}.")
        super().__init__(jitter, seed)
        self.step: int = int(step)

    def _edges(self, n_frames: int) -> NDArray[np.intp]:
        """Blocks of ``step`` frames, the last one short where the sequence does not divide."""
        return np.append(np.arange(0, n_frames, self.step, dtype=np.intp), np.intp(n_frames))

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Keep each frame whose position is a multiple of ``step``, or the shared bin walk.

        Unjittered, which block a frame falls in and whether it is that block's first frame are
        both answerable from the position alone -- so this needs neither the sequence's frame
        count nor a materialized set of every kept position, which for a long sequence at a small
        ``step`` would be millions of integers held to answer a modulo. Jittered, the pick within
        a block is not positional and the shared bin walk decides it.
        """
        if self.jitter:
            yield from super().select(frames)
            return
        for frame in frames:
            if frame.position % self.step == 0:
                yield FrameVerdict(frame.position)


class EvenlySpaced(_Binned):
    """Keep ``count`` frames per sequence, spread across its whole length.

    Where :class:`Stride` fixes the *spacing* and lets the count follow from how long a sequence
    is, this fixes the *count* and lets the spacing follow. That is what keeps a long video from
    drowning out a short one in any per-frame statistic, which is the usual reason to want it.

    The trade is that the same ``count`` means different things for different sequences: two
    videos of the same content at different lengths yield frames of different independence, since
    the shorter one's are drawn closer together.

    Parameters
    ----------
    count : int
        How many frames to keep per sequence. Must be at least 1. A sequence with fewer than
        ``count`` frames contributes every frame instead.
    jitter : bool, default False
        Draw a uniformly random position from each bin rather than always taking its first.
    seed : int or None, default None
        Seed for the jitter. ``None`` falls back to :func:`~dataeval.config.get_seed`, and
        selection is not reproducible when that is unset too. Ignored when ``jitter`` is False.

    Raises
    ------
    ValueError
        If ``count`` is less than 1.

    Warns
    -----
    UserWarning
        If a sequence has fewer than ``count`` frames (all of its frames are kept).

    See Also
    --------
    :class:`Stride` : Fix the spacing instead, and let the count follow the sequence's length

    Notes
    -----
    Each sequence is jittered from a generator keyed on its ``source_id``, so a sequence draws the
    same frames however the dataset around it is filtered or reordered -- see :class:`_Binned`.

    Examples
    --------
    >>> from dataeval.data import EvenlySpaced, SequenceFrames
    >>> frames = SequenceFrames(mot_dataset, EvenlySpaced(8))  # doctest: +SKIP

    One random frame from each of 8 equal spans, rather than the first of each:

    >>> frames = SequenceFrames(mot_dataset, EvenlySpaced(8, jitter=True, seed=0))  # doctest: +SKIP
    """

    def __init__(self, count: int, jitter: bool = False, seed: int | None = None) -> None:
        if count < 1:
            raise ValueError(f"EvenlySpaced: count must be at least 1; got {count}.")
        super().__init__(jitter, seed)
        self.count: int = int(count)

    def plan(self, info: SequenceInfo) -> NDArray[np.intp]:
        """Return one position per bin, decided from the sequence's frame count alone."""
        if info.n_frames < self.count:
            _logger.warning(
                "EvenlySpaced: sequence %d (id=%r) has %d frame(s), fewer than requested count %d; keeping all frames.",
                info.index,
                info.source_id,
                info.n_frames,
                self.count,
            )
            warnings.warn(
                f"EvenlySpaced: sequence {info.index} (id={info.source_id!r}) has {info.n_frames} frame(s), "
                f"fewer than requested count {self.count}; keeping all frames.",
                UserWarning,
                stacklevel=2,
            )
        return super().plan(info)

    def _edges(self, n_frames: int) -> NDArray[np.intp]:
        """``count`` bins of near-equal width, capped at one bin per frame for a short sequence."""
        n_bins = min(self.count, n_frames)
        edges = np.linspace(0, n_frames, n_bins + 1).round().astype(np.intp)
        # Rounding cannot collapse neighbouring edges once n_bins <= n_frames, but a collapsed bin
        # would be an empty draw range rather than a wrong answer, so it is ruled out rather than
        # reasoned about.
        return np.unique(edges)


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


_MEDIAN_SAMPLE = 2000
"""Frames used to estimate a sequence's median pairwise distance.

The estimate needs a length scale, not an exact quantile, and the full pairwise set is quadratic:
a 5-minute sequence at 30fps would spend gigabytes to refine a number that decides a kernel width.
The sample is taken by even stride, so it stays deterministic and spans the whole sequence.
"""

_DISTANCE_CHUNK = 512
"""Columns of the kernel matrix computed at once when accumulating the density estimate."""


def _sq_distances(x: NDArray[Any], y: NDArray[Any]) -> NDArray[Any]:
    """Squared euclidean distances between every row of ``x`` and every row of ``y``."""
    # `einsum` rather than `(x ** 2).sum(1)`: the row norms are wanted, not a squared copy of the
    # whole matrix, and this is called once per chunk over the same `x`.
    distances = np.einsum("ij,ij->i", x, x)[:, None] + np.einsum("ij,ij->i", y, y)[None, :] - 2 * x @ y.T
    # Expansion can land a hair below zero for near-identical rows; a negative squared distance
    # would come back NaN from the square root.
    return np.maximum(distances, 0.0)


def _median_bandwidth(embeddings: NDArray[Any]) -> float:
    """Estimate a kernel width by the median heuristic: ``median pairwise distance / sqrt(2)``.

    The same estimator :class:`~dataeval.shift.DriftMMD` uses (``sigma_median``), so a kernel width
    means the same thing whether a sequence is being sampled or compared.

    A sequence that is mostly one static shot has more than half its pairs at distance zero, and
    the plain median then comes back zero -- a zero-width kernel, which throws away the very
    frames that *do* differ. In that case alone the scale is read off the pairs that differ
    instead, leaving the estimate untouched wherever it already had a scale to report. Zero
    survives only when no two frames differ at all, which is what
    :meth:`Representative._indistinguishable` is for.
    """
    from scipy.spatial.distance import pdist

    # Ceiling division, so the sample really is capped at _MEDIAN_SAMPLE rows rather than at
    # nearly twice it.
    sample = embeddings[:: max(1, -(-len(embeddings) // _MEDIAN_SAMPLE))]
    distances = pdist(sample)
    median = float(np.median(distances)) if distances.size else 0.0
    if median == 0.0:
        distinct = distances[distances > 0]
        median = float(np.median(distinct)) if distinct.size else 0.0
    return median / np.sqrt(2.0)


def _kernel_herd(embeddings: NDArray[Any], count: int, bandwidth: float) -> tuple[list[int], NDArray[Any]]:
    """Greedily pick ``count`` rows whose kernel mean best matches that of the whole set.

    At each step the pick maximizes (mean similarity to every row) minus (mean similarity to what
    is already picked). The first term is a kernel density estimate, so picks are drawn toward
    well-populated regions; the second repels each pick from the last, so a dense region gets
    several spread-out representatives rather than the same row over and over.

    The kernel matrix is never materialized. Only its row means and one column per pick are
    needed, and both can be accumulated a chunk at a time -- which is the difference between
    working and exhausting memory on a sequence of more than a few thousand frames.

    Returns the chosen row indices, in selection order, and the weight of each: how many rows are
    nearer to it than to any other pick. Those weights sum to the number of rows by construction.
    """
    n = len(embeddings)
    count = min(count, n)
    gamma = 1.0 / (2 * bandwidth**2)

    density = np.zeros(n)
    for start in range(0, n, _DISTANCE_CHUNK):
        chunk = embeddings[start : start + _DISTANCE_CHUNK]
        density += np.exp(-gamma * _sq_distances(embeddings, chunk)).sum(1)
    density /= n

    chosen: list[int] = []
    columns: list[NDArray[Any]] = []
    picked_similarity = np.zeros(n)
    for step in range(count):
        # The subtraction always makes a fresh array, so `density` is never written to.
        scores = density - picked_similarity / max(step, 1)
        scores[chosen] = -np.inf  # never pick the same frame twice
        nearest = int(np.argmax(scores))
        chosen.append(nearest)
        # Kept rather than recomputed: this column is also what decides which pick owns each row.
        columns.append(_sq_distances(embeddings, embeddings[nearest : nearest + 1]).ravel())
        picked_similarity += np.exp(-gamma * columns[-1])

    owner = np.stack(columns, axis=1).argmin(axis=1)
    return chosen, np.bincount(owner, minlength=count).astype(np.float64)


class Representative(FrameSelector):
    """Keep ``count`` frames chosen to represent how a sequence *looks*, not where frames fall in it.

    The other selectors thin a sequence by position or timestamp, which spends the same number of
    frames on a minute of stillness as on a minute where everything changes. This one describes
    every frame, then picks the subset whose spread through descriptor space matches the whole
    sequence's -- so a long static stretch yields one frame and a busy stretch yields several,
    without anyone having to say in advance which is which.

    The method is kernel herding [1]_ [2]_: repeatedly take the frame that is most typical of the
    sequence and least like what has already been taken. Being a greedy argmax it is
    deterministic -- the same sequence and ``count`` always give the same frames.

    Because a kept frame stands for others scattered through the sequence rather than for a
    contiguous run, each verdict carries an explicit :attr:`FrameVerdict.weight`: the number of
    frames nearer to it than to any other kept frame. Those weights still sum to the sequence's
    frame count, so per-frame statistics stay correctly weighted.

    Parameters
    ----------
    count : int
        How many frames to keep per sequence. Must be at least 1. A sequence with fewer than
        ``count`` frames contributes every frame.
    extractor : FeatureExtractor or None, default None
        What describes a frame. ``None`` uses :class:`~dataeval.extractors.FlattenExtractor`,
        which needs no model but compares raw pixels; a pretrained
        :class:`~dataeval.extractors.TorchExtractor` is what makes "looks alike" mean anything
        beyond that, and is worth the cost here.
    bandwidth : float or None, default None
        Kernel width, in descriptor-space distance. ``None`` derives it per sequence by the
        median heuristic, which needs no tuning and adapts to how much a given sequence varies.
    batch_size : int or None, default None
        How many frames are described at once. ``None`` resolves the extractor's own batch size,
        then the global one -- see :func:`~dataeval.config.get_batch_size`.

    Raises
    ------
    ValueError
        If ``count`` is less than 1, or ``bandwidth`` is given and not positive.

    See Also
    --------
    :class:`Redundancy` : Drop frames carrying nothing new, decided one frame at a time
    :class:`EvenlySpaced` : Keep a fixed number of frames by position rather than by appearance

    Notes
    -----
    This is a buffering selector: it cannot choose any frame before it has described all of them,
    so each sequence is walked twice and the view cannot know its own length without walking it.
    Only descriptors are held between the passes, never pixels.

    Prefer a bandwidth left at ``None``. A kernel much wider than the median pairwise distance
    cannot tell neighbouring frames apart, and herding then picks *adjacent* frames at a
    sequence's extremes -- the opposite of what a representative subset is for.

    What is matched is the *distribution*, so budget follows mass rather than variety: a stretch
    holding two thirds of a sequence's frames draws about two thirds of the picks even if nothing
    in it moves. That is what makes the kept frames a stand-in for the whole sequence, but it means
    a long enough identical run can draw a second, redundant pick -- which then carries a weight of
    zero, since every frame nearer to it is nearer to the first. Use :class:`Redundancy` instead
    when collapsing repetition, rather than representing it, is the goal.

    References
    ----------
    .. [1] Chen, Y., Welling, M., & Smola, A. (2010). Super-Samples from Kernel Herding.
           *UAI 2010*, 109-116. arXiv:1203.3472
    .. [2] Bach, F., Lacoste-Julien, S., & Obozinski, G. (2012). On the Equivalence Between
           Herding and Conditional Gradient Algorithms. arXiv:1203.4523

    Examples
    --------
    >>> from dataeval.data import Representative, SequenceFrames
    >>> frames = SequenceFrames(mot_dataset, Representative(8))  # doctest: +SKIP

    Describing frames with a pretrained model rather than raw pixels:

    >>> from dataeval.extractors import TorchExtractor
    >>> selector = Representative(8, extractor=TorchExtractor(model))  # doctest: +SKIP
    """

    needs: FrameInput = FrameInput.PIXELS
    two_pass: bool = True

    def __init__(
        self,
        count: int,
        extractor: FeatureExtractor | None = None,
        bandwidth: float | None = None,
        batch_size: int | None = None,
    ) -> None:
        if count < 1:
            raise ValueError(f"Representative: count must be at least 1; got {count}.")
        if bandwidth is not None and not bandwidth > 0:  # `not >` rather than `<=`, so NaN is caught
            raise ValueError(f"Representative: bandwidth must be positive; got {bandwidth}.")
        from dataeval.extractors import FlattenExtractor

        self.count: int = int(count)
        self.extractor: FeatureExtractor = FlattenExtractor() if extractor is None else extractor
        self.bandwidth: float | None = None if bandwidth is None else float(bandwidth)
        if batch_size is not None:
            resolve_batch_size(batch_size)  # validated now; rejects a non-positive size early
        # The global fallback is resolved at selection time, not here: a selector is declared
        # before the configuration it will eventually run under.
        self.batch_size: int | None = batch_size

    def _describe(self, batch: list[NDArray[Any]]) -> NDArray[Any]:
        """Describe one batch of frames, flattened to one row each."""
        return as_numpy(self.extractor(np.stack(batch))).reshape(len(batch), -1)

    def _walk(self, frames: Iterator[FrameCandidate]) -> tuple[SequenceInfo | None, list[int], NDArray[Any]]:
        """Describe every frame of a sequence, holding descriptors and positions but never pixels.

        The extractor is called a batch at a time rather than a frame at a time, so a model pays
        its per-call overhead once per batch; each batch's pixels are released as soon as it has
        been described.
        """
        size = resolve_batch_size(self.batch_size, getattr(self.extractor, "batch_size", None))
        sequence: SequenceInfo | None = None
        positions: list[int] = []
        described: list[NDArray[Any]] = []
        batch: list[NDArray[Any]] = []
        for frame in frames:
            sequence = frame.sequence
            positions.append(frame.position)
            batch.append(frame.pixels)
            if len(batch) == size:
                described.append(self._describe(batch))
                batch = []
        if batch:
            described.append(self._describe(batch))
        embeddings = np.concatenate(described, dtype=np.float64) if described else np.empty((0, 0))
        return sequence, positions, embeddings

    def _indistinguishable(self, sequence: SequenceInfo, n_frames: int) -> tuple[list[int], NDArray[Any]]:
        """Keep the first frames when every frame describes identically.

        A zero-width kernel makes every subset equally representative and every score NaN, so the
        choice is made here rather than left to an argmax over undefined numbers.
        """
        kept = list(range(min(self.count, n_frames)))
        _logger.info(
            "Representative: every frame of sequence %d describes identically; kept the first %d "
            "rather than choosing between indistinguishable frames.",
            sequence.index,
            len(kept),
        )
        return kept, np.full(len(kept), n_frames / len(kept))

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Describe every frame of the sequence, then yield the representative subset in order."""
        sequence, positions, embeddings = self._walk(frames)
        if sequence is None:
            return

        bandwidth = self.bandwidth if self.bandwidth is not None else _median_bandwidth(embeddings)
        kept, weights = (
            _kernel_herd(embeddings, self.count, bandwidth)
            if bandwidth > 0
            else self._indistinguishable(sequence, len(positions))
        )

        weight_of = dict(zip(kept, weights, strict=True))
        for index in sorted(weight_of):
            yield FrameVerdict(positions[index], weight=float(weight_of[index]))
