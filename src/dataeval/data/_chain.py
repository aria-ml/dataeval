"""Composing frame selectors, each choosing among the frames the one before it kept."""

__all__ = []

from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from functools import reduce
from operator import or_
from typing import Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from dataeval.data._frames import SequenceFrames
from dataeval.data._selectors import (
    AllFrames,
    FrameCandidate,
    FrameInput,
    FrameSelector,
    FrameVerdict,
    SequenceInfo,
    _projected,
)


def _survivors(info: SequenceInfo, positions: NDArray[np.intp]) -> SequenceInfo:
    """Describe the sequence as a later stage sees it: only the frames at ``positions``, renumbered from 0."""
    target = None if info.target is None else _projected(info.target, positions)
    return SequenceInfo(info.index, info.source_id, len(positions), info.metadata, target)


def _stage_masses(masses: NDArray[np.float64], kept: NDArray[np.intp], excludes: bool) -> NDArray[np.float64]:
    """Apply the composition rule for one stage that declares no weights.

    ``masses`` holds the source mass of each of the stage's input frames, and ``kept`` the
    ascending ordinals it kept. Under a stage that excludes, a kept frame keeps its own mass;
    otherwise it takes the mass of every frame up to the next one kept, the first also taking
    every frame before it.
    """
    if excludes or not len(kept):
        return masses[kept]
    starts = kept.copy()
    starts[0] = 0
    return np.add.reduceat(masses, starts)


def _composed(
    stages: tuple[FrameSelector, ...],
    info: SequenceInfo,
    start: tuple[NDArray[np.intp], NDArray[np.float64]] | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.float64]] | None:
    """Plan each stage over the survivors of the one before, carrying their masses.

    The first stage plans over the source frames at the positions ``start`` names, at the masses
    it gives, or over every source frame at mass 1. Returns the frames the last stage keeps, as
    indices into those, and the mass of each; with no ``start`` the indices are source positions.
    Indices, so a frame ``start`` names twice stays two frames. With no stages, every frame. None
    when some stage returns no plan.
    """
    if start is None:
        positions = np.arange(info.n_frames, dtype=np.intp)
        masses = np.ones(info.n_frames, dtype=np.float64)
        stage_info = info
    else:
        positions, masses = start
        stage_info = _survivors(info, positions)
    taken = np.arange(len(positions), dtype=np.intp)
    for stage in stages:
        planned = stage.plan(stage_info)
        if planned is None:
            return None
        kept = SequenceFrames._normalized(planned, stage_info)
        taken, masses = taken[kept], _stage_masses(masses, kept, stage.excludes)
        stage_info = _survivors(info, positions[taken])
    return taken, masses


class _Mass:
    """A kept frame's mass in source frames, pending until every frame it stands for has arrived.

    A private protocol between ``Chain`` and :class:`~dataeval.data.SequenceFrames`. A streaming
    chain yields each final verdict before the frames after it are known, so it declares one of
    these as the verdict's weight. ``SequenceFrames._record`` reads it with ``float()`` only when
    it builds that frame's record, on the next kept frame or at the end of the sequence, and the
    chain has settled it by then. A read any sooner raises rather than returning a partial count.
    """

    __slots__ = ("settled", "total")

    def __init__(self) -> None:
        self.total = 0.0
        self.settled = False

    def __float__(self) -> float:
        if not self.settled:
            raise RuntimeError(
                "Chain: a frame's weight was read before it settled. A streaming stage is offered "
                "frames whose mass the stage before it is still counting, so its select must not "
                "decide from FrameCandidate.weight."
            )
        return self.total


class _Survivor(NamedTuple):
    """A frame that every stage so far has kept."""

    position: int
    """Its position in the source sequence."""
    candidate: FrameCandidate
    """As the last stage was offered it: the frame handle, whose pixels every stage shares one decode of."""
    weight: float | _Mass
    """Its mass as the last stage left it, which is the next stage's ``FrameCandidate.weight``."""
    factors: dict[str, tuple[FrameSelector, Any]]
    """Every factor a stage set on it, beside the stage that set it."""


class _StageRun:
    """One streaming stage's walk over one sequence, settling the masses of the frames it keeps.

    ``Chain.select`` builds one per stage per sequence, so nothing about a sequence outlives its
    walk or lands on the chain. A frame's mass is final once the frame after it reaches the stage,
    or the stage's input ends: the stage before has then kept that next frame, which closes every
    run up to it. So each kept frame's run is complete when the stage keeps the next one, and the
    last one's when its input ends, which is before the last frame is recorded.
    """

    def __init__(self, number: int, stage: FrameSelector, info: SequenceInfo) -> None:
        self.number = number
        self.stage = stage
        self.info = info
        self.pixels = bool(stage.needs & FrameInput.PIXELS)
        # The survivor the stage was offered last, renumbered: the one frame it holds.
        self.held: _Survivor | None = None
        self.ordinal = -1
        # Past the first stage, the walk may already have passed the held frame when the input
        # ends, as the stages before may have dropped the frames after it. So such a stage holds
        # it no longer, on every video alike.
        self.ended = False
        # Where a dropped frame's mass goes. Before anything is kept, a leading run the first
        # kept frame takes over if the default rule applies to it. Then the run of the last frame
        # kept under the default rule, and nowhere after a frame kept with a declared weight or
        # under excludes, as the frames after it are declared for or excluded.
        self.leading: _Mass | None = _Mass()
        self.run: _Mass | None = self.leading
        self.into: _Mass | None = None  # where the held frame's own mass goes
        self.open: _Mass | None = None  # the last kept frame's mass, closed when the next is kept
        self.mass_in = 0.0
        self.mass_out = 0.0

    def kept(  # noqa: C901
        self,
        upstream: Iterator[_Survivor],
        select: Callable[[Iterator[FrameCandidate]], Iterable[FrameVerdict]] | None = None,
    ) -> Iterator[_Survivor]:
        """Offer the stage each survivor, renumbered from 0, and yield each one it keeps as it keeps it.

        ``select`` replays verdicts the stage reached earlier, in place of its own ``select``.
        """
        offered = self._offered(upstream)
        for verdict in (self.stage.select if select is None else select)(offered):
            held = self.held
            if self.ended or held is None or verdict.position != self.ordinal:
                if self.ended:
                    reached = "after its input ended"
                elif held is None:
                    reached = "while it held no frame"
                else:
                    reached = f"while it held position {self.ordinal}"
                raise ValueError(
                    f"Chain: stage {self.number}, {self.stage!r}, yielded a verdict for position "
                    f"{verdict.position} of the frames reaching it from sequence {self.info.index} "
                    f"{reached}. A stage that decides a frame only after seeing later ones, or "
                    "after the end of its input behind another stage, must declare two_pass = True."
                )
            yield _Survivor(held.position, held.candidate, self._keep(verdict), self._merged(held, verdict))
        # A stage may stop reading early. The frames it never saw run to the last frame it kept.
        for _ in offered:
            pass
        self._finish()

    def _offered(self, upstream: Iterator[_Survivor]) -> Iterator[FrameCandidate]:
        for survivor in upstream:
            self._settle()
            self.ordinal += 1
            candidate = replace(
                survivor.candidate,
                sequence=self.info,
                position=self.ordinal,
                weight=cast(float, survivor.weight),
                _allow_pixels=self.pixels,
            )
            self.held, self.into = survivor._replace(candidate=candidate), self.run
            yield candidate
        self.ended = self.number > 1

    def _settle(self) -> None:
        """Count the held frame's mass, final now that the next frame has arrived or the input has ended."""
        if self.held is not None:
            mass = float(self.held.weight)
            self.mass_in += mass
            if self.into is not None:
                self.into.total += mass

    def _keep(self, verdict: FrameVerdict) -> float | _Mass:
        """Start the held frame's mass by the composition rule, closing the one kept before it."""
        if self.open is not None:
            self._close(self.open)
        mass: float | _Mass
        if verdict.weight is not None:
            # Declared, and already in source frames.
            mass, self.run, self.into, self.open = verdict.weight, None, None, None
            if not 0 <= float(mass) < np.inf:
                raise ValueError(
                    f"Chain: stage {self.number}, {self.stage!r}, declared a weight of {float(mass):g} "
                    f"for position {verdict.position} of the frames reaching it from sequence "
                    f"{self.info.index}. A FrameVerdict.weight counts the source frames a kept frame "
                    "stands for, so it must be finite and not negative."
                )
            self.mass_out += float(mass)
        elif self.stage.excludes:
            # Its own mass alone. The frames after it are excluded.
            mass = self.into = self.open = _Mass()
            self.run = None
        else:
            # Its own mass, and its run to the next kept frame, the first also taking those before it.
            mass = self.run = self.into = self.open = _Mass() if self.leading is None else self.leading
        self.leading = None
        return mass

    def _close(self, mass: _Mass) -> None:
        mass.settled = True
        self.mass_out += mass.total

    def _finish(self) -> None:
        """Settle the last frame's mass, and refuse a stage whose masses total more than its input."""
        self._settle()
        if self.open is not None:
            self._close(self.open)
        if self.mass_out > self.mass_in * (1 + 1e-9):
            raise ValueError(
                f"Chain: stage {self.number}, {self.stage!r}, gave the frames it kept of sequence "
                f"{self.info.index} weights totaling {self.mass_out:g} source frames, but the frames "
                f"reaching it stood for {self.mass_in:g}. A declared FrameVerdict.weight sums the "
                "FrameCandidate.weight of the frames it stands for, so a stage's weights cannot "
                "total more than its input."
            )

    def _merged(self, held: _Survivor, verdict: FrameVerdict) -> dict[str, tuple[FrameSelector, Any]]:
        """Add the stage's factors to those earlier stages set on the frame, refusing a name set twice."""
        clash = sorted(held.factors.keys() & verdict.factors.keys())
        if clash:
            raise ValueError(
                f"Chain: {held.factors[clash[0]][0]!r} and stage {self.number}, {self.stage!r}, both "
                f"set factor {clash[0]!r} on frame {held.candidate.frame_index} of sequence "
                f"{self.info.index}. A factor name can come from one stage only."
            )
        return {**held.factors, **{name: (self.stage, value) for name, value in verdict.factors.items()}}


def _prepended(first: FrameCandidate, rest: Iterator[FrameCandidate]) -> Iterator[FrameCandidate]:
    """Yield a peeked frame and then the rest, holding the peeked one only until the next is wanted."""
    yield first
    del first
    yield from rest


def _arriving(
    frames: Iterator[FrameCandidate],
    kept: Iterable[tuple[int, float, dict[str, tuple[FrameSelector, Any]]]],
) -> Iterator[_Survivor]:
    """Yield a survivor for each of ``kept``, as its source frame arrives, with its mass and any factors set so far.

    ``kept`` holds source positions in non-decreasing order, and one it holds twice arrives twice,
    as a stage may keep a frame more than once.
    """
    pending = iter(kept)
    entry = next(pending, None)
    for frame in frames:
        while entry is not None and entry[0] == frame.position:
            yield _Survivor(frame.position, frame, entry[1], entry[2])
            entry = next(pending, None)


def _settled(survivors: Iterator[_Survivor]) -> Iterator[_Survivor]:
    """Yield each survivor one behind, its mass a float: final once the next has arrived, or the input ended."""
    held: _Survivor | None = None
    for survivor in survivors:
        if held is not None:
            yield held._replace(weight=float(held.weight))
        held = survivor
    if held is not None:
        yield held._replace(weight=float(held.weight))


def _released(survivor: _Survivor) -> _Survivor:
    """Drop the survivor's frame handle and decoded pixels, so it can be held past its frame."""
    return survivor._replace(candidate=replace(survivor.candidate, _frame=None, _cache=[]))


def _verdicts(survivors: Iterator[_Survivor]) -> Iterator[FrameVerdict]:
    """Report each survivor of the last stage at its source position, with its weight and every stage's factors."""
    for survivor in survivors:
        factors = {name: value for name, (_, value) in survivor.factors.items()}
        yield FrameVerdict(survivor.position, cast(float, survivor.weight), factors)


def _refuse_misplaced_adjacency(stages: tuple[FrameSelector, ...]) -> None:
    """Refuse a stage that reads adjacent frames behind any stage that can drop frames."""
    dropper = next((k for k, stage in enumerate(stages) if type(stage) is not AllFrames), len(stages))
    for stage in stages[dropper + 1 :]:
        if stage._needs_adjacent_frames:
            raise ValueError(
                f"Chain: {stage!r} reads the frames on either side of each one, but follows "
                f"{stages[dropper]!r}, which can drop frames, so its neighbors would not be adjacent. "
                "Place it first, or behind AllFrames only."
            )


def _pass_boundary(stages: tuple[FrameSelector, ...], planning: tuple[bool, ...]) -> int | None:
    """Return the stage that ends pass 1, the first that needs its whole input, or None.

    A stage needs its whole input when it is ``two_pass``, or when it plans and a stage that does
    not plan runs earlier in the same pass, since its input count and its survivors' targets are
    unknown until that pass ends. After the boundary, planning stages resolve at once from its
    survivors, and the rest stream in pass 2. A second stage needing its whole input would need a
    third walk of every sequence, and is refused.
    """
    boundary: int | None = None
    streamed = False  # whether a stage that does not plan runs earlier in the current pass
    for k, (stage, plans) in enumerate(zip(stages, planning, strict=True)):
        if stage.two_pass or (plans and streamed):
            if boundary is not None:
                raise ValueError(
                    f"Chain: {stage!r} needs its whole input, but {stages[boundary]!r} already ends "
                    "the first pass, and a chain walks a sequence at most twice. A stage needs its "
                    "whole input when it is two_pass, or when it plans behind a stage that does not."
                )
            boundary, streamed = k, False
        else:
            streamed = streamed or not plans
    return boundary


class Chain(FrameSelector):
    """Apply frame selectors in turn, each choosing among the frames the one before it kept.

    ``Chain(Redundancy(), EvenlySpaced(8))`` spreads eight frames across what is left once
    redundant frames are gone, which neither selector does alone. A chain is itself a
    :class:`~dataeval.data.FrameSelector`, so it goes wherever one does.

    Parameters
    ----------
    *selectors : FrameSelector
        The stages, in the order they run. A ``Chain`` among them contributes its own stages.

    Attributes
    ----------
    stages : tuple[FrameSelector, ...]
        The stages, nested chains flattened.
    needs : FrameInput
        Everything any stage reads.
    invalidates : ImageStats
        Everything any stage invalidates.
    two_pass : bool
        Whether some stage needs its whole input: one that is ``two_pass``, or one that plans
        behind a stage that does not. Each sequence is then walked twice.

    Raises
    ------
    TypeError
        If a stage is not a :class:`~dataeval.data.FrameSelector` instance.
    ValueError
        If there are no stages, if a stage that reads adjacent frames follows any stage but
        :class:`~dataeval.data.AllFrames`, or if two stages need their whole input, which would
        take a third walk of every sequence.

    Notes
    -----
    **Each stage has its own coordinates.** A stage sees only the frames the stage before it
    kept, numbered ``0`` to ``m - 1`` in order. Its candidates' ``position``, the positions its
    :meth:`~dataeval.data.FrameSelector.plan` returns and the ``n_frames`` of the
    :class:`~dataeval.data.SequenceInfo` it plans over all count that input rather than the
    source sequence, and that ``SequenceInfo``'s ``target`` lists only the survivors' frame
    targets. A candidate's ``frame_index`` is still the source frame's number. The chain alone
    maps stage positions back to the source, so what it reports is in source positions like any
    other selector's.

    **A stage that does not plan decides frame by frame.** It starts before its input count is
    known, so its candidates carry the source sequence's :class:`~dataeval.data.SequenceInfo`
    without a ``target``, and a ``weight`` the stage before it may still be counting. Its
    :meth:`~dataeval.data.FrameSelector.select` decides from none of these, and a weight read
    before it settles raises :class:`RuntimeError`. A stage after the first must decide each
    frame before it asks for the next, even at the end of its input, since the stages before it
    may have dropped frames the walk has already passed. A stage that knows a frame is its last
    only once its input ends must declare ``two_pass``, or it raises :class:`ValueError` on every
    video. A stage reads pixels only when its own ``needs`` declares them. The factors every
    stage records on a frame are merged onto it, and two stages naming the same factor raise
    :class:`ValueError`.

    **Weights compose exactly.** Each frame reaching a stage carries the source frames it already
    stands for, and each stage applies the rule :class:`~dataeval.data.SequenceFrames` applies to
    a single selector. A weight the stage declares is taken as given. Under a stage that
    excludes, a kept frame keeps its own mass and a dropped frame's is excluded. Otherwise a kept
    frame takes the mass of every frame up to the next one kept, the first also taking those
    before it. The chain's weights are its last stage's, so per sequence they sum to the frame
    count less every frame some stage excluded. A chain always declares its weights, so its own
    ``excludes`` is False.

    **A chain plans when every stage plans.** Its plan composes theirs, so a view built on it
    knows its own length without decoding anything. Otherwise the stages ahead of the first that
    does not plan are planned over the source sequence, and the rest run as its frames arrive.

    **A stage that needs its whole input ends the first pass.** That is a ``two_pass`` stage, or
    a stage that plans behind one that does not, whose input count is unknown until every frame
    has passed the stages before it. The first walk of a sequence runs the stages up to that
    one, which then decides from every frame reaching it. A ``two_pass`` stage is offered its
    frames as a streaming stage is, but with each ``weight`` settled, so it may decide from it.
    The stages after it that plan resolve at once from its survivors, and the rest stream through
    the second walk, so no sequence is walked more than twice.

    Examples
    --------
    >>> from dataeval.data import Chain, EvenlySpaced, Stride
    >>> Chain(Stride(2), EvenlySpaced(8))
    Chain(Stride(step=2, jitter=False, seed=None), EvenlySpaced(count=8, jitter=False, seed=None))

    Eight frames spread across what remains once redundant frames are dropped:

    >>> from dataeval.data import Redundancy, SequenceFrames
    >>> frames = SequenceFrames(mot_dataset, Chain(Redundancy(), EvenlySpaced(8)))  # doctest: +SKIP
    """

    def __init__(self, *selectors: FrameSelector) -> None:
        for selector in selectors:
            if not isinstance(selector, FrameSelector):
                raise TypeError(f"Chain: every stage must be a FrameSelector instance; got {selector!r}.")
        stages = tuple(
            stage
            for selector in selectors
            for stage in (selector.stages if isinstance(selector, Chain) else (selector,))
        )
        if not stages:
            raise ValueError("Chain: needs at least one stage; got none.")
        self.stages: tuple[FrameSelector, ...] = stages
        self.needs = reduce(or_, (stage.needs for stage in stages))
        self.invalidates = reduce(or_, (stage.invalidates for stage in stages))
        # Decided from the class, as the classification below must be fixed at construction.
        self._planning: tuple[bool, ...] = tuple(type(stage).plan is not FrameSelector.plan for stage in stages)
        _refuse_misplaced_adjacency(stages)
        self._boundary: int | None = _pass_boundary(stages, self._planning)
        self.two_pass = self._boundary is not None

    def __repr__(self) -> str:
        return f"Chain({', '.join(repr(stage) for stage in self.stages)})"

    def _plan_weighted(self, info: SequenceInfo) -> tuple[NDArray[np.intp], NDArray[np.float64]] | None:
        """Plan each stage over the survivors of the one before, carrying their source positions and masses."""
        return _composed(self.stages, info) if all(self._planning) else None

    def plan(self, info: SequenceInfo) -> NDArray[np.intp] | None:
        """Return the source positions the last stage keeps, when every stage plans."""
        weighted = self._plan_weighted(info)
        return None if weighted is None else weighted[0]

    def _resolved(
        self,
        stages: tuple[FrameSelector, ...],
        info: SequenceInfo,
        start: tuple[NDArray[np.intp], NDArray[np.float64]] | None = None,
    ) -> tuple[list[int], list[float]]:
        """Plan ``stages`` as :func:`_composed` does, refusing a stage that returns no plan."""
        planned = _composed(stages, info, start)
        if planned is None:
            raise ValueError(
                f"Chain: a stage of {self!r} that plans returned no plan for sequence {info.index}. A "
                "chain applies a stage that implements plan through it, so it must plan every sequence."
            )
        return planned[0].tolist(), planned[1].tolist()

    def _streaming(self, start: int, stop: int) -> int:
        """Return the first stage from ``start`` that does not plan, or ``stop``."""
        return next((k for k in range(start, stop) if not self._planning[k]), stop)

    def _streamed(
        self, start: int, stop: int, survivors: Iterator[_Survivor], info: SequenceInfo
    ) -> Iterator[_Survivor]:
        """Run the stages from ``start`` up to ``stop`` frame by frame, each over what the one before keeps."""
        # A streaming stage starts before its input count is known, so it is handed the source
        # sequence without its targets, and decides from neither. Each stage holds one frame, so
        # the chain holds at most one per stage.
        streamed = replace(info, target=None)
        for number, stage in enumerate(self.stages[start:stop], start=start + 1):
            survivors = _StageRun(number, stage, streamed).kept(survivors)
        return survivors

    def _driven(self, boundary: int, survivors: Iterator[_Survivor], info: SequenceInfo) -> list[_Survivor]:
        """Drive a ``two_pass`` stage over every survivor, then apply its verdicts as a stage run would.

        Each survivor is held without its frame once offered, so only positions, masses and
        factors outlive the walk. The verdicts are replayed in order through a stage run, which
        composes masses and merges factors as it does for a streaming stage.
        """
        stage = self.stages[boundary]
        streamed = replace(info, target=None)
        pixels = bool(stage.needs & FrameInput.PIXELS)
        held: list[_Survivor] = []

        def offered() -> Iterator[FrameCandidate]:
            for ordinal, survivor in enumerate(_settled(survivors)):
                held.append(_released(survivor))
                yield replace(
                    survivor.candidate,
                    sequence=streamed,
                    position=ordinal,
                    weight=cast(float, survivor.weight),
                    _allow_pixels=pixels,
                )

        candidates = offered()
        chosen = {verdict.position: verdict for verdict in stage.select(candidates)}
        # A stage may stop reading early. The frames it never saw still run to the last frame it kept.
        for _ in candidates:
            pass
        replayed = _StageRun(boundary + 1, stage, streamed).kept(
            iter(held), lambda frames: (chosen[frame.position] for frame in frames if frame.position in chosen)
        )
        return list(replayed)

    def _second_pass(
        self, survivors: Iterator[_Survivor], info: SequenceInfo
    ) -> Callable[[Iterator[FrameCandidate]], Iterator[FrameVerdict]]:
        """End the first pass at the boundary stage, and return what runs the stages after it on the second.

        Planning stages after the boundary resolve here from its survivors. The rest stream over
        the second walk's source frames, which the returned callable narrows to those survivors.
        """
        boundary = cast(int, self._boundary)
        if self._planning[boundary]:
            # Its input count and its survivors' targets are known only now, so it plans over them.
            kept, after = [_released(survivor) for survivor in _settled(survivors)], boundary
        else:
            kept, after = self._driven(boundary, survivors, info), boundary + 1
        start, stop = self._streaming(after, len(self.stages)), len(self.stages)
        positions = np.array([survivor.position for survivor in kept], dtype=np.intp)
        masses = np.array([float(survivor.weight) for survivor in kept], dtype=np.float64)
        taken, resolved = self._resolved(self.stages[after:start], info, (positions, masses))
        # By index into what reached here, so each keep of a frame kept twice carries its own mass and factors.
        carried = [(kept[k].position, mass, kept[k].factors) for k, mass in zip(taken, resolved, strict=True)]
        return lambda frames: _verdicts(self._streamed(start, stop, _arriving(frames, carried), info))

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Plan the stages that plan ahead of the first that does not, then run the rest frame by frame.

        Each verdict names the source frame the walk holds, and declares that frame's composed
        weight, which the streaming stages finish settling before
        :class:`~dataeval.data.SequenceFrames` records the frame.

        A chain with a stage that needs its whole input runs only up to that stage here, and
        yields nothing. What this generator returns runs the rest over the second walk: a private
        protocol with :class:`~dataeval.data.SequenceFrames`, whose only user is ``Chain``.
        """
        first = next(frames, None)
        if first is None:
            return None
        info = first.sequence
        frames = _prepended(first, frames)
        del first  # peeked only for the sequence, so not held past the stages' use of it
        end = len(self.stages) if self._boundary is None else self._boundary
        split = self._streaming(0, end)
        positions, masses = self._resolved(self.stages[:split], info)
        arriving = _arriving(frames, ((position, mass, {}) for position, mass in zip(positions, masses, strict=True)))
        survivors = self._streamed(split, end, arriving, info)
        if self._boundary is not None:
            # Returning a value from a generator is the protocol: SequenceFrames reads it off StopIteration.
            return self._second_pass(survivors, info)  # noqa: B901
        yield from _verdicts(survivors)
