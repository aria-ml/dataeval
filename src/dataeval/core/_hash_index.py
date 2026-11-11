"""Hamming-radius search over packed perceptual hashes.

Perceptual hashes answer *how similar* two images are, in bits: a re-encode, a resize or a
mild color shift moves a handful of bits, an unrelated image moves about half of them. Using
them therefore means asking for every pair within some radius — a question a dictionary keyed
on the digest cannot answer at all, since it can only find pairs at radius zero.

The obstacle is scale. An all-pairs scan is quadratic, which for a hundred thousand hashes is
some five billion comparisons. :func:`hash_neighbors` avoids that with **multi-index hashing**:
split each code into ``radius + 1`` disjoint substrings, and observe that two codes differing in
at most ``radius`` bits must agree *exactly* on at least one of them, because ``radius`` differing
bits cannot fall into ``radius + 1`` disjoint parts without leaving one untouched. Grouping on
each substring in turn therefore produces a candidate set that provably contains every true
neighbor, and a popcount over those candidates alone decides the rest.

Recall is exact — this is a blocking scheme, not a sketch, and it discards no true pair.

References
----------
[1] Fast Search in Hamming Space with Multi-Index Hashing.
    Norouzi, M., Punjani, A., & Fleet, D. J. (2012). CVPR.
    https://www.cs.toronto.edu/~norouzi/research/papers/multi_index_hashing.pdf
"""

__all__ = []

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, NamedTuple, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from dataeval._experimental import warn_experimental_once
from dataeval._log import get_logger

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

_logger = get_logger(__name__)

_Batch: TypeAlias = tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]
"""One run of candidate pairs and their distances: ``(left, right, distance)``."""

_Classes: TypeAlias = tuple[NDArray[np.uint64], NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]
"""Equivalence classes of identical codes: ``(unique, members, starts, sizes)``."""


class _Labeled(NamedTuple):
    """Equivalence classes split by partition label into blocks, one run of ``members`` per (class, label).

    ``members`` lists the same positions as :func:`_classes`, class by class at the same ``starts``,
    but ordered inside each class by label and then by position. The block table lists the runs in
    ascending (class, label) order: ``block_class``, ``block_label`` (a label's rank, never the
    caller's value), ``block_starts`` into ``members`` and ``block_sizes``. ``histogram`` is the
    sparse classes-by-labels matrix of block sizes, so no dense table of every class and label is
    ever built, and ``groups`` is its column count, the number of distinct labels. ``sole_label``
    holds each class's label when all its members share one, else -1.
    """

    members: NDArray[np.intp]
    block_class: NDArray[np.intp]
    block_label: NDArray[np.intp]
    block_starts: NDArray[np.intp]
    block_sizes: NDArray[np.intp]
    histogram: "csr_matrix"
    sole_label: NDArray[np.intp]
    groups: int


_WORD_BITS = 64
_HEX_PER_WORD = _WORD_BITS // 4

_BRUTE_FORCE_CELLS = 65_536
"""Below this many ``n * n`` cells the blocked all-pairs scan wins on fixed overhead alone.

A few hundred codes is a couple of vectorized operations either way, and at that size the scan
skips the grouping the index would build. Above it the choice turns on the bands rather than on
the count — see :func:`_prefer_scan`. The two paths return identical results, so this only ever
trades one runtime for another.
"""

_MIN_BAND_YIELD = 1
"""Bucket-count-to-band-count ratio below which the bands stop paying for themselves.

A band ``w`` bits wide sorts the codes into ``2**w`` buckets, so each band proposes about
``n / 2**w`` candidates per code and all ``bands`` of them propose ``bands * n / 2**w``. Once
that reaches ``n`` -- that is, once ``2**w`` falls to ``bands`` -- the index is enumerating as
many candidates as the scan compares outright, and the scan's tighter inner loop wins.
"""

_MAX_PAIRS = 10_000_000
"""Default ceiling on the pairs :func:`hash_neighbors` will return before refusing.

Ten million pairs is a few hundred megabytes of result and several times that in transients.
Real perceptual-hash datasets produce orders of magnitude fewer; reaching this means the input
holds large sets of identical hashes, whose pair count is quadratic in their own size and for
which :func:`hash_groups` is almost always the question actually being asked.
"""

_EDGE_FLUSH = 500_000
"""Edges buffered before connectivity folds them into component labels and drops them."""

_BRUTE_FORCE_BLOCK_CELLS = 4_000_000
"""Cells per block in the all-pairs scan, which is what bounds its peak memory.

A fixed *row* count does not: the scan is also the path a radius at or above the digest width
takes, and that path has no size ceiling above it, so ``rows * n`` grows without bound with the
input. Budgeting cells instead holds every transient the block allocates -- the XOR, its
popcount gather and the reduction -- to a constant, whatever ``n`` is.
"""

_POPCOUNT8: NDArray[np.uint8] = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
    axis=1, dtype=np.uint8
)
"""Set-bit count of every byte value, so a popcount is a gather over the byte view of a XOR."""


class HashNeighborResult(TypedDict):
    """
    Pairs of hashes within a Hamming radius of one another.

    Attributes
    ----------
    pairs : NDArray[np.intp]
        Shape ``(M, 2)``. Each row is a pair of positions into the input, with the smaller
        first, sorted lexicographically. Every pair appears exactly once.
    distances : NDArray[np.intp]
        Shape ``(M,)``. Hamming distance of each pair, aligned with ``pairs``.
    """

    pairs: NDArray[np.intp]
    distances: NDArray[np.intp]


class HashGroupResult(TypedDict):
    """
    Connected groups of hashes linked by being within a Hamming radius of one another.

    Attributes
    ----------
    groups : Sequence[NDArray[np.intp]]
        One array of input positions per group, each sorted ascending and holding at least two
        members. Groups are ordered by their smallest member.
    labels : NDArray[np.intp]
        Shape ``(N,)``. The index into ``groups`` of each input position, or ``-1`` for a
        position in no group -- one with no neighbor, or one excluded as invalid.
    """

    groups: Sequence[NDArray[np.intp]]
    labels: NDArray[np.intp]


def _empty_result() -> HashNeighborResult:
    """Return the well-typed empty answer, so callers never special-case a no-neighbor input."""
    return HashNeighborResult(
        pairs=np.empty((0, 2), dtype=np.intp),
        distances=np.empty(0, dtype=np.intp),
    )


def _popcount(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.intp]:
    """Hamming distance between corresponding rows of two equal-shaped code arrays.

    Reduces over the *byte* view of the XOR rather than over 64-bit words: a 256-entry lookup
    table is both faster than a SWAR reduction in numpy and immune to the uint64/Python-int
    promotion rules that silently turn a bit-twiddling expression into float64.
    """
    xor = np.asarray(a ^ b)
    if xor.size == 0:
        return np.zeros(xor.shape[:-1], dtype=np.intp)
    return _POPCOUNT8[xor.view(np.uint8)].sum(axis=-1, dtype=np.intp)


def _band_bounds(bits: int, bands: int) -> list[tuple[int, int]]:
    """Partition ``bits`` into ``bands`` contiguous ranges, as evenly as the count allows.

    The remainder is spread one bit at a time across the leading bands rather than dumped on
    the last one, which keeps every band's bucket count within a factor of two of every other's.
    An uneven partition costs the pigeonhole argument nothing — it needs the bands disjoint and
    exhaustive, not equal.
    """
    base, remainder = divmod(bits, bands)
    widths = [base + 1] * remainder + [base] * (bands - remainder)
    bounds: list[tuple[int, int]] = []
    start = 0
    for width in widths:
        bounds.append((start, width))
        start += width
    return bounds


def _band_keys(codes: NDArray[np.uint64], start: int, width: int) -> NDArray[np.uint64]:
    """Extract the ``width`` bits beginning at global bit ``start`` from every row.

    Bit 0 is the most significant bit of word 0, matching the big-endian packing
    :func:`pack_hashes` produces, so a band's bits are contiguous in the same order the hex
    digest reads. ``width`` never exceeds 64 (see :func:`_band_count`), so a band spans at most
    two words and the second is only consulted when the first cannot supply the whole window.
    """
    word, offset = divmod(start, _WORD_BITS)
    window = codes[:, word]
    if offset:
        available = _WORD_BITS - offset
        window = window << np.uint64(offset)
        if width > available:
            # The low `offset` bits vacated by the shift are exactly the next word's leading
            # bits; without them the window would read zeros where the code continues.
            window = window | (codes[:, word + 1] >> np.uint64(available))
    return window >> np.uint64(_WORD_BITS - width)


def _band_count(bits: int, radius: int) -> int:
    """Return the band count: enough for the pigeonhole guarantee, and each within a word.

    ``radius + 1`` is what correctness requires. ``ceil(bits / 64)`` is what
    :func:`_band_keys` requires, and taking the larger of the two costs nothing — more bands
    than the pigeonhole minimum still guarantees one matches, it only narrows each band.
    """
    return max(radius + 1, -(-bits // _WORD_BITS))


def _grouped(keys: NDArray[np.uint64]) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Sort positions by key and return ``(order, starts, sizes)`` for the runs of equal keys.

    Groups are returned **largest first**. ``starts`` and ``sizes`` are only offsets into
    ``order``, so reordering them costs nothing and buys :func:`_pairs_at_offset` the property
    it needs: at any separation, the groups still large enough to contribute are a prefix.
    """
    order = np.argsort(keys, kind="stable").astype(np.intp)
    ordered = keys[order]
    starts = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1]))).astype(np.intp)
    sizes = np.diff(np.concatenate((starts, [len(order)]))).astype(np.intp)
    by_size = np.argsort(-sizes, kind="stable")
    return order, starts[by_size], sizes[by_size]


def _pairs_at_offset(
    order: NDArray[np.intp],
    starts: NDArray[np.intp],
    sizes: NDArray[np.intp],
    offset: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Every within-group pair whose members sit ``offset`` apart in sorted order.

    Enumerating a group's pairs by separation rather than by combination is what keeps this
    memory-safe: a group of size *s* is walked in *s* steps of at most *s* pairs each, so peak
    memory tracks the largest group rather than its pair count. Sweeping ``offset`` from 1 to
    ``sizes[0] - 1`` yields each pair exactly once.

    ``sizes`` must be descending, which makes the contributing groups a prefix found by binary
    search. Rescanning every group at every separation instead is what turns one oversized
    group into quadratic *scanning* work on top of the pairs it legitimately produces.
    """
    eligible = int(np.searchsorted(-sizes, -offset, side="left"))
    if not eligible:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    counts = sizes[:eligible] - offset
    total = int(counts.sum())
    base = np.repeat(starts[:eligible], counts)
    within = np.arange(total, dtype=np.intp) - np.repeat(np.cumsum(counts) - counts, counts)
    return order[base + within], order[base + within + offset]


def _sweep_offsets(
    order: NDArray[np.intp],
    starts: NDArray[np.intp],
    sizes: NDArray[np.intp],
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Walk every separation the largest group reaches, yielding each one's non-empty pairs.

    ``sizes`` is descending, so its first entry bounds the sweep: no group contributes a pair at
    a separation the largest one cannot reach. Together with :func:`_pairs_at_offset` this yields
    each within-group pair exactly once.
    """
    for offset in range(1, int(sizes[0]) if len(sizes) else 1):
        left, right = _pairs_at_offset(order, starts, sizes, offset)
        if len(left):
            yield left, right


def _classes(codes: NDArray[np.uint64]) -> _Classes:
    """Collapse identical codes into equivalence classes.

    Returns ``(unique, members, starts, sizes)``, where ``members[starts[c]:starts[c] + sizes[c]]``
    lists the input positions holding class *c*'s code, and ``sizes`` is descending.

    Every search below runs over ``unique`` rather than over the input, which matters precisely
    because a duplicate detector is pointed at data with duplicates in it. A thousand copies of
    one blank frame is one code here; left uncollapsed it is a thousand-member group in every
    band, rediscovering the same half-million pairs once per band before the dedupe throws all
    but one copy away.
    """
    unique, inverse = np.unique(codes, axis=0, return_inverse=True)
    inverse = np.ravel(inverse)
    sizes = np.bincount(inverse, minlength=len(unique)).astype(np.intp)
    by_size = np.argsort(-sizes, kind="stable")
    # Renumber so class order is descending by size, then lay members out in that same order.
    rank = np.empty(len(unique), dtype=np.intp)
    rank[by_size] = np.arange(len(unique), dtype=np.intp)
    sizes = sizes[by_size]
    members = np.argsort(rank[inverse], kind="stable").astype(np.intp)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.intp)
    return unique[by_size], members, starts, sizes


def _labeled_classes(members: NDArray[np.intp], sizes: NDArray[np.intp], labels: NDArray[np.intp]) -> _Labeled:
    """Split each class of :func:`_classes` into one contiguous block per partition label.

    Each class keeps its place in ``members``, so the ``starts`` of :func:`_classes` still hold.
    ``labels`` holds one label per position ``members`` lists. They are replaced by their ranks
    first, so a label is only ever compared and never used as an index: negative, huge and
    non-contiguous labels behave exactly as their ranks do.
    """
    from scipy.sparse import csr_matrix

    _, rank = np.unique(labels, return_inverse=True)
    rank = np.ravel(rank).astype(np.intp)
    owner = np.repeat(np.arange(len(sizes), dtype=np.intp), sizes)
    # Sorted by (class, label, position): every key is distinct, so no tie is left to the sort.
    members = members[np.lexsort((members, rank[members], owner))]
    label = rank[members]
    opens = np.concatenate(([True], (owner[1:] != owner[:-1]) | (label[1:] != label[:-1])))
    block_starts = np.flatnonzero(opens).astype(np.intp)
    block_sizes = np.diff(np.append(block_starts, len(members))).astype(np.intp)
    block_class, block_label = owner[block_starts], label[block_starts]
    groups = int(rank.max()) + 1
    histogram = csr_matrix((block_sizes.astype(np.int64), (block_class, block_label)), shape=(len(sizes), groups))
    per_class = np.bincount(block_class, minlength=len(sizes))
    sole_label = np.where(per_class == 1, block_label[np.cumsum(per_class) - per_class], -1).astype(np.intp)
    return _Labeled(members, block_class, block_label, block_starts, block_sizes, histogram, sole_label, groups)


def _pair_runs(
    members: NDArray[np.intp],
    left_starts: NDArray[np.intp],
    left_sizes: NDArray[np.intp],
    right_starts: NDArray[np.intp],
    right_sizes: NDArray[np.intp],
    distances: NDArray[np.intp],
) -> _Batch:
    """Pair every member of one run of ``members`` with every member of another, row by row.

    Row ``i`` pairs the ``left_sizes[i]`` members from ``left_starts[i]`` with the
    ``right_sizes[i]`` members from ``right_starts[i]``, all at ``distances[i]``.
    """
    counts = left_sizes * right_sizes
    total = int(counts.sum())
    if not total:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    flat = np.arange(total, dtype=np.intp) - np.repeat(np.cumsum(counts) - counts, counts)
    row, column = np.divmod(flat, np.repeat(right_sizes, counts))
    left = members[np.repeat(left_starts, counts) + row]
    right = members[np.repeat(right_starts, counts) + column]
    return left, right, np.repeat(distances, counts)


def _cross_class_pairs(
    labeled: _Labeled,
    starts: NDArray[np.intp],
    sizes: NDArray[np.intp],
    pairs: NDArray[np.intp],
    distances: NDArray[np.intp],
) -> _Batch:
    """Turn pairs of classes into the pairs of input positions whose labels differ.

    Each label block of a pair's first class pairs with the second class outside that label: the
    run of the second class before its block of the same label and the run after it, or the whole
    class when it holds no member with that label. A same-label pair is skipped over, never made.
    """
    first, second = pairs[:, 0], pairs[:, 1]
    per_class = np.bincount(labeled.block_class, minlength=len(sizes)).astype(np.intp)
    # One row per (class pair, label block of its first class); a class's blocks are contiguous.
    count = per_class[first]
    row = np.repeat(np.arange(len(pairs), dtype=np.intp), count)
    block = (
        np.repeat((np.cumsum(per_class) - per_class)[first], count)
        + np.arange(len(row), dtype=np.intp)
        - np.repeat(np.cumsum(count) - count, count)
    )
    other = second[row]
    # The block table is sorted by (class, label), so the same label's block in the other class is
    # found by one binary search, and is absent when the key there differs.
    groups = labeled.groups
    table = labeled.block_class.astype(np.int64) * groups + labeled.block_label
    wanted = other.astype(np.int64) * groups + labeled.block_label[block]
    match = np.minimum(np.searchsorted(table, wanted), len(table) - 1)
    shared = table[match] == wanted
    begin, end = starts[other], starts[other] + sizes[other]
    cut = np.where(shared, labeled.block_starts[match], begin)
    resume = np.where(shared, labeled.block_starts[match] + labeled.block_sizes[match], begin)
    left_starts, left_sizes = np.tile(labeled.block_starts[block], 2), np.tile(labeled.block_sizes[block], 2)
    right_starts = np.concatenate((begin, resume))
    right_sizes = np.concatenate((cut - begin, end - resume))
    return _pair_runs(labeled.members, left_starts, left_sizes, right_starts, right_sizes, np.tile(distances[row], 2))


def _within_class_pairs(
    members: NDArray[np.intp],
    starts: NDArray[np.intp],
    sizes: NDArray[np.intp],
    labeled: _Labeled | None = None,
) -> _Batch:
    """Every pair of input positions sharing one code, all at distance zero.

    With ``labeled``, only the pairs whose labels differ, and only those are made: a class's
    members are ordered by label, so a block pairs with the rest of its class after it.
    """
    if labeled is not None:
        after = labeled.block_starts + labeled.block_sizes
        rest = (starts + sizes)[labeled.block_class] - after
        zero = np.zeros(len(after), dtype=np.intp)
        return _pair_runs(labeled.members, labeled.block_starts, labeled.block_sizes, after, rest, zero)
    lefts: list[NDArray[np.intp]] = []
    rights: list[NDArray[np.intp]] = []
    for left, right in _sweep_offsets(members, starts, sizes):
        lefts.append(left)
        rights.append(right)
    left = np.concatenate(lefts) if lefts else np.empty(0, dtype=np.intp)
    right = np.concatenate(rights) if rights else np.empty(0, dtype=np.intp)
    return left, right, np.zeros(len(left), dtype=np.intp)


def _expand(
    members: NDArray[np.intp],
    starts: NDArray[np.intp],
    sizes: NDArray[np.intp],
    pairs: NDArray[np.intp],
    distances: NDArray[np.intp],
    labeled: _Labeled | None = None,
) -> _Batch:
    """Turn pairs of *classes* into pairs of input positions, one per member combination.

    With ``labeled``, one per combination whose labels differ (see :func:`_cross_class_pairs`).
    """
    if labeled is not None:
        return _cross_class_pairs(labeled, starts, sizes, pairs, distances)
    first, second = pairs[:, 0], pairs[:, 1]
    return _pair_runs(members, starts[first], sizes[first], starts[second], sizes[second], distances)


def _dedupe(
    left: NDArray[np.intp],
    right: NDArray[np.intp],
    distances: NDArray[np.intp],
    count: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Canonicalize each pair as ``(min, max)``, drop repeats, and sort lexicographically.

    A pair can agree on several bands and so be proposed several times; the distance is a
    function of the pair alone, so any one copy carries the right value.
    """
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    # One integer per pair, which is what makes the dedupe a single sort rather than a join.
    # count is bounded by the input length, so count**2 stays inside int64 for any real input.
    keys = low.astype(np.int64) * np.int64(count) + high.astype(np.int64)
    # np.unique returns its values already sorted, so the first-occurrence indices it hands back
    # are in ascending key order too -- which for this key is exactly lexicographic on the pair.
    _, order = np.unique(keys, return_index=True)
    return np.stack((low[order], high[order]), axis=1).astype(np.intp), distances[order].astype(np.intp)


def _brute_force(codes: NDArray[np.uint64], radius: int) -> Iterator[_Batch]:
    """All-pairs scan, yielding one row block of upper-triangle pairs within ``radius`` at a time.

    Each block compares its rows only against the codes at or after it, which is the half of the
    matrix the upper triangle actually needs. Scanning the full width and discarding the lower
    half afterwards costs twice the popcount and twice the peak memory for the same answer.
    """
    count = len(codes)
    rows_per_block = max(1, _BRUTE_FORCE_BLOCK_CELLS // max(1, count))
    for begin in range(0, count, rows_per_block):
        end = min(begin + rows_per_block, count)
        block = _popcount(codes[begin:end, None, :], codes[None, begin:, :])
        rows, columns = np.nonzero(block <= radius)
        keep = rows < columns
        rows, columns = rows[keep], columns[keep]
        if len(rows):
            yield (
                (rows + begin).astype(np.intp),
                (columns + begin).astype(np.intp),
                block[rows, columns].astype(np.intp),
            )


def _pruned(
    prune: Callable[[NDArray[np.intp], NDArray[np.intp]], NDArray[np.bool_]] | None,
    left: NDArray[np.intp],
    right: NDArray[np.intp],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Apply a caller's candidate filter, if it supplied one."""
    if prune is None or not len(left):
        return left, right
    keep = prune(left, right)
    return left[keep], right[keep]


def _multi_index(
    codes: NDArray[np.uint64],
    bits: int,
    radius: int,
    prune: Callable[[NDArray[np.intp], NDArray[np.intp]], NDArray[np.bool_]] | None = None,
) -> Iterator[_Batch]:
    """Multi-index hashing, yielding each band-and-separation batch of verified pairs.

    Yielding rather than accumulating is what bounds memory. The *answer* can be far larger than
    the input — low-entropy codes put much of the input in one band bucket and genuinely do have
    quadratically many near pairs — so a caller that only needs connectivity must be able to
    consume each batch and drop it. Batches may repeat a pair found in more than one band.
    """
    for start, width in _band_bounds(bits, _band_count(bits, radius)):
        order, starts, sizes = _grouped(_band_keys(codes, start, width))
        for candidate_left, candidate_right in _sweep_offsets(order, starts, sizes):
            left, right = _pruned(prune, candidate_left, candidate_right)
            if not len(left):
                continue
            distance = _popcount(codes[left], codes[right])
            keep = distance <= radius
            if keep.any():
                yield left[keep], right[keep], distance[keep]


def _checked_width(packed: int, bits: int | None, caller: str) -> int:
    """Resolve the significant width of a code, defaulting to every bit the packing holds.

    Unchecked, a width past the packed one indexes a word that is not there and the search fails
    with an ``IndexError`` from deep inside the banding rather than at its own door.
    """
    width = packed if bits is None else int(bits)
    if not 1 <= width <= packed:
        raise ValueError(f"{caller}: bits must be between 1 and the {packed} bit(s) codes actually holds; got {width}.")
    return width


def _participants(
    codes: NDArray[np.uint64], valid: NDArray[np.bool_] | None, caller: str
) -> tuple[NDArray[np.intp], NDArray[np.uint64]]:
    """Drop the positions marked invalid, keeping the map back to where each survivor came from."""
    if valid is None:
        return np.arange(len(codes), dtype=np.intp), codes

    checked = np.asarray(valid, dtype=np.bool_)
    if checked.ndim != 1 or len(checked) != len(codes):
        raise ValueError(
            f"{caller}: valid must be a 1D array of length {len(codes)}; got shape {checked.shape}. "
            "It indexes the same hashes as codes and must agree."
        )
    positions = np.flatnonzero(checked).astype(np.intp)
    return positions, codes[positions]


def _checked_partition(partition: object, count: int, caller: str) -> NDArray[np.intp]:
    """Return a caller's partition labels as ``intp``, one per code, refusing any other shape or dtype.

    Labels are only ever compared, so a cast that wraps a huge unsigned label still keeps every
    distinct label distinct.
    """
    checked = np.asarray(partition)
    if checked.ndim != 1 or len(checked) != count:
        got = f"shape {checked.shape} for {count} codes"
    elif not np.issubdtype(checked.dtype, np.integer):
        got = f"dtype {checked.dtype}"
    else:
        return checked.astype(np.intp)
    raise ValueError(f"{caller}: partition must be a 1D integer array with one label per code; got {got}.")


def _prepare(
    codes: NDArray[np.uint64],
    radius: int,
    valid: NDArray[np.bool_] | None,
    bits: int | None,
    caller: str,
) -> tuple[NDArray[np.intp], NDArray[np.uint64], int]:
    """Validate the arguments both entry points share and drop the positions that take no part."""
    if radius < 0:
        raise ValueError(f"{caller}: radius must be non-negative; got {radius}.")

    codes = np.ascontiguousarray(codes, dtype=np.uint64)
    if codes.ndim != 2 or codes.shape[1] < 1:
        raise ValueError(f"{caller}: codes must be a 2D (N, W) array with W >= 1; got shape {codes.shape}.")

    width = _checked_width(codes.shape[1] * _WORD_BITS, bits, caller)
    positions, codes = _participants(codes, valid, caller)
    return positions, codes, width


def _prefer_scan(count: int, width: int, radius: int) -> bool:
    """Whether the all-pairs scan beats multi-index hashing for this code count, width and radius.

    Size alone does not decide it. The scan is quadratic in ``count`` whatever the radius, while
    the index's cost turns on how finely its bands cut: ``radius + 1`` bands over ``width`` bits
    are ``width / (radius + 1)`` bits each, and a band that narrow sorts the codes into only
    ``2 ** that`` buckets. Choosing on ``count`` alone therefore hands the scan the case it loses
    worst -- a large input at the small radii perceptual hashing actually uses, where wide bands
    make the index tens of times faster.

    Three things send the search to the scan:

    - a radius at or above the digest width, where every pair is a neighbor and no blocking
      scheme can narrow anything, and which is also the only radius the pigeonhole argument
      cannot supply enough bands for;
    - an input small enough that the index's grouping costs more than the whole comparison;
    - bands too narrow to pay for themselves, i.e. fewer buckets per band than there are bands.
    """
    if radius >= width:
        return True
    if count * count <= _BRUTE_FORCE_CELLS:
        return True
    bands = _band_count(width, radius)
    # Clamped because a band spans at most a word, and 2**64 dwarfs any band count regardless.
    buckets = 1 << min(width // bands, _WORD_BITS)
    return buckets <= bands * _MIN_BAND_YIELD


def _search(
    unique: NDArray[np.uint64],
    width: int,
    radius: int,
    prune: Callable[[NDArray[np.intp], NDArray[np.intp]], NDArray[np.bool_]] | None = None,
) -> Iterator[_Batch]:
    """Yield batches of ``(left, right, distance)`` over pairs of *distinct* codes within ``radius``.

    Distances are all at least 1 by construction — two codes in different classes differ
    somewhere — so a caller adds the zero-distance within-class relationships itself, in whatever
    form it needs them. Batches may repeat a pair; a caller that cares deduplicates.

    ``prune`` is consulted on each batch of candidates *before* they are verified, and keeps the
    ones it returns True for. A caller tracking connectivity uses it to drop pairs whose answer
    cannot change anything, which is what makes low-entropy input — a static camera, where much
    of the input lands in one band bucket — tractable rather than merely bounded.
    """
    if radius == 0 or len(unique) < 2:
        return

    use_scan = _prefer_scan(len(unique), width, radius)
    _logger.debug(
        "hash search: %d distinct code(s), %d bits, radius %d, using %s",
        len(unique),
        width,
        radius,
        "all-pairs scan" if use_scan else "multi-index hashing",
    )
    yield from _brute_force(unique, radius) if use_scan else _multi_index(unique, width, radius, prune)


def _merge_components(labels: NDArray[np.intp], left: NDArray[np.intp], right: NDArray[np.intp]) -> NDArray[np.intp]:
    """Fold a buffer of edges into a component labeling and return the composed labeling.

    Composing after every flush is what keeps connectivity linear in space: the edges seen so far
    collapse to at most one label per node and are then discarded, so an input with quadratically
    many near pairs costs time proportional to them but never memory.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    size = int(labels.max()) + 1
    graph = coo_matrix(
        (np.ones(len(left), dtype=np.int8), (labels[left], labels[right])),
        shape=(size, size),
    )
    _, component = connected_components(graph, directed=False)
    return component.astype(np.intp)[labels]


def _connect(unique: NDArray[np.uint64], width: int, radius: int) -> NDArray[np.intp]:
    """Label each class with its connected component, folding edges in as they are found."""
    labels = np.arange(len(unique), dtype=np.intp)
    buffered_left: list[NDArray[np.intp]] = []
    buffered_right: list[NDArray[np.intp]] = []
    buffered = 0

    def unconnected(left: NDArray[np.intp], right: NDArray[np.intp]) -> NDArray[np.bool_]:
        """Candidates whose classes already share a component cannot change the answer.

        Consulted before verification, so near-identical content — a static camera, a stalled
        feed — pays a label comparison per redundant candidate instead of a popcount. Nothing is
        lost: connectivity is exactly what the surviving edges decide.
        """
        return labels[left] != labels[right]

    for left, right, _ in _search(unique, width, radius, unconnected):
        buffered_left.append(left)
        buffered_right.append(right)
        buffered += len(left)
        if buffered >= _EDGE_FLUSH:
            labels = _merge_components(labels, np.concatenate(buffered_left), np.concatenate(buffered_right))
            buffered_left, buffered_right, buffered = [], [], 0
    if buffered:
        labels = _merge_components(labels, np.concatenate(buffered_left), np.concatenate(buffered_right))
    return labels


def _consolidate(
    batches: Sequence[_Batch],
    count: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Concatenate and deduplicate a run of candidate batches."""
    return _dedupe(
        np.concatenate([batch[0] for batch in batches]),
        np.concatenate([batch[1] for batch in batches]),
        np.concatenate([batch[2] for batch in batches]),
        count,
    )


def _within_count(counts: NDArray[np.int64], labeled: _Labeled | None = None) -> int:
    """Pairs implied by the classes themselves: every combination inside each set of equal codes.

    With ``labeled``, only those whose labels differ, ``C(n, 2) - sum_g C(n_g, 2)`` per class: the
    combinations inside one (class, label) block share their label.
    """
    within = int((counts * (counts - 1) // 2).sum())
    if labeled is not None:
        blocks = labeled.block_sizes.astype(np.int64)
        within -= int((blocks * (blocks - 1) // 2).sum())
    return within


def _refuse(
    counts: NDArray[np.int64],
    radius: int,
    total: int,
    implied: int,
    max_pairs: int,
    labeled: _Labeled | None = None,
) -> None:
    """Raise, naming the reason the answer is quadratic and the function that is not."""
    within = _within_count(counts, labeled)
    if labeled is None:
        kind, spanning, sets = "pairs", "", int((counts > 1).sum())
        offer = "pass partition if only the pairs crossing between groups are needed, "
    else:
        kind, spanning, offer = "cross-group pairs", " that span more than one group", ""
        sets = int((labeled.sole_label < 0).sum())
    raise ValueError(
        f"hash_neighbors: radius {radius} over {total} hashes yields at least {implied} {kind}, "
        f"above max_pairs={max_pairs}. {within} of them come from {sets} "
        f"set(s) of identical hashes{spanning}, which are quadratic in their own size however they are "
        "found. Use dataeval.core.hash_groups for connected sets instead, which is bounded by "
        f"the number of hashes, {offer}or raise max_pairs if the pairs themselves are needed."
    )


def _check_budget(
    counts: NDArray[np.int64],
    class_pairs: NDArray[np.intp],
    within: int,
    radius: int,
    total: int,
    max_pairs: int,
    labeled: _Labeled | None = None,
) -> None:
    """Refuse if the pairs these classes imply would outgrow the budget.

    With ``labeled``, two classes imply ``n_a * n_b - sum_g n_a,g * n_b,g`` pairs: those whose labels differ.
    """
    across = int((counts[class_pairs[:, 0]] * counts[class_pairs[:, 1]]).sum()) if len(class_pairs) else 0
    if labeled is not None and len(class_pairs):
        histogram = labeled.histogram
        across -= int(histogram[class_pairs[:, 0]].multiply(histogram[class_pairs[:, 1]]).sum())
    if within + across > max_pairs:
        _refuse(counts, radius, total, within + across, max_pairs, labeled)


def _class_pairs(
    unique: NDArray[np.uint64],
    width: int,
    radius: int,
    counts: NDArray[np.int64],
    max_pairs: int,
    total: int,
    labeled: _Labeled | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Every pair of distinct codes within ``radius``, consolidated, staying inside a budget.

    The budget is checked against the *deduplicated* class pairs each time they are consolidated.
    Consolidating on overrun rather than refusing outright matters because a pair can be proposed
    by several bands: a raw overrun is not yet an answer that is too large.
    """
    within = _within_count(counts, labeled)
    found: list[_Batch] = []
    raw = 0
    for batch in _search(unique, width, radius):
        if labeled is not None:
            # Two classes whose members all hold one same label imply no reported pair. Carried, they
            # would cost nothing against the budget, so once they outnumbered it every later batch
            # would consolidate them all again.
            left, right, distance = batch
            sole = labeled.sole_label
            keep = (sole[left] != sole[right]) | (sole[left] < 0)
            batch = left[keep], right[keep], distance[keep]
        found.append(batch)
        raw += len(batch[0])
        if raw > max_pairs:
            pairs, distances = _consolidate(found, len(unique))
            _check_budget(counts, pairs, within, radius, total, max_pairs, labeled)
            found, raw = [(pairs[:, 0], pairs[:, 1], distances)], len(pairs)
    if not found:
        return np.empty((0, 2), dtype=np.intp), np.empty(0, dtype=np.intp)
    pairs, distances = _consolidate(found, len(unique))
    _check_budget(counts, pairs, within, radius, total, max_pairs, labeled)
    return pairs, distances


def pack_hashes(hashes: Sequence[str]) -> tuple[NDArray[np.uint64], NDArray[np.bool_]]:
    """
    Pack hex-encoded hashes into fixed-width 64-bit words for bitwise comparison.

    Parameters
    ----------
    hashes : Sequence[str]
        Hex digests, as :func:`~dataeval.core.phash`, :func:`~dataeval.core.dhash` and
        :func:`~dataeval.core.xxhash` return them. Every non-empty digest must have the same
        length. The empty string marks a hash that was never computed and is packed as zeros.

    Returns
    -------
    codes : NDArray[np.uint64]
        Shape ``(N, W)`` where ``W`` is the number of 64-bit words a digest occupies, most
        significant word first. A digest that does not fill its last word is padded with
        trailing zero bits, uniformly across every row, which leaves all Hamming distances
        unchanged.
    valid : NDArray[np.bool_]
        Shape ``(N,)``. False where the input was the empty string.

    Raises
    ------
    ValueError
        If the non-empty digests are not all the same length, or if any is not valid hex.

    See Also
    --------
    :func:`~dataeval.core.hash_neighbors` : Find every pair of codes within a Hamming radius

    Notes
    -----
    The empty string is how :func:`~dataeval.core.image_stats` reports a region it could not
    measure — an out-of-bounds box, an image its boxes cover completely, or a band group the
    datum cannot supply. Those must never be compared to one another: they would all pack
    identically and so read as mutual duplicates, which is a claim about regions there was no
    data for. ``valid`` is how a caller excludes them, and
    :func:`~dataeval.core.hash_neighbors` does so by default.

    Examples
    --------
    >>> from dataeval.core import pack_hashes
    >>> codes, valid = pack_hashes(["ff00ff00ff00ff00", "ff00ff00ff00ff01", ""])
    >>> codes.shape
    (3, 1)
    >>> valid
    array([ True,  True, False])
    """
    count = len(hashes)
    lengths = {len(digest) for digest in hashes if digest}
    if len(lengths) > 1:
        raise ValueError(
            f"pack_hashes: hashes must all be the same length; got lengths {sorted(lengths)}. "
            "Mixing hash methods in one call compares digests that do not describe the same bits."
        )

    valid = np.fromiter((bool(digest) for digest in hashes), dtype=np.bool_, count=count)
    if not lengths:
        # Nothing was measured anywhere. One zero word keeps the shape well-formed for callers
        # that index it before checking `valid`.
        return np.zeros((count, 1), dtype=np.uint64), valid

    hex_length = lengths.pop()
    words = -(-hex_length * 4 // _WORD_BITS)
    row_hex = words * _HEX_PER_WORD
    filler = "0" * (row_hex - hex_length)
    blank = "0" * row_hex

    try:
        raw = bytes.fromhex("".join((digest + filler) if digest else blank for digest in hashes))
    except ValueError as err:
        raise ValueError(f"pack_hashes: hashes must be hex-encoded strings; {err}") from err

    packed = np.frombuffer(raw, dtype=np.uint8).reshape(count, words * 8)
    return packed.view(">u8").astype(np.uint64), valid


def hash_groups(
    codes: NDArray[np.uint64],
    radius: int,
    *,
    valid: NDArray[np.bool_] | None = None,
    bits: int | None = None,
) -> HashGroupResult:
    """
    Group hashes into connected sets, where a link is being within a Hamming radius.

    Parameters
    ----------
    codes : NDArray[np.uint64]
        Shape ``(N, W)`` packed hashes, as :func:`~dataeval.core.pack_hashes` returns.
    radius : int
        Maximum Hamming distance, in bits, for two hashes to be linked. ``0`` links only
        identical hashes. Must not be negative.
    valid : NDArray[np.bool_] or None, default None
        Shape ``(N,)``. Positions marked False take no part and land in no group. When None,
        every position participates.
    bits : int or None, default None
        Significant bits per code. When None, all ``W * 64`` are used, which is correct for any
        digest that fills its words. Pass the true width for a digest padded to a word boundary,
        so the padding is not spent on bands of its own -- a band lying wholly in the padding
        reads the same value from every code, putting the whole input in one bucket. Must be
        between 1 and ``W * 64``.

    Returns
    -------
    HashGroupResult
        TypedDict containing:

        - **groups** (*Sequence[NDArray[np.intp]]*) -- one sorted array of input positions per
          group of two or more, ordered by smallest member.
        - **labels** (*NDArray[np.intp]*) -- shape ``(N,)``, each position's index into
          ``groups``, or ``-1`` for a position in no group.

    Raises
    ------
    ValueError
        If ``radius`` is negative, if ``codes`` is not a two-dimensional array of at least one
        word, if ``bits`` is outside the width ``codes`` holds, or if ``valid`` is not a
        one-dimensional array matching ``codes`` in length.

    See Also
    --------
    :func:`~dataeval.core.hash_neighbors` : The individual pairs, where the links themselves matter
    :func:`~dataeval.core.pack_hashes` : Pack hex digests into the expected form

    Notes
    -----
    **Prefer this to :func:`~dataeval.core.hash_neighbors` whenever the question is which hashes
    belong together.** Its output is bounded by ``N``, where the pairwise answer is bounded by
    ``N**2``: a thousand copies of one image are one group of a thousand here and half a million
    pairs there. Duplicate-heavy data is what a duplicate detector is pointed at, so that is the
    ordinary case rather than the pathological one.

    Grouping is transitive, which the pairwise relation is not: hashes 5 bits apart and 5 bits
    apart again land in one group at ``radius=5`` while being 10 bits apart themselves. That is
    the intended reading -- a chain of near-duplicates is one redundant set -- but it means a
    group is not a claim that every member is within ``radius`` of every other, and a large
    ``radius`` can chain unrelated content together.

    Examples
    --------
    >>> from dataeval.core import hash_groups, pack_hashes
    >>> codes, valid = pack_hashes(["ff00ff00ff00ff00", "ff00ff00ff00ff01", "0123456789abcdef", "0123456789abcdef"])
    >>> result = hash_groups(codes, radius=1, valid=valid)
    >>> [group.tolist() for group in result["groups"]]
    [[0, 1], [2, 3]]
    >>> result["labels"]
    array([0, 0, 1, 1])
    """
    total = len(codes)
    positions, codes, width = _prepare(codes, radius, valid, bits, "hash_groups")
    labels = np.full(total, -1, dtype=np.intp)
    if len(codes) < 2:
        return HashGroupResult(groups=[], labels=labels)

    if radius >= width:
        # No two codes can differ in more bits than the digest holds, so this radius links all of
        # them into one group. Answering it directly keeps a degenerate query from paying for a
        # scan that is quadratic in the input to rediscover a foregone conclusion.
        group = np.sort(positions)
        labels[group] = 0
        _logger.debug("hash_groups: radius %d spans the full %d-bit width; one group", radius, width)
        return HashGroupResult(groups=[group], labels=labels)

    # Union over *classes*, not positions: every member of a class is already linked to every
    # other at distance zero, so the class is one node and the search's pairs are its edges.
    unique, members, starts, sizes = _classes(codes)
    class_labels = _connect(unique, width, radius)

    # `members` lists positions class by class, so repeating each class's label by its size lines
    # the labels up with it without a join.
    member_label = class_labels[np.repeat(np.arange(len(sizes), dtype=np.intp), sizes)]
    order = np.argsort(member_label, kind="stable")
    ordered_label, ordered_members = member_label[order], positions[members[order]]
    boundary = np.flatnonzero(np.concatenate(([True], ordered_label[1:] != ordered_label[:-1]))).astype(np.intp)
    extent = np.diff(np.concatenate((boundary, [len(order)])))

    groups = [
        np.sort(ordered_members[begin : begin + size]) for begin, size in zip(boundary, extent, strict=True) if size > 1
    ]
    groups.sort(key=lambda group: int(group[0]))
    for index, group in enumerate(groups):
        labels[group] = index

    _logger.debug("hash_groups: %d group(s) over %d code(s) at radius %d", len(groups), len(codes), radius)
    return HashGroupResult(groups=groups, labels=labels)


def hash_neighbors(
    codes: NDArray[np.uint64],
    radius: int,
    *,
    valid: NDArray[np.bool_] | None = None,
    bits: int | None = None,
    max_pairs: int = _MAX_PAIRS,
    partition: NDArray[np.integer] | None = None,
) -> HashNeighborResult:
    """
    Find every pair of hashes within a Hamming radius of one another.

    Parameters
    ----------
    codes : NDArray[np.uint64]
        Shape ``(N, W)`` packed hashes, as :func:`~dataeval.core.pack_hashes` returns.
    radius : int
        Maximum Hamming distance, in bits, for a pair to be reported. ``0`` reports only
        identical codes. Must not be negative.
    valid : NDArray[np.bool_] or None, default None
        Shape ``(N,)``. Positions marked False take no part and appear in no pair. When None,
        every position participates.
    bits : int or None, default None
        Significant bits per code. When None, all ``W * 64`` are used, which is correct for any
        digest that fills its words -- every hash DataEval produces does. Pass the true width for
        a digest padded to a word boundary, so the padding is not spent on a band of its own --
        a band lying wholly in the padding reads the same value from every code, putting the
        whole input in one bucket. Must be between 1 and ``W * 64``.
    max_pairs : int, default 10000000
        Refuse rather than return more pairs than this. With ``partition``, only the pairs
        whose labels differ count. See Notes.
    partition : NDArray[np.integer] or None, default None
        Shape ``(N,)``. One integer label per position of ``codes``, including the positions
        ``valid`` excludes. When given, only the pairs whose two labels differ are reported:
        exactly the pairs of the call without it, less those inside one label, at the same
        distances. Labels are only compared, so they need not be contiguous. Two identical
        hashes under different labels are a pair at distance ``0``. A pair inside one label is
        never expanded, so a large set of identical hashes under a single label is not counted
        against ``max_pairs``. Passing it raises
        :class:`~dataeval.exceptions.ExperimentalWarning` once per process.

    Returns
    -------
    HashNeighborResult
        TypedDict containing:

        - **pairs** (*NDArray[np.intp]*) -- shape ``(M, 2)``, each row a pair of positions into
          ``codes`` with the smaller first, sorted lexicographically.
        - **distances** (*NDArray[np.intp]*) -- shape ``(M,)``, the Hamming distance of each pair.

    Raises
    ------
    ValueError
        If ``radius`` is negative, if ``codes`` is not a two-dimensional array of at least one
        word, if ``bits`` is outside the width ``codes`` holds, if ``valid`` is not a
        one-dimensional array matching ``codes`` in length, if ``partition`` is not a
        one-dimensional integer array of the same length, or if the answer would hold more
        than ``max_pairs`` pairs.

    See Also
    --------
    :func:`~dataeval.core.hash_groups` : The connected sets, when membership is the question
    :func:`~dataeval.core.pack_hashes` : Pack hex digests into the expected form
    :func:`~dataeval.core.hamming_distance` : Distance between two digests, one pair at a time

    Notes
    -----
    Recall is exact: every pair within ``radius`` is reported, and no pair outside it. Two
    strategies produce that same answer, chosen on size *and radius* and logged when the choice
    is made.

    Most inputs use **multi-index hashing** -- each code is split into ``radius + 1`` disjoint
    substrings, and two codes within ``radius`` bits must agree exactly on at least one of them,
    since ``radius`` differing bits cannot touch ``radius + 1`` disjoint parts. Grouping on each
    substring yields a candidate set that provably contains every true pair, which a popcount
    then filters.

    A blocked all-pairs scan takes over where that stops paying: an input of a few hundred codes, where
    the grouping costs more than the comparison, and a radius large enough that the bands are cut
    too fine to exclude anything. Multi-index hashing narrows as the radius grows -- bands shrink,
    so more of the input shares each one -- and at a radius approaching half the digest width it
    degenerates into the scan it replaced, which is not a defect but a statement about the
    question: at that radius most pairs *are* neighbors.

    **The answer can be quadratic even when the search is not.** *k* copies of one image are
    ``k * (k - 1) / 2`` pairs however cheaply they are found, so a few thousand duplicates --
    ordinary in the data a duplicate detector is aimed at -- is tens of millions of rows and
    gigabytes of memory. ``max_pairs`` is counted exactly rather than estimated, and counted
    *before the pairs are expanded*, which is the allocation that would dominate; the search
    itself still runs, and its own working set is bounded but not free. With ``partition`` the
    count is of the pairs whose labels differ, so a set of identical hashes under one label costs
    nothing against it; the search is not narrowed, so distinct hashes under one label are still
    compared. Exceeding the budget raises rather than truncating: a silently shortened neighbor
    list reads as a complete one. When the question is which hashes belong together rather than
    which pairs link them, :func:`~dataeval.core.hash_groups` answers it in space bounded by ``N``.

    Typical radii for a 64-bit perceptual hash follow the bands documented on
    :func:`~dataeval.core.hamming_distance`: ``0`` for identical, ``1-5`` for very similar,
    ``6-10`` for possibly similar.

    Examples
    --------
    >>> from dataeval.core import hash_neighbors, pack_hashes
    >>> codes, valid = pack_hashes(["ff00ff00ff00ff00", "ff00ff00ff00ff01", "0123456789abcdef"])
    >>> result = hash_neighbors(codes, radius=1, valid=valid)
    >>> result["pairs"]
    array([[0, 1]])
    >>> result["distances"]
    array([1])

    With ``partition``, only the pairs whose labels differ are reported. The pair inside label 0
    is left out, and the identical hashes under different labels are found at distance ``0``:

    >>> import numpy as np
    >>> codes, valid = pack_hashes(["ff00ff00ff00ff00", "ff00ff00ff00ff01", "ff00ff00ff00ff00", "0123456789abcdef"])
    >>> result = hash_neighbors(codes, radius=1, valid=valid, partition=np.array([0, 0, 1, 1]))
    >>> result["pairs"]
    array([[0, 2],
           [1, 2]])
    >>> result["distances"]
    array([0, 1])
    """
    # The labels index the caller's positions, so they are checked against the input before `valid` drops any.
    labels = None if partition is None else _checked_partition(partition, len(codes), "hash_neighbors")
    positions, codes, width = _prepare(codes, radius, valid, bits, "hash_neighbors")
    if labels is not None:
        warn_experimental_once("hash_neighbors(partition=...)")
        labels = labels[positions]
    if len(codes) < 2:
        return _empty_result()

    unique, members, starts, sizes = _classes(codes)
    labeled = None if labels is None else _labeled_classes(members, sizes, labels)
    counts = sizes.astype(np.int64)
    # Counted, never estimated, and counted before the expansion allocates anything.
    within = _within_count(counts, labeled)
    if within > max_pairs:
        _refuse(counts, radius, len(codes), within, max_pairs, labeled)
    pairs, distances = _class_pairs(unique, width, radius, counts, max_pairs, len(codes), labeled)

    left, right, zero = _within_class_pairs(members, starts, sizes, labeled)
    near = _expand(members, starts, sizes, pairs, distances, labeled)
    left = np.concatenate((left, near[0]))
    right = np.concatenate((right, near[1]))
    distances = np.concatenate((zero, near[2]))
    if not len(left):
        return _empty_result()

    # Within-class and cross-class pairs are disjoint by construction — two distinct codes are
    # never zero bits apart — so this sorts and canonicalizes rather than deduplicating.
    result_pairs, result_distances = _dedupe(left, right, distances, len(codes))
    _logger.debug("hash_neighbors: %d pair(s) within radius %d", len(result_pairs), radius)
    return HashNeighborResult(pairs=positions[result_pairs], distances=result_distances)
