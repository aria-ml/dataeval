"""Reading and writing the map from a factor's values to its codes.

The record itself is :class:`~dataeval.types.BinSpec` and
:class:`~dataeval.types.LevelSpec`. This module is the two things a record is *for*:
applying one to data it was not derived from, and moving one in and out of a file a
person can read.
"""

__all__ = []

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.types import BinSpec, LevelSpec

# Bumped when the on-disk shape changes in a way an older reader would misread. A
# descriptor is committed alongside code and read back months later, so it says which
# format it is rather than leaving a reader to guess.
DESCRIPTOR_VERSION = 1

FactorEncoding = BinSpec | LevelSpec

# JSON has no literal for an infinite float, and the values Python emits for one
# (``Infinity``) are not valid JSON that other tools will parse. Spelled as strings so the
# artifact stays readable by anything, and so a reviewer sees the word rather than a number
# that looks like a measurement.
_INFINITIES = {"-inf": float("-inf"), "inf": float("inf")}

# Stand-in key for a missing value inside the vocabulary index. ``NaN != NaN``, so a NaN
# cannot look itself up in a dict keyed by value: the lookup finds the slot by hash and
# then rejects it on equality, unless it happens to be the very same object. ``tolist()``
# builds fresh floats on every call, so it never is. One sentinel gives every spelling of
# missing one slot, which is also what the bin path does with ``BinSpec.missing_code``.
_MISSING_KEY = object()


def _index_key(value: Any) -> Any:
    """Key a value takes in the vocabulary index, collapsing every spelling of one value onto one slot.

    Two normalizations, both so that a vocabulary read back out of JSON still matches the
    values it was derived from. Every missing spelling -- ``NaN`` in an array, ``null`` in a
    descriptor -- takes the one sentinel slot. And a ``bytes`` value takes the slot of the
    string the renderer writes for it, since JSON has no byte string and a level read back
    as text would otherwise be a second level for the same value.
    """
    if value is None or (isinstance(value, float) and value != value):
        return _MISSING_KEY
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    return value


def apply_level_spec(
    data: NDArray[Any],
    spec: LevelSpec,
    *,
    strict: bool = False,
) -> tuple[NDArray[np.int64], LevelSpec]:
    """Encode values against a recorded vocabulary, appending anything it has not seen.

    This is where the append-only contract in :class:`~dataeval.types.LevelSpec` is
    actually executed. A value already in ``levels`` keeps the code it had; a value that is
    new takes the next one and goes on the end, *out* of sort order. Re-sorting is what
    ``np.unique`` does and is exactly what must not happen here — inserting ``bird`` ahead
    of ``cat`` would renumber every code above it, and every result computed under the old
    numbering would silently be describing different categories.

    Parameters
    ----------
    data : NDArray
        The values to encode.
    spec : LevelSpec
        The vocabulary to encode against.
    strict : bool, default False
        Whether a value the vocabulary does not contain is an error. Appending is the right
        default for extension — new data legitimately brings new categories — but a caller
        who declared a closed taxonomy wants to hear that the data left it, not to have the
        vocabulary quietly widened to fit.

    Returns
    -------
    tuple[NDArray[np.int64], LevelSpec]
        The codes, and the spec that describes them — the same object when nothing was
        appended, so an unchanged encoding stays identical rather than merely equal.

    Raises
    ------
    ValueError
        When ``strict`` and the data holds a value the vocabulary does not.
    """
    distinct, inverse = np.unique(data, return_inverse=True)
    # Once, not once per pass over it: ``tolist()`` rebuilds every scalar it returns, and
    # the identity of those scalars is what the NaN key above turns on.
    values = distinct.tolist()
    index = {_index_key(value): code for code, value in enumerate(spec.levels)}
    levels = list(spec.levels)
    if strict and (unseen := [value for value in values if _index_key(value) not in index]):
        raise ValueError(
            f"Values {sorted(unseen, key=repr)} are not in this factor's declared vocabulary "
            f"{list(spec.levels)}. Pass strict=False to admit them, or widen the declaration.",
        )
    for value in values:
        if (key := _index_key(value)) not in index:
            index[key] = len(levels)
            levels.append(value)
    codes = np.asarray([index[_index_key(value)] for value in values], dtype=np.int64)[inverse]
    grown = replace(spec, levels=tuple(levels)) if len(levels) != len(spec.levels) else spec
    return codes.reshape(data.shape), grown


def _as_plain(value: Any) -> Any:
    """Render one level as something JSON can hold.

    A record can be built from an array — ``factor_levels={"grade": np.arange(4)}`` is the
    natural spelling next to every other array-shaped argument — and a ``np.int64`` is not
    JSON-serializable. Unwrapping here rather than refusing the argument keeps the record
    writable however it was declared, and the digest reads through the same renderer, so
    an un-unwrapped level would have taken every bias evaluator down with it.

    Three shapes reach here that ``json`` cannot hold, and each has one honest spelling:

    - A NumPy scalar unwraps to the Python value it stands for.
    - A ``bytes`` level — a binary metadata column — becomes text. There is no byte string
      in JSON, and leaving it raised ``Object of type bytes is not JSON serializable`` out
      of :attr:`~dataeval.Metadata.encoding_digest`, which every bias evaluator reads
      *after* computing its result. :func:`_index_key` normalizes the same way, so the
      restored text level still matches the bytes the column holds.
    - A missing value becomes ``null``. ``json`` writes a bare ``NaN`` token for it, which
      is not JSON any other reader accepts — the same defect the infinite edges above are
      spelled as words to avoid, and reachable from any numeric factor with a gap in it.
    """
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    return None if isinstance(value, float) and value != value else value


def _level_from_json(level: Any) -> Any:
    """Read one level back, restoring the missing value ``null`` stands for."""
    return float("nan") if level is None else level


def _edge_to_json(edge: float) -> float | str | None:
    """Render one edge, spelling an infinity as a word."""
    if edge == float("-inf"):
        return "-inf"
    if edge == float("inf"):
        return "inf"
    return _as_plain(edge)  # `null` where a caller handed in a NaN, which is not an edge


def _edge_from_json(edge: float | str | None) -> float:
    """Read one edge back, accepting the words and the numbers alike."""
    if edge is None:
        return float("nan")
    if isinstance(edge, str):
        if edge not in _INFINITIES:
            raise ValueError(f"Bin edge {edge!r} is not a number; the only names accepted are 'inf' and '-inf'.")
        return _INFINITIES[edge]
    return float(edge)


def encoding_to_json(encodings: Mapping[str, FactorEncoding | None]) -> str:
    """Render an encoding as the artifact a person reviews.

    JSON with sorted keys and a fixed indent, so that the same encoding produces the same
    bytes and a change to one factor shows up in a diff as a change to one factor. That is
    the whole reason this is not a binary member inside the metadata archive: a descriptor
    is policy, committed alongside code and read in a pull request, and a format that
    cannot be read by eye defeats the review it exists for.

    Factors carrying no record are omitted rather than written as null — a descriptor names
    the factors it has something to say about, and the rest are encoded normally.
    """
    document = {"version": DESCRIPTOR_VERSION, "factors": encoding_to_mapping(encodings)}
    # ``allow_nan=False`` so that a value the renderer above has no spelling for stops the
    # write instead of putting a bare ``NaN`` or ``Infinity`` token into a file the whole
    # point of which is that other tools can read it.
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def encoding_to_mapping(encodings: Mapping[str, FactorEncoding | None]) -> dict[str, dict[str, Any]]:
    """Render the records as plain data, without committing to a file format.

    Shared by the reviewable descriptor and by the metadata archive's manifest, so the two
    cannot describe the same encoding differently — the descriptor is policy and the
    archive is state, but what a record *is* has to be one answer.
    """
    factors: dict[str, dict[str, Any]] = {}
    for name, spec in sorted(encodings.items()):
        if isinstance(spec, BinSpec):
            factors[name] = {
                "kind": "bins",
                "edges": [_edge_to_json(edge) for edge in spec.edges],
                "provenance": spec.provenance,
                "method": spec.method,
            }
        elif isinstance(spec, LevelSpec):
            factors[name] = {
                "kind": "levels",
                "levels": [_as_plain(level) for level in spec.levels],
                "provenance": spec.provenance,
            }
    return factors


def encoding_from_mapping(factors: Mapping[str, Mapping[str, Any]]) -> dict[str, FactorEncoding]:
    """Read records back from plain data. The reverse of :func:`encoding_to_mapping`."""
    return {name: _one_from_json(name, entry) for name, entry in factors.items()}


def _one_from_json(name: str, entry: Mapping[str, Any]) -> FactorEncoding:
    """Read one factor's entry, saying which factor is wrong when it is."""
    kind = entry.get("kind")
    if kind == "bins":
        return BinSpec(
            edges=tuple(_edge_from_json(edge) for edge in entry["edges"]),
            provenance=entry.get("provenance", "edges"),
            method=entry.get("method"),
        )
    if kind == "levels":
        levels = tuple(_level_from_json(level) for level in entry["levels"])
        return LevelSpec(levels=levels, provenance=entry.get("provenance", "declared"))
    raise ValueError(f"Encoding for factor {name!r} has kind {kind!r}; expected 'bins' or 'levels'.")


def encoding_from_json(text: str) -> dict[str, FactorEncoding]:
    """Read a descriptor back into records.

    The reverse of :func:`encoding_to_json`, and the half that makes the loop close: what
    was exported has to be accepted as input, or locking an encoding in and applying it to
    the next dataset is two different vocabularies again.
    """
    document = json.loads(text)
    version = document.get("version")
    if version != DESCRIPTOR_VERSION:
        raise ValueError(
            f"Encoding descriptor is version {version!r}; this DataEval reads version {DESCRIPTOR_VERSION}.",
        )
    return encoding_from_mapping(document.get("factors", {}))


def bins_to_mapping(bins: Mapping[str, int | Sequence[float]]) -> dict[str, int | list[float | str | None]]:
    """Render a ``continuous_factor_bins`` declaration as plain data.

    Kept as the caller wrote it rather than resolved into a :class:`~dataeval.types.BinSpec`,
    because a *count* is not a cut: it says how finely to divide, and where the edges land
    is still read off the values. Resolving it would need the binning pass to have run,
    which is the one thing an unread instance has not done. Restoring the count instead
    re-derives the same edges, since the archive holds the same values it was derived from.
    """
    return {
        name: spec if isinstance(spec, int) else [_edge_to_json(float(edge)) for edge in spec]
        for name, spec in bins.items()
    }


def bins_from_mapping(bins: Mapping[str, Any]) -> dict[str, int | list[float]]:
    """Read a ``continuous_factor_bins`` declaration back. The reverse of :func:`bins_to_mapping`."""
    return {
        name: spec if isinstance(spec, int) else [_edge_from_json(edge) for edge in spec] for name, spec in bins.items()
    }


def declared_levels(factor_levels: Mapping[str, Sequence[Any]]) -> dict[str, LevelSpec]:
    """Turn a caller's vocabularies into records, with the codes fixed before any data is seen.

    The categorical counterpart to ``continuous_factor_bins``: a fixed taxonomy, or a schema
    from upstream, said once so that every dataset encoded against it agrees. Order is the
    caller's — code ``i`` means ``levels[i]`` — which is what lets two datasets share an
    alphabet without either having been structured first.
    """
    return {
        name: LevelSpec(levels=tuple(_as_plain(level) for level in levels), provenance="declared")
        for name, levels in factor_levels.items()
    }


def read_encoding(source: str | Path | Mapping[str, FactorEncoding]) -> dict[str, FactorEncoding]:
    """Take an encoding from wherever the caller has one.

    A path is read as a descriptor; a mapping is taken as already-parsed records, which is
    what ``md.encoding()`` returns, so one metadata's encoding can be handed to the next
    without a file in between.
    """
    if isinstance(source, str | Path):
        return encoding_from_json(Path(source).read_text(encoding="utf-8"))
    invalid = sorted(name for name, spec in source.items() if not isinstance(spec, BinSpec | LevelSpec))
    if invalid:
        raise TypeError(f"Encoding entries for {invalid} are not a BinSpec or a LevelSpec.")
    return dict(source)
