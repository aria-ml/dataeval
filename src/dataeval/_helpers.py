__all__ = []

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from typing_extensions import TypeIs

from dataeval.exceptions import DeprecatedWarning
from dataeval.protocols import AnyMetadataLike, LabelsLike, MetadataLike, ValuedMetadataLike
from dataeval.types._factors import BinSpec, LevelSpec

IGNORE_KEYS = {"self", "config", "__class__"}


@runtime_checkable
class _LegacyMetadataLike(Protocol):
    """The pre-1.1 :class:`~dataeval.protocols.MetadataLike` shape, with ``is_discrete``.

    Lives here rather than beside the protocol it shadows because :func:`is_metadata_like`
    is its only consumer, so the whole deprecation — this class, the branch that checks it,
    and the warning — is deleted from one file in 1.2.

    A container written against the 1.0 protocol has to be *recognized* before it can be
    told it is out of date. Failing the ``isinstance`` check instead would report it as not
    being metadata at all, which says nothing about what to fix.
    """

    @property
    def factor_names(self) -> Sequence[str]:
        """Names of the metadata factors."""
        ...

    @property
    def factor_data(self) -> NDArray[np.int64]:
        """Metadata factors in array of shape (n_samples, n_factors)."""
        ...

    @property
    def class_labels(self) -> NDArray[np.intp]:
        """Flat array of class labels with one entry per target/detection."""
        ...

    @property
    def is_discrete(self) -> Sequence[bool]:
        """Whether each factor is discrete (True) or continuous (False)."""
        ...


def get_overrides(local_vars: dict[str, Any], exclude: set[str] | None = None) -> dict[str, Any]:
    """
    Extract explicit arguments from locals() to create a config override dictionary.

    Removes 'self', 'config', and any variable that is None.
    """
    # 1. Standard things to always ignore in __init__
    exclude = IGNORE_KEYS | (exclude or set())
    return {key: value for key, value in local_vars.items() if key not in exclude and value is not None}


def apply_config(obj: Any, config: BaseModel, exclude: set[str] | None = None) -> None:
    """Apply attributes onto obj from config, excluding specified keys."""
    exclude = IGNORE_KEYS | (exclude or set())
    obj.config = config
    for key, value in config.model_dump().items():
        if key not in exclude:
            setattr(obj, key, value)


def is_metadata_like(candidate: Any) -> TypeIs[MetadataLike]:
    """Whether ``candidate`` satisfies :class:`~dataeval.protocols.MetadataLike`.

    Answers from the concrete :class:`~dataeval.Metadata` class first, which is what
    keeps the common case both cheap and safe. ``MetadataLike`` is a
    :func:`~typing.runtime_checkable` protocol, and ``isinstance`` against one consults
    every member the protocol names. Python 3.12+ resolves those with
    :func:`inspect.getattr_static` and touches nothing, but 3.10 and 3.11 use a plain
    ``hasattr``, which *calls* each property getter — so for a
    :class:`~dataeval.Metadata` the check structures and bins the entire dataset from
    inside a type test. Worse, at a view above :attr:`~dataeval.Metadata.label_level`
    :attr:`~dataeval.Metadata.class_labels` deliberately raises, and ``hasattr`` only
    swallows :class:`AttributeError`, so the :class:`ValueError` escapes ``isinstance``
    rather than it returning False.

    Recognizing by class the one type that reaches here most avoids both. A
    third-party container still pays the protocol check, which costs nothing because
    its members are plain attributes.

    Parameters
    ----------
    candidate : Any
        Object to test.

    Returns
    -------
    TypeIs[MetadataLike]
        True when ``candidate`` is a :class:`~dataeval.Metadata`, or otherwise
        satisfies the protocol. Spelled as a :class:`~typing.TypeIs` rather than a plain
        ``bool`` so that it narrows a union in *both* branches, exactly as the
        ``isinstance`` call it replaces did — every caller here is a dispatch that hands
        the other branch to :class:`~dataeval.Metadata`, which does not accept a
        ``MetadataLike``.
    """
    # Imported here rather than at module scope: dataeval.types.Evaluator imports this
    # module and Metadata imports dataeval.types, so a module-level import closes the
    # cycle. By the time this is called the module is already imported.
    from dataeval import Metadata

    if isinstance(candidate, (Metadata, MetadataLike)):
        return True
    if isinstance(candidate, _LegacyMetadataLike):
        _warn_missing_is_binned(type(candidate))
        return True
    return False


def is_any_metadata_like(candidate: Any) -> TypeIs[AnyMetadataLike]:
    """Whether ``candidate`` carries factors in *either* representation.

    What :class:`~dataeval.bias.Balance` dispatches on, since it is the one evaluator that
    reads codes or measured values depending on
    :attr:`~dataeval.bias.Balance.factor_source`. A container providing only
    ``factor_values`` is metadata as far as it is concerned, and answering False would send
    it to :class:`~dataeval.Metadata`'s constructor as though it were a dataset.

    Returns
    -------
    TypeIs[AnyMetadataLike]
        True when ``candidate`` satisfies either metadata protocol.
    """
    return is_metadata_like(candidate) or isinstance(candidate, ValuedMetadataLike)


def is_labels_like(candidate: Any) -> TypeIs[LabelsLike]:
    """Whether ``candidate`` carries class labels, whatever else it does or does not.

    What the evaluators that never look at a factor dispatch on. Requiring the full
    metadata protocol from them asked for members they do not read, so a container built
    for :class:`~dataeval.scope.Representation` had to declare factors it has none of.

    The concrete class is answered first for the same reason as in
    :func:`is_metadata_like`: on Python 3.10 and 3.11 an ``isinstance`` against a
    :func:`~typing.runtime_checkable` protocol *calls* the property, and
    :attr:`~dataeval.Metadata.class_labels` deliberately raises above the label level --
    a :class:`ValueError`, which ``hasattr`` does not swallow.

    Returns
    -------
    TypeIs[LabelsLike]
        True when ``candidate`` is a :class:`~dataeval.Metadata`, or otherwise provides
        ``class_labels``.
    """
    from dataeval import Metadata

    return isinstance(candidate, (Metadata, LabelsLike))


# Types already told about `is_binned`. A container's author writes one class and reuses
# it, so the class is the unit the message is about; warning per instance would repeat it
# for every row of a loop without saying anything new.
_WARNED_LEGACY_METADATA: set[type] = set()


def _warn_missing_is_binned(candidate_type: type) -> None:
    """Tell a pre-1.1 container's author to add ``is_binned``, once per class.

    Raised from :func:`is_metadata_like` rather than from the point of use, because that is
    the one call every evaluator makes before touching a metadata object, and it runs on the
    user's own line.
    """
    if candidate_type in _WARNED_LEGACY_METADATA:
        return
    _WARNED_LEGACY_METADATA.add(candidate_type)
    warnings.warn(
        f"{candidate_type.__name__} implements MetadataLike with `is_discrete` but no "
        "`is_binned`. Support for `is_discrete` is removed in v1.2.0. Add an `is_binned` "
        "property returning, per factor and aligned with `factor_names`, whether that "
        "factor's entries in `factor_data` are bin indices (True) rather than codes "
        "standing for the values themselves (False).\n"
        "`is_discrete` is being read in its place, which is right for every factor except "
        "a discrete numeric one you binned anyway -- reported as discrete, but its codes "
        "cover ranges. That factor is currently scored against a ceiling it does not have, "
        "which moves its values in `Balance`'s `factors` output.",
        DeprecatedWarning,
        stacklevel=3,
    )


def _get_item_indices(metadata: MetadataLike) -> Sequence[int]:
    """Get item indices from metadata, generating default if not available."""
    item_indices = getattr(metadata, "item_indices", None)
    if item_indices is not None:
        return item_indices
    return list(range(len(metadata.class_labels)))


def _get_index2label(metadata: MetadataLike) -> dict[int, str]:
    """Get index2label mapping, generating default if not available."""
    index2label = getattr(metadata, "index2label", None)
    if index2label:
        return dict(index2label)
    return {int(i): str(i) for i in np.unique(metadata.class_labels)}


class LabelAxis(NamedTuple):
    """The variable a class-conditional statistic conditions on.

    Attributes
    ----------
    values : NDArray[np.intp]
        One densely numbered code per row, in the metadata's own row order.
    names : Mapping[int, str]
        Display name for each code, for the ``class_name`` column of an output.
    label : str
        What the axis is called, e.g. ``"class_label"`` or ``"weather"``.
    excluded : tuple[int, ...]
        Indices of the factor columns serving as the axis, which the caller must drop
        from the factors it analyses — a factor left in place correlates perfectly with
        itself and reports 1.0.
    """

    values: NDArray[np.intp]
    names: Mapping[int, str]
    label: str
    excluded: tuple[int, ...]


def _recorded_encodings(metadata: MetadataLike, names: Sequence[str]) -> dict[str, BinSpec | LevelSpec | None]:
    """Read each named factor's recorded encoding, where the container keeps one.

    :class:`~dataeval.Metadata` records how every factor's values became codes; a bare
    :class:`~dataeval.protocols.MetadataLike` carries codes and nothing about their
    origin. Same optional-member degradation as :func:`_get_item_indices` and
    :func:`_get_index2label`, and for the same reason: the protocol stays four members.

    ``_factor_info`` rather than the public property, which warns about a level rename on
    the paths that put a :class:`~dataeval.types.FactorInfo` in a user's hands. Naming a
    group is not one of those.
    """
    factor_info = getattr(metadata, "_factor_info", None)
    if factor_info is None:
        return {}
    wanted = set(names)
    return {name: info.encoding for name, info in factor_info.items() if name in wanted}


# The window where fixed notation is the readable choice. ``%g`` switches to an exponent at
# 1e6, which is exactly where it starts hiding the difference between neighbouring edges;
# above 1e16 a float64 has no digits left below the exponent, so the value really is
# scientific and writing it out in full would be sixteen true digits and a tail of noise.
_FIXED_FLOOR, _FIXED_CEIL = 1e6, 1e16


def _edge_format(edges: Sequence[float]) -> str:
    """Pick one format for every edge in a cut, so no two bins print the same name.

    ``%g`` alone carries six significant figures, which suits most factors and leaves
    too little for the ones with large magnitudes. A capture time in epoch
    milliseconds sits near 1.787e12, so every bound of every bin rendered ``1.78711e+12``
    and an eight-bin cut produced labels a reader cannot tell apart -- and those labels key
    :attr:`~dataeval.bias.ParityOutput.insufficient_data` and name the groups of a ``label=``
    axis, so the output that exists to say *which subset to collect more of* said nothing.

    Two regimes:

    - **Large magnitudes** write out in full, with as many decimals as it takes to keep the
      finite edges distinct. This is where the exponent does the damage and where the digits
      it hides are the informative ones -- epoch times, pixel counts, byte sizes.
    - **Everything else** takes the fewest significant figures that keep the finite edges
      distinct, starting at six so an ordinary factor's labels are exactly what they were.
      Raising precision only as far as distinctness needs it avoids the opposite failure,
      where ``0.1`` renders as ``0.10000000000000001``.

    One format for the whole cut rather than one per edge, so that neighbouring bounds are
    written to the same precision and an interval reads as an interval. :func:`_distinguish`
    still guards the result: two edges can round to the same text under any precision this
    is willing to reach.
    """
    finite = sorted({edge for edge in edges if np.isfinite(edge)})
    if not finite:
        return ".6g"
    if _FIXED_FLOOR <= max(abs(edge) for edge in finite) < _FIXED_CEIL:
        return _distinct_format(finite, "f", range(7), ".6f")
    return _distinct_format(finite, "g", range(6, 18), ".17g")


def _distinct_format(finite: Sequence[float], kind: str, widths: Iterable[int], fallback: str) -> str:
    """First width in ``widths`` that renders every edge differently, or ``fallback``."""
    for width in widths:
        if len({f"{edge:.{width}{kind}}" for edge in finite}) == len(finite):
            return f".{width}{kind}"
    return fallback


def _edge_label(low: float, high: float, fmt: str = ".6g") -> str:
    """Name the interval ``[low, high)`` as a reader would write it.

    An infinite bound is stated as the open comparison it is -- ``< 12.4`` rather than
    ``[-inf, 12.4)`` -- because the infinity is an artifact of needing a bound, not a value
    anyone measured. Both finite, and the half-open interval is written out, so that a
    reader can see which side each boundary belongs to.

    Both infinite, and there is no comparison left to state: the bin holds every observed
    value, which is what a single-bin spec means. ``"clusters"`` emits one on a factor whose
    values are tied enough that no cut survives, and ``< inf`` would read as a bound rather
    than as the absence of one.
    """
    if np.isneginf(low) and np.isposinf(high):
        return "all"
    if np.isneginf(low):
        return f"< {high:{fmt}}"
    if np.isposinf(high):
        return f">= {low:{fmt}}"
    return f"[{low:{fmt}}, {high:{fmt}})"


def _bin_name(spec: BinSpec, code: int, fmt: str = ".6g") -> str:
    """Name one bin from the edges that produced it."""
    edges = spec.edges
    if code == spec.missing_code:
        return "missing"
    if code == 0:
        # Reachable only where the caller supplied an edge list that does not open at -inf:
        # digitizing cannot return 0 against a -inf first edge, since nothing is below it.
        return f"< {edges[0]:{fmt}}"
    if code >= len(edges):
        # Digitizing is right-open, so +inf falls past a +inf outer edge rather than into
        # the top bin. It is an observed extreme, not a missing value.
        return "inf" if np.isposinf(edges[-1]) else f">= {edges[-1]:{fmt}}"
    return _edge_label(edges[code - 1], edges[code], fmt)


def _names_from_bin_spec(spec: BinSpec, uniques: NDArray[Any]) -> dict[int, str]:
    """Name each bin from the record rather than from the rows it happens to hold.

    Naming a bin after its contents made the label move with the sample -- the same policy
    over a different draw printed a different name -- and, worse, hid a declared cutoff
    from its own label: ``{"temp_c": [-inf, 0.0, inf]}`` rendered as ``[-40, -0.3]``, with
    nothing saying that zero was where the meaning was.

    A descending edge list gets its codes back. ``np.digitize`` reverses the comparison for
    one -- code ``i`` means ``edges[i] <= x < edges[i - 1]`` -- so the arithmetic here would
    name the wrong interval, with nothing in the label to show it. The code is what is
    left to name it by, which is what a container with no record gets for the same reason.
    """
    # An empty edge list places no cut, so there is no interval to name a code after -- the
    # same honest fallback a descending list gets. `_reachable_bins` guards the same case.
    if not spec.edges or any(later < earlier for earlier, later in zip(spec.edges, spec.edges[1:], strict=False)):
        return {int(code): str(int(code)) for code in uniques}
    # One format for the whole cut, chosen from all of its edges: a per-edge choice would
    # render neighbouring bounds to different precisions and print `[1.787e+15, 1787011240000000)`.
    fmt = _edge_format(spec.edges)
    return {int(code): _bin_name(spec, int(code), fmt) for code in uniques}


def _names_from_level_spec(spec: LevelSpec, uniques: NDArray[Any]) -> dict[int, str]:
    """Name each code by the value it stands for.

    Exact, and available without reaching the raw column -- which is what lets a factor
    name its own codes from the record alone.
    """
    levels = spec.levels
    return {int(code): str(levels[int(code)]) if int(code) < len(levels) else str(code) for code in uniques}


def has_own_alphabet(metadata: MetadataLike, indices: Sequence[int]) -> list[bool]:
    """Say, per factor, whether its set of values belongs to the factor or to the binning.

    A factor that was digitized keeps one code per distinct value, so its alphabet is the
    factor's own: a category, a count, a rating. A factor that was **binned** has an
    alphabet that came out of where the cuts fell, and :class:`~dataeval.Metadata` derives
    the number of those cuts from the data rather than taking it as a setting — so the
    alphabet's size is a property of the draw, not of the variable.

    Two callers ask this same question for different reasons, which is why it is one
    function rather than two. :func:`resolve_label_axis` asks in order to *name* a group:
    a code covering a range cannot be named after one of its members. :class:`.Balance`
    asks in order to decide whether the factor's entropy is a legitimate **ceiling** to
    divide a pairwise association by. Both reduce to whether one code stands for one value.

    Parameters
    ----------
    metadata : MetadataLike
        Metadata the factors were read from.
    indices : Sequence[int]
        Positions into :attr:`~dataeval.Metadata.factor_names` of the factors to answer
        for. Positions rather than names because both callers hold a *subset* of the
        factors — the label axis's own columns, or the ones left after dropping them — and
        the protocol's sequences are aligned with the full list.

    Returns
    -------
    list[bool]
        True where one code stands for one value, aligned with ``indices``.

    Notes
    -----
    Answered by ``is_binned``, which :class:`~dataeval.protocols.MetadataLike` requires as
    of 1.1. A container written against the older protocol has only ``is_discrete``, which
    is read instead; :func:`is_metadata_like` has already warned its author by this point.
    The two disagree for a discrete numeric factor binned for carrying more levels than the
    sample supports: it reports ``is_discrete=True`` while its codes cover ranges like any
    other binned factor, so it is credited with a ceiling it does not have.

    A factor beyond the end of either sequence is treated as having its own alphabet, which
    is the conservative answer: it keeps the entropy ceiling and so cannot inflate a
    reported association.
    """
    binned = getattr(metadata, "is_binned", None)
    if binned is not None:
        flags = [not bool(value) for value in binned]
    else:
        flags = [bool(value) for value in getattr(metadata, "is_discrete", ())]
    return [flags[index] if index < len(flags) else True for index in indices]


class FactorChannel(NamedTuple):
    """One representation of the factors, with what each column turned out to be.

    Attributes
    ----------
    data : NDArray[np.float64]
        Shape (n_samples, n_factors), one column per kept factor in the order they were
        kept. Float throughout even where every column holds codes, since the two
        representations mix in one matrix under ``"auto"``.
    own_alphabet : list[bool]
        Whether each column's set of values belongs to the factor rather than to a cut,
        which is what decides if its entropy is a legitimate ceiling to divide by.
    coded : list[bool]
        Whether each column holds codes rather than measurements. Read from the values
        with the same test the statistic itself uses, so the two cannot disagree about
        what was handed over.
    """

    data: NDArray[np.float64]
    own_alphabet: list[bool]
    coded: list[bool]


def _reads_codes(
    metadata: Any,
    name: str,
    index: int,
    column: NDArray[Any],
    records: Mapping[str, Any] | None,
) -> bool:
    """Whether one factor is read as codes rather than as measurements under ``"auto"``.

    Three independent reasons to keep the codes, and any one is enough.

    **There is nothing to read instead.** A factor whose measured values are already
    integral is coded either way -- a category, a count, an identifier -- so reading them
    natively recovers no resolution. What it does lose is the cardinality cap that binning
    applied: a per-image ``id`` taking fifty values across ninety-three rows tabulates as a
    fifty-level factor, determines every image-level factor outright, and reports 1.0
    against all of them. That is arithmetically correct and tells a reader nothing, which
    is what capping the level count exists to prevent. The unbinned read is worth having only where
    the values carry something a cut threw away, which means where they are not integers.

    **The measurement is missing.** A row with no value for a factor has no native reading
    at all: ``factor_values`` reports the NaN the column holds, and the neighbor estimator
    refuses one outright rather than skipping the row. The coded channel is the only
    representation that carries such a row -- binning reserves a code for exactly this --
    so a column with any non-finite entry is read as codes however the cut was chosen.

    **Somebody made a claim.** A cut that was declared, asked for by count, or ratified with
    :meth:`~dataeval.Metadata.accept` is honored, because reading past it would discard the
    claim the record exists to carry and answer a question the caller did not ask. A cut
    nobody chose carries no claim, so the values are read. The record is consulted first
    among the three sources, then ``is_binned``, then the column itself -- each more
    specific than the next, and the first two exist only where somebody said something.

    Parameters
    ----------
    metadata : Any
        The container, consulted for ``is_binned`` where it keeps no record.
    name : str
        Factor name, for the record lookup.
    index : int
        The factor's position in the container's full factor list.
    column : NDArray
        The factor's measured values.
    records : Mapping or None
        Every factor's recorded encoding, read once by :func:`resolve_factor_channel`
        rather than per column -- ``Metadata.encoding()`` rebuilds the whole mapping on
        each call, so asking it per factor is quadratic in the factor count.
    """
    # Imported inside, not at module scope: `dataeval.types` imports this module, and
    # `dataeval.core` imports `dataeval.types`, so a module-level import closes the cycle.
    from dataeval.core._mutual_info import _is_coded

    if _is_coded(column) or not np.all(np.isfinite(column)):
        return True

    spec = records.get(name) if records is not None else None
    if isinstance(spec, BinSpec | LevelSpec):
        return spec.provenance != "derived"

    binned = getattr(metadata, "is_binned", None)
    if binned is not None and index < len(binned):
        return not bool(binned[index])

    return False


def _all_records(metadata: Any) -> Mapping[str, Any] | None:
    """Every factor's recorded encoding, or None where the container keeps none.

    Read once per :func:`resolve_factor_channel` call. ``Metadata.encoding()`` builds a
    fresh mapping over every visible factor each time it is called, so consulting it once
    per column costs a rebuild -- and a sort of the factor names -- per factor.
    """
    encoding: Any = getattr(metadata, "encoding", None)
    records = encoding() if callable(encoding) else None
    return records if isinstance(records, Mapping) else None


def _numeric_channel(metadata: Any) -> NDArray[np.float64] | None:
    """Return the measured representation, or None where the container has none."""
    values = getattr(metadata, "factor_values", None)
    return None if values is None else np.asarray(values, dtype=np.float64)


def _no_channel(metadata: Any, source: str, member: str, alternative: str) -> str:
    """Say that a named channel is not there, and which one is."""
    return (
        f"factor_source={source!r} reads `{member}`, which {type(metadata).__name__} does not "
        f"provide. Pass a container carrying that representation, use "
        f"factor_source={alternative!r} to read the one it has, or leave factor_source at "
        f"'auto' to read whichever is available per factor."
    )


def _coded_channel(metadata: Any, source: str, kept: Sequence[int], codes: NDArray[np.float64] | None) -> FactorChannel:
    """Read every kept factor off ``factor_data``, or say the container has none."""
    if codes is None:
        raise ValueError(_no_channel(metadata, source, "factor_data", "values"))
    return FactorChannel(codes[:, list(kept)], has_own_alphabet(metadata, kept), [True] * len(kept))


def _mixed_channel(
    metadata: Any,
    source: str,
    names: Sequence[str],
    kept: Sequence[int],
    codes: NDArray[np.float64] | None,
    values: NDArray[np.float64],
) -> FactorChannel:
    """Read each kept factor in whichever representation :func:`_reads_codes` picks for it.

    One column at a time, because ``"auto"`` mixes: a declared cut contributes its codes to
    the same matrix a derived factor contributes its values to.
    """
    from dataeval.core._mutual_info import _is_coded  # see `_reads_codes` for why

    columns: list[NDArray[np.float64]] = []
    own: list[bool] = []
    coded: list[bool] = []
    alphabets = has_own_alphabet(metadata, kept)
    records = _all_records(metadata) if source == "auto" else None
    for position, (index, name) in enumerate(zip(kept, names, strict=False)):
        measured = values[:, index]
        # `codes is not None` is part of the question, not a guard on the answer: "read the
        # codes instead" is only available where there are codes to read. A values-only
        # container carrying counts is integral throughout, so every column would otherwise
        # resolve to a channel it does not have -- on the *default* source.
        as_codes = source == "auto" and codes is not None and _reads_codes(metadata, name, index, measured, records)
        column = codes[:, index].astype(np.float64) if as_codes and codes is not None else measured
        columns.append(column)
        is_coded = _is_coded(column)
        coded.append(is_coded)
        # A column read as codes keeps whatever the container said about its alphabet. One
        # read as measurements has no cut behind it to disown, so its alphabet is its own
        # exactly when it turned out to hold codes at all -- a count, an ordinal.
        own.append(alphabets[position] if as_codes else is_coded)

    data = np.column_stack(columns) if columns else np.empty((values.shape[0], 0), dtype=np.float64)
    # Reachable only where the codes were not available to fall back on -- `"values"` was
    # named, or the container carries no `factor_data` -- since `_reads_codes` sends any
    # column with a missing measurement to the coded channel wherever there is one. Said
    # here because sklearn's own refusal names a matrix the caller never assembled.
    if unreadable := [name for name, column in zip(names, columns, strict=False) if not np.all(np.isfinite(column))]:
        raise ValueError(
            f"Factors {sorted(unreadable)} hold missing measurements, and the estimator that reads "
            "measured values has no reading for a row that has none — unlike the coded channel, "
            "which reserves a code for it. Use factor_source='coded', or leave factor_source at "
            "'auto' on a container that also carries codes, or drop the rows with where().",
        )
    return FactorChannel(data, own, coded)


def resolve_factor_channel(
    metadata: Any,
    source: str,
    names: Sequence[str],
    kept: Sequence[int],
) -> FactorChannel:
    """Choose which representation of each factor the statistic reads.

    Parameters
    ----------
    metadata : Any
        The container, in either representation or both.
    source : {"coded", "values", "auto"}
        Which representation to read. Named for the two channels rather than for binning:
        ``factor_data`` holds *codes*, of which bin indices are only one kind -- a category
        is coded and was never binned. ``"auto"`` decides per factor, see :func:`_reads_codes`.
    names : Sequence[str]
        The kept factors' names, aligned with ``kept``.
    kept : Sequence[int]
        Positions of the kept factors in the container's full factor list, which is what
        the protocol's own sequences are aligned with.

    Returns
    -------
    FactorChannel

    Raises
    ------
    ValueError
        When ``"values"`` is asked of a container that has no ``factor_values``. Refused
        rather than silently answered from the codes, because the two are different
        numbers and a caller who named the channel is entitled to get it or hear why not.
    """
    raw = getattr(metadata, "factor_data", None)
    codes = None if raw is None else np.asarray(raw, dtype=np.float64)

    # Asked for only where it might be read. Building it is a projection of every factor
    # over every row, which `"coded"` then throws away -- and on a `Metadata` that
    # projection reads the whole store.
    if source == "coded":
        return _coded_channel(metadata, source, kept, codes)

    values = _numeric_channel(metadata)

    # Both directions, symmetrically: a caller who named a channel gets it or hears why not,
    # rather than an IndexError out of an empty array that names neither the channel nor the
    # container missing it.
    if source == "values" and values is None:
        raise ValueError(_no_channel(metadata, "values", "factor_values", "coded"))

    if values is None:
        return _coded_channel(metadata, source, kept, codes)

    return _mixed_channel(metadata, source, names, kept, codes, values)


def scored_as(coded: Sequence[bool], own_alphabet: Sequence[bool], i: int, j: int) -> str:
    """Name the regime a pair of factors was scored under.

    Three branches, and which one runs is not visible in the number it produces. Two
    containers over the same data can report different associations for the same pair, so
    the output says which read produced each one.
    """
    if not (coded[i] and coded[j]):
        return "estimator"
    return "table" if (own_alphabet[i] or own_alphabet[j]) else "linfoot"


def _distinguish(names: dict[int, str]) -> dict[int, str]:
    """Make sure no two codes answer to the same name.

    Names are rendered with ``%g``, which carries six significant figures, so two genuinely
    different cutoffs can print identically -- ``1000000.4`` and ``1000000.6`` are both
    ``1e+06``, and the bin between them reads ``[1e+06, 1e+06)`` exactly like its neighbor.
    A caller keying a dict by these names then loses one of the two entries outright:
    :attr:`~dataeval.bias.ParityOutput.insufficient_data` reported one under-sampled level
    where the statistic had found two.

    The colliding names take their code alongside; the ones that were already unique are
    left exactly as they read.
    """
    seen: dict[str, int] = {}
    for name in names.values():
        seen[name] = seen.get(name, 0) + 1
    return {code: f"{name} (code {code})" if seen[name] > 1 else name for code, name in names.items()}


def _code_names(codes: NDArray[Any], encoding: "BinSpec | LevelSpec | None") -> dict[int, str]:
    """Map each distinct code of one factor to a display name.

    Keyed by the code itself rather than by position, so a composite axis can look each
    component up independently of how the combinations were numbered.

    Without a record there is nothing to name a code with, so the code names itself -- what
    a bare :class:`~dataeval.protocols.MetadataLike` has always got.
    """
    uniques = np.unique(codes)
    if isinstance(encoding, BinSpec):
        return _distinguish(_names_from_bin_spec(encoding, uniques))
    if isinstance(encoding, LevelSpec):
        return _distinguish(_names_from_level_spec(encoding, uniques))
    return {int(code): str(code) for code in uniques}


def factor_code_names(
    metadata: MetadataLike,
    factor_data: NDArray[Any],
    names: Sequence[str],
) -> list[dict[int, str]]:
    """Display name for every code present, one lookup per named factor.

    What :func:`resolve_label_axis` does for the axis, for the factors an evaluator reports
    *about*. A code is opaque on its own — ``illum_lux = 3`` says nothing — and the record
    is what turns it back into ``[0, 12.4)`` or ``rain``.

    Empty per factor where the container kept no record, in which case the caller keeps
    whatever it was showing before.
    """
    encodings = _recorded_encodings(metadata, names)
    return [_code_names(factor_data[:, position], encodings.get(name)) for position, name in enumerate(names)]


def _axis_codes(metadata: Any, requested: Sequence[str]) -> Any:
    """Return the coded factors a label axis groups by, or say why there are none.

    A label axis is a *grouping*, and measured values do not form groups. Said here rather
    than left as an ``AttributeError`` from the indexing that follows, which would name the
    attribute and not the reason it is needed.
    """
    codes = getattr(metadata, "factor_data", None)
    if codes is None:
        raise ValueError(
            f"`label` names {list(requested)}, and conditioning on a factor reads its codes — a "
            f"label axis groups rows, which measured values cannot do. {type(metadata).__name__} "
            "provides only `factor_values`. Pass label=None to condition on the class labels.",
        )
    return codes


def resolve_label_axis(metadata: Any, label: str | Sequence[str] | None) -> LabelAxis:
    """Resolve what a class-conditional statistic should condition on.

    ``label=None`` reads :attr:`~dataeval.Metadata.class_labels`, which is what every
    caller got before this existed and is still the default. Naming one or more factors
    instead conditions on those, which is the only way to ask a bias question at a view
    above :attr:`~dataeval.Metadata.label_level` — a frame, a track or a sequence has no
    single class label, and ``class_labels`` deliberately refuses to invent one. Derive
    the coarse-level label as a factor, then name it here.

    Resolution mirrors ``split_dataset``'s ``split_on=``, which also selects factors by
    name out of :attr:`~dataeval.Metadata.factor_data` and combines several into one
    grouping. The two are deliberately not the same call: ``split_on`` ignores a name the
    metadata does not have and never needs the groups named, while this rejects the name
    and names every group, so sharing an implementation would make each pay for the
    other's contract.

    Parameters
    ----------
    metadata : MetadataLike
        Metadata to read the axis from.
    label : str or Sequence[str] or None
        Factor name, several factor names to combine, or None for the class labels.

    Returns
    -------
    LabelAxis
        The axis values, their names, what it is called, and the factor columns to drop.

    Raises
    ------
    ValueError
        When ``label`` is an empty sequence, or names a factor the metadata does not have.
    """
    if label is None:
        labels = np.asarray(metadata.class_labels, dtype=np.intp)
        return LabelAxis(labels, _get_index2label(metadata), "class_label", ())

    requested = [label] if isinstance(label, str) else list(label)
    if not requested:
        raise ValueError("`label` names the factor(s) to condition on; an empty sequence names none.")

    factor_names = list(metadata.factor_names)
    if unknown := [name for name in requested if name not in factor_names]:
        raise ValueError(
            f"Label factor(s) {unknown} are not among this metadata's factors {factor_names}. "
            "`label` names a factor to condition on; pass None to use the class labels.",
        )

    indices = tuple(factor_names.index(name) for name in requested)
    data = np.asarray(_axis_codes(metadata, requested))[:, indices]
    # The record says what each code means, and says it the same way whichever kind of
    # factor produced it — a BinSpec names a range, a LevelSpec names a value. Neither
    # needs the raw column, which is why a container that kept its encoding can name its
    # own codes for the first time.
    encodings = _recorded_encodings(metadata, requested)
    lookups = [_code_names(data[:, position], encodings.get(name)) for position, name in enumerate(requested)]
    if len(indices) == 1:
        # Renumbered densely rather than used as-is: the downstream counters index by
        # label value, so a factor whose codes are sparse would size every contingency
        # table by its largest code.
        uniques, codes = np.unique(data[:, 0], return_inverse=True)
        names = {index: lookups[0][int(value)] for index, value in enumerate(uniques)}
    else:
        uniques, codes = np.unique(data, axis=0, return_inverse=True)
        names = {
            index: " × ".join(lookup[int(value)] for lookup, value in zip(lookups, row, strict=True))
            for index, row in enumerate(uniques)
        }
    return LabelAxis(codes.astype(np.intp).ravel(), names, " × ".join(requested), indices)


def kept_factors(metadata: Any, excluded: Sequence[int]) -> tuple[list[str], list[int]]:
    """Name the factors left after the label axis takes its own columns.

    Names and positions only, with no array read, so that it answers for a container in
    either representation -- :class:`~dataeval.bias.Balance` decides which one to read
    *after* this, and one carrying only measured values has no ``factor_data`` to slice.

    Returns
    -------
    tuple[list[str], list[int]]
        The survivors' names, and the positions they held in the metadata's full factor
        list. Those positions are what :func:`has_own_alphabet` and anything else reading a
        protocol sequence needs, since such a sequence is aligned with the full list rather
        than with this subset.

    Raises
    ------
    ValueError
        When the axis consumed every factor, leaving nothing to measure against it.
    """
    names = list(metadata.factor_names)
    if not excluded:
        return names, list(range(len(names)))
    dropped = set(excluded)
    kept = [index for index in range(len(names)) if index not in dropped]
    if not kept:
        # Caught here rather than downstream: every caller checks that the metadata has
        # factors *before* the axis is resolved, so an axis that consumes all of them
        # otherwise reaches the statistics as an empty matrix and reports on nothing.
        raise ValueError(
            f"The label axis names every factor this metadata has ({names}), leaving nothing to "
            "measure against it — a factor serving as the axis is dropped from the factors "
            "analysed, since it correlates perfectly with itself. Name fewer factors in `label`, "
            "or pass None to use the class labels.",
        )
    return [names[index] for index in kept], kept


def factors_excluding(metadata: MetadataLike, excluded: Sequence[int]) -> tuple[NDArray[Any], list[str], list[int]]:
    """Drop the label axis's own columns from the metadata's coded factors.

    :func:`kept_factors` with the array, for the evaluators that read one representation
    only.

    Returns
    -------
    tuple[NDArray[Any], list[str], list[int]]
        Factor data and factor names with ``excluded`` dropped from each, and the positions
        the survivors held in the metadata's full factor list.

    Raises
    ------
    ValueError
        When the axis consumed every factor, leaving nothing to measure against it.
    """
    names, kept = kept_factors(metadata, excluded)
    data = np.asarray(metadata.factor_data)
    # Guarded on `excluded`, not on a shape comparison: an empty metadata's factor_data has
    # no second axis to compare against.
    return (data[:, kept] if excluded else data), names, kept


def reject_filtered_metadata(candidate: Any, caller: str) -> None:
    """Refuse a filtered :class:`~dataeval.Metadata` wherever embeddings are involved.

    A filtered instance holds a subset of its rows and the **whole** of its dataset, so
    anything computed from that dataset describes more rows than the metadata does. Pairing
    the two silently misaligns them: the arrays are different lengths in the lucky case and
    the same length in the unlucky one, where a row is compared against another row's
    embedding and nothing raises.

    Refused rather than warned, and refused whenever a filtered metadata is involved *at
    all* rather than only when embeddings are passed in — an evaluator that computes its own
    embeddings from the bound dataset produces exactly the same mismatch, so there is no
    case where this pairing is safe.

    Parameters
    ----------
    candidate : Any
        Whatever was handed in. Anything that is not a filtered metadata is ignored, so a
        caller can offer a dataset, an array or None without checking first.
    caller : str
        Name of the evaluator, for the message.

    Raises
    ------
    ValueError
        When ``candidate`` reports :attr:`~dataeval.Metadata.is_filtered`.
    """
    if not getattr(candidate, "is_filtered", False):
        return
    raise ValueError(
        f"{caller} pairs metadata with embeddings, and this metadata has been filtered by where() "
        "or having() while its dataset has not — so the embeddings would describe rows this "
        "metadata no longer holds. Bring the dataset into correspondence first:\n"
        "    items = metadata.selected_items()\n"
        "    matching = View(dataset, Indices(items.tolist()))\n"
        "then build the embeddings from `matching`. Metadata.selected_items() raises if the "
        "filter cut below the item level, where no dataset subset can match.",
    )
