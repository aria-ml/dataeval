__all__ = []

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from typing_extensions import TypeIs

from dataeval.exceptions import DeprecatedWarning
from dataeval.protocols import MetadataLike

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


def _raw_factor_columns(metadata: MetadataLike, names: Sequence[str]) -> dict[str, Sequence[Any]]:
    """Read the named factors' values before binning, when the metadata can reach them.

    :attr:`~dataeval.Metadata.factor_data` holds bin indices, so an axis built from a
    factor can only name its groups ``0, 1, 2`` on its own. A
    :class:`~dataeval.Metadata` also holds the pre-binning columns and can do better;
    a bare :class:`~dataeval.protocols.MetadataLike` cannot, and gets the indices. Same
    optional-member degradation as :func:`_get_item_indices` and
    :func:`_get_index2label`, and for the same reason: the protocol stays four members.

    Every requested factor is read off one frame. :meth:`~dataeval.Metadata.rows_at`
    filters the whole dataframe on each call, so a composite axis asking per factor would
    pay that filter once per name.
    """
    rows_at, view = getattr(metadata, "rows_at", None), getattr(metadata, "view", None)
    if rows_at is None or view is None:
        return {}
    rows = rows_at(view)
    return {name: rows[name].to_list() for name in names if name in rows.columns}


def _binned_names(codes: NDArray[Any], raw: Sequence[Any], uniques: NDArray[Any]) -> dict[int, str]:
    """Name each bin of a *continuous* factor by the span of values it actually holds.

    A bin covers a range, so naming it after one of its members — whichever row happened
    to come first — reads as an exact value that most of the group does not have. The
    observed low and high bound the group honestly.

    Grouped by sorting once rather than by scanning the codes per bin, so the cost is
    ``n log n`` in the rows rather than ``n`` per distinct bin.
    """
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        # Declared continuous but not numerically readable: the code is the only honest
        # name left, and is what a bare MetadataLike gets for the same reason.
        return {int(code): str(code) for code in uniques}
    order = np.argsort(codes, kind="stable")
    sorted_codes, sorted_values = codes[order], values[order]
    starts = np.searchsorted(sorted_codes, uniques, side="left")
    ends = np.searchsorted(sorted_codes, uniques, side="right")
    return {
        int(code): f"[{sorted_values[start:end].min():g}, {sorted_values[start:end].max():g}]"
        for code, start, end in zip(uniques, starts, ends, strict=True)
    }


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


def _code_names(codes: NDArray[Any], raw: Sequence[Any] | None, *, discrete: bool) -> dict[int, str]:
    """Map each distinct bin of one factor to a display name.

    Keyed by the bin's own code rather than by position, so a composite axis can look
    each component up independently of how the combinations were numbered.

    Parameters
    ----------
    codes : NDArray[Any]
        The factor's bin index for each row.
    raw : Sequence[Any] or None
        The factor's pre-binning values, in the same row order, or None when the metadata
        cannot reach them.
    discrete : bool
        Whether the factor's codes each stand for a single value. A discrete factor's
        group is named by that value; a continuous one's covers a range, and is named by
        :func:`_binned_names`.

    Returns
    -------
    dict[int, str]
        Display name for every code present in ``codes``.
    """
    uniques, first = np.unique(codes, return_index=True)
    if raw is None:
        return {int(code): str(code) for code in uniques}
    if not discrete:
        return _binned_names(codes, raw, uniques)
    # One code, one value, so the first row carrying the code names the whole group.
    return {int(code): str(raw[int(index)]) for code, index in zip(uniques, first, strict=True)}


def resolve_label_axis(metadata: MetadataLike, label: str | Sequence[str] | None) -> LabelAxis:
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
    data = np.asarray(metadata.factor_data)[:, indices]
    # How a group is named turns on whether its code stands for one value or a range, not
    # on the factor's discrete/continuous kind — a binned factor covers a range whichever
    # kind it is. See :func:`has_own_alphabet`.
    single_valued = has_own_alphabet(metadata, indices)
    raw_columns = _raw_factor_columns(metadata, requested)
    lookups = [
        _code_names(data[:, position], raw_columns.get(name), discrete=single_valued[position])
        for position, name in enumerate(requested)
    ]
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


def factors_excluding(metadata: MetadataLike, excluded: Sequence[int]) -> tuple[NDArray[Any], list[str], list[int]]:
    """Drop the label axis's own columns from the metadata's factors.

    Returns
    -------
    tuple[NDArray[Any], list[str], list[int]]
        Factor data and factor names with ``excluded`` dropped from each, and the positions
        the survivors held in the metadata's full factor list. Those positions are what
        :func:`has_own_alphabet` and anything else reading a protocol sequence needs, since
        such a sequence is aligned with the full list rather than with this subset.

    Raises
    ------
    ValueError
        When the axis consumed every factor, leaving nothing to measure against it.
    """
    names = list(metadata.factor_names)
    data = np.asarray(metadata.factor_data)
    if not excluded:
        return data, names, list(range(len(names)))
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
    return data[:, kept], [names[index] for index in kept], kept


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
