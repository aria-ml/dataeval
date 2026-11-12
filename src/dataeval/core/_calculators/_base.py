__all__ = []

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Flag, auto
from typing import Any, Generic, NamedTuple, TypeVar

TFlag = TypeVar("TFlag", bound=Flag)


class ViewKind(Flag):
    """Which view of a datum a statistic is defined over.

    A datum is measured more than once per row. The whole thing, the scene behind its
    annotations, a named group of its bands — each is a *view*, and not every statistic
    means something over every one. A width does not change when bands are dropped; a
    hash is not stable when pixels are masked to NaN. Declaring which views a statistic
    answers for is what keeps ``rgb_width`` and ``background_xxhash`` from being emitted.

    Attributes
    ----------
    WHOLE : ViewKind
        The datum as given — the default pass, over the full image or one of its boxes.
        Every statistic answers here.
    MASK : ViewKind
        A region with part of it masked out to NaN, which is how ``per_background``
        measures the scene behind the annotations. Only NaN-aware reductions survive it.
    BAND : ViewKind
        A named subset of the channel axis. Statistics that reduce over pixel values
        answer differently per band and so belong here; pure geometry restates the
        unprefixed value under a new name and does not.
    """

    WHOLE = auto()
    MASK = auto()
    BAND = auto()


# Every view, for a statistic that is a NaN-aware reduction over the values themselves —
# a masked region is simply not counted and a band subset is a different set to reduce.
ALL_VIEWS = ViewKind.WHOLE | ViewKind.MASK | ViewKind.BAND


class ContextNeed(Flag):
    """What a statistic needs beyond its own box, declared per statistic on its :class:`Handler`.

    A statistic that compares a box with its peers cannot be computed from the one converted box a calculator sees.
    Declaring the need lets `image_stats` build the per-image annotation context, and read a dataset's labels, only
    for a request that includes such a statistic. A request that includes none builds nothing and reads nothing.

    Attributes
    ----------
    NONE : ContextNeed
        The statistic reads only its own box. The default.
    CONTEXT : ContextNeed
        The statistic reads the image's other boxes through ``cache.context``.
    LABELS : ContextNeed
        The statistic also reads their class labels. Any non-empty need builds the context; this one additionally
        reads a dataset's labels.
    """

    NONE = 0
    CONTEXT = auto()
    LABELS = auto()


class Handler(NamedTuple):
    """One statistic: its output name, how to compute it, and the views it is defined over.

    The views are declared here, beside the callable, rather than as a property of the
    calculator. A calculator is not the right granularity for the question — a dimension
    calculator emits both band-invariant geometry and a band-*variant* bit depth, and a
    single answer for the pair produces either a meaningless ``rgb_width`` or a whole-cube
    depth contradicting the group it sits beside.

    Attributes
    ----------
    name : str
        The statistic's name in the output, before any view prefix is applied.
    compute : Callable[[], list[Any]]
        Computes the statistic, returning one value per row the calculator emits.
    views : ViewKind, default ViewKind.WHOLE
        Views this statistic is defined over, combined with ``|``. ``WHOLE`` alone —
        the default — means it describes the datum as given and is not recomputed for
        any masked region or band group.
    needs : ContextNeed, default ContextNeed.NONE
        What the statistic reads beyond its own box. Declared per statistic, not per calculator: a calculator
        can hold one statistic that compares a box with its peers and another that also compares their classes,
        and only a request that includes the second should read labels.
    """

    name: str
    compute: Callable[[], list[Any]]
    views: ViewKind = ViewKind.WHOLE
    needs: ContextNeed = ContextNeed.NONE


class Calculator(ABC, Generic[TFlag]):
    """
    Abstract base class for stateful statistics calculators.

    Calculators are responsible for computing specific categories of statistics
    on data. They are stateful and can cache intermediate results for efficiency.

    Each calculator:
    - Declares which flags it can handle via get_applicable_flags()
    - Receives raw datum + DatumProcessor instance during initialization
    - Computes statistics via compute() method
    - Can cache intermediate results as instance attributes or cached_properties

    Parameters
    ----------
    datum : Any
        The raw data element to compute statistics on.
    cache : CalculatorCache
        The per-datum cache providing preprocessed/transformed views of the datum.
    """

    @abstractmethod
    def get_applicable_flags(self) -> TFlag:
        """
        Return which flags this calculator can handle.

        Returns
        -------
        Flag
            A flag enum value representing all flags this calculator can process.
            Typically a group flag like ImageStats.PIXEL or TextStats.SENTIMENT.
        """

    @classmethod
    def flags_for_view(cls, flags: TFlag, view: ViewKind) -> TFlag:
        """
        Narrow `flags` to those this calculator's statistics are defined over for `view`.

        Asked once per view before any datum is read, so that a view runs only the
        statistics that mean something over it. An empty result means the calculator has
        nothing to say about this view and can be skipped entirely.

        Parameters
        ----------
        flags : Flag
            The requested flags, already narrowed to this calculator by the registry.
        view : ViewKind
            The view about to be measured.

        Returns
        -------
        Flag
            The subset of `flags` defined over `view`; falsy if there is none.
        """
        kept = type(flags)(0)
        for flag, handler in cls._handlers_for(flags).items():
            if view in handler.views:
                kept |= flag
        return kept

    @classmethod
    def _handlers_for(cls, flags: TFlag) -> dict[TFlag, Handler]:
        """Return the handlers `flags` selects, without constructing the calculator.

        ``__new__`` rather than a real instance: the answer is a property of the class and
        the constructor wants a datum. Kept here so that the trick — and the assumption it
        rests on, that ``get_handlers`` reads no instance state — lives beside the class it
        is a fact about, rather than being reproduced by every module that wants to ask.
        """
        blank = cls.__new__(cls)
        return {flag: handler for flag, handler in blank.get_handlers().items() if flag in flags}

    @classmethod
    def context_needs(cls, flags: TFlag) -> ContextNeed:
        """Return the union of what the statistics `flags` select read beyond their own box."""
        needs = ContextNeed.NONE
        for handler in cls._handlers_for(flags).values():
            needs |= handler.needs
        return needs

    @classmethod
    def flags_with_needs(cls, flags: TFlag) -> TFlag:
        """Return the subset of `flags` whose statistics read the image's other boxes, so have no value off a box."""
        kept = type(flags)(0)
        for flag, handler in cls._handlers_for(flags).items():
            if handler.needs:
                kept |= flag
        return kept

    @classmethod
    def stat_names(cls, flags: TFlag) -> set[str]:
        """Return the statistic names `flags` would produce from this calculator."""
        return {handler.name for handler in cls._handlers_for(flags).values()}

    @abstractmethod
    def get_handlers(self) -> dict[TFlag, Handler]:
        """
        Return a mapping of flags to the statistic each one produces.

        Returns
        -------
        dict[Flag, Handler]
            A dictionary mapping each flag this calculator can handle to a
            :class:`Handler` carrying the statistic's output name, the callable that
            computes it, and the views it is defined over.
        """

    def get_empty_values(self) -> dict[str, Any]:
        """
        Return empty values for statistics when they don't apply to certain channels.

        By default, all statistics use np.nan as the empty value. Override this method
        to provide custom empty values for specific statistics (e.g., arrays, strings).

        Returns
        -------
        dict[str, Any]
            A dictionary mapping stat names to their empty values.
            If a stat is not in this dict, np.nan is used as the default.

        Examples
        --------
        For a calculator with array-valued statistics:

        >>> def get_empty_values(self) -> dict[str, Any]:
        ...     return {
        ...         "center": [np.nan, np.nan],
        ...         "histogram": [np.nan] * 256,
        ...     }
        """
        return {}

    def compute(self, flags: TFlag, view: ViewKind = ViewKind.WHOLE) -> dict[str, list[Any]]:
        """
        Compute statistics for the requested flags.

        Parameters
        ----------
        flags : Flag
            The specific flags to compute. This will be a subset of the flags
            returned by get_applicable_flags(), representing what the user requested.
        view : ViewKind, default ViewKind.WHOLE
            Which view of the datum the cache is presenting. Statistics not defined over
            it are skipped — see :class:`Handler`.

        Returns
        -------
        dict[str, list[Any]]
            Dictionary mapping stat names to lists of values. Every statistic reduces its
            view to a single reading, so each list holds exactly one value, e.g. ``[42.0]``
            — the list is what lets a name a calculator does not produce stay absent rather
            than become a value. A longer list is rejected by ``_reconcile_stats``.

            The processor framework will reconcile outputs from multiple calculators.
        """
        stats: dict[str, list[Any]] = {}

        for flag, handler in self.get_handlers().items():
            if flag in flags and view in handler.views:
                stats[handler.name] = handler.compute()

        return stats

    def details(self, flags: TFlag, view: ViewKind = ViewKind.WHOLE) -> dict[str, list[Any]]:  # noqa: ARG002
        """
        Return per-row facts about how the statistics for `flags` were measured.

        Called after :meth:`compute` on the same instance, so it can read what computing
        cached. Shaped like :meth:`compute`'s output, one single-element list per name, and
        kept beside the statistics rather than among them, so a status or a support count never
        becomes a factor. A calculator returns the same names on every row it measures, and
        :meth:`get_empty_values` supplies a detail's value on rows it does not measure, as it
        does for a statistic. Identifier details are int64, with -1 for none.

        Empty by default. Only a calculator with something to report overrides it.

        Parameters
        ----------
        flags : Flag
            The flags :meth:`compute` was given.
        view : ViewKind, default ViewKind.WHOLE
            The view :meth:`compute` was given.

        Returns
        -------
        dict[str, list[Any]]
            Detail names mapped to one-element lists of values.
        """
        return {}
