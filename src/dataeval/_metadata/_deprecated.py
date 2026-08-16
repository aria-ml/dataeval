"""The v1.2.0 removals, gathered into one expiring layer.

Every member here is a stateless forward to a supported call plus a
:class:`DeprecationWarning`, and every one of them is removed in v1.2.0. Kept as a mixin
rather than left in the class body for exactly that reason: they share a removal date, so
they are a *layer with an expiry* — retiring them is deleting this file and one name from
:class:`~dataeval.Metadata`'s base list, rather than a scattered diff through a class that
is still being developed underneath them.

The cost is that these members are not greppable in ``_metadata.py``. That is paid once,
on members nobody should be reading, and is why nothing that is *not* scheduled for
removal belongs in this file.
"""

__all__ = []

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval.types import FactorInfo, FactorLevel


class DeprecatedMetadataAPI:
    """Deprecated members of :class:`~dataeval.Metadata`, all removed in v1.2.0."""

    if TYPE_CHECKING:
        # Declared, not defined. This is a mixin into :class:`~dataeval.Metadata` and
        # these are the members it borrows from its host — listing them is what lets a
        # type checker see the mixin as complete, and doubles as the statement of how
        # much of Metadata a deprecated shim is allowed to reach into. Nothing here is
        # assigned, so there is no runtime attribute to shadow the real ones. Each is
        # declared in the *form* Metadata defines it, property for property, so that the
        # override stays compatible rather than narrowing a property to its return type.
        _target_factors_only: bool

        @property
        def _item_level(self) -> FactorLevel: ...

        @property
        def _label_level(self) -> FactorLevel: ...

        @property
        def _view_level(self) -> FactorLevel: ...

        @property
        def multi_target(self) -> bool: ...

        @property
        def factor_names(self) -> Sequence[str]: ...

        def _structure(self) -> None: ...

        def _reset_view_dependent_state(self) -> None: ...

        def rows_at(self, level: FactorLevel | Literal["target", "image"]) -> pl.DataFrame: ...

        def filter_by_factor(self, condition: Callable[[str, FactorInfo], bool]) -> NDArray[np.float64]: ...

    @property
    def target_factors_only(self) -> bool:
        """Whether factors above the target level are dropped, on multi-target tasks.

        .. deprecated::
            Two knobs in one, and neither of them this. Use ``md.at(level)`` to choose
            which rows are read, and :attr:`~dataeval.Metadata.inherited` to choose whether
            ancestor factors count. Removed in v1.2.0.

        Notes
        -----
        Retains its v1.1 semantics exactly, including the part that reads like a bug:
        it is a no-op unless :attr:`~dataeval.Metadata.multi_target`, so on image
        classification it has never done anything. :attr:`~dataeval.Metadata.inherited`
        does not carry that exemption over — it means what it says on every task — which
        is why this is kept as its own flag rather than forwarded to it.
        """
        warnings.warn(
            "Metadata.target_factors_only is deprecated and will be removed in v1.2.0. "
            "Use Metadata.at(level) to choose the rows and Metadata.inherited to choose "
            "whether ancestor factors count.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._target_factors_only

    @target_factors_only.setter
    def target_factors_only(self, value: bool) -> None:
        warnings.warn(
            "Metadata.target_factors_only is deprecated and will be removed in v1.2.0. "
            "Use Metadata.at(level) to choose the rows and Metadata.inherited to choose "
            "whether ancestor factors count.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._target_factors_only != value:
            self._target_factors_only = value
            self._reset_view_dependent_state()

    @property
    def raw_data(self) -> NDArray[Any]:
        """Raw factor values before binning or digitization.

        .. deprecated::
            This is ``rows_at(md.view).select(factor_names).to_numpy()``, and going
            through the dataframe keeps the per-factor dtypes that this array flattens
            to ``object`` the moment factors of different types are mixed. Use
            :meth:`~dataeval.Metadata.rows_at` for raw values, or
            :meth:`~dataeval.Metadata.filter_by_factor` for a float array.
            Removed in v1.2.0.

        Returns
        -------
        NDArray[Any]
            Array with shape (n_samples, n_factors) containing original factor
            values, taken at the :attr:`~dataeval.Metadata.view` level. Returns empty
            array when no factors are available.
        """
        warnings.warn(
            "Metadata.raw_data is deprecated and will be removed in v1.2.0. Use "
            "Metadata.rows_at(md.view).select(md.factor_names).to_numpy() instead, or "
            "Metadata.filter_by_factor() for a float array.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.factor_names:
            return np.array([], dtype=np.float64)

        return self.rows_at(self._view_level).select(self.factor_names).to_numpy()

    @property
    def image_data(self) -> pl.DataFrame:
        """Dataframe containing only image-level rows.

        .. deprecated::
            The name only tells the truth when a dataset item is an image, so it is
            defined for image-based tasks alone and raises for every other task. Use
            ``rows_at(md.item_level)``. Removed in v1.2.0.

        Returns
        -------
        pl.DataFrame
            Image-level metadata, exactly as previous releases returned it — which,
            on image classification, means the *labelled* rows rather than the image
            rows. See Notes.

        Raises
        ------
        ValueError
            When the bound dataset's items are not images.

        Notes
        -----
        Bug-for-bug with v1.1, on purpose. There, a classification dataset had a
        single block of rows and this property returned it; the level restructure
        split that block into image rows and instance rows, and returning the image
        rows here would silently hand existing callers nulls where ``class_label``,
        ``score`` and ``target_index`` used to be. So it still returns the labelled
        rows for a single-target task and the image rows for a multi-target one.
        ``rows_at(md.item_level)`` is the spelling that means image rows on every task.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> metadata.rows_at(metadata.item_level).select("item_index", "time_of_day", "weather", "location").head(5)
        shape: (5, 4)
        ┌────────────┬─────────────┬─────────┬──────────┐
        │ item_index ┆ time_of_day ┆ weather ┆ location │
        │ ---        ┆ ---         ┆ ---     ┆ ---      │
        │ i64        ┆ str         ┆ str     ┆ str      │
        ╞════════════╪═════════════╪═════════╪══════════╡
        │ 0          ┆ dusk        ┆ cloudy  ┆ urban    │
        │ 1          ┆ night       ┆ rainy   ┆ suburban │
        │ 2          ┆ night       ┆ cloudy  ┆ urban    │
        │ 3          ┆ dawn        ┆ clear   ┆ maritime │
        │ 4          ┆ dusk        ┆ cloudy  ┆ urban    │
        └────────────┴─────────────┴─────────┴──────────┘
        """
        warnings.warn(
            "Metadata.image_data is deprecated and will be removed in v1.2.0. Use "
            "Metadata.rows_at(md.item_level) for image rows. Note that on a "
            "single-target task this property returns the labelled rows, not the "
            "image rows, to match what v1.1 returned.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._structure()
        if self._item_level != "unit":
            raise ValueError(
                "Metadata.image_data is only defined for image-based tasks, but this dataset has "
                f"items at the {self._item_level!r} level. "
                'Use Metadata.rows_at("unit") for the image rows, '
                "or Metadata.rows_at(md.item_level) for item-level rows.",
            )
        return self.rows_at(self._item_level if self.multi_target else self._label_level)

    @property
    def target_data(self) -> pl.DataFrame:
        """Dataframe containing only label-level rows.

        .. deprecated::
            One spelling per level does not scale, and this one names a level that no
            longer exists. Use ``rows_at(md.label_level)`` for the rows the labels are
            on, or ``rows_at(md.view)`` for the rows the array accessors project.
            Removed in v1.2.0.

        Returns
        -------
        pl.DataFrame
            Dataframe with label-level metadata. Each row represents a single
            labelled thing with its associated class, score, and bounding box
            information: a detection for object detection, the image itself for
            classification.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> metadata.rows_at(metadata.label_level).select("item_index", "target_index", "class_label").head(5)
        shape: (5, 3)
        ┌────────────┬──────────────┬─────────────┐
        │ item_index ┆ target_index ┆ class_label │
        │ ---        ┆ ---          ┆ ---         │
        │ i64        ┆ i64          ┆ i64         │
        ╞════════════╪══════════════╪═════════════╡
        │ 0          ┆ 0            ┆ 0           │
        │ 1          ┆ 0            ┆ 3           │
        │ 1          ┆ 1            ┆ 2           │
        │ 1          ┆ 2            ┆ 1           │
        │ 2          ┆ 0            ┆ 1           │
        └────────────┴──────────────┴─────────────┘
        """
        warnings.warn(
            "Metadata.target_data is deprecated and will be removed in v1.2.0. Use "
            "Metadata.rows_at(md.label_level) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._structure()
        return self.rows_at(self._label_level)

    def get_image_factors(self, image_idx: int) -> dict[str, Any]:
        """Get all factors for a specific image.

        .. deprecated::
            A single row lookup is one dataframe filter, and phrasing it as a method
            per level does not scale past the two levels that happen to exist today.
            Use ``rows_at("unit").filter(pl.col("item_index") == image_idx)``.
            Removed in v1.2.0.

        Parameters
        ----------
        image_idx : int
            Index of the image to retrieve factors for

        Returns
        -------
        dict[str, Any]
            Dictionary mapping factor names to their values for the specified image

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> factors = metadata.get_image_factors(0)
        >>> factors["time_of_day"]
        'dusk'
        >>> factors["weather"]
        'cloudy'
        >>> factors["location"]
        'urban'
        """
        warnings.warn(
            "Metadata.get_image_factors() is deprecated and will be removed in v1.2.0. Use "
            'Metadata.rows_at("unit").filter(pl.col("item_index") == image_idx) instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        self._structure()
        row = self.rows_at(self._item_level).filter(pl.col("item_index") == image_idx)
        if row.height == 0:
            raise ValueError(f"No image found with index {image_idx}")
        return row.to_dicts()[0]

    def get_target_factors(self, image_idx: int, target_idx: int) -> dict[str, Any]:
        """Get all factors for a specific target within an item.

        .. deprecated::
            A single row lookup is one dataframe filter, and phrasing it as a method
            per level does not scale past the two levels that happen to exist today.
            Use ``target_data.filter(...)``. Removed in v1.2.0.

        Parameters
        ----------
        image_idx : int
            Index of the item containing the target
        target_idx : int
            Index of the target within the item (0-indexed per item)

        Returns
        -------
        dict[str, Any]
            Dictionary mapping factor names to their values for the specified target

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> factors = metadata.get_target_factors(1, 1)
        >>> factors["item_index"]
        1
        >>> factors["target_index"]
        1
        >>> factors["class_label"]
        2
        """
        warnings.warn(
            "Metadata.get_target_factors() is deprecated and will be removed in v1.2.0. Use "
            'Metadata.rows_at(md.label_level).filter((pl.col("item_index") == image_idx) & '
            '(pl.col("target_index") == target_idx)) instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        self._structure()
        rows = self.rows_at(self._label_level)
        row = rows.filter((pl.col("item_index") == image_idx) & (pl.col("target_index") == target_idx))
        if row.height == 0:
            raise ValueError(f"No target found with item_index={image_idx}, target_index={target_idx}")
        return row.to_dicts()[0]

    def has_targets(self) -> bool:
        """Check if the source dataset has targets.

        .. deprecated::
            Renamed for what it actually reports. Use
            :attr:`~dataeval.Metadata.multi_target`. Removed in v1.2.0.

        Returns
        -------
        bool
            True for object detection, False for image classification — unchanged
            from v1.1.

        Notes
        -----
        No expression over the row counts reproduces this, which is why the
        replacement is a property and not one. ``level_counts["instance"] !=
        level_counts["unit"]`` is False for a detection dataset with one detection
        per image and True for a classification dataset with an unlabeled item, so it
        gets the answer wrong in both directions. Nor is it ``label_level !=
        item_level``: every task now names its labelled level ``instance`` and its
        item level ``unit``, so that comparison is true even for classification.
        """
        warnings.warn(
            "Metadata.has_targets() is deprecated and will be removed in v1.2.0. Use Metadata.multi_target instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.multi_target

    def filter_by_factor_type(
        self,
        factor_type: Literal["categorical", "discrete", "continuous"],
    ) -> NDArray[np.float64]:
        """Filter metadata factors by factor type.

        .. deprecated::
            One predicate over :meth:`~dataeval.Metadata.filter_by_factor` is the whole of
            this method, and keeping a named wrapper per :class:`FactorInfo` field does not
            scale as the class grows fields. Use
            ``filter_by_factor(lambda _, fi: fi.factor_type == factor_type)``.

        Parameters
        ----------
        factor_type : "categorical", "discrete" or "continuous"
            The factor type to include in the output.

        Returns
        -------
        NDArray[np.float64]
            Array with shape (n_samples, n_factors) where the factors
            are filtered by the user provided factor type. Rows are taken at
            the ``instance`` level; see :meth:`~dataeval.Metadata.filter_by_factor` for
            which representation of each factor the values come from.
        """
        warnings.warn(
            "Metadata.filter_by_factor_type() is deprecated and will be removed in v1.2.0. "
            "Use Metadata.filter_by_factor(lambda _, fi: fi.factor_type == factor_type) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.filter_by_factor(lambda _, fi: fi.factor_type == factor_type)
