"""Evaluate how completely a dataset covers its embedding space, per class.

Where :class:`Representation` measures coverage of an ontology's *label* space (which
sanctioned classes are populated), :class:`Coverage` measures coverage of the
*embedding* space (which samples sit in sparse regions, and how visually varied each
class is). The two are orthogonal: a dataset can have every class present yet have a
class whose examples are near-duplicates.

Per class, three complementary variety signals describe *how* a class fills space, which raw
label counts cannot reveal:

* **intra-class dispersion** — the *magnitude* of spread, the class's mean distance to its own
  centroid relative to a typical (median) class.
* **isotropy** — the *shape* of that spread, in how many independent directions the class
  varies (via :func:`~dataeval.core.completeness`). A class can spread far yet vary along a
  single axis — all images shot from one angle, lighting or zoom aside — which dispersion's
  single radius cannot distinguish.
* **near-duplicate fraction** — the *redundancy*, the share of a class's images whose nearest
  neighbor is unusually close (repeated stock frames), derived from the same computation.

Alongside them, the global :func:`~dataeval.core.coverage_adaptive` result flags individual
samples in under-sampled regions of the whole space.

The global coverage radius is computed *once* over the full dataset and then aggregated per
class — never re-run per class, which would be statistically invalid for the small classes
that matter most. The per-class spread signals (dispersion, isotropy, near-duplicate fraction)
are class-local statistics, so they are computed per class above ``min_class_samples``; classes
below it are reported but left unassessed.
"""

__all__ = ["Coverage", "CoverageOutput"]

from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import PositiveFloat, PositiveInt

from dataeval import Metadata
from dataeval._helpers import BareLabels, is_labels_like, reject_filtered_metadata
from dataeval._log import get_logger
from dataeval.core._completeness import CompletenessResult
from dataeval.core._completeness import completeness as _completeness
from dataeval.core._coverage import CoverageResult, coverage_adaptive, coverage_naive
from dataeval.exceptions import ShapeMismatchError
from dataeval.protocols import AnnotatedDataset, Array, ArrayLike, FeatureExtractor, LabelsLike
from dataeval.types import ClassAxis, DataFrameOutput, Evaluator, EvaluatorConfig, set_metadata
from dataeval.types._config import OpenUnitInterval, UnitInterval

_logger = get_logger(__name__)

MethodType = Literal["naive", "adaptive"]

_PER_CLASS_SCHEMA: dict[str, pl.DataType] = {
    "class": pl.String(),
    "count": pl.Int64(),
    "uncovered": pl.Int64(),
    "uncovered_fraction": pl.Float64(),
    "dispersion": pl.Float64(),
    "isotropy": pl.Float64(),
    "near_duplicate_fraction": pl.Float64(),
    "assessable": pl.Boolean(),
    # Appended, so the columns released before them keep their positions.
    "is_clustered": pl.Boolean(),
    "is_one_dimensional": pl.Boolean(),
    "has_near_duplicates": pl.Boolean(),
}


def _mean_dispersion(points: NDArray[np.float64]) -> float:
    """Mean Euclidean distance of ``points`` to their centroid (their spread)."""
    return float(np.linalg.norm(points - points.mean(axis=0), axis=1).mean())


def _nearest_distances(result: CompletenessResult, n: int) -> NDArray[np.float64]:
    """Each of the ``n`` points' distance to its nearest other point, from ``completeness``'s pairs.

    The pairs list two mutual nearest neighbors once, so they count pairs, not points. A point's
    nearest-neighbor distance is the smallest among the pairs it is in.
    """
    nearest = np.full(n, np.inf)
    distances = np.asarray(result["nearest_neighbor_distances"], dtype=np.float64)
    for points in np.asarray(result["nearest_neighbor_pairs"], dtype=np.intp).reshape(-1, 2).T:
        np.minimum.at(nearest, points, distances)
    return nearest


class CoverageOutput(DataFrameOutput):
    """
    A dataset's per-class embedding-space coverage.

    The wrapped DataFrame is the per-class breakdown — one row per class, sorted with the
    lowest-dispersion assessable classes first (the ones most worth collecting varied data
    for) and unassessable classes last — with columns ``class``, ``count``, ``uncovered``,
    ``uncovered_fraction``, ``dispersion``, ``isotropy``, ``near_duplicate_fraction``,
    ``assessable``, ``is_clustered``, ``is_one_dimensional``, and ``has_near_duplicates``.
    ``isotropy`` and ``near_duplicate_fraction`` are null for classes below the sample floors
    that make them meaningful. The sample-level coverage detail hangs off it as attributes.

    The last three columns are a verdict on each variety signal, with cutoffs that are
    parameters of :class:`Coverage`. They are independent, and a class can trip more than one:

    - is_clustered: bool - True if dispersion < dispersion_factor (default 0.5): the class
      spreads less than half as far from its centroid as the median class.
    - is_one_dimensional: bool - True if isotropy < isotropy_factor (default 0.5) x the median
      isotropy of the classes that report one, the class itself included. Isotropy is not
      normalized by the evaluator, since it depends on the embedding dimension and on any
      reduction applied first, so it is judged against its peers.
    - has_near_duplicates: bool - True if near_duplicate_fraction > near_duplicate_threshold
      (default 0.1).

    A flag is null where it cannot judge, which is new: the flags of
    :class:`~dataeval.bias.Balance` and :class:`~dataeval.bias.Diversity` are never null. A
    flag is null where its signal is null, that is, for a class below ``min_class_samples`` or,
    for ``is_one_dimensional``, below the isotropy sample floor. It is also null on every row
    where the median it is judged against is not positive. A median spread of 0 leaves
    ``dispersion`` in raw embedding units, and a median isotropy of 0 would flag no class at
    all. With a single assessable class the median is the class itself, so neither
    ``is_clustered`` nor ``is_one_dimensional`` can trip however flat the class is.

    The defaults are conservative, so a flag means a class has collapsed, not that it sits a
    little below its peers. Tighten a factor to catch a partial collapse. They are tuned for
    deep embeddings, and two signals depend on more than the data:

    - Isotropy rises with class size, so a small class reads lower beside large ones.
      Tighten ``isotropy_factor`` only for classes of similar size.
    - The near-duplicate share of a class with no duplicates rises as the embedding dimension
      falls. On simulated Gaussian classes of 100 to 400 images it exceeded 0.1 for most
      classes in 2 dimensions, for about half in 3 and for a few up to 7, and for none in 8
      or more. Raise ``near_duplicate_threshold`` before reading ``has_near_duplicates`` on a
      low-dimensional embedding, such as a few principal components.

    .. versionadded:: 1.2
        The ``is_clustered``, ``is_one_dimensional`` and ``has_near_duplicates`` columns,
        after ``assessable`` so the earlier columns keep their positions.

    Attributes
    ----------
    uncovered_indices : NDArray[np.intp]
        Indices of individual samples sitting in under-sampled regions of the full
        embedding space (from :func:`~dataeval.core.coverage_adaptive` /
        :func:`~dataeval.core.coverage_naive`).
    coverage_radius : float
        The radius threshold separating covered from uncovered samples.
    critical_value_radii : NDArray[np.float32]
        Per-sample distance to the ``num_observations``-th nearest neighbor — the raw
        density signal the uncovered set is thresholded from.
    class_axis : ClassAxis or None
        What the ``class`` column of the breakdown holds: the dataset's own labels, or an
        axis a caller defined with :meth:`~dataeval.Metadata.classed_by`. Read
        ``class_axis.source`` to tell the two apart — the column reads the same either way,
        and ``rain`` is a perfectly plausible class name. None where the labels arrived as a
        raw array, which carries no record of where it came from.

        .. versionadded:: 1.2
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        uncovered_indices: NDArray[np.intp],
        coverage_radius: float,
        critical_value_radii: NDArray[np.float32],
        class_axis: ClassAxis | None = None,
    ) -> None:
        super().__init__(data)
        self.uncovered_indices = uncovered_indices
        self.coverage_radius = coverage_radius
        self.critical_value_radii = critical_value_radii
        self.class_axis = class_axis


class Coverage(Evaluator):
    """
    Evaluate a dataset's embedding-space coverage and per-class variety.

    Computes the global coverage radius *once* over the full set of image embeddings (each
    class's share of globally under-sampled samples), then describes each class with three
    complementary, class-local variety signals label counts cannot give:

    * **dispersion** — magnitude of spread (mean distance to centroid, relative to a typical
      class); low means clustered even if numerous.
    * **isotropy** — shape of that spread (in how many independent directions the class
      varies); low means it varies along few axes even when dispersion is high.
    * **near_duplicate_fraction** — redundancy (share of the class's images whose nearest
      neighbor is unusually close); high means repeated / near-identical samples.

    Each signal comes with a boolean verdict, ``is_clustered``, ``is_one_dimensional`` and
    ``has_near_duplicates``, so the output table is the per-class gap report. See
    :class:`CoverageOutput` for the exact rules and when a flag is null.

    Parameters
    ----------
    extractor : FeatureExtractor or None, default None
        Feature extractor used to compute embeddings from a dataset. Optional only when
        pre-computed embeddings are passed to :meth:`evaluate`.
    batch_size : int or None, default None
        Batch size for embedding computation. When None, uses the global batch size.
    method : {"naive", "adaptive"}, default "adaptive"
        Coverage radius method — fixed analytic radius (``naive``) or data-adaptive cutoff
        on the ``percent`` most sparsely-neighbored samples (``adaptive``).
    num_observations : int, default 20
        Neighbors required for a sample to be considered covered (20-50 is typical).
    percent : float, default 0.01
        Fraction of samples to flag as uncovered (``adaptive`` method only).
    min_class_samples : int, default 20
        A class needs at least this many samples for its per-class signals to be assessed;
        smaller classes are reported with ``assessable=False`` and null ``dispersion`` /
        ``isotropy`` / ``near_duplicate_fraction``.
    isotropy_min_samples : int or None, default None
        A class needs at least this many samples for its ``isotropy`` to be reported (the
        effective-dimension estimate is degenerate when samples do not exceed dimensions).
        When None, defaults to one more than the embedding dimensionality.
    near_duplicate_factor : float, default 0.5
        An image counts as a near-duplicate when its nearest neighbor in its class is no farther
        than this fraction of a typical class's median nearest-neighbor distance (scale-free).
        A scale for a distance, which sets ``near_duplicate_fraction``. Not to be confused with
        ``near_duplicate_threshold``, the share of images that sets the flag.
    dispersion_factor : float, default 0.5
        ``is_clustered`` is True where ``dispersion < dispersion_factor``, that is, where a class
        spreads less than this share of the median class's spread. Must be in (0, 1); a factor
        near 1 flags about half the classes by construction.

        .. versionadded:: 1.2
    isotropy_factor : float, default 0.5
        ``is_one_dimensional`` is True where ``isotropy < isotropy_factor * median(isotropy)``,
        the median taken over the classes that report an isotropy. Must be in (0, 1).

        .. versionadded:: 1.2
    near_duplicate_threshold : float, default 0.1
        ``has_near_duplicates`` is True where ``near_duplicate_fraction`` is above this share of
        the class's images. Must be in [0, 1]: 0 flags any near-duplicate, and 1 turns the flag
        off. A share, where ``near_duplicate_factor`` is a distance scale.

        .. versionadded:: 1.2
    config : Coverage.Config or None, default None
        Optional configuration object; parameters passed directly to ``__init__`` override
        its values.

    See Also
    --------
    dataeval.scope.Representation : The label-space counterpart (ontology coverage).
    dataeval.scope.Prioritize : Rank individual samples for labeling in embedding space.
    dataeval.core.coverage_adaptive : The underlying sample-level coverage computation.

    Notes
    -----
    ``dispersion`` is the class's mean distance-to-centroid divided by the *median* of that
    same measure across assessable classes: ``~1`` is a typical class, ``< 1`` means the
    class is unusually clustered (low variety / near-duplicate), ``> 1`` means it is more
    spread out than its peers. Normalizing by the peer median (rather than the global
    spread) keeps the scale meaningful even when classes are well-separated in embedding
    space. ``isotropy`` is the class's effective dimensionality relative to the subspace it
    spans (via :func:`~dataeval.core.completeness`): ``~1`` means it varies evenly in every
    direction it occupies, low means its variation collapses onto a few axes — orthogonal to
    ``dispersion``, which only measures how far it spreads. The effective dimensionality is
    ``exp(H)`` over the class's normalized covariance eigenvalues — the eigenvalue form of
    Roy & Vetterli's effective rank — and is degenerate unless the class has more samples
    than the embedding has dimensions, which is what ``isotropy_min_samples`` guards; on a
    high-dimensional extractor most classes fall below that floor and report ``isotropy``
    as null rather than zero. Reduce the embedding dimensionality (for example with PCA) if
    you need the signal on classes of a few hundred samples. ``near_duplicate_fraction`` is the
    share of a class's images whose nearest other image in the class is within
    ``near_duplicate_factor`` x the *median* across assessable classes of each class's median
    nearest-neighbor distance, surfacing repeated / near-identical samples that inflate counts
    without adding variety. Both images of a near-identical pair count, and an exact copy counts
    the same as one that differs by float noise. Like ``dispersion``, the scale comes from the
    peers rather than the class itself, so a tight clump that is most of its class still shows;
    the cost is that a class whose images sit that close together by nature reads high too.
    With a single assessable class there are no peers, and a clump of more than half of it goes
    unflagged. Embeddings are auto-rescaled to the unit interval for the coverage
    computation.

    The flag defaults are conservative and tuned for deep embeddings: at 0.5 a flag means a
    class has collapsed. A flag is null where its signal is null, and on every row where the
    median it is judged against is not positive. On an embedding of fewer than about 8
    dimensions, healthy classes can have a near-duplicate share above the 0.1 default, so raise
    ``near_duplicate_threshold`` there. Isotropy rises with class size, so tighten
    ``isotropy_factor`` only for classes of similar size.

    This evaluator assumes one embedding per label (image classification). For
    object detection, wrap the dataset with :class:`~dataeval.data.DetectionCrops` to present
    its boxes as an image-classification dataset (one crop per detection, aligned 1:1 with the
    labels) and evaluate that, or supply detection-level embeddings you have computed yourself.

    References
    ----------
    [1] Identifying Insufficient Data Coverage for Ordinal Continuous-Valued Attributes.
        Asudeh, A., Shahbazi, N., Jin, Z., & Jagadish, H. V. (2021). Proceedings of the 2021 International Conference on
        Management of Data (SIGMOD '21). doi: 10.1145/3448016.3457315 (Source of the coverage radius. The per-class
        dispersion, isotropy and near-duplicate signals this class adds are not from that work.)
        https://dl.acm.org/doi/abs/10.1145/3448016.3457315
    [2] The effective rank: A measure of effective dimensionality.
        Roy, O., & Vetterli, M. (2007). Proceedings of the 15th European Signal Processing Conference (EUSIPCO),
        606-610. (Closest antecedent of the effective-dimensionality estimate behind ``isotropy``; see
        :func:`~dataeval.core.completeness` for the full lineage.)
        https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf

    Examples
    --------
    >>> from dataeval.scope import Coverage
    >>> evaluator = Coverage(extractor, num_observations=20, min_class_samples=20)
    >>> result = evaluator.evaluate(dataset)  # doctest: +SKIP
    >>> result.data()  # per-class breakdown, lowest-dispersion classes first  # doctest: +SKIP
    >>> result.uncovered_indices  # individual samples in sparse regions  # doctest: +SKIP

    Pass pre-computed embeddings to skip extraction:

    >>> result = evaluator.evaluate(dataset, embeddings=embeddings)  # doctest: +SKIP

    The per-class ``dispersion`` column is the signal raw label counts cannot give:
    two classes with identical counts can differ sharply when one's examples are
    near-duplicates (``dispersion`` well below ``1``).
    """

    class Config(EvaluatorConfig):
        """
        Configuration for the :class:`Coverage` evaluator.

        Attributes
        ----------
        extractor : FeatureExtractor or None
            Feature extractor for computing embeddings.
        batch_size : int or None
            Batch size for embedding computation.
        method : {"naive", "adaptive"}, default "adaptive"
            Coverage radius method.
        num_observations : int, default 20
            Neighbors required for a sample to be covered.
        percent : float, default 0.01
            Fraction of samples to flag as uncovered (adaptive method).
        min_class_samples : int, default 20
            Minimum samples for a class's per-class signals to be assessed.
        isotropy_min_samples : int or None, default None
            Minimum samples for a class's isotropy to be reported (None: dims + 1).
        near_duplicate_factor : float, default 0.5
            Fraction of a typical class's median nearest-neighbor distance within which an
            image's nearest neighbor makes it a near-duplicate (a distance scale, which sets
            ``near_duplicate_fraction``).
        dispersion_factor : float, default 0.5
            ``is_clustered`` is True where ``dispersion`` is below this, in (0, 1).
        isotropy_factor : float, default 0.5
            ``is_one_dimensional`` is True where ``isotropy`` is below this share of the median
            isotropy, in (0, 1).
        near_duplicate_threshold : float, default 0.1
            ``has_near_duplicates`` is True where ``near_duplicate_fraction`` is above this
            share of images, in [0, 1].
        """

        extractor: FeatureExtractor | None = None
        batch_size: PositiveInt | None = None
        method: MethodType = "adaptive"
        num_observations: PositiveInt = 20
        percent: OpenUnitInterval = 0.01
        min_class_samples: PositiveInt = 20
        isotropy_min_samples: PositiveInt | None = None
        near_duplicate_factor: PositiveFloat = 0.5
        dispersion_factor: OpenUnitInterval = 0.5
        isotropy_factor: OpenUnitInterval = 0.5
        near_duplicate_threshold: UnitInterval = 0.1

    # Set by apply_config from Config.
    extractor: FeatureExtractor | None
    batch_size: int | None
    method: MethodType
    num_observations: int
    percent: float
    min_class_samples: int
    isotropy_min_samples: int | None
    near_duplicate_factor: float
    dispersion_factor: float
    isotropy_factor: float
    near_duplicate_threshold: float
    config: Config

    def __init__(
        self,
        extractor: FeatureExtractor | None = None,
        *,
        batch_size: int | None = None,
        method: MethodType | None = None,
        num_observations: int | None = None,
        percent: float | None = None,
        min_class_samples: int | None = None,
        isotropy_min_samples: int | None = None,
        near_duplicate_factor: float | None = None,
        dispersion_factor: float | None = None,
        isotropy_factor: float | None = None,
        near_duplicate_threshold: float | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())

    def _embeddings(
        self, dataset: AnnotatedDataset[Any] | LabelsLike | ArrayLike, embeddings: Array | None
    ) -> NDArray[np.float64]:
        """Use pre-computed embeddings if given, else extract them from the dataset."""
        if embeddings is not None:
            return np.asarray(embeddings, dtype=np.float64)
        if self.extractor is None:
            raise ValueError("Provide pre-computed embeddings, or configure an extractor to compute them.")
        from dataeval._embeddings import Embeddings as _Embeddings

        # A Metadata holds factors, not images: embed the dataset it describes. Checked by type,
        # not by probing for `.dataset`, which a torch Subset also has and which names its parent.
        if isinstance(dataset, Metadata):
            if dataset.dataset is None:
                raise ValueError(
                    "This Metadata has no dataset bound (as after Metadata.load), so there are no images "
                    "to embed. Pass embeddings=..., or the dataset it was built from."
                )
            dataset = cast("AnnotatedDataset[Any]", dataset.dataset)
        # Reached only when no embeddings were supplied, i.e. ``dataset`` is a real image
        # dataset (raw labels / LabelsLike are rejected upstream unless embeddings are given).
        image_source = cast("AnnotatedDataset[Any]", dataset)
        return np.asarray(
            _Embeddings(image_source, extractor=self.extractor, batch_size=self.batch_size), dtype=np.float64
        )

    def _labels(
        self,
        dataset: AnnotatedDataset[Any] | LabelsLike | ArrayLike,
        embeddings: Array | None,
    ) -> LabelsLike:
        """Return the labels ``dataset`` carries, as a labels container :meth:`from_embeddings` reads.

        A dataset becomes its :class:`~dataeval.Metadata`, and a raw label array is wrapped. Any
        other labels container passes through as it is, so an axis it records is kept.
        """
        if isinstance(dataset, AnnotatedDataset):
            dataset = Metadata(dataset)
        if is_labels_like(dataset):
            return dataset
        if embeddings is None:
            raise ValueError(
                "Raw class labels require pre-computed embeddings. Pass embeddings=..., or provide a "
                "dataset/Metadata so labels and embeddings can be read/computed from it."
            )
        return BareLabels(np.asarray(dataset, dtype=np.intp).reshape(-1))

    def _coverage(self, embeddings: NDArray[np.float64]) -> CoverageResult:
        """Run the configured global coverage computation (auto-rescaling to unit interval)."""
        if self.method == "naive":
            return coverage_naive(embeddings, self.num_observations, force_unit_interval=True)
        return coverage_adaptive(embeddings, self.num_observations, self.percent, force_unit_interval=True)

    def _isotropy_floor(self, dim: int) -> int:
        """Minimum class size for a non-degenerate isotropy estimate (samples must exceed dims)."""
        return self.isotropy_min_samples if self.isotropy_min_samples is not None else dim + 1

    def _per_class(
        self,
        embeddings: NDArray[np.float64],
        class_labels: NDArray[np.intp],
        index2label: Mapping[int, str],
        uncovered_indices: NDArray[np.intp],
    ) -> list[dict[str, Any]]:
        """Aggregate the single global coverage result by class, assessing intra-class dispersion."""
        is_uncovered = np.zeros(len(class_labels), dtype=bool)
        is_uncovered[uncovered_indices] = True

        masks = {int(cls): class_labels == cls for cls in np.unique(class_labels)}
        counts = {cls: int(mask.sum()) for cls, mask in masks.items()}
        # Raw spread (mean distance to centroid) of each class with enough samples to assess.
        raw_spread = {
            cls: _mean_dispersion(embeddings[mask])
            for cls, mask in masks.items()
            if counts[cls] >= self.min_class_samples
        }
        # Normalize by the median assessable class, so a typical class scores ~1 regardless of
        # how far apart the classes sit in the embedding space (the global spread would conflate
        # within-class variety with between-class separation). Avoid /0.
        median_spread = float(np.median(list(raw_spread.values()))) if raw_spread else 0.0
        reference = median_spread or 1.0

        # One completeness pass per assessable class yields both directional variety (isotropy)
        # and the nearest-neighbor distances that expose near-duplicates.
        results = {
            cls: _completeness(embeddings[mask]) for cls, mask in masks.items() if counts[cls] >= self.min_class_samples
        }
        # Isotropy is orthogonal to dispersion's magnitude: a class can spread far yet vary along
        # a single axis (shot from one angle). The effective-dimension SVD degenerates when samples
        # don't exceed dimensions, so it needs more samples than dispersion (default: more than dims).
        isotropy_floor = max(self.min_class_samples, self._isotropy_floor(embeddings.shape[1]))
        isotropy = {cls: float(r["isotropy"]) for cls, r in results.items() if counts[cls] >= isotropy_floor}
        # Isotropy is not normalized, so its flag compares each class with the median class instead.
        median_isotropy = float(np.median(list(isotropy.values()))) if isotropy else 0.0
        # Each image's distance to its nearest other image in its class; a class of one has none.
        nearest = {cls: _nearest_distances(r, counts[cls]) for cls, r in results.items() if counts[cls] > 1}
        # Like dispersion, measure against a typical (median) class's nearest-neighbor spacing rather than
        # the class's own, which a tight clump covering most of the class would set. `<=` keeps exact
        # duplicates when that spacing is itself zero.
        spacing = float(np.median([np.median(d) for d in nearest.values()])) if nearest else 0.0
        near_duplicate = {cls: float(np.mean(d <= self.near_duplicate_factor * spacing)) for cls, d in nearest.items()}

        rows: list[dict[str, Any]] = []
        for cls, mask in masks.items():
            count = counts[cls]
            uncovered = int(is_uncovered[mask].sum())
            assessable = cls in raw_spread
            dispersion = raw_spread[cls] / reference if assessable else None
            class_isotropy = isotropy.get(cls)
            duplicates = near_duplicate.get(cls)
            rows.append({
                "class": index2label.get(cls, str(cls)),
                "count": count,
                "uncovered": uncovered,
                "uncovered_fraction": uncovered / count if count else 0.0,
                # Spread relative to a typical class; < 1 == clustered / near-duplicate.
                "dispersion": dispersion,
                # In how many independent directions the class varies; null when too few samples.
                "isotropy": class_isotropy,
                # Share of the class's images whose nearest neighbor is a near-duplicate.
                "near_duplicate_fraction": duplicates,
                "assessable": assessable,
                # A flag is null where its signal is, or where the median it is judged against is not positive.
                "is_clustered": (
                    dispersion < self.dispersion_factor if dispersion is not None and median_spread > 0 else None
                ),
                "is_one_dimensional": (
                    class_isotropy < self.isotropy_factor * median_isotropy
                    if class_isotropy is not None and median_isotropy > 0
                    else None
                ),
                "has_near_duplicates": duplicates > self.near_duplicate_threshold if duplicates is not None else None,
            })
        # Lowest-dispersion assessable classes first (most worth broadening); unassessable last.
        return sorted(rows, key=lambda row: (not row["assessable"], row["dispersion"] or 0.0, row["class"]))

    @set_metadata(state=["class_axis", "class_axis_source", "class_axis_level"])
    def evaluate(
        self,
        dataset: AnnotatedDataset[Any] | LabelsLike | ArrayLike,
        embeddings: Array | None = None,
    ) -> CoverageOutput:
        """
        Evaluate a dataset's embedding-space coverage, broken down by class.

        Parameters
        ----------
        dataset : AnnotatedDataset, LabelsLike, or ArrayLike of int
            The source of class labels, accepted in four forms:

            * a full :class:`~dataeval.protocols.AnnotatedDataset` — class labels and their
              ``index2label`` names are read from it, and embeddings are computed from it via
              the configured extractor unless ``embeddings`` is passed;
            * a :class:`~dataeval.Metadata` — labels/names come straight from it, and
              embeddings are computed from the dataset it describes
              (:attr:`~dataeval.Metadata.dataset`) unless ``embeddings`` is passed. A
              Metadata with no dataset bound needs ``embeddings``;
            * any :class:`~dataeval.protocols.LabelsLike` object (e.g. a custom lightweight
              metadata container) — its ``class_labels`` are used, and ``index2label`` if it
              exposes one;
            * a raw sequence/array of integer class labels (one per embedding) — the minimal
              form for when you already hold ``embeddings`` and just a label array, requiring
              no MAITE dataset or Metadata. Classes then name themselves by their integer
              index. Only valid when ``embeddings`` is provided.
        embeddings : Array or None, default None
            Pre-computed embeddings, one per label. When omitted, ``dataset`` must be a full
            dataset or a Metadata bound to one, and an extractor must be configured so embeddings can be computed
            from it. Required when ``dataset`` is a raw label array.

        Returns
        -------
        CoverageOutput
            The per-class breakdown (``count`` / ``uncovered_fraction`` / ``dispersion`` /
            ``assessable``) with sample-level ``uncovered_indices``, ``coverage_radius``,
            and ``critical_value_radii``.

        Examples
        --------
        >>> evaluator = Coverage(crop_extractor, min_class_samples=5)
        >>> result = evaluator.evaluate(cropped_dataset)

        ``data()`` is the per-class breakdown, sorted with the lowest-dispersion (least
        visually varied) classes first. ``car`` here has plenty of crops but the least
        spread — the signal raw counts cannot give:

        >>> result.data().select("class", "count", "uncovered", "dispersion")
        shape: (4, 4)
        ┌────────┬───────┬───────────┬────────────┐
        │ class  ┆ count ┆ uncovered ┆ dispersion │
        │ ---    ┆ ---   ┆ ---       ┆ ---        │
        │ str    ┆ i64   ┆ i64       ┆ f64        │
        ╞════════╪═══════╪═══════════╪════════════╡
        │ car    ┆ 24    ┆ 0         ┆ 0.395929   │
        │ boat   ┆ 22    ┆ 0         ┆ 0.805483   │
        │ plane  ┆ 27    ┆ 0         ┆ 1.194517   │
        │ person ┆ 20    ┆ 1         ┆ 1.488841   │
        └────────┴───────┴───────────┴────────────┘

        The flags give the verdict. ``car`` spreads less than half as far as the median class.
        ``is_one_dimensional`` is null because no class has more crops than the embedding has
        dimensions, so none reports an isotropy:

        >>> result.data().select("class", "is_clustered", "is_one_dimensional", "has_near_duplicates")
        shape: (4, 4)
        ┌────────┬──────────────┬────────────────────┬─────────────────────┐
        │ class  ┆ is_clustered ┆ is_one_dimensional ┆ has_near_duplicates │
        │ ---    ┆ ---          ┆ ---                ┆ ---                 │
        │ str    ┆ bool         ┆ bool               ┆ bool                │
        ╞════════╪══════════════╪════════════════════╪═════════════════════╡
        │ car    ┆ true         ┆ null               ┆ false               │
        │ boat   ┆ false        ┆ null               ┆ false               │
        │ plane  ┆ false        ┆ null               ┆ false               │
        │ person ┆ false        ┆ null               ┆ false               │
        └────────┴──────────────┴────────────────────┴─────────────────────┘
        """
        reject_filtered_metadata(dataset, "Coverage")

        labels = self._labels(dataset, embeddings)
        return self.from_embeddings(self._embeddings(dataset, embeddings), labels)

    @set_metadata(state=["class_axis", "class_axis_source", "class_axis_level"])
    def from_embeddings(self, embeddings: Array, labels: LabelsLike | None = None) -> CoverageOutput:
        """
        Measure coverage from embeddings, and class labels, you already hold.

        :meth:`evaluate` reads or computes both from a dataset and calls this. Call it
        directly when the embeddings are already computed, for example shared with other
        evaluators, so they are not extracted again.

        .. versionadded:: 1.2

        Parameters
        ----------
        embeddings : Array
            One embedding per item, shape ``(n_items, n_features)``.
        labels : LabelsLike or None, default None
            One class label per embedding, such as a :class:`~dataeval.Metadata` of an
            image-classification dataset. Its ``index2label``, where it has one, names the
            classes. When None, every item is one class, ``0``, and the table has one row.

        Returns
        -------
        CoverageOutput
            The same output :meth:`evaluate` returns for the same embeddings and labels.

        Raises
        ------
        ShapeMismatchError
            If ``labels`` does not hold one label per embedding, as with a detection dataset,
            which has one label per target.
        ValueError
            If ``labels`` is a filtered :class:`~dataeval.Metadata`.

        See Also
        --------
        evaluate : Measure a dataset, computing its embeddings with the extractor.

        Examples
        --------
        >>> import numpy as np
        >>> embeddings = np.random.default_rng(0).random((100, 8))
        >>> result = Coverage().from_embeddings(embeddings)
        >>> result.data()["class"].to_list()
        ['0']
        """
        reject_filtered_metadata(labels, "Coverage")
        emb = np.asarray(embeddings, dtype=np.float64)
        # Recorded off whatever container the labels came from: a pivoted Metadata reports its own
        # axis here, and a raw label array, or no labels, reports none.
        self._axis_record = getattr(labels, "class_axis_info", None)
        if labels is None:
            class_labels, index2label = np.zeros(len(emb), dtype=np.intp), {}
        else:
            class_labels = np.asarray(labels.class_labels, dtype=np.intp)
            index2label = getattr(labels, "index2label", None) or {}
        if len(emb) != len(class_labels):
            raise ShapeMismatchError(
                f"Got {len(emb)} embeddings for {len(class_labels)} labels. Coverage assumes one embedding per "
                "label (image classification). For object detection, wrap the dataset with "
                "dataeval.data.DetectionCrops to get one crop per detection aligned 1:1 with the labels, or supply "
                "detection-level embeddings you have computed yourself, aligned 1:1 with the class labels."
            )
        coverage = self._coverage(emb)
        rows = self._per_class(emb, class_labels, index2label, coverage["uncovered_indices"])
        return CoverageOutput(
            pl.DataFrame(rows, schema=_PER_CLASS_SCHEMA),
            uncovered_indices=coverage["uncovered_indices"],
            coverage_radius=coverage["coverage_radius"],
            critical_value_radii=coverage["critical_value_radii"],
            class_axis=self._axis_record,
        )
