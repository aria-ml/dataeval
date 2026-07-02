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
* **near-duplicate fraction** — the *redundancy*, the share of a class sitting in unusually
  tight nearest-neighbor pairs (repeated stock frames), derived from the same computation.

Alongside them, the global :func:`~dataeval.core.coverage_adaptive` result flags individual
samples in under-sampled regions of the whole space.

The global coverage radius is computed *once* over the full dataset and then aggregated per
class — never re-run per class, which would be statistically invalid for the small classes
that matter most. The per-class spread signals (dispersion, isotropy, near-duplicate fraction)
are class-local statistics, so they are computed per class above ``min_class_samples``; classes
below it are reported but left unassessed.
"""

__all__ = ["Coverage", "CoverageOutput"]

import logging
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval import Metadata
from dataeval.core._completeness import completeness as _completeness
from dataeval.core._coverage import CoverageResult, coverage_adaptive, coverage_naive
from dataeval.exceptions import ShapeMismatchError
from dataeval.protocols import AnnotatedDataset, Array, FeatureExtractor
from dataeval.types import DataFrameOutput, Evaluator, EvaluatorConfig, set_metadata

_logger = logging.getLogger(__name__)

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
}


def _mean_dispersion(points: NDArray[np.float64]) -> float:
    """Mean Euclidean distance of ``points`` to their centroid (their spread)."""
    return float(np.linalg.norm(points - points.mean(axis=0), axis=1).mean())


class CoverageOutput(DataFrameOutput):
    """
    A dataset's per-class embedding-space coverage.

    The wrapped DataFrame is the per-class breakdown — one row per class, sorted with the
    lowest-dispersion assessable classes first (the ones most worth collecting varied data
    for) and unassessable classes last — with columns ``class``, ``count``, ``uncovered``,
    ``uncovered_fraction``, ``dispersion``, ``isotropy``, ``near_duplicate_fraction``, and
    ``assessable``. ``isotropy`` and ``near_duplicate_fraction`` are null for classes below
    the sample floors that make them meaningful. The sample-level coverage detail hangs off
    it as attributes.

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
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        uncovered_indices: NDArray[np.intp],
        coverage_radius: float,
        critical_value_radii: NDArray[np.float32],
    ) -> None:
        super().__init__(data)
        self.uncovered_indices = uncovered_indices
        self.coverage_radius = coverage_radius
        self.critical_value_radii = critical_value_radii


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
    * **near_duplicate_fraction** — redundancy (share of the class in unusually tight
      nearest-neighbor pairs); high means repeated / near-identical samples.

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
        A nearest-neighbor pair counts as a near-duplicate when its distance is below this
        fraction of the class's median nearest-neighbor distance (scale-free).
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
    ``dispersion``, which only measures how far it spreads. ``near_duplicate_fraction`` is the
    share of within-class nearest-neighbor pairs closer than ``near_duplicate_factor`` x the
    class median, surfacing repeated / near-identical samples that inflate counts without
    adding variety. Embeddings are auto-rescaled to the unit interval for the coverage
    computation. This evaluator assumes one embedding per label (image classification). For
    object detection, wrap the dataset with :class:`~dataeval.data.DetectionCrops` to present
    its boxes as an image-classification dataset (one crop per detection, aligned 1:1 with the
    labels) and evaluate that, or supply detection-level embeddings you have computed yourself.

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
            Fraction of the class median nearest-neighbor distance below which a pair is a
            near-duplicate.
        """

        extractor: FeatureExtractor | None = None
        batch_size: int | None = None
        method: MethodType = "adaptive"
        num_observations: int = 20
        percent: float = 0.01
        min_class_samples: int = 20
        isotropy_min_samples: int | None = None
        near_duplicate_factor: float = 0.5

    # Set by apply_config from Config.
    extractor: FeatureExtractor | None
    batch_size: int | None
    method: MethodType
    num_observations: int
    percent: float
    min_class_samples: int
    isotropy_min_samples: int | None
    near_duplicate_factor: float
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
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())

    def _embeddings(self, dataset: AnnotatedDataset[Any] | Metadata, embeddings: Array | None) -> NDArray[np.float64]:
        """Use pre-computed embeddings if given, else extract them from the dataset."""
        if embeddings is not None:
            return np.asarray(embeddings, dtype=np.float64)
        if self.extractor is None:
            raise ValueError("Provide pre-computed embeddings, or configure an extractor to compute them.")
        from dataeval._embeddings import Embeddings as _Embeddings

        return np.asarray(_Embeddings(dataset, extractor=self.extractor, batch_size=self.batch_size), dtype=np.float64)

    def _coverage(self, embeddings: NDArray[np.float64]) -> CoverageResult:
        """Run the configured global coverage computation (auto-rescaling to unit interval)."""
        if self.method == "naive":
            return coverage_naive(embeddings, self.num_observations, force_unit_interval=True)
        return coverage_adaptive(embeddings, self.num_observations, self.percent, force_unit_interval=True)

    def _isotropy_floor(self, dim: int) -> int:
        """Minimum class size for a non-degenerate isotropy estimate (samples must exceed dims)."""
        return self.isotropy_min_samples if self.isotropy_min_samples is not None else dim + 1

    def _near_duplicate_fraction(self, distances: Any) -> float | None:
        """Share of nearest-neighbor pairs closer than ``near_duplicate_factor`` x the class median.

        Relative-to-median so it is scale-free. A zero median means at least half the pairs are
        exact duplicates, so the exact-duplicate share is reported instead.
        """
        dists = np.asarray(distances, dtype=np.float64)
        if dists.size == 0:
            return None
        median = float(np.median(dists))
        if median == 0.0:
            return float(np.mean(dists == 0.0))
        return float(np.mean(dists < self.near_duplicate_factor * median))

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
        reference = (float(np.median(list(raw_spread.values()))) if raw_spread else 1.0) or 1.0

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
        near_duplicate = {
            cls: self._near_duplicate_fraction(r["nearest_neighbor_distances"]) for cls, r in results.items()
        }

        rows: list[dict[str, Any]] = []
        for cls, mask in masks.items():
            count = counts[cls]
            uncovered = int(is_uncovered[mask].sum())
            assessable = cls in raw_spread
            rows.append({
                "class": index2label.get(cls, str(cls)),
                "count": count,
                "uncovered": uncovered,
                "uncovered_fraction": uncovered / count if count else 0.0,
                # Spread relative to a typical class; < 1 == clustered / near-duplicate.
                "dispersion": raw_spread[cls] / reference if assessable else None,
                # In how many independent directions the class varies; null when too few samples.
                "isotropy": isotropy.get(cls),
                # Share of the class sitting in near-duplicate nearest-neighbor pairs.
                "near_duplicate_fraction": near_duplicate.get(cls),
                "assessable": assessable,
            })
        # Lowest-dispersion assessable classes first (most worth broadening); unassessable last.
        return sorted(rows, key=lambda row: (not row["assessable"], row["dispersion"] or 0.0, row["class"]))

    @set_metadata
    def evaluate(self, dataset: AnnotatedDataset[Any] | Metadata, embeddings: Array | None = None) -> CoverageOutput:
        """
        Evaluate a dataset's embedding-space coverage, broken down by class.

        Parameters
        ----------
        dataset : AnnotatedDataset or Metadata
            The dataset to evaluate. Class labels are read from it; embeddings are computed
            from it via the configured extractor unless provided directly.
        embeddings : Array or None, default None
            Pre-computed embeddings, one per label. When omitted, an extractor must be
            configured and embeddings are computed from ``dataset``.

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
        """
        metadata = dataset if isinstance(dataset, Metadata) else Metadata(dataset)
        emb = self._embeddings(dataset, embeddings)
        class_labels = np.asarray(metadata.class_labels)
        if len(emb) != len(class_labels):
            raise ShapeMismatchError(
                f"Got {len(emb)} embeddings for {len(class_labels)} labels. Coverage assumes one embedding per "
                "label (image classification). For object detection, wrap the dataset with "
                "dataeval.data.DetectionCrops to get one crop per detection aligned 1:1 with the labels, or supply "
                "detection-level embeddings you have computed yourself, aligned 1:1 with metadata.class_labels."
            )
        coverage = self._coverage(emb)
        rows = self._per_class(emb, class_labels, metadata.index2label, coverage["uncovered_indices"])
        return CoverageOutput(
            pl.DataFrame(rows, schema=_PER_CLASS_SCHEMA),
            uncovered_indices=coverage["uncovered_indices"],
            coverage_radius=coverage["coverage_radius"],
            critical_value_radii=coverage["critical_value_radii"],
        )
