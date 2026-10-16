"""Closed-set drift detector for data that should still belong to one of several known reference domains."""

__all__ = []

import json
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypedDict, cast

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from pydantic import Field, field_validator
from scipy.stats import fisher_exact
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from dataeval import Metadata
from dataeval._experimental import experimental
from dataeval.config import get_max_processes, get_seed
from dataeval.exceptions import NotFittedError, ShapeMismatchError
from dataeval.protocols import AnnotatedDataset, FeatureExtractor, Threshold
from dataeval.shift._drift._base import BaseDrift, ChunkableMixin, DriftOutput, apply_correction
from dataeval.shift._drift._domain_spec import DomainSpec, item_values
from dataeval.types import EvaluatorConfig, set_metadata
from dataeval.utils._array import flatten_samples
from dataeval.utils.thresholds import ConstantThreshold

_MIN_DOMAINS = 3
_FORMAT = 2
_METADATA_KEY = "dataeval.DriftKnownDomains"
_DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 40,
    "num_leaves": 15,
    "max_depth": 4,
    "learning_rate": 0.05,
}

ThresholdMode = Literal["balanced", "conservative", "sensitive"]


class _DriftKnownDomainsStats(TypedDict):
    p_val: float
    scores: NDArray[np.float32]
    closest_domain: str
    domain_probabilities: dict[str, float]
    predicted_domains: NDArray[np.str_]
    domains: pl.DataFrame


def _select_threshold(
    scores: NDArray[np.floating[Any]],
    is_novel: NDArray[np.floating[Any]],
    mode: ThresholdMode,
    tolerance: float,
) -> tuple[float, float, float]:
    """Pick the score at which a sample is flagged as novel, from known (0) and novel (1) scores.

    Returns the threshold with its false positive and true positive rate.
    """
    false_pos, true_pos, thresholds = roc_curve(is_novel, scores)
    if mode == "conservative":  # the most sensitive threshold that still keeps FPR within tolerance
        index = np.flatnonzero(false_pos <= tolerance)[-1]
    elif mode == "sensitive":  # the strictest threshold that still reaches the target TPR
        index = np.flatnonzero(true_pos >= 1 - tolerance)[0]
    else:
        index = np.argmax(true_pos - false_pos)
    # sklearn's leading threshold is infinity; a score never exceeds 1, so 1 flags nothing just the same
    return float(min(thresholds[index], 1.0)), float(false_pos[index]), float(true_pos[index])


def _novelty(model: LGBMClassifier, x: NDArray[np.float32]) -> NDArray[np.float64]:
    """Novelty score ``1 - max(P)`` of each row."""
    return 1.0 - np.asarray(model.predict_proba(x), dtype=np.float64).max(axis=1)


def _stack_domains(
    reference_data: Mapping[str, Any], min_samples: int
) -> tuple[tuple[str, ...], NDArray[np.float32], NDArray[np.intp]]:
    """Validate the named domains and stack them into one feature matrix and its domain labels."""
    domains = {str(k): np.atleast_2d(np.asarray(v, dtype=np.float32)) for k, v in reference_data.items()}
    if len(domains) < _MIN_DOMAINS:
        raise ValueError(f"DriftKnownDomains needs at least three reference domains, got {len(domains)}.")
    widths = {d.shape[1] for d in domains.values()}
    if len(widths) != 1:
        raise ShapeMismatchError(f"Reference domains have different feature counts: {sorted(widths)}.")
    smallest = min(len(d) for d in domains.values())
    if smallest < min_samples:
        raise ValueError(f"Every domain needs at least n_folds={min_samples} samples, smallest has {smallest}.")
    y = np.repeat(np.arange(len(domains)), [len(d) for d in domains.values()]).astype(np.intp)
    return tuple(domains), np.concatenate(list(domains.values())), y


_TABLE_SCHEMA = {
    "domain": pl.String,
    "n": pl.Int64,
    "flagged_fraction": pl.Float64,
    "novelty_p_val": pl.Float64,
    "agreement": pl.Float64,
    "agreement_p_val": pl.Float64,
    "closest_domain": pl.String,
    "drifted": pl.Boolean,
}


def _greater(bad: int, n: int, ref_bad: int, ref_n: int) -> float:
    """Return the one-sided Fisher's exact p-value that ``bad / n`` exceeds ``ref_bad / ref_n``."""
    table = [[bad, n - bad], [ref_bad, ref_n - ref_bad]]
    # float(): scipy returns np.float64, and a comparison with it gives an np.bool_ that polars
    # builds into a float column. cast(): the stubs type the result as a bare tuple.
    return float(cast(float, fisher_exact(table, alternative="greater")[1]))


@experimental
class DriftKnownDomains(ChunkableMixin, BaseDrift[_DriftKnownDomainsStats]):
    """Drift detector for data that should still belong to one of several known domains.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Each known *domain* (for example a sensor, a site or a season) is either a named
    reference dataset, or the samples of one reference dataset whose metadata factor falls
    in that domain's group of values (``domains=``).
    A multi-class LightGBM classifier learns to tell the domains apart, and a sample's
    novelty score is ``1 - max(P)``: how unsure the classifier is that it belongs to any
    one of them. Data from a domain the classifier has not seen is the data it is least
    sure about.

    :meth:`fit` sets the score threshold from the reference alone. Each domain is scored by
    a classifier trained without it (leave-one-domain-out) to stand in for novel data, and
    every reference sample is scored out-of-fold to stand in for known data. :meth:`predict`
    flags the samples at or above that threshold and tests whether a larger share of them
    is flagged than of the held-out reference data of the known domain they are closest to.
    When the test data declares its domains, through its metadata or ``domain_labels``,
    each declared domain is also tested on its own: for novelty against that domain's
    held-out reference rate, and for *agreement*, whether the classifier still names the
    declared domain as often as it did for the reference. The tests are corrected for
    multiple testing by ``correction``.

    Unlike :class:`~dataeval.shift.DriftDomainClassifier`, which compares the test data with
    the reference as a whole, this reports which known domain the test data is closest to,
    and a fitted detector can be written to an ONNX file with :meth:`save` and restored
    with :meth:`load` to warm start monitoring without the reference data.

    :meth:`chunked` splits the test data into chunks and gives each the test :meth:`predict`
    runs on undeclared data: its p-value against the closest domain's held-out reference
    rate, flagged below ``p_val`` (a :class:`~dataeval.utils.thresholds.ConstantThreshold`).
    A threshold drawn from reference chunks is refused, because the classifier was trained
    on them and they look far less novel than unseen data from the same domains.

    .. versionadded:: 1.2

    Parameters
    ----------
    n_folds : int, default 5
        Number of cross-validation folds used to score known data out-of-fold.
    mode : {"conservative", "balanced", "sensitive"}, default "conservative"
        How the score threshold is chosen. ``"conservative"`` keeps the false positive rate
        within ``tolerance``. ``"balanced"`` maximizes true positive minus false positive
        rate (Youden's J). ``"sensitive"`` keeps the true positive rate at or above
        ``1 - tolerance``. The last two place the threshold among the confident scores that
        most known data shares, where a small difference between the cross-validated
        classifiers and the final one moves the flag rate a great deal, so they raise
        false alarms; ``"conservative"`` places it in the sparse tail.
    tolerance : float, default 0.05
        The false positive rate (``"conservative"``) or missed-detection rate
        (``"sensitive"``) allowed. Unused by ``"balanced"``.
    p_val : float, default 0.05
        Significance level for the test that more samples are flagged than expected.
    hyperparameters : dict or None, default None
        LightGBM hyperparameters. When None, a shallow, regularized configuration is
        used; when given, it replaces that configuration entirely.
    domains : str, dict or None, default None
        The metadata factor that splits one reference dataset into domains. A factor name
        alone makes each distinct value a domain. ``{factor: {name: group}}`` names the
        domains, where a group is a value, a ``(low, high)`` range (half-open, None at
        either end unbounded), a list of those, or None for every value no other group
        matches. ``{factor: [group, ...]}`` names them from their values. The same groups
        read the test data's metadata in :meth:`predict`.
    extractor : FeatureExtractor or None, default None
        Feature extractor applied to data before detection, so :meth:`fit` and
        :meth:`predict` can take raw data such as an annotated dataset. Not written by
        :meth:`save`; pass it again to :meth:`load`.
    correction : {"bonferroni", "fdr"}, default "bonferroni"
        Multiple testing correction over the per-domain tests. ``"bonferroni"`` controls
        the family-wise error rate; ``"fdr"`` (Benjamini-Hochberg) is less conservative.
    config : DriftKnownDomains.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    Notes
    -----
    The score measures how unsure the classifier is, so it finds data that is confusable
    between known domains, such as a blend of two of them or a new domain that lies
    between them. Tree ensembles stay confident far outside their training range, so data
    beyond every known domain scores as confident and is not flagged. Use
    :class:`~dataeval.shift.DriftDomainClassifier` for a shift of that kind.

    In ``domains=``, a tuple is always a range and a list is a group of values, as in
    :class:`~dataeval.types.Remap`: ``(1, 2)`` is ``1 <= x < 2``, so write ``[1, 2]`` for
    two camera IDs. A number matches however the factor stores it, so ``3`` matches ``3.0``
    in a float column, but a number never matches a boolean.

    Examples
    --------
    Split one reference dataset into domains by a metadata factor:

    >>> from dataeval import Metadata
    >>> rng = np.random.default_rng(0)
    >>> x = np.concatenate([rng.normal(c, 1.0, size=(100, 4)) for c in (0.0, 4.0, 8.0)])
    >>> time = ["morning"] * 50 + ["noon"] * 50 + ["dusk"] * 100 + ["night"] * 100
    >>> spec = {"time": {"day": ["morning", "noon"], "dusk": "dusk", "night": "night"}}
    >>> detector = DriftKnownDomains(n_folds=3, domains=spec).fit(x, metadata=Metadata.from_factors({"time": time}))
    >>> detector.domain_names
    ('day', 'dusk', 'night')

    Test data that declares itself "dusk" but now looks like night:

    >>> result = detector.predict(rng.normal(8.0, 1.0, size=(40, 4)), domain_labels=["dusk"] * 40)
    >>> result.details["domains"]["closest_domain"].to_list()
    ['night']

    Or fit on named reference datasets directly:

    >>> rng = np.random.default_rng(0)
    >>> reference = {
    ...     name: rng.normal(center, 1.0, size=(100, 4))
    ...     for name, center in {"day": 0.0, "dusk": 4.0, "night": 8.0}.items()
    ... }
    >>> detector = DriftKnownDomains(n_folds=3).fit(reference)
    >>> result = detector.predict(rng.normal(4.0, 1.0, size=(50, 4)))
    >>> result.details["closest_domain"]
    'dusk'
    """

    class Stats(_DriftKnownDomainsStats):
        """Statistics from known-domains drift detection.

        Attributes
        ----------
        p_val : float
            Smallest uncorrected p-value in ``domains``. With no declared domains, the
            one-sided Fisher's exact p-value that a larger share of the samples is flagged
            than of the closest domain's held-out reference data.
        scores : NDArray[np.float32]
            Novelty score ``1 - max(P)`` of each sample, shape ``(n_samples,)``.
        closest_domain : str
            Name of the domain with the highest mean probability.
        domain_probabilities : dict[str, float]
            Mean probability the classifier gives each domain over the samples.
        predicted_domains : NDArray[np.str_]
            The most likely domain of each sample, shape ``(n_samples,)``.
        domains : pl.DataFrame
            One row per declared domain in the batch, in domain order, then one with a null
            ``domain`` for undeclared samples. Columns: ``domain``; ``n`` samples;
            ``flagged_fraction``; ``novelty_p_val``, against the declared domain's held-out
            reference rate (the closest domain's, when undeclared); ``agreement``, the share
            whose most likely domain is the declared one, and ``agreement_p_val``, that it
            is lower than for the reference (both null when undeclared);
            ``closest_domain``, by mean probability over the row; and ``drifted``, whether
            either p-value falls below the corrected threshold.
        """

    class Config(EvaluatorConfig):
        """
        Configuration for DriftKnownDomains detector.

        Attributes
        ----------
        n_folds : int, default 5
            Number of cross-validation folds.
        mode : {"conservative", "balanced", "sensitive"}, default "conservative"
            How the score threshold is chosen.
        tolerance : float, default 0.05
            The rate allowed by the ``"conservative"`` and ``"sensitive"`` modes.
        p_val : float, default 0.05
            Significance level for the drift test.
        hyperparameters : dict or None, default None
            LightGBM hyperparameters. Replaces the default configuration when given.
        domains : str, dict or None, default None
            Metadata factor, and optionally its value groups, that splits one reference
            dataset into domains. Held in its named form.
        extractor : FeatureExtractor or None, default None
            Feature extractor applied to data before detection.
        correction : {"bonferroni", "fdr"}, default "bonferroni"
            Multiple testing correction over the per-domain tests.
        """

        n_folds: Annotated[int, Field(ge=2)] = 5
        mode: ThresholdMode = "conservative"
        tolerance: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
        p_val: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
        hyperparameters: dict[str, Any] | None = None
        # Any, not a union of scalars: pydantic would coerce np.int64(1) to 1.0, and matching
        # compares types. DomainSpec.parse validates and normalizes instead.
        domains: Any = None
        extractor: FeatureExtractor | None = None
        correction: Literal["bonferroni", "fdr"] = "bonferroni"

        @field_validator("domains")
        @classmethod
        def _normalize_domains(cls, value: Any) -> Any:
            return None if value is None else DomainSpec.parse(value).canonical()

    def __init__(
        self,
        n_folds: int | None = None,
        mode: ThresholdMode | None = None,
        tolerance: float | None = None,
        p_val: float | None = None,
        hyperparameters: dict[str, Any] | None = None,
        domains: Any = None,
        extractor: FeatureExtractor | None = None,
        correction: Literal["bonferroni", "fdr"] | None = None,
        config: Config | None = None,
    ) -> None:
        base_config = config or DriftKnownDomains.Config()
        self.config: DriftKnownDomains.Config = DriftKnownDomains.Config(
            n_folds=n_folds if n_folds is not None else base_config.n_folds,
            mode=mode if mode is not None else base_config.mode,
            tolerance=tolerance if tolerance is not None else base_config.tolerance,
            p_val=p_val if p_val is not None else base_config.p_val,
            hyperparameters=hyperparameters if hyperparameters is not None else base_config.hyperparameters,
            domains=domains if domains is not None else base_config.domains,
            extractor=extractor if extractor is not None else base_config.extractor,
            correction=correction if correction is not None else base_config.correction,
        )
        self._n_folds = self.config.n_folds
        self._mode = self.config.mode
        self._tolerance = self.config.tolerance
        self._p_val = self.config.p_val
        self._hyperparameters = self.config.hyperparameters
        self._spec = None if self.config.domains is None else DomainSpec.parse(self.config.domains)
        self._extractor = self.config.extractor
        self._correction = self.config.correction

        BaseDrift.__init__(self)
        self._metric_name = "flagged_fraction"
        self._domains: tuple[str, ...] = ()
        self._n_features = 0
        self._score_threshold = 1.0
        self._null_counts = np.zeros((0, 3), dtype=np.int64)  # per domain, out of fold: held out, flagged, agreed
        self._model: LGBMClassifier | None = None  # set by fit
        self._session: Any = None  # set by load, in place of a model

    @property
    def reference_data(self) -> NDArray[np.float32]:
        """Reference embeddings of every domain, stacked in domain order, as :meth:`fit` saw them.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        RuntimeError
            If the detector was restored by :meth:`load`, which does not hold the reference.
        """
        if self._reference_data is None and self._fitted:
            raise RuntimeError("This detector was restored by load() and holds no reference data. Call fit().")
        return super().reference_data

    @property
    def domain_names(self) -> tuple[str, ...]:
        """Names of the reference domains, in the order their probabilities are reported."""
        return self._domains

    @property
    def score_threshold(self) -> float:
        """Novelty score at or above which a sample is flagged, set by :meth:`fit`."""
        if not self._fitted:
            raise NotFittedError("Must call fit() before accessing score_threshold.")
        return self._score_threshold

    def _train(self, x: NDArray[np.float32], y: NDArray[np.intp]) -> LGBMClassifier:
        params = {"verbosity": -1, **(self._hyperparameters or _DEFAULT_HYPERPARAMETERS)}
        return LGBMClassifier(**params, n_jobs=get_max_processes(), random_state=get_seed()).fit(x, y)

    def _encode(self, data: Any) -> NDArray[np.float32]:
        """Embeddings of ``data``: through the extractor when there is one, one row per sample."""
        if self._extractor is not None:
            data = self._extractor(data)
        try:
            x = np.asarray(data, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "`data` must be Array-like or provide an `extractor` to convert your data to an array."
            ) from e
        # A 1-D array is one sample, as before; anything higher is flattened per sample.
        return np.atleast_2d(x) if x.ndim < 2 else flatten_samples(x)

    def _score_reference(
        self, x: NDArray[np.float32], y: NDArray[np.intp]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Score every reference sample as known data (out of fold) and as novel data (its domain left out).

        Also returns whether each out-of-fold prediction named the sample's own domain.
        """
        novel = np.empty(len(x), dtype=np.float64)
        for k in range(y.max() + 1):
            held_out = y == k
            remaining = y[~held_out]
            model = self._train(x[~held_out], remaining - (remaining > k))  # relabel to 0..n-2
            novel[held_out] = _novelty(model, x[held_out])

        known = np.empty(len(x), dtype=np.float64)
        agreed = np.empty(len(x), dtype=np.bool_)
        for train, test in StratifiedKFold(self._n_folds, shuffle=True, random_state=get_seed()).split(x, y):
            probabilities = np.asarray(self._train(x[train], y[train]).predict_proba(x[test]), dtype=np.float64)
            known[test] = 1.0 - probabilities.max(axis=1)
            agreed[test] = probabilities.argmax(axis=1) == y[test]
        return known, novel, agreed

    def _metadata_of(self, data: Any, metadata: Any) -> Any:
        """Return the metadata given, else one built from an annotated dataset, else None."""
        if metadata is not None:
            return metadata
        if isinstance(data, AnnotatedDataset):
            return Metadata(data)
        return None

    def _split_by_spec(self, reference_data: Any, metadata: Any) -> dict[str, NDArray[np.float32]]:  # noqa: C901
        """Split one reference dataset into the spec's domains, dropping samples in none of them."""
        spec = cast(DomainSpec, self._spec)
        md = self._metadata_of(reference_data, metadata)
        if md is None:  # before encoding, so a missing source does not cost a full extractor pass
            raise ValueError("`domains=` needs metadata: pass `metadata=` or an annotated dataset.")
        x = self._encode(reference_data)
        values = item_values(md, spec.factor, len(x), "DriftKnownDomains.fit")
        names = spec.assign(values)
        dropped = Counter(
            "null" if value is None else str(value) for value, name in zip(values, names, strict=True) if name is None
        )
        if dropped:
            warnings.warn(
                f"DriftKnownDomains.fit dropped {sum(dropped.values())} reference samples whose "
                f"{spec.factor!r} is in no domain: {dict(dropped)}.",
                UserWarning,
                stacklevel=3,
            )
        assigned = np.array([name is not None for name in names])
        domains: dict[str, NDArray[np.float32]] = {}
        for name in spec.domain_order(values):
            mask = assigned & np.array([n == name for n in names])
            if not mask.any():
                if spec.groups is not None and dict(spec.groups)[name] is None:
                    raise ValueError(
                        f"Domain {name!r} is the catch-all, and every reference sample matched another domain, "
                        f"so it has no samples. Remove it or give it data.",
                    )
                raise ValueError(f"Domain {name!r} has no reference samples.")
            domains[name] = x[mask]
        return domains

    def fit(self, reference_data: Any, *, metadata: Any = None) -> Self:
        """Fit on the known domains: named reference datasets, or one dataset split by ``domains=``.

        Parameters
        ----------
        reference_data : Any
            Either a mapping of domain name to that domain's data, or, when the detector
            has ``domains=``, one dataset whose metadata factor splits it into domains.
            Data is array-like with shape ``(n_samples, ...)``, or anything the
            ``extractor`` accepts (an annotated dataset, :class:`~dataeval.Embeddings`).
            At least three domains are needed so that leaving one out still leaves two to
            tell apart.
        metadata : Metadata or None, default None
            Metadata of ``reference_data`` for ``domains=`` to read. When None, it is built
            from ``reference_data`` if that is an annotated dataset. Reference samples whose
            value is in no domain, or missing, are dropped with a warning.

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            If there are fewer than three domains, or a domain has fewer samples than
            ``n_folds``; if a mapping is given alongside ``domains=`` or ``metadata=``; if
            one dataset is given without ``domains=``, or without metadata to read; if a
            domain of ``domains=`` has no reference samples; or if the metadata is filtered,
            lacks the factor, or records it below the item level.
        ShapeMismatchError
            If the domains do not share a feature count, or the metadata's items do not pair
            one to one with the samples.
        """
        if isinstance(reference_data, Mapping):
            if self._spec is not None:
                raise ValueError(
                    "The mapping already defines the domains; remove `domains=` from the detector, "
                    "or pass one dataset for `domains=` to split."
                )
            if metadata is not None:
                raise ValueError(
                    "`metadata=` is for splitting one dataset with `domains=`; a mapping is already split."
                )
            domains = {str(k): self._encode(v) for k, v in reference_data.items()}
        else:
            if self._spec is None:
                raise ValueError(
                    "fit() takes a mapping of domain name to data, or one dataset with `domains=` set on the "
                    "detector to split it by a metadata factor."
                )
            domains = self._split_by_spec(reference_data, metadata)
        names, x, y = _stack_domains(domains, self._n_folds)
        known, novel, agreed = self._score_reference(x, y)
        scores = np.concatenate([known, novel])
        is_novel = np.concatenate([np.zeros(len(known)), np.ones(len(novel))])
        threshold, _, _ = _select_threshold(scores, is_novel, self.config.mode, self.config.tolerance)

        self._domains = names
        self._n_features = x.shape[1]
        self._reference_data = x
        self._score_threshold = threshold
        # Per domain, since a middle domain is ambiguous by position and a pooled rate would flag it
        # far above the edge domains.
        self._null_counts = np.column_stack([
            np.bincount(y),
            np.bincount(y, weights=(known >= threshold)),
            np.bincount(y, weights=agreed),
        ]).astype(np.int64)
        self._model = self._train(x, y)
        self._session = None
        self._fitted = True
        return self

    def _predict_proba(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        if self._model is not None:
            return np.asarray(self._model.predict_proba(x), dtype=np.float64)
        return np.asarray(self._session.run(["probabilities"], {"features": x})[0], dtype=np.float64)

    def _prepare_data(self, data: Any) -> NDArray[np.float32]:
        """Encode test data and check it has the reference's feature count."""
        x = self._encode(data)
        if x.shape[1] != self._n_features:
            raise ShapeMismatchError("Reference and test embeddings have different number of features")
        return x

    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Return the chunk's closest-domain test p-value, the one :meth:`predict` runs on undeclared data."""
        probabilities = self._predict_proba(chunk_data)
        flagged = 1.0 - probabilities.max(axis=1) >= self._score_threshold
        everything = np.ones(len(chunk_data), dtype=np.bool_)
        return self._row(-1, everything, probabilities, probabilities.argmax(axis=1), flagged)["novelty_p_val"]

    def _default_chunk_threshold(self) -> Threshold:
        """Flag a chunk whose p-value falls below ``p_val``."""
        return ConstantThreshold(lower=self._p_val)

    def _check_chunk_threshold(self, threshold: Threshold) -> None:
        """Refuse a threshold derived from reference chunks, whose baseline would be in-sample.

        ``ChunkedDrift`` scores reference chunks with the final classifier, which was trained
        on them, so they look far less novel than unseen data from the same domains; bounds
        drawn from their spread would flag known data. A p-value against the out-of-fold
        reference rate needs no baseline, so only a constant threshold is meaningful.
        """
        if not isinstance(threshold, ConstantThreshold):
            raise ValueError(
                f"DriftKnownDomains chunks are judged by a p-value, so they take a ConstantThreshold, e.g. "
                f"ConstantThreshold(lower=0.01); {threshold!r} would derive bounds from in-sample reference "
                f"chunks, which look far less novel than unseen data."
            )

    def _declared(  # noqa: C901
        self, data: Any, metadata: Any, domain_labels: Sequence[Any] | None, n: int
    ) -> NDArray[np.intp]:
        """Each sample's declared domain index, or -1 where it declares none the detector knows."""
        index = {name: i for i, name in enumerate(self._domains)}
        if domain_labels is not None:
            labels = list(domain_labels)
            if len(labels) != n:
                raise ShapeMismatchError(f"domain_labels has {len(labels)} labels for {n} samples.")
            # repr for non-text labels, so a raw value 1 does not read as the domain name "1"
            unknown = sorted({
                repr(str(label)) if isinstance(label, str) else repr(label)
                for label in labels
                if label is not None and label not in index
            })
            if unknown:
                warnings.warn(
                    f"domain_labels names domains the detector does not know: [{', '.join(unknown)}]; labels are "
                    f"domain names, from {list(self._domains)}. Those samples are tested as undeclared.",
                    UserWarning,
                    stacklevel=3,
                )
            return np.array([index.get(label, -1) if label is not None else -1 for label in labels], dtype=np.intp)

        undeclared = np.full(n, -1, dtype=np.intp)
        if metadata is not None and self._spec is None:
            raise ValueError(
                "predict(metadata=) applies the detector's `domains=` spec, and this detector has none; "
                "pass domain_labels= instead."
            )
        md = self._metadata_of(data, metadata) if self._spec is not None else None
        if md is None:
            return undeclared
        spec = cast(DomainSpec, self._spec)
        if metadata is None and spec.factor not in md.factor_info:
            warnings.warn(
                f"The test metadata has no {spec.factor!r}; reporting the closest-domain test only. If the "
                f"factor is derived with a correction, pass metadata= built with the same corrections.",
                UserWarning,
                stacklevel=3,
            )
            return undeclared
        names = spec.assign(item_values(md, spec.factor, n, "DriftKnownDomains.predict"))
        declared = np.array([index.get(name, -1) if name is not None else -1 for name in names], dtype=np.intp)
        if (declared == -1).all():
            warnings.warn(
                f"The test metadata's {spec.factor!r} matched no sample to a known domain. Exact values must "
                f"match the column's type: 3 does not match a float column's 3.0.",
                UserWarning,
                stacklevel=3,
            )
        return declared

    def _row(
        self,
        k: int,
        mask: NDArray[np.bool_],
        probabilities: NDArray[np.float64],
        predicted: NDArray[np.intp],
        flagged: NDArray[np.bool_],
    ) -> dict[str, Any]:
        """Novelty and agreement for the samples declaring domain ``k`` (-1: undeclared)."""
        n = int(mask.sum())
        bad = int(flagged[mask].sum())
        closest = int(probabilities[mask].mean(axis=0).argmax())
        held_out, ref_flagged, ref_agreed = (int(c) for c in self._null_counts[k if k >= 0 else closest])
        row: dict[str, Any] = {
            "domain": self._domains[k] if k >= 0 else None,
            "n": n,
            "flagged_fraction": bad / n,
            "novelty_p_val": _greater(bad, n, ref_flagged, held_out),
            "agreement": None,
            "agreement_p_val": None,
            "closest_domain": self._domains[closest],
            "drifted": False,
        }
        if k >= 0:
            agreed = int((predicted[mask] == k).sum())
            row["agreement"] = agreed / n
            row["agreement_p_val"] = _greater(n - agreed, n, held_out - ref_agreed, held_out)
        return row

    @set_metadata
    def predict(
        self, data: Any, *, metadata: Any = None, domain_labels: Sequence[Any] | None = None
    ) -> DriftOutput["DriftKnownDomains.Stats"]:
        """
        Test whether data still looks like the known domains, per declared domain when known.

        Each sample's declared domain comes from the first of: ``domain_labels``;
        ``metadata`` read through ``domains=``; the metadata of ``data`` when it is an
        annotated dataset. Without one, every sample is undeclared and the batch is tested
        against the known domain it is closest to.

        Parameters
        ----------
        data : Any
            Data to test: array-like with shape ``(n_samples, ...)``, or anything the
            ``extractor`` accepts.
        metadata : Metadata or None, default None
            Metadata of ``data`` for ``domains=`` to read. Samples whose value is in no
            known domain, or missing, are tested as undeclared.
        domain_labels : Sequence or None, default None
            The declared domain name of each sample. Names the detector does not know, and
            None, are undeclared.

        Returns
        -------
        DriftOutput[DriftKnownDomains.Stats]
            ``distance`` is the fraction of samples flagged as novel and ``metric_name`` is
            ``"flagged_fraction"``. ``details["domains"]`` holds one row per declared domain
            in the batch, and a row with a null ``domain`` for undeclared samples. Each
            declared row has a novelty test against its own domain's held-out reference rate
            and an agreement test; the undeclared row has a novelty test against its closest
            domain. ``threshold`` is ``p_val`` corrected over all of them by ``correction``,
            and ``drifted`` is whether any test falls below it.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit` or :meth:`load`.
        ValueError
            If both ``metadata`` and ``domain_labels`` are given; if ``metadata`` is given to
            a detector without ``domains=``; or if ``metadata`` lacks the factor.
        ShapeMismatchError
            If the feature count differs from the reference, or ``domain_labels`` or the
            metadata does not have one entry per sample.

        Warns
        -----
        UserWarning
            If ``domain_labels`` names unknown domains, if an annotated dataset's metadata
            lacks the factor (only the closest-domain test is reported), or if the metadata
            matches no sample to a known domain.
        """
        if not self._fitted:
            raise NotFittedError("Must call fit() or load() before predict().")
        if metadata is not None and domain_labels is not None:
            raise ValueError("Pass metadata= or domain_labels=, not both.")
        x = self._prepare_data(data)

        probabilities = self._predict_proba(x)
        scores = 1.0 - probabilities.max(axis=1)
        predicted = probabilities.argmax(axis=1)
        flagged = scores >= self._score_threshold
        declared = self._declared(data, metadata, domain_labels, len(x))

        present = [k for k in range(len(self._domains)) if (declared == k).any()]
        if (declared == -1).any():
            present.append(-1)
        rows = [self._row(k, declared == k, probabilities, predicted, flagged) for k in present]
        row_p_vals = [[p for p in (r["novelty_p_val"], r["agreement_p_val"]) if p is not None] for r in rows]
        p_vals = np.array([p for ps in row_p_vals for p in ps], dtype=np.float64)
        drifted, threshold = apply_correction(p_vals, self._p_val, self._correction)
        for row, ps in zip(rows, row_p_vals, strict=True):
            row["drifted"] = bool(min(ps) < threshold)

        mean = probabilities.mean(axis=0)
        return DriftOutput(
            drifted=drifted,
            threshold=threshold,
            distance=float(flagged.mean()),
            metric_name="flagged_fraction",
            details=_DriftKnownDomainsStats(
                p_val=float(p_vals.min()),
                scores=scores.astype(np.float32),
                closest_domain=self._domains[int(mean.argmax())],
                domain_probabilities={name: float(p) for name, p in zip(self._domains, mean, strict=True)},
                predicted_domains=np.asarray(self._domains)[predicted],
                domains=pl.DataFrame(rows, schema=_TABLE_SCHEMA),
            ),
        )

    def save(self, path: str | Path) -> None:
        """Write the fitted detector to an ONNX file, to be restored with :meth:`load`.

        The file holds the classifier and, as ONNX metadata, everything :meth:`predict`
        needs: the domain names and ``domains=`` spec, score threshold, per-domain reference
        counts and configuration. It does not hold the reference data or the extractor.
        Requires ``onnxmltools``.

        Parameters
        ----------
        path : str or Path
            Destination of the ``.onnx`` file.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        RuntimeError
            If the detector was restored by :meth:`load`, whose file already exists.
        ImportError
            If ``onnxmltools`` is not installed.
        """
        if not self._fitted:
            raise NotFittedError("Must call fit() before save().")
        if self._model is None:
            raise RuntimeError("This detector was restored from a file; there is no model to write again. Call fit().")
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType
        except ImportError as e:
            raise ImportError(
                "onnxmltools is required to save a DriftKnownDomains. Install it with: pip install onnxmltools"
            ) from e

        model = convert_lightgbm(
            self._model,
            initial_types=[("features", FloatTensorType([None, self._n_features]))],
            zipmap=False,
        )
        entry = model.metadata_props.add()
        entry.key = _METADATA_KEY
        entry.value = json.dumps({
            "format": _FORMAT,
            # The extractor is code, not data, and the spec needs its ranges tagged to survive JSON.
            "config": self.config.model_dump(exclude={"extractor", "domains"}),
            "domain_spec": None if self._spec is None else self._spec.to_json(),
            "domains": list(self._domains),
            "n_features": self._n_features,
            "score_threshold": self._score_threshold,
            "null_counts": self._null_counts.tolist(),
        })
        Path(path).write_bytes(model.SerializeToString())

    @classmethod
    def load(cls, path: str | Path, extractor: FeatureExtractor | None = None) -> Self:
        """Restore a detector written by :meth:`save`, ready to :meth:`predict` without refitting.

        A restored detector predicts from the ONNX file alone, with ONNX Runtime. Call
        :meth:`fit` on it to train afresh from new reference data.

        Parameters
        ----------
        path : str or Path
            Path of an ``.onnx`` file written by :meth:`save`.
        extractor : FeatureExtractor or None, default None
            Feature extractor for the restored detector, which :meth:`save` does not write.

        Returns
        -------
        Self
            A fitted detector with the saved domains, threshold and configuration.

        Raises
        ------
        ValueError
            If the file was not written by :meth:`save`, or by an incompatible version.
        ImportError
            If ``onnxruntime`` is not installed.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required to load a DriftKnownDomains. Install it with: pip install onnxruntime"
            ) from e

        options = ort.SessionOptions()
        options.log_severity_level = 3  # the exported label output has a dynamic shape ORT warns about
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        raw = session.get_modelmeta().custom_metadata_map.get(_METADATA_KEY)
        if raw is None:
            raise ValueError(f"{path} was not written by DriftKnownDomains.save: no '{_METADATA_KEY}' metadata.")
        meta = json.loads(raw)
        if meta["format"] != _FORMAT:
            raise ValueError(f"{path} is DriftKnownDomains format {meta['format']}; this version reads {_FORMAT}.")

        spec = meta["domain_spec"]
        config = cls.Config(**meta["config"], domains=None if spec is None else DomainSpec.from_json(spec))
        detector = cls(config=config, extractor=extractor)
        detector._domains = tuple(meta["domains"])
        detector._n_features = int(meta["n_features"])
        detector._score_threshold = float(meta["score_threshold"])
        detector._null_counts = np.asarray(meta["null_counts"], dtype=np.int64)
        detector._session = session
        detector._fitted = True
        return detector
