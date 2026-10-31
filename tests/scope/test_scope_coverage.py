import numpy as np
import polars as pl
import pytest
import torch

from dataeval import Metadata
from dataeval.exceptions import ShapeMismatchError
from dataeval.extractors import FlattenExtractor, TorchExtractor
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.scope import _coverage as coverage_module
from dataeval.scope._coverage import _PER_CLASS_SCHEMA, Coverage, CoverageOutput


class ClassificationDataset:
    """A minimal image-classification dataset with one-hot targets and index2label."""

    def __init__(self, class_names: list[str], per_class: int, shape: tuple[int, ...] = (3, 4, 4)) -> None:
        targets = []
        for i in range(len(class_names)):
            for _ in range(per_class):
                onehot = np.zeros(len(class_names))
                onehot[i] = 1
                targets.append(onehot.copy())
        self._targets = targets
        self._data = np.random.default_rng(0).random((len(targets), *shape)).astype(np.float32)
        self.metadata = DatasetMetadata(id="d", index2label=dict(enumerate(class_names)))

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, DatumMetadata]:
        return self._data[i], self._targets[i], DatumMetadata(id=i)

    def __len__(self) -> int:
        return len(self._targets)


def _two_class_embeddings(rng, class_a, class_b):
    """Stack two per-class point clouds into one embedding/label array."""
    embeddings = np.vstack([class_a, class_b]).astype(np.float64)
    class_labels = np.array([0] * len(class_a) + [1] * len(class_b), dtype=np.intp)
    index2label = {0: "a", 1: "b"}
    return embeddings, class_labels, index2label


def _rows_by_class(cov, embeddings, class_labels, index2label):
    """Per-class rows keyed by class label (no globally-uncovered samples)."""
    rows = cov._per_class(embeddings, class_labels, index2label, np.array([], dtype=np.intp))
    return {row["class"]: row for row in rows}


@pytest.mark.required
class TestCoverageIsotropy:
    def test_isotropic_class_scores_higher_than_anisotropic(self):
        """An isotropic class fills its directions; a near-collinear one does not."""
        rng = np.random.default_rng(0)
        d = 3
        isotropic = rng.normal(size=(120, d))
        # Anisotropic: variance concentrated on one axis (shot from one angle).
        anisotropic = rng.normal(size=(120, d)) * np.array([5.0, 0.02, 0.02])
        embeddings, class_labels, index2label = _two_class_embeddings(rng, isotropic, anisotropic)

        cov = Coverage(min_class_samples=10)
        rows = _rows_by_class(cov, embeddings, class_labels, index2label)

        assert rows["a"]["isotropy"] is not None
        assert rows["b"]["isotropy"] is not None
        assert rows["a"]["isotropy"] > rows["b"]["isotropy"]

    def test_isotropy_null_below_floor(self):
        """A class with fewer samples than the isotropy floor reports null isotropy."""
        rng = np.random.default_rng(1)
        d = 3
        embeddings, class_labels, index2label = _two_class_embeddings(
            rng, rng.normal(size=(30, d)), rng.normal(size=(30, d))
        )

        cov = Coverage(min_class_samples=10, isotropy_min_samples=1000)
        rows = _rows_by_class(cov, embeddings, class_labels, index2label)

        assert rows["a"]["isotropy"] is None
        assert rows["b"]["isotropy"] is None


@pytest.mark.required
class TestCoverageNearDuplicates:
    def test_duplicated_class_scores_higher_than_varied(self):
        """A class of twinned points is mostly near-duplicates; a varied class is not."""
        rng = np.random.default_rng(0)
        d = 4
        varied = rng.random((60, d))
        # Each point repeated once: every point's nearest neighbor is its exact twin.
        twins = np.repeat(rng.random((30, d)), 2, axis=0)
        embeddings, class_labels, index2label = _two_class_embeddings(rng, varied, twins)

        cov = Coverage(min_class_samples=10)
        rows = _rows_by_class(cov, embeddings, class_labels, index2label)

        assert rows["b"]["near_duplicate_fraction"] >= 0.9
        assert rows["a"]["near_duplicate_fraction"] < rows["b"]["near_duplicate_fraction"]

    def test_factor_widens_what_counts_as_duplicate(self):
        """A larger near_duplicate_factor flags more pairs as duplicates."""
        rng = np.random.default_rng(3)
        d = 4
        points = rng.random((80, d))
        embeddings, class_labels, index2label = _two_class_embeddings(rng, points, rng.random((40, d)))

        strict = Coverage(min_class_samples=10, near_duplicate_factor=0.25)
        loose = Coverage(min_class_samples=10, near_duplicate_factor=0.9)
        strict_rows = _rows_by_class(strict, embeddings, class_labels, index2label)
        loose_rows = _rows_by_class(loose, embeddings, class_labels, index2label)

        assert loose_rows["a"]["near_duplicate_fraction"] > strict_rows["a"]["near_duplicate_fraction"]

    def test_fraction_is_a_share_of_images(self):
        """Both images of a near-identical pair count, so the fraction is a share of the class's images."""
        grid = np.stack(np.meshgrid(np.arange(6.0), np.arange(6.0)), axis=-1).reshape(-1, 2)  # spacing 1
        class_a = np.vstack([grid, grid[:4] + [0.01, 0.0]])  # 4 copies: 8 of 40 images in tight pairs
        embeddings, class_labels, index2label = _two_class_embeddings(None, class_a, grid + 100.0)

        rows = _rows_by_class(Coverage(min_class_samples=10), embeddings, class_labels, index2label)

        assert rows["a"]["near_duplicate_fraction"] == pytest.approx(8 / 40)
        assert rows["b"]["near_duplicate_fraction"] == 0.0

    def test_exact_copies_score_like_perturbed_copies(self):
        """Bit-identical copies and copies off by float noise are the same duplicates (device independence)."""
        rng = np.random.default_rng(6)
        originals, peer = rng.random((200, 8)), rng.random((250, 8))
        exact = np.vstack([originals, originals[:50]])
        perturbed = np.vstack([originals, originals[:50] + rng.normal(scale=1e-6, size=(50, 8))])
        cov = Coverage(min_class_samples=10)

        exact_rows = _rows_by_class(cov, *_two_class_embeddings(rng, exact, peer))
        perturbed_rows = _rows_by_class(cov, *_two_class_embeddings(rng, perturbed, peer))

        assert exact_rows["a"]["near_duplicate_fraction"] == perturbed_rows["a"]["near_duplicate_fraction"]
        assert exact_rows["a"]["near_duplicate_fraction"] >= 100 / 250

    def test_clump_covering_most_of_a_class_is_flagged(self):
        """A tight clump of 60% of a class is measured against its peers, so it cannot set its own threshold."""
        rng = np.random.default_rng(7)
        clump = rng.random(8) + rng.normal(scale=1e-3, size=(60, 8))
        class_a = np.vstack([rng.random((40, 8)), clump])
        embeddings = np.vstack([class_a, rng.random((200, 8))])
        class_labels = np.repeat(np.arange(3, dtype=np.intp), 100)
        cov = Coverage(min_class_samples=10)

        rows = _rows_by_class(cov, embeddings, class_labels, {0: "a", 1: "b", 2: "c"})
        # Alone there is no peer, so the reference is the class's own median, which the clump sets.
        alone = _rows_by_class(cov, class_a, np.zeros(100, dtype=np.intp), {0: "a"})

        assert 0.6 <= rows["a"]["near_duplicate_fraction"] < 0.7
        assert alone["a"]["near_duplicate_fraction"] < 0.1

    def test_varied_classes_score_near_zero(self):
        rng = np.random.default_rng(8)
        rows = _rows_by_class(
            Coverage(min_class_samples=10), rng.random((600, 8)), np.repeat(np.arange(3), 200), dict(enumerate("abc"))
        )
        assert all(row["near_duplicate_fraction"] < 0.05 for row in rows.values())

    def test_class_of_one_reports_null(self):
        """A single-image class has no neighbor to compare with, even when it is assessable."""
        rng = np.random.default_rng(9)
        embeddings, class_labels, index2label = _two_class_embeddings(rng, rng.random((30, 4)), rng.random((1, 4)))

        rows = _rows_by_class(Coverage(min_class_samples=1), embeddings, class_labels, index2label)

        assert rows["b"]["near_duplicate_fraction"] is None
        assert rows["a"]["near_duplicate_fraction"] is not None

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_identical_class_is_all_near_duplicates(self):
        """A class of one repeated image is entirely near-duplicates, alone or beside a varied class."""
        rng = np.random.default_rng(10)
        same = np.ones((30, 4))
        cov = Coverage(min_class_samples=10)

        alone = _rows_by_class(cov, same, np.zeros(30, dtype=np.intp), {0: "a"})
        rows = _rows_by_class(cov, *_two_class_embeddings(rng, same, rng.random((30, 4))))

        assert alone["a"]["near_duplicate_fraction"] == 1.0
        assert rows["a"]["near_duplicate_fraction"] == 1.0

    def test_near_duplicate_fraction_null_below_min_class_samples(self):
        """A class too small to assess reports null near_duplicate_fraction."""
        rng = np.random.default_rng(4)
        d = 4
        embeddings, class_labels, index2label = _two_class_embeddings(rng, rng.random((40, d)), rng.random((5, d)))

        cov = Coverage(min_class_samples=10)
        rows = _rows_by_class(cov, embeddings, class_labels, index2label)

        assert rows["b"]["near_duplicate_fraction"] is None


_FLAGS = ["is_clustered", "is_one_dimensional", "has_near_duplicates"]


def _signal_table(monkeypatch, spreads, isotropies):
    """Classes of 30 points whose raw spread and isotropy are set by hand, one class per entry.

    Class k's points carry k in their first coordinate, which the stand-ins read back.
    """
    labels = np.repeat(np.arange(len(spreads), dtype=np.intp), 30)
    embeddings = np.column_stack([labels, np.random.default_rng(0).random((len(labels), 3))])
    completeness = coverage_module._completeness
    monkeypatch.setattr(coverage_module, "_mean_dispersion", lambda points: spreads[int(points[0, 0])])
    monkeypatch.setattr(
        coverage_module,
        "_completeness",
        lambda points: {**completeness(points), "isotropy": isotropies[int(points[0, 0])]},
    )
    return embeddings, labels, dict(enumerate("abcd"))


def _two_grids():
    """Class a: a unit grid plus 4 near-copies, so 8 of its 40 images sit in tight pairs. Class b: the grid alone."""
    grid = np.stack(np.meshgrid(np.arange(6.0), np.arange(6.0)), axis=-1).reshape(-1, 2)
    return _two_class_embeddings(None, np.vstack([grid, grid[:4] + [0.01, 0.0]]), grid + 100.0)


@pytest.mark.required
class TestCoverageFlags:
    @pytest.mark.parametrize(("spread", "flagged"), [(0.49, True), (0.5, False), (0.51, False)])
    def test_is_clustered_is_strict_at_the_factor(self, monkeypatch, spread, flagged):
        """dispersion < dispersion_factor, where the other two classes set the median spread to 1."""
        rows = _rows_by_class(Coverage(), *_signal_table(monkeypatch, [spread, 1.0, 1.0], [0.9, 0.9, 0.9]))
        assert rows["a"]["dispersion"] == spread
        assert rows["a"]["is_clustered"] is flagged

    @pytest.mark.parametrize(("isotropy", "flagged"), [(0.39, True), (0.4, False), (0.41, False)])
    def test_is_one_dimensional_is_strict_at_the_factor(self, monkeypatch, isotropy, flagged):
        """isotropy < isotropy_factor * median isotropy, here 0.5 * 0.8."""
        rows = _rows_by_class(Coverage(), *_signal_table(monkeypatch, [1.0, 1.0, 1.0], [isotropy, 0.8, 0.8]))
        assert rows["a"]["is_one_dimensional"] is flagged

    @pytest.mark.parametrize(("threshold", "flagged"), [(0.19, True), (0.2, False), (0.21, False)])
    def test_has_near_duplicates_is_strict_at_the_threshold(self, threshold, flagged):
        """near_duplicate_fraction > near_duplicate_threshold, around a share of 8 / 40."""
        rows = _rows_by_class(Coverage(min_class_samples=10, near_duplicate_threshold=threshold), *_two_grids())
        assert rows["a"]["near_duplicate_fraction"] == 0.2
        assert rows["a"]["has_near_duplicates"] is flagged

    def test_flags_are_set_on_every_assessable_row(self):
        """polars fills a schema column the rows lack with null, so a misspelled key would pass a value check."""
        rng = np.random.default_rng(11)
        scales = np.repeat([1.0, 1.0, 0.2], 50)[:, None]  # class 2 is clustered
        embeddings = np.vstack([rng.normal(size=(150, 4)) * scales, rng.random((5, 4))])
        labels = np.repeat(np.arange(4), [50, 50, 50, 5])
        data = Coverage(num_observations=5).evaluate(labels, embeddings=embeddings).data()

        assert data.columns == list(_PER_CLASS_SCHEMA)
        assert data.filter(pl.col("assessable")).select(_FLAGS).null_count().row(0) == (0, 0, 0)
        assert data.filter(~pl.col("assessable")).select(_FLAGS).null_count().row(0) == (1, 1, 1)
        assert data.filter(pl.col("class") == "2")["is_clustered"].item() is True

    def test_flags_null_where_their_signal_is_null(self):
        """An unassessable class has no flags; a class below the isotropy floor has no is_one_dimensional."""
        rng = np.random.default_rng(12)
        embeddings = rng.random((70, 4))
        labels = np.repeat(np.arange(3), [40, 25, 5])
        rows = _rows_by_class(Coverage(min_class_samples=10, isotropy_min_samples=30), embeddings, labels, {})

        assert all(rows["0"][flag] is not None for flag in _FLAGS)
        assert rows["1"]["is_one_dimensional"] is None
        assert rows["1"]["is_clustered"] is not None
        assert rows["1"]["has_near_duplicates"] is not None
        assert all(rows["2"][flag] is None for flag in _FLAGS)

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_no_assessable_class_leaves_every_flag_null(self):
        rng = np.random.default_rng(13)
        cov = Coverage(min_class_samples=100)
        rows = _rows_by_class(cov, *_two_class_embeddings(rng, rng.random((30, 4)), rng.random((30, 4))))
        assert all(row[flag] is None for row in rows.values() for flag in _FLAGS)

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_zero_median_spread_leaves_is_clustered_null(self, monkeypatch):
        """Dispersion falls back to raw units when the median spread is 0, so a factor of it means nothing."""
        rows = _rows_by_class(Coverage(), *_signal_table(monkeypatch, [0.0, 0.0, 1.0], [0.9, 0.9, 0.9]))
        assert all(row["is_clustered"] is None for row in rows.values())
        assert all(row["is_one_dimensional"] is False for row in rows.values())

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_zero_median_isotropy_leaves_is_one_dimensional_null(self, monkeypatch):
        """Against a median of 0 no class could be flagged, so the flag says it cannot judge."""
        rows = _rows_by_class(Coverage(), *_signal_table(monkeypatch, [1.0, 1.0, 1.0], [0.0, 0.0, 0.93]))
        assert all(row["is_one_dimensional"] is None for row in rows.values())
        assert all(row["is_clustered"] is False for row in rows.values())

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_no_isotropy_leaves_is_one_dimensional_null(self):
        rng = np.random.default_rng(14)
        cov = Coverage(min_class_samples=10, isotropy_min_samples=1000)
        rows = _rows_by_class(cov, *_two_class_embeddings(rng, rng.random((30, 4)), rng.random((30, 4))))
        assert all(row["is_one_dimensional"] is None for row in rows.values())
        assert all(row["is_clustered"] is False for row in rows.values())

    def test_one_class_is_never_flagged_against_itself(self):
        """A lone class is its own median, and x < 0.5x is false however flat the class is."""
        rng = np.random.default_rng(15)
        flat = rng.normal(size=(60, 4)) * [5.0, 0.01, 0.01, 0.01]
        rows = _rows_by_class(Coverage(min_class_samples=10), flat, np.zeros(60, dtype=np.intp), {0: "a"})
        assert rows["a"]["isotropy"] < 0.5
        assert rows["a"]["is_one_dimensional"] is False
        assert rows["a"]["is_clustered"] is False

    @pytest.mark.parametrize(("low", "flagged"), [(0.29, True), (0.31, False)])
    def test_two_classes_flag_the_low_one_below_a_third_of_the_other(self, monkeypatch, low, flagged):
        """The median of two is their mean, so x < 0.5 * (x + 0.9) / 2 holds only for x < 0.3."""
        rows = _rows_by_class(Coverage(), *_signal_table(monkeypatch, [low, 0.9], [low, 0.9]))
        assert rows["a"]["is_clustered"] is flagged
        assert rows["a"]["is_one_dimensional"] is flagged
        assert rows["b"]["is_clustered"] is False
        assert rows["b"]["is_one_dimensional"] is False

    def test_identical_class_is_one_dimensional(self):
        """A class of one repeated image has isotropy 0.0, which is below any positive median."""
        rng = np.random.default_rng(16)
        cov = Coverage(min_class_samples=10)
        rows = _rows_by_class(cov, *_two_class_embeddings(rng, np.ones((30, 4)), rng.random((30, 4))))
        assert rows["a"]["isotropy"] == 0.0
        assert rows["a"]["is_one_dimensional"] is True

    def test_arguments_and_config_give_the_same_flags(self):
        rng = np.random.default_rng(17)
        embeddings = rng.normal(size=(150, 4)) * np.repeat([1.0, 1.0, 0.7], 50)[:, None]
        labels = np.repeat(np.arange(3), 50)
        config = Coverage.Config(dispersion_factor=0.9, isotropy_factor=0.9, near_duplicate_threshold=0.0)
        from_arguments = Coverage(dispersion_factor=0.9, isotropy_factor=0.9, near_duplicate_threshold=0.0)
        tuned = from_arguments.evaluate(labels, embeddings=embeddings).data()

        assert tuned.equals(Coverage(config=config).evaluate(labels, embeddings=embeddings).data())
        assert tuned.filter(pl.col("class") == "2")["is_clustered"].item() is True
        default = Coverage().evaluate(labels, embeddings=embeddings).data()
        assert default.filter(pl.col("class") == "2")["is_clustered"].item() is False


@pytest.mark.required
class TestCoverageSchema:
    def test_flags_follow_the_released_columns(self):
        """Appended after assessable, so a reader that unpacks rows by position still gets the v1.1 columns."""
        released = [
            "class",
            "count",
            "uncovered",
            "uncovered_fraction",
            "dispersion",
            "isotropy",
            "near_duplicate_fraction",
            "assessable",
        ]
        assert list(_PER_CLASS_SCHEMA) == released + _FLAGS
        assert all(_PER_CLASS_SCHEMA[flag] == pl.Boolean for flag in _FLAGS)

    def test_rows_with_null_signals_build_dataframe(self):
        """The per-class schema (used by evaluate) accepts null isotropy / near_duplicate_fraction."""
        rng = np.random.default_rng(5)
        d = 4
        embeddings, class_labels, index2label = _two_class_embeddings(rng, rng.random((40, d)), rng.random((5, d)))
        cov = Coverage(min_class_samples=10)
        rows = cov._per_class(embeddings, class_labels, index2label, np.array([], dtype=np.intp))

        df = pl.DataFrame(rows, schema=_PER_CLASS_SCHEMA)
        assert df.schema["isotropy"] == pl.Float64
        assert df.schema["near_duplicate_fraction"] == pl.Float64
        # The 5-sample class is below min_class_samples, so both signals are null for it.
        assert df["near_duplicate_fraction"].null_count() >= 1


@pytest.mark.required
class TestEvaluate:
    def test_evaluate_with_precomputed_embeddings(self):
        ds = ClassificationDataset(["a", "b"], per_class=25)
        emb = np.random.default_rng(1).random((len(ds), 4))
        res = Coverage(num_observations=5, min_class_samples=10).evaluate(ds, embeddings=emb)

        assert isinstance(res, CoverageOutput)
        assert set(res.data()["class"].to_list()) == {"a", "b"}
        assert res.coverage_radius > 0
        assert len(res.critical_value_radii) == len(ds)
        # uncovered_indices index into the sample set
        assert all(0 <= i < len(ds) for i in res.uncovered_indices)

    def test_evaluate_naive_method(self):
        ds = ClassificationDataset(["a", "b"], per_class=25)
        emb = np.random.default_rng(2).random((len(ds), 4))
        res = Coverage(num_observations=5, min_class_samples=10, method="naive").evaluate(ds, embeddings=emb)
        assert res.coverage_radius > 0

    def test_evaluate_computes_embeddings_from_extractor(self):
        ds = ClassificationDataset(["a", "b"], per_class=25)
        extractor = TorchExtractor(torch.nn.Flatten(), device="cpu")
        res = Coverage(extractor=extractor, batch_size=16, num_observations=5, min_class_samples=10).evaluate(ds)
        assert set(res.data()["class"].to_list()) == {"a", "b"}

    def test_evaluate_shape_mismatch_raises(self):
        ds = ClassificationDataset(["a", "b"], per_class=25)
        emb = np.random.default_rng(3).random((10, 4))  # far fewer embeddings than labels
        with pytest.raises(ShapeMismatchError, match="one embedding per"):
            Coverage(num_observations=5, min_class_samples=10).evaluate(ds, embeddings=emb)


@pytest.mark.required
class TestEmbeddingsResolution:
    def test_precomputed_embeddings_returned_as_float64(self):
        cov = Coverage()
        emb = cov._embeddings(dataset=None, embeddings=np.ones((4, 3), dtype=np.float32))  # type: ignore
        assert emb.dtype == np.float64
        assert emb.shape == (4, 3)

    def test_missing_extractor_and_embeddings_raises(self):
        with pytest.raises(ValueError, match="Provide pre-computed embeddings"):
            Coverage()._embeddings(dataset=None, embeddings=None)  # type: ignore


class _SpyExtractor(FlattenExtractor):
    """Records the per-item shape the extractor was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, data):
        self.shapes.extend(np.asarray(item).shape for item in data)
        return super().__call__(data)


@pytest.mark.required
def test_bound_metadata_embeds_images_not_factor_rows():
    """A bound Metadata is embedded through its dataset, identically to passing the dataset."""
    ds = ClassificationDataset(["a", "b"], per_class=10)
    spy = _SpyExtractor()
    from_metadata = Coverage(spy, num_observations=5, min_class_samples=5, batch_size=8).evaluate(Metadata(ds))
    assert set(spy.shapes) == {(3, 4, 4)}
    from_dataset = Coverage(FlattenExtractor(), num_observations=5, min_class_samples=5, batch_size=8).evaluate(ds)
    assert from_metadata.coverage_radius == pytest.approx(from_dataset.coverage_radius)
    assert from_metadata.data().equals(from_dataset.data())


@pytest.mark.required
def test_unbound_metadata_without_embeddings_raises():
    """No dataset behind the Metadata means no images to embed: say so instead of embedding factors."""
    md = Metadata.from_factors({"w": np.arange(20.0)}, class_labels=np.repeat([0, 1], 10))
    with pytest.raises(ValueError, match="embeddings=") as excinfo:
        Coverage(FlattenExtractor(), num_observations=5, min_class_samples=5, batch_size=8).evaluate(md)
    assert "the dataset it was built from" in str(excinfo.value)


@pytest.mark.required
def test_unbound_metadata_with_embeddings_still_works():
    """The documented notebook path (Metadata plus precomputed embeddings) is unaffected."""
    md = Metadata.from_factors({"w": np.arange(20.0)}, class_labels=np.repeat([0, 1], 10))
    embeddings = np.random.default_rng(0).random((20, 8))
    result = Coverage(num_observations=5, min_class_samples=5).evaluate(md, embeddings=embeddings)
    assert result.data()["count"].sum() == 20


@pytest.mark.required
def test_subset_like_dataset_is_not_unwrapped():
    """A dataset carrying its own `.dataset` (torch Subset style) is embedded as itself, not its parent."""
    parent = ClassificationDataset(["a", "b"], per_class=10)

    class _Subset:
        def __init__(self, dataset, indices):
            self.dataset, self.indices = dataset, indices
            self.metadata = dataset.metadata

        def __getitem__(self, i):
            return self.dataset[self.indices[i]]

        def __len__(self):
            return len(self.indices)

    subset = _Subset(parent, list(range(0, 20, 2)) + list(range(1, 20, 2))[:6])
    spy = _SpyExtractor()
    Coverage(spy, num_observations=5, min_class_samples=5, batch_size=8).evaluate(subset)
    assert len(spy.shapes) == len(subset)
