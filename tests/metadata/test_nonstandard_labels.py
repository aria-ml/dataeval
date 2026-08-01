"""Tests for Metadata with non-0-based and non-contiguous class labels.

Two scenarios users encounter:

1. **Correct construction** — one-hot / pseudo-prob vectors are full-width, with
   zeros for every position that doesn't correspond to a real class.  argmax
   recovers the *real* label key (e.g. 1, 3).  ``index2label`` keys match
   those label values.

2. **Collapsed construction** — user compresses the pseudo-probs so only real
   classes have columns.  argmax now produces 0-based positional indices that
   do NOT match the ``index2label`` keys.  Metadata must not crash; unmapped
   labels get fallback names like ``UNDEFINED_CLASS_<int>``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from dataeval import Metadata
from dataeval.bias import Balance, Diversity, Parity
from dataeval.core._label_stats import label_stats
from dataeval.protocols import DatumMetadata

# ---------------------------------------------------------------------------
# Test dataset helpers
# ---------------------------------------------------------------------------


@dataclass
class SimpleODTarget:
    """Minimal ObjectDetectionTarget for testing."""

    boxes: Any
    labels: Any
    scores: Any


class ODDatasetWithLabels:
    """OD dataset that directly provides arbitrary label values."""

    def __init__(
        self,
        n_images: int,
        labels_per_image: list[list[int]],
        index2label: dict[int, str] | None = None,
    ):
        self._n_images = n_images
        self._labels = labels_per_image
        self._images = [np.random.rand(3, 32, 32).astype(np.float32) for _ in range(n_images)]

        all_labels = sorted({lbl for img_labels in labels_per_image for lbl in img_labels})
        if index2label is None:
            index2label = {lbl: str(lbl) for lbl in all_labels}
        self._index2label = index2label
        self._id = f"od_nonstandard_{n_images}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {"id": self._id, "index2label": self._index2label}

    def __len__(self):
        return self._n_images

    def __getitem__(self, idx: int) -> tuple[NDArray, SimpleODTarget, DatumMetadata]:
        img_labels = self._labels[idx]
        n_det = len(img_labels)
        boxes = np.random.rand(n_det, 4).astype(np.float32) * 50
        n_score_cols = len(self._index2label)
        scores = np.random.rand(n_det, n_score_cols).astype(np.float32)
        target = SimpleODTarget(boxes=boxes, labels=np.array(img_labels), scores=scores)
        return self._images[idx], target, {"id": idx, "brightness": float(idx) / self._n_images}  # type: ignore


class ICDatasetCorrect:
    """IC dataset with *correct* full-width one-hot targets.

    One-hot vector length = max(label_key) + 1, with zeros for positions
    that don't correspond to any real class.  argmax recovers the original
    label key (e.g. 1, 3).
    """

    def __init__(self, n_images: int, labels: list[int], index2label: dict[int, str]):
        assert len(labels) == n_images
        self._n_images = n_images
        self._labels = labels
        self._images = [np.random.rand(3, 32, 32).astype(np.float32) for _ in range(n_images)]
        self._n_classes = max(labels) + 1
        self._index2label = index2label
        self._id = f"ic_correct_{n_images}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {"id": self._id, "index2label": self._index2label}

    def __len__(self):
        return self._n_images

    def __getitem__(self, idx: int) -> tuple[NDArray, NDArray, DatumMetadata]:
        one_hot = np.zeros(self._n_classes, dtype=np.float32)
        one_hot[self._labels[idx]] = 1.0
        return self._images[idx], one_hot, {"id": idx, "brightness": float(idx) / self._n_images}  # type: ignore


class ICDatasetCollapsed:
    """IC dataset with *collapsed* one-hot targets.

    One-hot vector length = number of real classes.  argmax produces 0-based
    positional indices (0, 1, 2 …) that do NOT match the ``index2label``
    keys (e.g. 1, 3, 5).
    """

    def __init__(self, n_images: int, positional_labels: list[int], index2label: dict[int, str]):
        assert len(positional_labels) == n_images
        self._n_images = n_images
        self._labels = positional_labels  # 0-based positional
        self._images = [np.random.rand(3, 32, 32).astype(np.float32) for _ in range(n_images)]
        self._n_classes = len(index2label)  # collapsed width
        self._index2label = index2label
        self._id = f"ic_collapsed_{n_images}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {"id": self._id, "index2label": self._index2label}

    def __len__(self):
        return self._n_images

    def __getitem__(self, idx: int) -> tuple[NDArray, NDArray, DatumMetadata]:
        one_hot = np.zeros(self._n_classes, dtype=np.float32)
        one_hot[self._labels[idx]] = 1.0
        return self._images[idx], one_hot, {"id": idx, "brightness": float(idx) / self._n_images}  # type: ignore


# ---------------------------------------------------------------------------
# Fixtures — correct construction
# ---------------------------------------------------------------------------


@pytest.fixture
def od_1based():
    """OD dataset with correctly constructed 1-based labels."""
    labels_per_image = [[1, 2], [2, 3], [1, 3], [1, 2], [2, 3]]
    return ODDatasetWithLabels(5, labels_per_image, {1: "cat", 2: "dog", 3: "bird"})


@pytest.fixture
def od_noncontiguous():
    """OD dataset with correctly constructed non-contiguous labels."""
    labels_per_image = [[1, 3], [3, 5], [1, 5], [3, 5], [1, 3]]
    return ODDatasetWithLabels(5, labels_per_image, {1: "cat", 3: "dog", 5: "bird"})


@pytest.fixture
def ic_1based():
    """IC dataset with correctly constructed 1-based one-hot targets."""
    labels = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    return ICDatasetCorrect(10, labels, {1: "cat", 2: "dog", 3: "bird"})


@pytest.fixture
def ic_noncontiguous():
    """IC dataset with correctly constructed non-contiguous one-hot targets."""
    labels = [1, 3, 5, 1, 3, 5, 1, 3, 5, 1]
    return ICDatasetCorrect(10, labels, {1: "cat", 3: "dog", 5: "bird"})


# ---------------------------------------------------------------------------
# Fixtures — collapsed construction
# ---------------------------------------------------------------------------


@pytest.fixture
def od_collapsed_1based():
    """OD dataset with collapsed 0-based labels but 1-based index2label."""
    labels_per_image = [[0, 1], [1, 2], [0, 2], [0, 1], [1, 2]]
    return ODDatasetWithLabels(5, labels_per_image, {1: "cat", 2: "dog", 3: "bird"})


@pytest.fixture
def od_collapsed_noncontiguous():
    """OD dataset with collapsed 0-based labels but non-contiguous index2label."""
    labels_per_image = [[0, 1], [1, 2], [0, 2], [0, 1], [1, 2]]
    return ODDatasetWithLabels(5, labels_per_image, {1: "cat", 3: "dog", 5: "bird"})


@pytest.fixture
def ic_collapsed_1based():
    """IC dataset with collapsed 3-wide one-hot but 1-based index2label."""
    positional = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    return ICDatasetCollapsed(10, positional, {1: "cat", 2: "dog", 3: "bird"})


@pytest.fixture
def ic_collapsed_noncontiguous():
    """IC dataset with collapsed 3-wide one-hot but non-contiguous index2label."""
    positional = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    return ICDatasetCollapsed(10, positional, {1: "cat", 3: "dog", 5: "bird"})


# ===================================================================
# SCENARIO 1: Correct construction — labels match index2label keys
# ===================================================================


class TestCorrectOD:
    """OD datasets where target.labels values ARE the index2label keys."""

    def test_1based_class_labels(self, od_1based):
        """1-based labels should be preserved as provided."""
        md = Metadata(od_1based, exclude=["id"])
        assert set(md.class_labels) == {1, 2, 3}
        assert len(md.class_labels) == 10  # 5 images * 2 detections

    def test_1based_index2label(self, od_1based):
        """index2label should preserve the user-provided mapping."""
        md = Metadata(od_1based, exclude=["id"])
        assert md.index2label == {1: "cat", 2: "dog", 3: "bird"}

    def test_1based_factor_data(self, od_1based):
        md = Metadata(od_1based, exclude=["id"])
        assert md.factor_data.shape[0] == len(md.class_labels)

    def test_1based_dataframe(self, od_1based):
        md = Metadata(od_1based, exclude=["id"])
        assert "class_label" in md.dataframe.columns
        np.testing.assert_array_equal(md.target_data["class_label"].to_numpy(), md.class_labels)

    def test_noncontiguous_class_labels(self, od_noncontiguous):
        """Non-contiguous labels should be preserved as provided."""
        md = Metadata(od_noncontiguous, exclude=["id"])
        assert set(md.class_labels) == {1, 3, 5}
        assert len(md.class_labels) == 10

    def test_noncontiguous_index2label(self, od_noncontiguous):
        md = Metadata(od_noncontiguous, exclude=["id"])
        assert md.index2label == {1: "cat", 3: "dog", 5: "bird"}

    def test_noncontiguous_factor_data(self, od_noncontiguous):
        md = Metadata(od_noncontiguous, exclude=["id"])
        assert md.factor_data.shape[0] == len(md.class_labels)

    def test_noncontiguous_item_indices(self, od_noncontiguous):
        md = Metadata(od_noncontiguous, exclude=["id"])
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        np.testing.assert_array_equal(md.item_indices, expected)


class TestCorrectIC:
    """IC datasets where argmax of full-width one-hot matches index2label keys."""

    def test_1based_class_labels(self, ic_1based):
        """1-based labels recovered via argmax should be preserved."""
        md = Metadata(ic_1based, exclude=["id"])
        assert set(md.class_labels) == {1, 2, 3}
        assert len(md.class_labels) == 10

    def test_1based_index2label(self, ic_1based):
        md = Metadata(ic_1based, exclude=["id"])
        assert md.index2label == {1: "cat", 2: "dog", 3: "bird"}

    def test_1based_factor_data(self, ic_1based):
        md = Metadata(ic_1based, exclude=["id"])
        assert md.factor_data.shape[0] == len(md.class_labels)

    def test_noncontiguous_class_labels(self, ic_noncontiguous):
        md = Metadata(ic_noncontiguous, exclude=["id"])
        assert set(md.class_labels) == {1, 3, 5}
        assert len(md.class_labels) == 10

    def test_noncontiguous_index2label(self, ic_noncontiguous):
        md = Metadata(ic_noncontiguous, exclude=["id"])
        assert md.index2label == {1: "cat", 3: "dog", 5: "bird"}

    def test_noncontiguous_factor_data(self, ic_noncontiguous):
        md = Metadata(ic_noncontiguous, exclude=["id"])
        assert md.factor_data.shape[0] == len(md.class_labels)

    def test_noncontiguous_item_indices(self, ic_noncontiguous):
        md = Metadata(ic_noncontiguous, exclude=["id"])
        np.testing.assert_array_equal(md.item_indices, np.arange(10))


# ===================================================================
# SCENARIO 2: Collapsed construction — labels are 0-based positional,
#              index2label keys don't match
# ===================================================================


class TestCollapsedOD:
    """OD datasets where target.labels are 0-based but index2label keys are not.

    The user mistakenly collapsed labels to [0, 1, 2] while index2label
    has keys like {1, 2, 3} or {1, 3, 5}.  Metadata should not crash;
    labels stay 0-based and unmapped labels get UNDEFINED_CLASS_<int> names.
    """

    def test_collapsed_1based_class_labels(self, od_collapsed_1based):
        """Labels are already 0-based, should stay as provided."""
        md = Metadata(od_collapsed_1based, exclude=["id"])
        assert set(md.class_labels) == {0, 1, 2}
        assert len(md.class_labels) == 10

    def test_collapsed_1based_index2label(self, od_collapsed_1based):
        """Labels matching a provided key get that name; others get fallback."""
        md = Metadata(od_collapsed_1based, exclude=["id"])
        i2l = md.index2label
        # Labels are [0, 1, 2], provided index2label is {1: cat, 2: dog, 3: bird}
        # 0 has no match → fallback; 1 matches → "cat"; 2 matches → "dog"
        assert i2l[0] == "UNDEFINED_CLASS_0"
        assert i2l[1] == "cat"
        assert i2l[2] == "dog"

    def test_collapsed_1based_factor_data(self, od_collapsed_1based):
        """factor_data should not crash."""
        md = Metadata(od_collapsed_1based, exclude=["id"])
        assert md.factor_data.shape[0] == len(md.class_labels)

    def test_collapsed_noncontiguous_class_labels(self, od_collapsed_noncontiguous):
        md = Metadata(od_collapsed_noncontiguous, exclude=["id"])
        assert set(md.class_labels) == {0, 1, 2}
        assert len(md.class_labels) == 10

    def test_collapsed_noncontiguous_index2label(self, od_collapsed_noncontiguous):
        """Labels matching a provided key get that name; others get fallback."""
        md = Metadata(od_collapsed_noncontiguous, exclude=["id"])
        i2l = md.index2label
        # Labels are [0, 1, 2], provided index2label is {1: cat, 3: dog, 5: bird}
        # 0 has no match → fallback; 1 matches → "cat"; 2 has no match → fallback
        assert i2l[0] == "UNDEFINED_CLASS_0"
        assert i2l[1] == "cat"
        assert i2l[2] == "UNDEFINED_CLASS_2"

    def test_collapsed_noncontiguous_factor_data(self, od_collapsed_noncontiguous):
        md = Metadata(od_collapsed_noncontiguous, exclude=["id"])
        assert md.factor_data.shape[0] == len(md.class_labels)


class TestCollapsedIC:
    """IC datasets where collapsed one-hot produces 0-based argmax that
    doesn't match the non-0-based index2label keys.

    User provides 3-wide one-hot for 3 classes, argmax -> [0, 1, 2],
    but index2label has keys {1, 2, 3} or {1, 3, 5}.
    """

    def test_collapsed_1based_class_labels(self, ic_collapsed_1based):
        md = Metadata(ic_collapsed_1based, exclude=["id"])
        assert set(md.class_labels) == {0, 1, 2}
        assert len(md.class_labels) == 10

    def test_collapsed_1based_index2label(self, ic_collapsed_1based):
        """Labels matching a provided key get that name; others get fallback."""
        md = Metadata(ic_collapsed_1based, exclude=["id"])
        i2l = md.index2label
        # argmax produces [0, 1, 2], provided index2label is {1: cat, 2: dog, 3: bird}
        assert i2l[0] == "UNDEFINED_CLASS_0"
        assert i2l[1] == "cat"
        assert i2l[2] == "dog"

    def test_collapsed_1based_factor_data(self, ic_collapsed_1based):
        md = Metadata(ic_collapsed_1based, exclude=["id"])
        assert md.factor_data.shape[0] == len(md.class_labels)

    def test_collapsed_noncontiguous_class_labels(self, ic_collapsed_noncontiguous):
        md = Metadata(ic_collapsed_noncontiguous, exclude=["id"])
        assert set(md.class_labels) == {0, 1, 2}
        assert len(md.class_labels) == 10

    def test_collapsed_noncontiguous_index2label(self, ic_collapsed_noncontiguous):
        md = Metadata(ic_collapsed_noncontiguous, exclude=["id"])
        i2l = md.index2label
        # argmax produces [0, 1, 2], provided index2label is {1: cat, 3: dog, 5: bird}
        assert i2l[0] == "UNDEFINED_CLASS_0"
        assert i2l[1] == "cat"
        assert i2l[2] == "UNDEFINED_CLASS_2"

    def test_collapsed_noncontiguous_factor_data(self, ic_collapsed_noncontiguous):
        md = Metadata(ic_collapsed_noncontiguous, exclude=["id"])
        assert md.factor_data.shape[0] == len(md.class_labels)


# ===================================================================
# Cross-cutting: label consistency, key types, no-op for 0-based
# ===================================================================


class TestLabelConsistency:
    """Invariants that hold across both correct and collapsed scenarios."""

    def test_0based_labels_unchanged(self):
        """Labels already 0-based contiguous with matching index2label stay as-is."""
        labels_per_image = [[0, 1], [1, 2], [0, 2]]
        ds = ODDatasetWithLabels(3, labels_per_image, {0: "cat", 1: "dog", 2: "bird"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        assert set(md.class_labels) == {0, 1, 2}
        assert md.index2label == {0: "cat", 1: "dog", 2: "bird"}

    def test_index2label_keys_are_plain_int_od(self):
        """index2label keys must be plain int, not np.int64."""
        ds = ODDatasetWithLabels(3, [[1, 2], [2, 3], [1, 3]], {1: "a", 2: "b", 3: "c"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        for key in md.index2label:
            assert type(key) is int, f"index2label key {key!r} is {type(key).__name__}, expected int"

    def test_index2label_keys_are_plain_int_ic(self):
        """IC index2label keys must be plain int."""
        ds = ICDatasetCorrect(5, [1, 2, 3, 1, 2], {1: "a", 2: "b", 3: "c"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        for key in md.index2label:
            assert type(key) is int, f"index2label key {key!r} is {type(key).__name__}, expected int"

    def test_index2label_keys_are_plain_int_auto_generated(self):
        """Auto-generated index2label (no dataset mapping) should have plain int keys.

        When dataset metadata omits index2label, Metadata builds one from
        np.unique(labels).  Keys must be plain int, not np.int64.
        """
        labels_per_image = [[0, 1], [1, 2], [0, 2]]
        ds = ODDatasetWithLabels(3, labels_per_image, {0: "a", 1: "b", 2: "c"})
        original_metadata = ds.metadata

        class NoI2LDataset:
            @property
            def metadata(self):
                return {"id": original_metadata["id"]}

            def __len__(self):
                return len(ds)

            def __getitem__(self, idx):
                return ds[idx]

        md = Metadata(NoI2LDataset(), exclude=["id"])  # type: ignore
        for key in md.index2label:
            assert type(key) is int, f"index2label key {key!r} is {type(key).__name__}, expected int"

    def test_dataframe_matches_class_labels_od(self, od_1based):
        """DataFrame class_label column should match class_labels property."""
        md = Metadata(od_1based, exclude=["id"])
        np.testing.assert_array_equal(md.target_data["class_label"].to_numpy(), md.class_labels)

    def test_item_count_od(self, od_1based):
        """item_count should be number of images, not detections."""
        md = Metadata(od_1based, exclude=["id"])
        assert md.item_count == 5


# ===================================================================
# Downstream integration: bias evaluators and label_stats
# ===================================================================


class TestDownstreamCorrect:
    """Bias evaluators and label_stats with correctly constructed datasets."""

    @pytest.mark.parametrize(
        ("labels_per_image", "index2label"),
        [
            ([[1, 2], [2, 3], [1, 3], [1, 2], [2, 3]], {1: "a", 2: "b", 3: "c"}),
            ([[1, 3], [3, 5], [1, 5], [3, 5], [1, 3]], {1: "a", 3: "b", 5: "c"}),
        ],
        ids=["1-based", "non-contiguous"],
    )
    def test_od_balance(self, labels_per_image, index2label):
        n = len(labels_per_image)
        ds = ODDatasetWithLabels(n, labels_per_image, index2label)
        md = Metadata(ds, exclude=["id"])  # type: ignore
        result = Balance(num_neighbors=2).evaluate(md)
        assert result.balance.shape[0] > 0

    @pytest.mark.parametrize(
        ("labels_per_image", "index2label"),
        [
            ([[1, 2], [2, 3], [1, 3], [1, 2], [2, 3]], {1: "a", 2: "b", 3: "c"}),
            ([[1, 3], [3, 5], [1, 5], [3, 5], [1, 3]], {1: "a", 3: "b", 5: "c"}),
        ],
        ids=["1-based", "non-contiguous"],
    )
    def test_od_diversity(self, labels_per_image, index2label):
        n = len(labels_per_image)
        ds = ODDatasetWithLabels(n, labels_per_image, index2label)
        md = Metadata(ds, exclude=["id"])  # type: ignore
        result = Diversity().evaluate(md)
        assert result.factors.shape[0] > 0

    @pytest.mark.parametrize(
        ("labels_per_image", "index2label"),
        [
            ([[1, 2], [2, 3], [1, 3], [1, 2], [2, 3]], {1: "a", 2: "b", 3: "c"}),
            ([[1, 3], [3, 5], [1, 5], [3, 5], [1, 3]], {1: "a", 3: "b", 5: "c"}),
        ],
        ids=["1-based", "non-contiguous"],
    )
    def test_od_parity(self, labels_per_image, index2label):
        n = len(labels_per_image)
        ds = ODDatasetWithLabels(n, labels_per_image, index2label)
        md = Metadata(ds, exclude=["id"])  # type: ignore
        result = Parity().evaluate(md)
        assert result.factors.shape[0] > 0

    @pytest.mark.parametrize(
        ("labels", "index2label"),
        [
            ([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], {1: "a", 2: "b", 3: "c"}),
            ([1, 3, 5, 1, 3, 5, 1, 3, 5, 1], {1: "a", 3: "b", 5: "c"}),
        ],
        ids=["1-based", "non-contiguous"],
    )
    def test_ic_balance(self, labels, index2label):
        ds = ICDatasetCorrect(len(labels), labels, index2label)
        md = Metadata(ds, exclude=["id"])  # type: ignore
        result = Balance(num_neighbors=2).evaluate(md)
        assert result.balance.shape[0] > 0

    @pytest.mark.parametrize(
        ("labels", "index2label"),
        [
            ([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], {1: "a", 2: "b", 3: "c"}),
            ([1, 3, 5, 1, 3, 5, 1, 3, 5, 1], {1: "a", 3: "b", 5: "c"}),
        ],
        ids=["1-based", "non-contiguous"],
    )
    def test_ic_diversity(self, labels, index2label):
        ds = ICDatasetCorrect(len(labels), labels, index2label)
        md = Metadata(ds, exclude=["id"])  # type: ignore
        result = Diversity().evaluate(md)
        assert result.factors.shape[0] > 0

    @pytest.mark.parametrize(
        ("labels", "index2label"),
        [
            ([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], {1: "a", 2: "b", 3: "c"}),
            ([1, 3, 5, 1, 3, 5, 1, 3, 5, 1], {1: "a", 3: "b", 5: "c"}),
        ],
        ids=["1-based", "non-contiguous"],
    )
    def test_ic_parity(self, labels, index2label):
        ds = ICDatasetCorrect(len(labels), labels, index2label)
        md = Metadata(ds, exclude=["id"])  # type: ignore
        result = Parity().evaluate(md)
        assert result.factors.shape[0] > 0

    @pytest.mark.parametrize(
        ("labels_per_image", "index2label"),
        [
            ([[1, 2], [2, 3], [1, 3], [1, 2], [2, 3]], {1: "a", 2: "b", 3: "c"}),
            ([[1, 3], [3, 5], [1, 5], [3, 5], [1, 3]], {1: "a", 3: "b", 5: "c"}),
        ],
        ids=["1-based", "non-contiguous"],
    )
    def test_od_label_stats(self, labels_per_image, index2label):
        n = len(labels_per_image)
        ds = ODDatasetWithLabels(n, labels_per_image, index2label)
        md = Metadata(ds, exclude=["id"])  # type: ignore
        stats = label_stats(md.class_labels, md.item_indices, md.index2label, image_count=md.item_count)
        assert len(stats["label_counts_per_image"]) == n

    @pytest.mark.parametrize(
        ("labels", "index2label"),
        [
            ([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], {1: "a", 2: "b", 3: "c"}),
            ([1, 3, 5, 1, 3, 5, 1, 3, 5, 1], {1: "a", 3: "b", 5: "c"}),
        ],
        ids=["1-based", "non-contiguous"],
    )
    def test_ic_label_stats(self, labels, index2label):
        n = len(labels)
        ds = ICDatasetCorrect(n, labels, index2label)
        md = Metadata(ds, exclude=["id"])  # type: ignore
        stats = label_stats(md.class_labels, md.item_indices, md.index2label, image_count=md.item_count)
        assert len(stats["label_counts_per_image"]) == n


class TestDownstreamCollapsed:
    """Bias evaluators should not crash on collapsed (mismatched) datasets."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["od_collapsed_1based", "od_collapsed_noncontiguous"],
        ids=["1-based", "non-contiguous"],
    )
    def test_od_balance(self, fixture_name, request):
        ds = request.getfixturevalue(fixture_name)
        md = Metadata(ds, exclude=["id"])  # type: ignore
        result = Balance(num_neighbors=2).evaluate(md)
        assert result.balance.shape[0] > 0

    @pytest.mark.parametrize(
        "fixture_name",
        ["od_collapsed_1based", "od_collapsed_noncontiguous"],
        ids=["1-based", "non-contiguous"],
    )
    def test_od_diversity(self, fixture_name, request):
        ds = request.getfixturevalue(fixture_name)
        md = Metadata(ds, exclude=["id"])
        result = Diversity().evaluate(md)
        assert result.factors.shape[0] > 0

    @pytest.mark.parametrize(
        "fixture_name",
        ["od_collapsed_1based", "od_collapsed_noncontiguous"],
        ids=["1-based", "non-contiguous"],
    )
    def test_od_parity(self, fixture_name, request):
        ds = request.getfixturevalue(fixture_name)
        md = Metadata(ds, exclude=["id"])
        result = Parity().evaluate(md)
        assert result.factors.shape[0] > 0

    @pytest.mark.parametrize(
        "fixture_name",
        ["ic_collapsed_1based", "ic_collapsed_noncontiguous"],
        ids=["1-based", "non-contiguous"],
    )
    def test_ic_balance(self, fixture_name, request):
        ds = request.getfixturevalue(fixture_name)
        md = Metadata(ds, exclude=["id"])
        result = Balance(num_neighbors=2).evaluate(md)
        assert result.balance.shape[0] > 0

    @pytest.mark.parametrize(
        "fixture_name",
        ["ic_collapsed_1based", "ic_collapsed_noncontiguous"],
        ids=["1-based", "non-contiguous"],
    )
    def test_ic_diversity(self, fixture_name, request):
        ds = request.getfixturevalue(fixture_name)
        md = Metadata(ds, exclude=["id"])
        result = Diversity().evaluate(md)
        assert result.factors.shape[0] > 0

    @pytest.mark.parametrize(
        "fixture_name",
        ["ic_collapsed_1based", "ic_collapsed_noncontiguous"],
        ids=["1-based", "non-contiguous"],
    )
    def test_ic_parity(self, fixture_name, request):
        ds = request.getfixturevalue(fixture_name)
        md = Metadata(ds, exclude=["id"])
        result = Parity().evaluate(md)
        assert result.factors.shape[0] > 0

    @pytest.mark.parametrize(
        "fixture_name",
        ["od_collapsed_1based", "od_collapsed_noncontiguous"],
        ids=["1-based", "non-contiguous"],
    )
    def test_od_label_stats(self, fixture_name, request):
        ds = request.getfixturevalue(fixture_name)
        md = Metadata(ds, exclude=["id"])
        stats = label_stats(md.class_labels, md.item_indices, md.index2label, image_count=md.item_count)
        assert len(stats["label_counts_per_image"]) == len(ds)

    @pytest.mark.parametrize(
        "fixture_name",
        ["ic_collapsed_1based", "ic_collapsed_noncontiguous"],
        ids=["1-based", "non-contiguous"],
    )
    def test_ic_label_stats(self, fixture_name, request):
        ds = request.getfixturevalue(fixture_name)
        md = Metadata(ds, exclude=["id"])
        stats = label_stats(md.class_labels, md.item_indices, md.index2label, image_count=md.item_count)
        assert len(stats["label_counts_per_image"]) == len(ds)


# ===================================================================
# Edge cases: empty targets, single class, no index2label, etc.
# ===================================================================


class TestEdgeCases:
    """Edge cases for label handling: single class, no index2label, empty targets."""

    # --- Single-class datasets ---

    def test_single_class_ic(self):
        """IC dataset where every image has the same label."""
        labels = [2] * 10
        ds = ICDatasetCorrect(10, labels, {2: "only_class"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        assert set(md.class_labels) == {2}
        assert len(md.class_labels) == 10
        assert md.index2label == {2: "only_class"}

    def test_single_class_od(self):
        """OD dataset where every detection is the same class."""
        labels_per_image = [[3], [3], [3, 3], [3]]
        ds = ODDatasetWithLabels(4, labels_per_image, {3: "only_class"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        assert set(md.class_labels) == {3}
        assert md.index2label == {3: "only_class"}
        assert len(md.class_labels) == 5  # 1+1+2+1 detections

    # --- No index2label provided ---

    def test_no_index2label_ic(self):
        """IC dataset with no index2label: auto-generated uses str(label_value)."""
        labels = [1, 3, 5, 1, 3, 5, 1, 3, 5, 1]

        class _ICNoI2L:
            @property
            def metadata(self):
                return {"id": "no_i2l_ic"}

            def __len__(self):
                return len(labels)

            def __getitem__(self, idx):
                n_classes = max(labels) + 1
                one_hot = np.zeros(n_classes, dtype=np.float32)
                one_hot[labels[idx]] = 1.0
                img = np.random.rand(3, 32, 32).astype(np.float32)
                return img, one_hot, {"id": idx, "brightness": float(idx)}

        md = Metadata(_ICNoI2L(), exclude=["id"])  # type: ignore
        # Auto-generated: keys are label values, names are str(value)
        assert md.index2label == {1: "1", 3: "3", 5: "5"}
        for key in md.index2label:
            assert type(key) is int

    def test_no_index2label_od(self):
        """OD dataset with no index2label: auto-generated uses str(label_value)."""
        labels_per_image = [[1, 2], [2, 3], [1, 3]]

        class _ODNoI2L:
            @property
            def metadata(self):
                return {"id": "no_i2l_od"}

            def __len__(self):
                return len(labels_per_image)

            def __getitem__(self, idx):
                img_labels = labels_per_image[idx]
                n_det = len(img_labels)
                boxes = np.random.rand(n_det, 4).astype(np.float32) * 50
                scores = np.random.rand(n_det, 4).astype(np.float32)
                target = SimpleODTarget(boxes=boxes, labels=np.array(img_labels), scores=scores)
                img = np.random.rand(3, 32, 32).astype(np.float32)
                return img, target, {"id": idx, "brightness": float(idx)}

        md = Metadata(_ODNoI2L(), exclude=["id"])  # type: ignore
        assert md.index2label == {1: "1", 2: "2", 3: "3"}
        for key in md.index2label:
            assert type(key) is int

    # --- OD with some images having zero detections ---

    def test_od_some_empty_targets(self):
        """OD dataset where some images have zero detections."""
        labels_per_image = [[1, 2], [], [1, 3], [], [2, 3]]
        ds = ODDatasetWithLabels(5, labels_per_image, {1: "a", 2: "b", 3: "c"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        # Only 3 images contribute detections: 2+2+2 = 6
        assert len(md.class_labels) == 6
        assert set(md.class_labels) == {1, 2, 3}
        # item_indices should skip empty images
        assert 1 not in md.item_indices
        assert 3 not in md.item_indices
        assert md.item_count == 5

    def test_od_all_empty_targets(self):
        """OD dataset where every image has zero detections."""
        labels_per_image: list[list[int]] = [[], [], []]
        ds = ODDatasetWithLabels(3, labels_per_image, {0: "a", 1: "b"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        assert len(md.class_labels) == 0
        assert md.index2label == {0: "a", 1: "b"}  # provided mapping preserved even with no observations
        assert md.item_count == 3

    # --- IC with some empty targets ---

    def test_ic_some_empty_targets(self):
        """IC dataset where some images have empty target arrays."""
        labels = [1, None, 3, None, 5]

        class _ICPartialEmpty:
            @property
            def metadata(self):
                return {"id": "ic_partial_empty", "index2label": {1: "a", 3: "b", 5: "c"}}

            def __len__(self):
                return len(labels)

            def __getitem__(self, idx):
                img = np.random.rand(3, 32, 32).astype(np.float32)
                lbl = labels[idx]
                if lbl is None:
                    return img, np.array([]), {"id": idx}
                n_classes = 6
                one_hot = np.zeros(n_classes, dtype=np.float32)
                one_hot[lbl] = 1.0
                return img, one_hot, {"id": idx}

        md = Metadata(_ICPartialEmpty(), exclude=["id"])  # type: ignore
        assert len(md.class_labels) == 3
        assert set(md.class_labels) == {1, 3, 5}
        # Items 1 and 3 are skipped
        assert 1 not in md.item_indices
        assert 3 not in md.item_indices

    # --- Single-image datasets ---

    def test_single_image_ic(self):
        """IC dataset with exactly one image."""
        ds = ICDatasetCorrect(1, [2], {2: "only"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        assert len(md.class_labels) == 1
        assert md.class_labels[0] == 2
        assert md.index2label == {2: "only"}
        assert md.item_count == 1

    def test_single_image_od(self):
        """OD dataset with exactly one image and one detection."""
        ds = ODDatasetWithLabels(1, [[5]], {5: "singleton"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        assert len(md.class_labels) == 1
        assert md.class_labels[0] == 5
        assert md.index2label == {5: "singleton"}
        assert md.item_count == 1

    # --- Large label gap ---

    def test_large_label_gap_ic(self):
        """IC dataset with a large gap in label values (0 and 100)."""
        labels = [0, 100, 0, 100, 0, 100, 0, 100, 0, 100]
        ds = ICDatasetCorrect(10, labels, {0: "low", 100: "high"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        assert set(md.class_labels) == {0, 100}
        assert md.index2label == {0: "low", 100: "high"}

    def test_large_label_gap_od(self):
        """OD dataset with a large gap in label values."""
        labels_per_image = [[0, 100], [100, 0], [0, 100], [0, 100], [100, 0]]
        ds = ODDatasetWithLabels(5, labels_per_image, {0: "low", 100: "high"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        assert set(md.class_labels) == {0, 100}
        assert md.index2label == {0: "low", 100: "high"}

    # --- Extra index2label keys (more keys than observed labels) ---

    def test_extra_index2label_keys(self):
        """index2label has keys for classes not present in the data."""
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ds = ICDatasetCorrect(10, labels, {0: "unused_0", 1: "used", 2: "unused_2"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        # Full provided mapping is preserved even for unobserved classes
        assert md.index2label == {0: "unused_0", 1: "used", 2: "unused_2"}
        assert set(md.class_labels) == {1}

    # --- Downstream: single-class with bias evaluators ---

    def test_single_class_diversity(self):
        """Diversity should not crash on a single-class dataset."""
        labels = [2] * 10
        ds = ICDatasetCorrect(10, labels, {2: "only_class"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        result = Diversity().evaluate(md)
        assert result.factors.shape[0] > 0

    def test_single_class_label_stats(self):
        """label_stats should work with a single class."""
        labels = [2] * 10
        ds = ICDatasetCorrect(10, labels, {2: "only_class"})
        md = Metadata(ds, exclude=["id"])  # type: ignore
        stats = label_stats(md.class_labels, md.item_indices, md.index2label, image_count=md.item_count)
        assert len(stats["label_counts_per_image"]) == 10


# ===================================================================
# Regression: provided index2label mapping must be the source of truth
# ===================================================================


class TestIndex2LabelSourceOfTruth:
    """Regression tests ensuring the dataset's index2label mapping is fully
    preserved, even when not all classes appear in the observed data.

    Previously, Metadata only kept index2label entries for observed labels,
    which meant a dataset declaring 10 classes but only containing 7 in the
    data would report only 7 classes.
    """

    def test_ic_10_classes_7_observed(self):
        """IC dataset with 10-class mapping but only 7 classes observed."""
        full_mapping = {i: f"class_{i}" for i in range(10)}
        # Only classes 0-6 appear in the data; classes 7, 8, 9 are absent
        observed_labels = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1]
        ds = ICDatasetCorrect(len(observed_labels), observed_labels, full_mapping)
        md = Metadata(ds, exclude=["id"])  # type: ignore

        # All 10 classes from the provided mapping must be in index2label
        assert len(md.index2label) == 10
        assert md.index2label == full_mapping
        # Only 7 classes are actually observed in class_labels
        assert set(md.class_labels) == set(range(7))

    def test_od_10_classes_7_observed(self):
        """OD dataset with 10-class mapping but only 7 classes observed."""
        full_mapping = {i: f"class_{i}" for i in range(10)}
        labels_per_image = [
            [0, 1, 2],
            [3, 4],
            [5, 6],
            [0, 3, 6],
            [1, 4, 5],
        ]
        ds = ODDatasetWithLabels(5, labels_per_image, full_mapping)
        md = Metadata(ds, exclude=["id"])  # type: ignore

        assert len(md.index2label) == 10
        assert md.index2label == full_mapping
        assert set(md.class_labels) == set(range(7))

    def test_ic_unobserved_classes_with_noncontiguous_keys(self):
        """Non-contiguous index2label with unobserved classes preserved."""
        mapping = {1: "cat", 3: "dog", 5: "bird", 7: "fish", 9: "frog"}
        # Only classes 1 and 3 observed
        observed_labels = [1, 3, 1, 3, 1]
        ds = ICDatasetCorrect(len(observed_labels), observed_labels, mapping)
        md = Metadata(ds, exclude=["id"])  # type: ignore

        assert len(md.index2label) == 5
        assert md.index2label == mapping
        assert set(md.class_labels) == {1, 3}

    def test_od_unobserved_plus_undefined_labels(self):
        """OD: some labels in data don't appear in mapping, and some mapping
        entries don't appear in data — both should be handled."""
        mapping = {0: "cat", 1: "dog", 2: "bird"}
        # Label 99 is observed but not in mapping; class 2 is in mapping but not observed
        labels_per_image = [[0, 1], [0, 99], [1, 0]]
        ds = ODDatasetWithLabels(3, labels_per_image, mapping)
        md = Metadata(ds, exclude=["id"])  # type: ignore

        # All provided mapping entries preserved + fallback for 99
        assert md.index2label[0] == "cat"
        assert md.index2label[1] == "dog"
        assert md.index2label[2] == "bird"  # unobserved but in mapping
        assert md.index2label[99] == "UNDEFINED_CLASS_99"  # observed but not in mapping
        assert len(md.index2label) == 4
