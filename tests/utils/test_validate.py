"""Tests for the MAITE dataset shape validator.

Covers :func:`dataeval.utils.data.validate_dataset`, the
``@requires_maite_dataset`` decorator, and integration with the
public entry points it guards (:class:`Embeddings`, :class:`Metadata`,
:class:`Select` + :class:`ClassFilter`, ``split_dataset``, ``unzip_dataset``).
"""

from typing import Any

import numpy as np
import pytest

from dataeval._embeddings import Embeddings
from dataeval._metadata import Metadata
from dataeval.data import ClassFilter, Limit, Select, Shuffle, split_dataset, unzip_dataset
from dataeval.exceptions import MaiteShapeError
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.utils.data import requires_maite_dataset, validate_dataset

# ---------- fixtures ----------


class _ImageOnly:
    """Bare-image dataset: dataset[i] is a (3, 8, 8) array."""

    def __init__(self, n: int = 4) -> None:
        self.data = np.zeros((n, 3, 8, 8), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.data[i]


class _ICDataset:
    """Image classification MAITE dataset."""

    metadata = DatasetMetadata(id="ic_test")

    def __init__(self, n: int = 4, k: int = 3) -> None:
        self.data = np.zeros((n, 3, 8, 8), dtype=np.float32)
        self.y = np.eye(k)[np.arange(n) % k].astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, DatumMetadata]:
        return self.data[i], self.y[i], DatumMetadata(id=i)


class _ODTarget:
    def __init__(self) -> None:
        self.boxes = np.zeros((1, 4), dtype=np.float32)
        self.labels = np.array([0], dtype=np.intp)
        self.scores = np.array([[1.0]], dtype=np.float32)


class _ODDataset:
    metadata: DatasetMetadata = DatasetMetadata(id="od_test")

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[np.ndarray, _ODTarget, DatumMetadata]:
        return np.zeros((3, 8, 8), dtype=np.float32), _ODTarget(), DatumMetadata(id=i)


# ---------- validate_dataset: happy paths ----------


class TestValidateDatasetHappy:
    def test_image_only_accepts_bare(self) -> None:
        assert validate_dataset(_ImageOnly(), expected="image_only") == "image_only"

    def test_image_only_accepts_tuple(self) -> None:
        assert validate_dataset(_ICDataset(), expected="image_only") == "image_only"

    def test_classification(self) -> None:
        assert validate_dataset(_ICDataset(), expected="classification") == "classification"

    def test_object_detection(self) -> None:
        assert validate_dataset(_ODDataset(), expected="object_detection") == "object_detection"

    def test_any_target_resolves_concrete_kind(self) -> None:
        assert validate_dataset(_ICDataset(), expected="any_target") == "classification"
        assert validate_dataset(_ODDataset(), expected="any_target") == "object_detection"


# ---------- validate_dataset: failure modes ----------


class TestValidateDatasetFailures:
    def test_bare_image_fails_target_kind(self) -> None:
        with pytest.raises(MaiteShapeError, match="3-tuple"):
            validate_dataset(_ImageOnly(), expected="classification")

    def test_bare_image_fails_any_target(self) -> None:
        with pytest.raises(MaiteShapeError, match="3-tuple"):
            validate_dataset(_ImageOnly(), expected="any_target")

    def test_ic_target_rejected_as_od(self) -> None:
        with pytest.raises(MaiteShapeError, match="ObjectDetectionTarget"):
            validate_dataset(_ICDataset(), expected="object_detection")

    def test_od_target_rejected_as_ic(self) -> None:
        with pytest.raises(MaiteShapeError, match="Array of class scores"):
            validate_dataset(_ODDataset(), expected="classification")

    def test_empty_dataset_is_allowed(self) -> None:
        # Empty datasets are legal (e.g. after filtering) — no probe, no rejection.
        class _Empty:
            def __len__(self) -> int:
                return 0

            def __getitem__(self, i: int) -> Any:
                raise IndexError

        assert validate_dataset(_Empty(), expected="any_target") == "image_only"
        assert validate_dataset(_Empty(), expected="object_detection") == "object_detection"

    def test_unsized(self) -> None:
        class _NoLen:
            def __getitem__(self, i: int) -> int:
                return i

        with pytest.raises(MaiteShapeError, match="not Sized"):
            validate_dataset(_NoLen(), expected="image_only")

    def test_wrong_tuple_arity(self) -> None:
        class _TwoTuple:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, i: int) -> tuple[Any, Any]:
                return np.zeros((3, 8, 8)), {"id": i}

        with pytest.raises(MaiteShapeError, match="3-tuple"):
            validate_dataset(_TwoTuple(), expected="any_target")

    def test_unknown_kind_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="unknown expected"):
            validate_dataset(_ICDataset(), expected="bogus")  # type: ignore[arg-type]


# ---------- @requires_maite_dataset ----------


class TestRequiresMaiteDataset:
    def test_param_name_passed_through(self) -> None:
        @requires_maite_dataset("ds", expected="image_only")
        def f(ds: Any) -> int:
            return len(ds)

        assert f(_ImageOnly()) == 4

    def test_failure_message_includes_qualname(self) -> None:
        @requires_maite_dataset(expected="object_detection")
        def f(dataset: Any) -> None: ...

        with pytest.raises(MaiteShapeError, match="f:"):
            f(_ImageOnly())

    def test_none_is_skipped(self) -> None:
        @requires_maite_dataset(expected="any_target")
        def f(dataset: Any = None) -> str:
            return "ok"

        assert f() == "ok"
        assert f(dataset=None) == "ok"

    def test_keyword_passing_works(self) -> None:
        @requires_maite_dataset(expected="classification")
        def f(dataset: Any) -> int:
            return len(dataset)

        assert f(dataset=_ICDataset()) == 4

    def test_decorator_rejects_missing_param(self) -> None:
        with pytest.raises(TypeError, match="no parameter named 'dataset'"):

            @requires_maite_dataset(expected="any_target")
            def f(x: Any) -> None: ...


# ---------- integration: public entry points ----------


class TestIntegrationEmbeddings:
    def test_image_only_dataset_accepted(self) -> None:
        # Embeddings explicitly supports Dataset[ArrayLike]
        Embeddings(_ImageOnly())

    def test_maite_dataset_accepted(self) -> None:
        Embeddings(_ICDataset())

    def test_none_dataset_accepted(self) -> None:
        # Unbound construction is supported
        Embeddings()


class TestIntegrationMetadata:
    def test_maite_dataset_accepted(self) -> None:
        Metadata(_ICDataset())
        Metadata(_ODDataset())

    def test_bare_image_rejected(self) -> None:
        with pytest.raises(MaiteShapeError, match="3-tuple"):
            Metadata(_ImageOnly())  # pyright: ignore[reportArgumentType]

    def test_none_dataset_accepted(self) -> None:
        Metadata(None)


class TestIntegrationSelect:
    def test_target_agnostic_filters_skip_validation(self) -> None:
        # Limit/Shuffle don't read targets; image-only datasets must keep working.
        assert len(Select(_ImageOnly(10), selections=[Limit(size=3)])) == 3
        assert len(Select(_ImageOnly(10), selections=[Shuffle()])) == 10

    def test_classfilter_on_bare_image_fails_fast(self) -> None:
        with pytest.raises(MaiteShapeError, match="3-tuple"):
            Select(_ImageOnly(), selections=[ClassFilter(classes=[0])])

    def test_classfilter_on_maite_dataset_works(self) -> None:
        # Should not raise; whether anything is selected depends on labels.
        Select(_ICDataset(), selections=[ClassFilter(classes=[0, 1, 2])])


class TestIntegrationUnzip:
    def test_per_target_requires_od(self) -> None:
        with pytest.raises(MaiteShapeError, match="ObjectDetectionTarget"):
            unzip_dataset(_ICDataset(), per_target=True)

    def test_per_target_false_accepts_bare_image(self) -> None:
        images, targets = unzip_dataset(_ImageOnly(), per_target=False)
        assert targets is None
        assert len(list(images)) == 4

    def test_per_target_true_on_od_works(self) -> None:
        images, targets = unzip_dataset(_ODDataset(), per_target=True)
        assert targets is not None
        assert len(list(images)) == 4


class TestIntegrationSplit:
    def test_bare_image_dataset_rejected(self) -> None:
        with pytest.raises(MaiteShapeError, match="3-tuple"):
            split_dataset(_ImageOnly(20), num_folds=2, val_frac=0.0)  # pyright: ignore[reportArgumentType]
