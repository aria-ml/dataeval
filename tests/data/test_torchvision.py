"""Tests for the TorchvisionTransform escape-hatch view operation."""

import subprocess
import sys
import warnings

import numpy as np
import pytest

from dataeval.data import Shuffle, TorchvisionTransform, View
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata

torch = pytest.importorskip("torch")
v2 = pytest.importorskip("torchvision.transforms.v2")

pytestmark = pytest.mark.optional


class _ODTarget:
    def __init__(self, boxes, labels) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.ones(len(self.labels), dtype=np.float64)


class _ODDataset:
    """Object-detection dataset with per-datum ids that differ from positions."""

    def __init__(self, shape=(20, 40), boxes=None, labels=None, n: int = 4, ids=None, metadata=None) -> None:
        self._shape = shape
        self._boxes = boxes if boxes is not None else [[4.0, 4.0, 8.0, 8.0]]
        self._labels = labels if labels is not None else [0] * len(self._boxes)
        self._n = n
        self._ids = list(ids) if ids is not None else list(range(n))
        self._metadata = metadata or {}
        self.metadata = DatasetMetadata(id="toy", index2label={0: "a", 1: "b"})

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self._ids[index])
        image = (rng.integers(0, 256, size=(3, *self._shape))).astype(np.uint8)
        return image, _ODTarget(self._boxes, self._labels), {"id": self._ids[index], **self._metadata}


class _NoIdDataset:
    """Non-conformant source whose datum metadata omits the protocol-required id."""

    metadata = {"id": "no-id"}

    def __init__(self, n: int = 3) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        rng = np.random.default_rng(index)
        return (rng.integers(0, 256, size=(3, 8, 8))).astype(np.uint8), np.array([1.0, 0.0]), {}


def _jitter(seed: int = 0) -> TorchvisionTransform:
    return TorchvisionTransform(v2.ColorJitter(brightness=(0.2, 1.8)), seed=seed)


def _images_of(view) -> list[np.ndarray]:
    return [np.asarray(datum[0] if isinstance(datum, tuple) else datum) for datum in view]


@pytest.mark.required
class TestLazyImport:
    def test_importing_dataeval_data_does_not_import_torchvision(self):
        code = "import sys, dataeval.data; print('torchvision' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False"

    def test_missing_torchvision_raises_a_clear_error(self, monkeypatch):
        op = _jitter()
        monkeypatch.setitem(sys.modules, "torchvision", None)
        view = View(_ODDataset(), [op])
        with pytest.raises(ImportError, match="torchvision"):
            _ = view[0]


class TestGeometry:
    def test_boxes_track_a_geometric_transform(self):
        # 20x40 (hxw) source into 10x10: sx = 10/40 = 0.25, sy = 10/20 = 0.5
        op = TorchvisionTransform(v2.Resize((10, 10)))
        view = View(_ODDataset((20, 40), [[4.0, 4.0, 8.0, 8.0]]), [op])
        image, target, _ = view[0]
        assert np.asarray(image).shape == (3, 10, 10)
        assert np.asarray(target.boxes).tolist() == [[1.0, 2.0, 2.0, 4.0]]

    def test_labels_and_scores_survive_a_geometric_transform(self):
        op = TorchvisionTransform(v2.Resize((10, 10)))
        view = View(_ODDataset((20, 40), [[4.0, 4.0, 8.0, 8.0], [12.0, 4.0, 16.0, 8.0]], [0, 1]), [op])
        _, target, _ = view[0]
        assert np.asarray(target.labels).tolist() == [0, 1]
        assert np.asarray(target.scores).tolist() == [1.0, 1.0]

    def test_image_classification_datum_is_transformed_without_boxes(self):
        op = TorchvisionTransform(v2.Resize((4, 4)))
        view = View(_NoIdDataset(), [op])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image, target, _ = view[0]
        assert np.asarray(image).shape == (3, 4, 4)
        assert np.asarray(target).tolist() == [1.0, 0.0]


class TestDetectionDropping:
    def test_sanitize_drops_detections_and_masks_metadata_in_step(self):
        # A degenerate box is dropped by SanitizeBoundingBoxes; its metadata must go too.
        boxes = [[4.0, 4.0, 8.0, 8.0], [10.0, 10.0, 10.0, 12.0], [12.0, 4.0, 16.0, 8.0]]
        dataset = _ODDataset((20, 40), boxes, [0, 1, 0], metadata={"track": [100, 200, 300]})
        op = TorchvisionTransform(v2.SanitizeBoundingBoxes())
        view = View(dataset, [op])
        _, target, metadata = view[0]
        assert len(np.asarray(target.boxes)) == 2
        assert np.asarray(target.labels).tolist() == [0, 0]
        assert metadata["track"] == [100, 300]

    def test_out_of_frame_crop_drops_detections(self):
        boxes = [[0.0, 0.0, 4.0, 4.0], [30.0, 10.0, 38.0, 18.0]]
        dataset = _ODDataset((20, 40), boxes, [0, 1], metadata={"track": [7, 9]})
        op = TorchvisionTransform(v2.Compose([v2.CenterCrop((20, 10)), v2.SanitizeBoundingBoxes()]))
        view = View(dataset, [op])
        _, target, metadata = view[0]
        assert len(np.asarray(target.boxes)) == len(metadata["track"])


class TestDeterminism:
    def test_iterating_twice_yields_byte_identical_output(self):
        view = View(_ODDataset(n=4), [_jitter()])
        first, second = _images_of(view), _images_of(view)
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))

    def test_a_random_transform_actually_varies_across_datums(self):
        # Guards the test above from passing on a no-op transform.
        source = _ODDataset(n=4)
        plain = [np.asarray(source[i][0]) for i in range(4)]
        jittered = _images_of(View(source, [_jitter()]))
        assert any(not np.array_equal(a, b) for a, b in zip(plain, jittered, strict=True))

    def test_augmentation_follows_the_datum_id_not_its_position(self):
        source = _ODDataset(n=4, ids=[10, 11, 12, 13])
        direct = View(source, [_jitter()])
        shuffled = View(View(source, [Shuffle(seed=7)]), [_jitter()])

        by_id = {datum[2]["id"]: np.asarray(datum[0]) for datum in direct}
        for datum in shuffled:
            assert np.array_equal(np.asarray(datum[0]), by_id[datum[2]["id"]])

    def test_shuffling_upstream_actually_reorders(self):
        # Guards the test above from passing because Shuffle did nothing.
        source = _ODDataset(n=4, ids=[10, 11, 12, 13])
        shuffled = View(View(source, [Shuffle(seed=7)]), [_jitter()])
        assert [datum[2]["id"] for datum in shuffled] != [10, 11, 12, 13]

    def test_output_is_identical_under_different_python_hash_seeds(self):
        code = (
            "import numpy as np;"
            "from torchvision.transforms import v2;"
            "from dataeval.data import TorchvisionTransform, View;"
            "import xxhash as xxh;"
            "img = lambda i: (np.random.default_rng(i).integers(0, 256, (3, 8, 8))).astype(np.uint8);"
            "ds = type('D', (), {"
            "  'metadata': {'id': 'd'},"
            "  '__len__': lambda s: 3,"
            "  '__getitem__': lambda s, i: (img(i), np.array([1.0]), {'id': f'item-{i}'}),"
            "})();"
            "op = TorchvisionTransform(v2.ColorJitter(brightness=(0.2, 1.8)), seed=3);"
            "out = np.stack([np.asarray(d[0]) for d in View(ds, [op])]);"
            "print(xxh.xxh64_hexdigest(out.tobytes()))"
        )
        digests = []
        for hash_seed in ("0", "12345"):
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": hash_seed, "HOME": "/tmp"},
            )
            digests.append(result.stdout.strip())
        assert digests[0] == digests[1]

    def test_two_chained_ops_with_distinct_seeds_decorrelate(self):
        source = _ODDataset(n=3)
        distinct = _images_of(View(source, [_jitter(seed=0), _jitter(seed=1)]))
        repeated = _images_of(View(source, [_jitter(seed=0), _jitter(seed=0)]))
        assert any(not np.array_equal(a, b) for a, b in zip(distinct, repeated, strict=True))


class TestMissingDatumId:
    def test_missing_id_warns_once(self):
        view = View(_NoIdDataset(n=3), [_jitter()])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _images_of(view)
        matching = [w for w in record if "id" in str(w.message)]
        assert len(matching) == 1

    def test_missing_id_falls_back_to_a_content_hash_and_round_trips(self):
        view = View(_NoIdDataset(n=3), [_jitter()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first, second = _images_of(view), _images_of(view)
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))

    def test_content_hash_gives_different_datums_different_augmentations(self):
        view = View(_NoIdDataset(n=3), [_jitter()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            images = _images_of(view)
        assert not np.array_equal(images[0], images[1])


class TestNormalizationGuard:
    def test_warns_on_a_chain_ending_in_normalize(self):
        op = TorchvisionTransform(
            v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        )
        view = View(_ODDataset(n=3), [op])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _images_of(view)
        matching = [w for w in record if "transforms=" in str(w.message)]
        assert len(matching) == 1

    def test_does_not_warn_on_a_geometric_transform(self):
        view = View(_ODDataset(n=3), [TorchvisionTransform(v2.Resize((10, 10)))])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _images_of(view)
        assert not [w for w in record if "transforms=" in str(w.message)]


@pytest.mark.required
class TestInvalidates:
    def test_defaults_to_everything(self):
        assert TorchvisionTransform(object()).invalidates is ImageStats.ALL

    def test_is_user_narrowable(self):
        op = TorchvisionTransform(object(), invalidates=ImageStats.DIMENSION)
        assert op.invalidates is ImageStats.DIMENSION

    def test_declaration_reaches_the_invalidation_walk(self):
        from dataeval.data._invalidates import invalidated_stats

        op = TorchvisionTransform(object(), invalidates=ImageStats.VISUAL_SHARPNESS)
        assert invalidated_stats(View(_ODDataset(), [op])) is ImageStats.VISUAL_SHARPNESS

    def test_docstring_warns_that_a_view_may_not_be_reconstructable(self):
        doc = TorchvisionTransform.__doc__ or ""
        assert "sidecar" in doc.lower()
        assert "reconstruct" in doc.lower()
