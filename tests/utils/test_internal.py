import logging
from pathlib import PosixPath

import numpy as np
import pytest
import torch

from dataeval.exceptions import ShapeMismatchError
from dataeval.utils._internal import (
    channels_first_to_last,
    ensure_embeddings,
    flatten_metadata,
    flatten_samples,
    iter_images,
    merge_metadata,
    rescale_array,
    simplify_type,
    to_numpy,
    to_numpy_iter,
    unwrap_image,
)


@pytest.mark.optional
class TestInterop:
    def test_torch_to_numpy(self):
        t = torch.tensor([1, 2, 3, 4, 5])
        n = to_numpy(t)
        assert list(n) == list(t)
        assert n.dtype == np.int64

    def test_torch_non_tensor_to_numpy(self):
        t = torch.int
        n = to_numpy(t)  # type: ignore
        assert n.shape == ()

    def test_to_numpy_iter(self):
        t = [torch.tensor([1]), torch.tensor([2, 3]), torch.tensor([4, 5, 6])]
        count = 0
        for n in to_numpy_iter(t):
            count += 1
            assert len(n) == count
            assert isinstance(n, np.ndarray)
        assert count == 3


@pytest.mark.required
class TestUnwrapImage:
    def test_passthrough_non_tuple(self):
        img = np.zeros((3, 4, 4))
        assert unwrap_image(img) is img

    def test_strips_maite_tuple(self):
        img = np.zeros((3, 4, 4))
        target = {"label": 1}
        metadata = {"id": "x"}
        assert unwrap_image((img, target, metadata)) is img

    def test_strips_two_tuple(self):
        img = np.zeros((3, 4, 4))
        assert unwrap_image((img, 1)) is img  # type: ignore

    def test_passthrough_ndarray(self):
        # ndarrays are Arrays, not tuples; passed through.
        item = np.asarray([[0, 1], [2, 3]])
        assert unwrap_image(item) is item


@pytest.mark.required
class TestIterImages:
    def test_yields_bare_images(self):
        imgs = [np.zeros((3, 4, 4)), np.ones((3, 4, 4))]
        assert list(iter_images(imgs)) == imgs

    def test_unwraps_maite_dataset(self):
        imgs = [np.zeros((3, 4, 4)), np.ones((3, 4, 4))]
        dataset = [(img, i, {"id": i}) for i, img in enumerate(imgs)]
        assert list(iter_images(dataset)) == imgs

    def test_unwraps_two_tuple(self):
        imgs = [np.zeros((3, 4, 4)), np.ones((3, 4, 4))]
        dataset = list(zip(imgs, [0, 1], strict=True))
        assert list(iter_images(dataset)) == imgs  # type: ignore

    def test_mixed_iterable(self):
        bare = np.zeros((3, 4, 4))
        wrapped = np.ones((3, 4, 4))
        result = list(iter_images([bare, (wrapped, 1, {})]))
        assert result == [bare, wrapped]

    def test_empty_iterable(self):
        assert list(iter_images([])) == []

    def test_consumes_generator(self):
        imgs = [np.zeros((3, 4, 4)), np.ones((3, 4, 4))]

        def gen():
            yield from imgs

        assert list(iter_images(gen())) == imgs


@pytest.mark.required
class TestInteropLogging:
    def test_logging(self, tmp_path: PosixPath):
        log = logging.getLogger("dataeval")
        log.setLevel(logging.DEBUG)
        handler = logging.FileHandler(filename=tmp_path / "test.log", mode="w")
        formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)
        t = torch.tensor([1, 2, 3, 4, 5])
        to_numpy(t)
        assert (tmp_path / "test.log").exists()


array_native = [[0, 1], [2, 3]]
array_expected = np.asarray(array_native)


@pytest.mark.optional
class TestInteropArrayLike:
    @pytest.mark.parametrize(
        ("param", "expected"),
        [
            (array_native, array_expected),
            (np.array(array_native), array_expected),
            (torch.Tensor(array_native), array_expected),
            (None, np.array([])),
        ],
    )
    def test_to_numpy(self, param, expected):
        actual = to_numpy(param)
        np.testing.assert_equal(actual, expected)
        assert len(actual) == len(expected)


@pytest.mark.required
class TestEnsureEmbeddings:
    tt = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    na = np.array([[5, 6], [7, 8]], dtype=np.int64)

    def test_ensure_embeddings_torch(self):
        assert type(ensure_embeddings(self.tt) is torch.Tensor)

    def test_ensure_embeddings_torch_dtype(self):
        emb = ensure_embeddings(self.tt, dtype=torch.int32)
        assert emb.dtype == torch.int32
        assert type(emb) is torch.Tensor

    def test_ensure_embeddings_torch_unit_interval(self):
        with pytest.raises(ValueError, match="Embeddings must be unit interval"):
            ensure_embeddings(self.tt, dtype=torch.int64, unit_interval=True)

    def test_ensure_embeddings_torch_force_unit_interval(self):
        emb = ensure_embeddings(self.tt, dtype=torch.float32, unit_interval="force")
        assert emb.dtype == torch.float32
        assert emb.min() >= 0.0
        assert emb.max() <= 1.0
        assert type(emb) is torch.Tensor

    def test_ensure_embeddings_numpy(self):
        assert type(ensure_embeddings(self.na) is np.ndarray)

    def test_ensure_embeddings_numpy_dtype(self):
        emb = ensure_embeddings(self.na, dtype=np.int32)
        assert emb.dtype == np.int32
        assert type(emb) is np.ndarray

    def test_ensure_embeddings_numpy_unit_interval(self):
        with pytest.raises(ValueError, match="Embeddings must be unit interval"):
            ensure_embeddings(self.na, dtype=np.int64, unit_interval=True)

    def test_ensure_embeddings_numpy_force_unit_interval(self):
        emb = ensure_embeddings(self.na, dtype=np.float32, unit_interval="force")
        assert emb.dtype == np.float32
        assert emb.min() >= 0.0
        assert emb.max() <= 1.0
        assert type(emb) is np.ndarray


@pytest.mark.required
class TestRescaleArray:
    tt = torch.rand((4, 1, 16, 16)) * 100
    na = np.random.random((4, 1, 16, 16)) * 100

    def test_rescale_torch(self):
        rescaled = rescale_array(self.tt)
        assert rescaled.shape == (4, 1, 16, 16)
        assert rescaled.min() >= 0.0
        assert rescaled.max() <= 1.0

    def test_rescale_numpy(self):
        rescaled = rescale_array(self.na)
        assert rescaled.shape == (4, 1, 16, 16)
        assert rescaled.min() >= 0.0
        assert rescaled.max() <= 1.0

    def test_invalid_input(self):
        with pytest.raises(TypeError, match="Unsupported type: <class 'str'>"):
            rescale_array("invalid input")  # type: ignore


@pytest.mark.required
class TestFlatten:
    tt = torch.rand((4, 1, 16, 16))
    na = np.random.random((4, 1, 16, 16))

    def test_flatten_torch(self):
        flat = flatten_samples(self.tt)
        assert flat.shape == (4, 256)
        assert type(flat) is torch.Tensor

    def test_flatten_numpy(self):
        flat = flatten_samples(self.na)
        assert flat.shape == (4, 256)
        assert type(flat) is np.ndarray

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            flatten_samples("invalid input")  # type: ignore


@pytest.mark.required
class TestChannelsFirstToLast:
    tt = torch.rand((1, 4, 8))
    na = np.random.random((1, 4, 8))

    def test_channels_first_to_last_torch(self):
        flipped = channels_first_to_last(self.tt)
        assert flipped.shape == (4, 8, 1)

    def test_channels_first_to_last_numpy(self):
        flipped = channels_first_to_last(self.na)
        assert flipped.shape == (4, 8, 1)

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            channels_first_to_last("invalid input")  # type: ignore


@pytest.mark.required
class TestRequiredNdim:
    def test_validate_ndim_1d(self):
        arr = [1, 2, 3]
        result = to_numpy(arr, required_ndim=1)
        assert result.ndim == 1

    def test_validate_ndim_2d(self):
        arr = [[1, 2], [3, 4]]
        result = to_numpy(arr, required_ndim=2)
        assert result.ndim == 2

    def test_validate_ndim_3d(self):
        arr = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        result = to_numpy(arr, required_ndim=3)
        assert result.ndim == 3

    def test_validate_ndim_mismatch(self):
        arr = [[1, 2], [3, 4]]
        with pytest.raises(ShapeMismatchError, match="Array has 2 dimensions, expected 1"):
            to_numpy(arr, required_ndim=1)

    def test_validate_ndim_mismatch_3d_to_2d(self):
        arr = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        with pytest.raises(ShapeMismatchError, match="Array has 3 dimensions, expected 2"):
            to_numpy(arr, required_ndim=2)

    @pytest.mark.optional
    def test_validate_ndim_torch_tensor(self):
        arr = torch.tensor([[1, 2], [3, 4]])
        result = to_numpy(arr, required_ndim=2)
        assert result.ndim == 2

    @pytest.mark.optional
    def test_validate_ndim_torch_tensor_mismatch(self):
        arr = torch.tensor([1, 2, 3])
        with pytest.raises(ShapeMismatchError, match="Array has 1 dimensions, expected 2"):
            to_numpy(arr, required_ndim=2)


@pytest.mark.required
class TestRequiredShape:
    def test_validate_shape_1d(self):
        arr = [1, 2, 3]
        result = to_numpy(arr, required_shape=(3,))
        assert result.shape == (3,)

    def test_validate_shape_2d(self):
        arr = [[1, 2, 3], [4, 5, 6]]
        result = to_numpy(arr, required_shape=(2, 3))
        assert result.shape == (2, 3)

    def test_validate_shape_3d(self):
        arr = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        result = to_numpy(arr, required_shape=(2, 2, 2))
        assert result.shape == (2, 2, 2)

    def test_validate_shape_mismatch(self):
        arr = [[1, 2], [3, 4]]
        with pytest.raises(ShapeMismatchError, match=r"Array has shape \(2, 2\), expected \(3, 3\)"):
            to_numpy(arr, required_shape=(3, 3))

    def test_validate_shape_mismatch_dimensions(self):
        arr = [1, 2, 3]
        with pytest.raises(ShapeMismatchError, match=r"Array has shape \(3,\), expected \(1, 3\)"):
            to_numpy(arr, required_shape=(1, 3))

    @pytest.mark.optional
    def test_validate_shape_torch_tensor(self):
        arr = torch.tensor([[1, 2], [3, 4], [5, 6]])
        result = to_numpy(arr, required_shape=(3, 2))
        assert result.shape == (3, 2)

    @pytest.mark.optional
    def test_validate_shape_torch_tensor_mismatch(self):
        arr = torch.tensor([[1, 2], [3, 4]])
        with pytest.raises(ShapeMismatchError, match=r"Array has shape \(2, 2\), expected \(2, 3\)"):
            to_numpy(arr, required_shape=(2, 3))


@pytest.mark.required
class TestRequiredNdimAndShape:
    def test_validate_both_correct(self):
        arr = [[1, 2, 3], [4, 5, 6]]
        result = to_numpy(arr, required_ndim=2, required_shape=(2, 3))
        assert result.ndim == 2
        assert result.shape == (2, 3)

    def test_validate_both_ndim_fails(self):
        arr = [1, 2, 3]
        with pytest.raises(ShapeMismatchError, match="Array has 1 dimensions, expected 2"):
            to_numpy(arr, required_ndim=2, required_shape=(2, 3))

    def test_validate_both_shape_fails(self):
        arr = [[1, 2], [3, 4]]
        with pytest.raises(ShapeMismatchError, match=r"Array has shape \(2, 2\), expected \(2, 3\)"):
            to_numpy(arr, required_ndim=2, required_shape=(2, 3))

    @pytest.mark.optional
    def test_validate_both_torch_correct(self):
        arr = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = to_numpy(arr, required_ndim=3, required_shape=(2, 2, 2))
        assert result.ndim == 3
        assert result.shape == (2, 2, 2)


@pytest.mark.required
class TestUtilsMetadata:
    duplicate_keys = {
        "a": 1,
        "b": {
            "b1": "b1",
            "b2": "b2",
        },
        "c": {
            "d": [
                {"e": 1, "f": 2, "g": 3},
                {"e": 4, "f": 5, "g": 6},
                {"e": 7, "f": 8, "g": 9, "z": 0},
            ],
            "h": [1.1, 1.2, 1.3],
        },
        "d": {
            "d": {"e": 4, "f": 5, "g": 6},
            "h": 1,
        },
    }

    inconsistent_keys = [
        {"a": 1, "b": [1], "c": [1, 2]},
        {"a": 2},
        {"a": 3, "d": [{"e": {"f": [{"g": 1, "h": 2}]}}]},
    ]

    numpy_value = [{"time": np.array([1.2, 3.4, 5.6]), "altitude": [235, 6789, 101112], "point": 4}]

    voc_test = [
        {
            "annotation": {
                "folder": "VOC2011",
                "filename": "2008_000009.jpg",
                "source": {"database": "The VOC2008 Database", "annotation": "PASCAL VOC2008", "image": "flickr"},
                "size": {"width": "600", "height": "300", "depth": "3"},
                "segmented": "0",
                "object": [
                    {
                        "name": "cat",
                        "pose": "Unspecified",
                        "truncated": "0",
                        "occluded": "1",
                        "bndbox": {"xmin": "53", "ymin": "87", "xmax": "471", "ymax": "420"},
                        "difficult": "0",
                    },
                    {
                        "name": "dog",
                        "pose": "Unspecified",
                        "truncated": "1",
                        "occluded": "0",
                        "bndbox": {"xmin": "158", "ymin": "44", "xmax": "289", "ymax": "167"},
                        "difficult": "0",
                    },
                    {
                        "name": "person",
                        "pose": "Right",
                        "truncated": "1",
                        "occluded": "0",
                        "bndbox": {"xmin": "158", "ymin": "44", "xmax": "289", "ymax": "167"},
                        "difficult": "0",
                    },
                ],
            },
        },
        {
            "annotation": {
                "folder": "VOC2011",
                "filename": "2008_000036.jpg",
                "source": {"database": "The VOC2008 Database", "annotation": "PASCAL VOC2008", "image": "flickr"},
                "size": {"width": "500", "height": "375", "depth": "3"},
                "segmented": "0",
                "object": [
                    {
                        "name": "bicycle",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "0",
                        "bndbox": {"xmin": "120", "ymin": "1", "xmax": "203", "ymax": "35"},
                        "difficult": "0",
                    },
                    {
                        "name": "bicycle",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "117", "ymin": "38", "xmax": "273", "ymax": "121"},
                        "difficult": "0",
                    },
                    {
                        "name": "person",
                        "pose": "Left",
                        "truncated": "0",
                        "occluded": "0",
                        "bndbox": {"xmin": "206", "ymin": "74", "xmax": "395", "ymax": "237"},
                        "difficult": "0",
                        "part": [
                            {"name": "head", "bndbox": {"xmin": "321", "ymin": "75", "xmax": "359", "ymax": "122"}},
                            {"name": "foot", "bndbox": {"xmin": "205", "ymin": "183", "xmax": "240", "ymax": "222"}},
                            {"name": "foot", "bndbox": {"xmin": "209", "ymin": "208", "xmax": "250", "ymax": "237"}},
                            {"name": "hand", "bndbox": {"xmin": "371", "ymin": "204", "xmax": "396", "ymax": "219"}},
                        ],
                    },
                    {
                        "name": "boat",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "24", "ymin": "2", "xmax": "500", "ymax": "188"},
                        "difficult": "0",
                    },
                    {
                        "name": "boat",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "1", "ymin": "187", "xmax": "500", "ymax": "282"},
                        "difficult": "0",
                    },
                ],
            },
        },
        {
            "annotation": {
                "folder": "VOC2011",
                "filename": "2008_000128.jpg",
                "source": {"database": "The VOC2008 Database", "annotation": "PASCAL VOC2008", "image": "flickr"},
                "size": {"width": "500", "height": "375", "depth": "3"},
                "segmented": "0",
                "object": [
                    {
                        "name": "sofa",
                        "pose": "Left",
                        "truncated": "0",
                        "occluded": "1",
                        "bndbox": {"xmin": "11", "ymin": "29", "xmax": "500", "ymax": "375"},
                        "difficult": "0",
                    },
                    {
                        "name": "person",
                        "pose": "Unspecified",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "1", "ymin": "85", "xmax": "361", "ymax": "375"},
                        "difficult": "0",
                        "part": [
                            {"name": "head", "bndbox": {"xmin": "243", "ymin": "88", "xmax": "358", "ymax": "225"}},
                            {"name": "hand", "bndbox": {"xmin": "168", "ymin": "209", "xmax": "216", "ymax": "257"}},
                            {"name": "hand", "bndbox": {"xmin": "94", "ymin": "252", "xmax": "128", "ymax": "308"}},
                        ],
                    },
                    {
                        "name": "person",
                        "pose": "Unspecified",
                        "truncated": "0",
                        "occluded": "1",
                        "bndbox": {"xmin": "92", "ymin": "173", "xmax": "212", "ymax": "357"},
                        "difficult": "0",
                    },
                ],
            },
        },
    ]

    def test_ignore_lists(self):
        a, d = merge_metadata([self.duplicate_keys], return_dropped=True, ignore_lists=True)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1],
            "b1": ["b1"],
            "b2": ["b2"],
            "e": [4],
            "f": [5],
            "g": [6],
            "h": [1],
            "_image_index": [0],
        }
        assert d == {"c_d": ["nested_list"], "c_h": ["nested_list"]}

    def test_fully_qualified_keys(self):
        a, d = merge_metadata([self.duplicate_keys], return_dropped=True, fully_qualified=True)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1, 1, 1],
            "b_b1": ["b1", "b1", "b1"],
            "b_b2": ["b2", "b2", "b2"],
            "c_d_e": [1, 4, 7],
            "c_d_f": [2, 5, 8],
            "c_d_g": [3, 6, 9],
            "c_h": [1.1, 1.2, 1.3],
            "d_d_e": [4, 4, 4],
            "d_d_f": [5, 5, 5],
            "d_d_g": [6, 6, 6],
            "d_h": [1, 1, 1],
            "_image_index": [0, 0, 0],
        }
        assert d == {"c_d_z": ["inconsistent_key"]}

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_duplicate_keys(self, return_numpy):
        a = merge_metadata([self.duplicate_keys], return_numpy=return_numpy)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1, 1, 1],
            "b1": ["b1", "b1", "b1"],
            "b2": ["b2", "b2", "b2"],
            "c_d_e": [1, 4, 7],
            "c_d_f": [2, 5, 8],
            "c_d_g": [3, 6, 9],
            "c_h": [1.1, 1.2, 1.3],
            "d_d_e": [4, 4, 4],
            "d_d_f": [5, 5, 5],
            "d_d_g": [6, 6, 6],
            "d_h": [1, 1, 1],
            "_image_index": [0, 0, 0],
        }

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_inconsistent_keys(self, return_numpy):
        a, d = merge_metadata(self.inconsistent_keys, return_dropped=True, return_numpy=return_numpy)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1, 2, 3],
            "_image_index": [0, 1, 2],
        }
        assert d == {"b": ["inconsistent_key"], "c": ["inconsistent_size"], "d_e_f": ["nested_list"]}

    def test_inconsistent_key(self):
        list_metadata = [{"common": 1, "target": [{"a": 1, "b": 3, "c": 5}, {"a": 2, "b": 4}], "source": "example"}]
        reorganized_metadata, dropped_keys = merge_metadata(list_metadata, return_dropped=True)
        assert reorganized_metadata == {
            "common": [1, 1],
            "a": [1, 2],
            "b": [3, 4],
            "source": ["example", "example"],
            "_image_index": [0, 0],
        }
        assert dropped_keys == {"target_c": ["inconsistent_key"]}

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_voc_test(self, return_numpy):
        a = merge_metadata(self.voc_test, return_numpy=return_numpy)
        assert {k: list(v) for k, v in a.items()} == {
            "folder": [
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
            ],
            "filename": [
                "2008_000009.jpg",
                "2008_000009.jpg",
                "2008_000009.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000128.jpg",
                "2008_000128.jpg",
                "2008_000128.jpg",
            ],
            "database": [
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
            ],
            "annotation": [
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
            ],
            "image": [
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
            ],
            "width": [600, 600, 600, 500, 500, 500, 500, 500, 500, 500, 500],
            "height": [300, 300, 300, 375, 375, 375, 375, 375, 375, 375, 375],
            "depth": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "segmented": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "name": [
                "cat",
                "dog",
                "person",
                "bicycle",
                "bicycle",
                "person",
                "boat",
                "boat",
                "sofa",
                "person",
                "person",
            ],
            "pose": [
                "Unspecified",
                "Unspecified",
                "Right",
                "Left",
                "Left",
                "Left",
                "Left",
                "Left",
                "Left",
                "Unspecified",
                "Unspecified",
            ],
            "truncated": [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
            "occluded": [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
            "xmin": [53, 158, 158, 120, 117, 206, 24, 1, 11, 1, 92],
            "ymin": [87, 44, 44, 1, 38, 74, 2, 187, 29, 85, 173],
            "xmax": [471, 289, 289, 203, 273, 395, 500, 500, 500, 361, 212],
            "ymax": [420, 167, 167, 35, 121, 237, 188, 282, 375, 375, 357],
            "difficult": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "_image_index": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
        }

    @pytest.mark.filterwarnings("error")
    def test_flatten_metadata_no_dropped_no_warn(self):
        flatten_metadata({"a": {"b": 1, "c": 2}}, return_dropped=False)

    def test_flatten_metadata_no_dropped_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            flatten_metadata(self.inconsistent_keys[0], return_dropped=False)
        assert "Metadata entries were dropped" in caplog.text

    @pytest.mark.filterwarnings("error")
    def test_merge_metadata_no_dropped_no_warn(self):
        merge_metadata([{"a": {"b": 1, "c": 2}}], return_dropped=False)

    def test_merge_metadata_no_dropped_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            merge_metadata(self.inconsistent_keys, return_dropped=False)
        assert "Metadata entries were dropped" in caplog.text

    def test_handle_numpy(self):
        output, dropped = merge_metadata(self.numpy_value, return_dropped=True)
        assert output == {
            "time": [1.2, 3.4, 5.6],
            "altitude": [235, 6789, 101112],
            "point": [4, 4, 4],
            "_image_index": [0, 0, 0],
        }
        assert dropped == {}

    def test_targets_per_image_mismatch(self):
        targets_per_image = [1]
        with pytest.raises(ValueError, match="Number of targets per image must be equal"):
            merge_metadata([{"a": 1}, {"a": 2}], targets_per_image=targets_per_image)

    def test_image_index_key_exists_in_output(self):
        merge_metadatad = merge_metadata([{"a": {"b": 1, "c": 2, "foo": 0}}], image_index_key="foo")
        assert merge_metadatad["foo"] == [0]

    def test_merge_metadata_drop_no_targets(self):
        merge_metadatad = merge_metadata([{"a": 1}, {"a": 2}, {"a": 3}], targets_per_image=[1, 0, 1])
        assert merge_metadatad["a"] == [1, 3]


@pytest.mark.required
class TestCastSimplify:
    @pytest.mark.parametrize(
        ("value", "output"),
        [
            ("123", 123),
            ("12.3", 12.3),
            ("foo", "foo"),
            ([123, "12.3"], [123.0, 12.3]),
            ([123, "foo"], ["123", "foo"]),
            (["123", "456"], [123, 456]),
        ],
    )
    def test_convert_type(self, value, output):
        assert output == simplify_type(value)
