import hashlib

import pytest

from dataeval.utils.data.datasets._base import DataLocation
from dataeval.utils.data.datasets._mnist import MNIST
from dataeval.utils.data.datasets._voc import VOCSegmentation
from dataeval.utils.data.types import SegmentationTarget

TEMP_MD5 = "d149274109b50d5147c09d6fc7e80c71"
TEMP_SHA256 = "2b749913055289cb3a5c602a17196b5437dc59bba50e986ea449012a303f7201"


def get_tmp_hash(fpath, chunk_size=65535):
    hasher = hashlib.md5()
    with open(fpath, "rb") as fpath_file:
        while chunk := fpath_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


# @pytest.mark.optional
# class TestDatasetSelection:
#     @pytest.mark.parametrize(
#         "size, from_back, balance, trunc, expected",
#         [
#             (-1, False, False, False, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8, 9])),
#             (3, True, True, False, np.array([9, 7, 6])),
#             (1, False, True, False, np.array([0, 1, 2])),
#             (15, False, True, False, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8])),
#             (15, False, False, False, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8, 9])),
#             (5, True, False, False, np.array([9, 7, 6, 8, 5])),
#             (-1, False, False, True, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8])),
#         ],
#     )
#     def test_ic_data_subselection(self, size, from_back, balance, trunc, expected):
#         labels = ["0", "1", "2", "2", "1", "0", "1", "0", "2", "2"]
#         if trunc:
#             labels = labels[:-1]
#         out = BaseClassificationDataset._ic_data_subselection(labels, {0, 1, 2}, size, from_back, balance, False)
#         assert np.all(out == expected)

#     @pytest.mark.parametrize(
#         "size, from_back, balance, expected",
#         [
#             (15, False, True, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8])),
#             (15, True, True, np.array([9, 7, 6, 8, 5, 4, 3, 0, 1])),
#             (15, False, False, np.array([0, 1, 2, 5, 4, 3, 7, 6, 8, 9])),
#         ],
#     )
#     def test_ic_data_subselection_warning(self, size, from_back, balance, expected):
#         labels = ["0", "1", "2", "2", "1", "0", "1", "0", "2", "2"]
#         if balance:
#             warn_msg = (
#                 f"Because of dataset limitations, only {9} samples will be returned, instead of the desired {size}."
#             )
#         else:
#             sent2 = "Adjusting down to raw dataset size."
#             warn_msg = f"Asked for more samples, {size}, than the raw dataset contains, {10}. " + sent2
#         with pytest.warns(UserWarning, match=warn_msg):
#             out = BaseClassificationDataset._ic_data_subselection(labels, {0, 1, 2}, 15, from_back, balance, True)
#         assert np.all(out == expected)


@pytest.mark.optional
class TestBaseDataset:
    @pytest.mark.parametrize("verbose", [True, False])
    def test_get_resource(self, capsys, dataset_nested_folder, mnist_npy, verbose, monkeypatch):
        def mock_resources(dataset_nested_folder, mnist_npy):
            resources = [
                DataLocation(
                    url="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
                    filename="mnist.npz",
                    md5=False,
                    checksum=get_tmp_hash(mnist_npy / "mnist.npz"),
                ),
                DataLocation(
                    url="https://zenodo.org/record/3239543/files/mnist_c.zip",
                    filename="mnist_c.zip",
                    md5=True,
                    checksum=get_tmp_hash(dataset_nested_folder),
                ),
            ]
            return resources

        monkeypatch.setattr(MNIST, "_resources", mock_resources(dataset_nested_folder, mnist_npy))
        datasetA = MNIST(root=dataset_nested_folder.parent, download=False, corruption="translate", verbose=verbose)
        assert len(datasetA) == 5000
        img, *_ = datasetA[0]
        assert img.shape == (1, 28, 28)
        if verbose:
            captured = capsys.readouterr()
            assert "Determining if mnist_c.zip needs to be downloaded." in captured.out
            assert "mnist_c.zip already exists, skipping download." in captured.out
        datasetB = MNIST(root=mnist_npy, download=False, verbose=verbose)
        assert len(datasetB) == 50000
        img, *_ = datasetA[0]
        assert img.shape == (1, 28, 28)
        if verbose:
            captured = capsys.readouterr()
            assert "Determining if mnist.npz needs to be downloaded." in captured.out
            assert "No download needed, loaded data successfully." in captured.out
        print(datasetA)
        captured = capsys.readouterr()
        assert "Dataset" in captured.out

    # def test_dataset_unit_interval(self, mnist_npy):
    #     """Test unit_interval functionality"""
    #     dataset = MNIST(root=mnist_npy, size=1000, unit_interval=True, verbose=False)
    #     images = np.vstack([img for img, _, _ in dataset])
    #     assert np.all((images >= 0) & (images <= 1))

    # def test_dataset_normalize(self, mnist_npy):
    #     """Test normalization functionality."""
    #     dataset = MNIST(root=mnist_npy, size=1000, unit_interval=True, normalize=(0.5, 0.5), dtype=np.float32)
    #     images = np.vstack([img for img, _, _ in dataset])
    #     assert np.all((images >= -1) & (images <= 1))
    #     assert np.min(images) == -1

    # def test_dataset_flatten(self, mnist_npy):
    #     """Test flattening functionality."""
    #     dataset = MNIST(root=mnist_npy, size=1000, flatten=True)
    #     images = np.vstack([img for img, _, _ in dataset])
    #     assert len(images) == len(dataset)
    #     assert images.shape == (1000, 784)

    # @pytest.mark.parametrize(
    #     "channels, expected_img, expected_scene",
    #     [
    #         ("channels_first", (3, 10, 10), (3, 1500, 1250)),
    #         ("channels_last", (10, 10, 3), (1500, 1250, 3)),
    #     ],
    # )
    # def test_dataset_channels(self, ship_fake, channels, expected_img, expected_scene):
    #     """Test channels_first functionality."""
    #     dataset = Ships(root=str(ship_fake), size=1000, channels=channels)
    #     img, _, _ = dataset[0]
    #     assert img.shape == expected_img
    #     scene = dataset.get_scene(0)
    #     assert scene.shape == expected_scene


# @pytest.mark.optional
# class TestBaseICDataset:
#     @pytest.mark.parametrize("verbose", [True, False])
#     def test_dataset_preprocess(self, capsys, wrong_mnist, verbose):
#         """Test selecting different sized datasets."""
#         if not verbose:
#             dataset = MNIST(root=wrong_mnist, size=5000, randomize=False, verbose=verbose)
#             assert len(dataset) == 5000
#         else:
#             dataset = MNIST(root=wrong_mnist, size=5000, randomize=False, slice_back=False, verbose=verbose)
#             captured = capsys.readouterr()
#             assert "Running data preprocessing steps" in captured.out
#             assert len(dataset) == 5000

#     @pytest.mark.parametrize("balance, from_back", [(True, False), (False, True), (False, False)])
#     def test_dataset_preprocess_metadata(self, ship_fake, wrong_mnist, balance, from_back):
#         """Test selecting different sized datasets with and without metadata."""
#         dataset = Ships(root=ship_fake, balance=balance, slice_back=from_back)
#         assert dataset._datum_metadata != {}
#         dataset = MNIST(root=wrong_mnist, balance=balance, randomize=False, slice_back=from_back)
#         assert dataset._datum_metadata == {}

#     @pytest.mark.parametrize("ship, expected", [(True, (4000, None)), (False, (49680, 100))])
#     def test_dataset_slice_back(self, ship_fake, mnist_npy, ship, expected):
#         """Test the functionality of slicing from the back."""
#         if ship:
#             datasetA = Ships(root=ship_fake, size=-1, slice_back=True, randomize=False)
#             datasetB = Ships(root=ship_fake, size=1000, slice_back=True, balance=False, randomize=True)
#             indicesA = np.arange(len(datasetA))
#             indicesB = np.arange(len(datasetB))
#         else:
#             datasetA = MNIST(root=mnist_npy, size=-1, slice_back=True, randomize=False)
#             datasetB = MNIST(root=mnist_npy, size=1000, slice_back=True, balance=False, randomize=True)
#             indicesA = datasetA._indices
#             indicesB = datasetB._indices

#         label_arrayA = np.array(datasetA._annotations, dtype=np.uintp)
#         label_arrayB = np.array(datasetB._annotations, dtype=np.uintp)

#         assert indicesA is not None
#         assert indicesB is not None
#         _, countsA = np.unique_counts(label_arrayA[indicesA[:1000]])
#         _, countsB = np.unique_counts(label_arrayB[indicesB])
#         assert np.all(countsA == countsB)

#         assert indicesA.size == expected[0]
#         assert expected[1] is None or np.all(countsB == expected[1])

#     def test_ic_dataset(self, mnist_npy):
#         """Test dataset properties."""
#         dataset = MNIST(root=mnist_npy, size=1000)
#         if isinstance(dataset, MNIST):
#             assert dataset._resources is not None
#             assert dataset.index2label != {}
#             assert dataset.label2index != {}
#             assert "id" in dataset.metadata
#             assert len(dataset) == 1000
#             img, *_ = dataset[0]
#             assert img.shape == (1, 28, 28)
#             this = dataset.info()
#             assert "Train\n-----\n" in this


@pytest.mark.optional
class TestBaseVOCDataset:
    def mock_resources(self, base):
        resources = [
            DataLocation(
                url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
                filename="VOCtrainval_11-May-2012.tar",
                md5=True,
                checksum=get_tmp_hash(base / "VOCtrainval_11-May-2012.tar"),
            ),
        ]
        return resources

    def test_seg_dataset(self, voc_fake, monkeypatch):
        "Test to make sure the BaseSegDataset has all the required parts"
        monkeypatch.setattr(VOCSegmentation, "_resources", self.mock_resources(voc_fake))
        dataset = VOCSegmentation(root=voc_fake)
        if isinstance(dataset, VOCSegmentation):
            assert dataset._resources is not None
            assert dataset.index2label != {}
            assert dataset.label2index != {}
            assert "id" in dataset.metadata
            assert len(dataset) == 3
            img, target, datum_meta = dataset[1]
            assert img.shape == (3, 10, 10)
            assert isinstance(target, SegmentationTarget)
            assert "pose" in datum_meta

    def test_voc_wrong_year(self, voc_fake):
        """Test ask for test set with wrong year"""
        err_msg = "The only test set available is for the year 2007, not 2012."
        with pytest.raises(ValueError) as e:
            VOCSegmentation(root=voc_fake, image_set="test")
        assert err_msg in str(e.value)

    def test_voc_2007_test(self, voc_fake, monkeypatch):
        """Test correctly ask for test set"""
        monkeypatch.setattr(VOCSegmentation, "_resources", self.mock_resources(voc_fake))
        dataset = VOCSegmentation(root=voc_fake, year="2007", image_set="test")
        assert dataset.path.stem == "VOC2007"
