import numpy as np
import pytest

from dataeval.utils.data import Embeddings, Images, Metadata
from dataeval.utils.data._dataset import _validate_data, to_image_classification_dataset, to_object_detection_dataset


@pytest.fixture(scope="module")
def images():
    return [np.random.random((3, 16, 16)) for _ in range(10)]


@pytest.fixture(scope="module")
def ic_labels():
    return [np.random.randint(0, 9) for _ in range(10)]


@pytest.fixture(scope="module")
def od_labels():
    return [[np.random.randint(0, 9) for _ in range(i + 1)] for i in range(10)]


@pytest.fixture(scope="module")
def bboxes():
    return [[(0.0, 0.0, 10.0, 10.0) for _ in range(i + 1)] for i in range(10)]


@pytest.fixture(scope="module")
def ic_metadata():
    return [{"foo": i} for i in range(10)]


@pytest.fixture(scope="module")
def od_metadata():
    return [{"foo": i, "bar": [i for _ in range(i + 1)]} for i in range(10)]


@pytest.fixture(scope="module")
def classes():
    return [str(i) for i in range(10)]


class TestDatasetValidateData:
    bad_images = [np.random.random((16, 16)) for _ in range(10)]
    bad_ic_labels = [np.random.randint(0, 9) for _ in range(11)]
    bad_labels_type = [str(i) for i in range(10)]
    bad_boxes = [[(0.0, 0.0, 10.0, 10.0) for _ in range(i + 1)] for i in range(11)]
    bad_boxes_type = [[(0.0, 0.0, 10.0) for _ in range(i + 1)] for i in range(10)]
    bad_metadata = [{"foo": i} for i in range(11)]

    def test_validate_data_invalid_image_shape(self, ic_labels):
        with pytest.raises(ValueError, match="Images must be a sequence or array"):
            _validate_data("ic", self.bad_images, ic_labels, None, None)

    def test_validate_data_invalid_label_length(self, images):
        with pytest.raises(ValueError, match="Number of labels"):
            _validate_data("ic", images, self.bad_ic_labels, None, None)

    def test_validate_data_invalid_bbox_length(self, images, ic_labels):
        with pytest.raises(ValueError, match="Number of bboxes"):
            _validate_data("od", images, ic_labels, self.bad_boxes, None)

    def test_validate_data_invalid_metadata_length(self, images, ic_labels):
        with pytest.raises(ValueError, match="Number of metadata"):
            _validate_data("ic", images, ic_labels, None, self.bad_metadata)

    def test_validate_data_ic_labels_invalid_type(self, images):
        with pytest.raises(TypeError, match="Labels must be a sequence of integers"):
            _validate_data("ic", images, self.bad_labels_type, None, None)  # type: ignore

    def test_validate_data_od_labels_invalid_type(self, images, bboxes):
        with pytest.raises(TypeError, match="Labels must be a sequence of sequences of integers"):
            _validate_data("od", images, self.bad_labels_type, bboxes, None)  # type: ignore

    def test_validate_data_bboxes_invalid_type(self, images, od_labels):
        with pytest.raises(TypeError, match="Boxes must be a sequence of sequences"):
            _validate_data("od", images, od_labels, self.bad_boxes_type, None)  # type: ignore

    def test_validate_data_unknown_datum_type(self, images, ic_labels):
        with pytest.raises(ValueError, match="Unknown datum type"):
            _validate_data("unknown", images, ic_labels, None, None)  # type: ignore


class TestDatasetFactoryFunctions:
    def test_to_image_classification_dataset(self, images, ic_labels, classes):
        ds = to_image_classification_dataset(images, ic_labels, None, classes)
        assert len(Images(ds).to_list()) == 10
        assert len(Embeddings(ds, batch_size=10).to_tensor()) == 10
        assert len(Metadata(ds).image_indices) == 10

    def test_to_image_classification_dataset_no_classes(self, images, ic_labels):
        ds = to_image_classification_dataset(images, ic_labels, None, None)
        assert len(Images(ds).to_list()) == 10
        assert len(Embeddings(ds, batch_size=10).to_tensor()) == 10
        assert len(Metadata(ds).image_indices) == 10

    def test_to_image_classification_dataset_with_meta(self, images, ic_labels, ic_metadata, classes):
        ds = to_image_classification_dataset(images, ic_labels, ic_metadata, classes)
        assert len(Images(ds).to_list()) == 10
        assert len(Embeddings(ds, batch_size=10).to_tensor()) == 10
        assert len(Metadata(ds).image_indices) == 10

    def test_to_image_classification_dataset_with_name(self, images, ic_labels, classes):
        name = "Test_IC_Dataset"
        ds = to_image_classification_dataset(images, ic_labels, None, classes, name)
        assert name in ds.__repr__()
        assert name in ds.__class__.__name__

    def test_to_object_detection_dataset(self, images, od_labels, bboxes, classes):
        ds = to_object_detection_dataset(images, od_labels, bboxes, None, classes)
        assert len(Images(ds).to_list()) == 10
        assert len(Embeddings(ds, batch_size=10).to_tensor()) == 10
        assert len(Metadata(ds).image_indices) == 55  # 1 + 2 + 3 + ... + 10

    def test_to_object_detection_dataset_no_classes(self, images, od_labels, bboxes):
        ds = to_object_detection_dataset(images, od_labels, bboxes, None, None)
        assert len(Images(ds).to_list()) == 10
        assert len(Embeddings(ds, batch_size=10).to_tensor()) == 10
        assert len(Metadata(ds).image_indices) == 55  # 1 + 2 + 3 + ... + 10

    def test_to_object_detection_dataset_with_meta(self, images, od_labels, bboxes, od_metadata, classes):
        ds = to_object_detection_dataset(images, od_labels, bboxes, od_metadata, classes)
        assert len(Images(ds).to_list()) == 10
        assert len(Embeddings(ds, batch_size=10).to_tensor()) == 10
        assert len(Metadata(ds).image_indices) == 55  # 1 + 2 + 3 + ... + 10

    def test_to_object_detection_dataset_with_name(self, images, od_labels, bboxes, classes):
        name = "Test_OD_Dataset"
        ds = to_object_detection_dataset(images, od_labels, bboxes, None, classes, name=name)
        assert name in ds.__repr__()
        assert name in ds.__class__.__name__
