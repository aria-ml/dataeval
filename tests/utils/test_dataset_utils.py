import numpy as np
import pytest

from dataeval.utils.torch import read_dataset


class TestDatasetReader:
    """
    Tests the dataset reader aggregates data into separate List[ArrayLike] from tuple return
    """

    @pytest.mark.parametrize(
        "data",
        [
            # Returns a single list
            [0, 1, 2],  # List of scalars
            [np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3))],  # List of images
            np.ones((10, 3, 3)),  # Array of images
        ],
    )
    def test_single_input(self, data):
        """Tests common input dataset output"""

        class ImageDataset:
            """Basic form of torch.utils.data.Dataset"""

            def __init__(self, data) -> None:
                self.data = data

            def __getitem__(self, idx):
                return self.data[idx]

        ds = ImageDataset(data)

        result = read_dataset(ds)  # type: ignore -> dont need to subclass from torch.utils.data.Dataset

        assert isinstance(result, list)
        assert len(result) == 1
        # assert result[0] == data

    @pytest.mark.parametrize(
        "data, labels",
        [
            # # Returns two lists
            [[0, 1, 2], [3, 4, 5]],  # List of scalar pairs e.g. preds & scores
            [np.ones((10, 3, 3)), np.ones((10,))],  # Array of images and labels
            [np.ones((10, 3, 3)), np.ones((10, 3, 3))],  # Array of images and images (AE)
        ],
    )
    def test_double_input(self, data, labels):
        """Tests common (input, target) dataset output"""

        class ICDataset:
            """Basic form of torch.utils.data.Dataset"""

            def __init__(self, data, labels):
                self.data = data
                self.labels = labels

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]

        ds = ICDataset(data, labels)

        result = read_dataset(ds)  # type: ignore -> dont need to subclass from torch.utils.data.Dataset

        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.parametrize(
        "data, labels, metadata",
        [
            # Returns three lists
            [(0, 1, 2), (3, 4, 5), (6, 7, 8)],  # Note: reader does not care about type
            [np.ones((10, 3, 3)), np.ones((10, 3, 3)), np.ones((10, 3, 3))],  # 3 sets of images
            [np.ones((10, 3, 3)), np.ones((10,)), [{i: i} for i in range(10)]],  # Images, labels, metadata
            [
                # Images, ObjectDetectionTarget, Metadata
                np.ones((10, 3, 3)),
                [{"labels": [0, 1, 2], "boxes": [[0, 1, 2], [3, 4, 5]]} for _ in range(10)],
                [{i: i} for i in range(10)],
            ],
        ],
    )
    def test_triple_input(self, data, labels, metadata):
        """Tests common (input, target, metadata) dataset output"""

        class ODDataset:
            """Basic form of torch.utils.data.Dataset"""

            def __init__(self, data, labels, metadata):
                self.data = data
                self.labels = labels
                self.metadata = metadata

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx], self.metadata[idx]

        ds = ODDataset(data, labels, metadata)

        result = read_dataset(ds)  # type: ignore -> dont need to subclass from torch.utils.data.Dataset

        assert isinstance(result, list)
        assert len(result) == 3

    def test_list_of_dicts(self):
        """Tests MAITE specific return behavior for List[Dict[str, Any]"""

        class DictDataset:
            items = {"meta1": 0, "meta2": "1", "meta3": {"metameta": 2.0}}

            def __init__(self):
                self.class_dict = [self.items for _ in range(5)]

            def __getitem__(self, idx):
                return self.class_dict[idx]

            def __len__(self):
                return len(self.class_dict)

        td = DictDataset()

        result = read_dataset(td)  # type: ignore -> dont need to subclass from torch.utils.data.Dataset

        assert len(result) == 1  # Only 1 object return
        assert len(result[0]) == 5  # 5 datapoints

        assert all(r == {"meta1": 0, "meta2": "1", "meta3": {"metameta": 2.0}} for r in result[0])
