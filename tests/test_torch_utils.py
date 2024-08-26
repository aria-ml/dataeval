from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval._internal.interop import is_arraylike
from dataeval._internal.utils import _validate_getitem
from dataeval.utils import read_dataset


class TestDatasetReader:
    """
    Tests the dataset reader aggregates data into two separate List[ArrayLike]

    MagicMock.__iter__ is used to mock `torch.utils.data.Dataset` __getitem__ behavior
    """

    @pytest.mark.parametrize(
        "return_value",
        [
            [(np.ones(shape=(1, 16, 16)), np.ones(shape=(1))) for _ in range(10)],
            [(np.ones(shape=(1, 16, 16)), np.ones(shape=(1)), {"a": 1, "b": 2}) for _ in range(10)],
            [(np.ones(shape=(1, 16 + i, 16 + i)), np.ones(shape=(1))) for i in range(10)],
        ],
    )
    def test_multiple_returns(self, return_value):
        """Tests Tuple[ArrayLike, ArrayLike] is valid regardless of additional values or image shapes"""
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = return_value

        images, labels = read_dataset(mock_dataset)

        assert isinstance(images, list)
        assert isinstance(labels, list)
        assert is_arraylike(images[0])
        assert is_arraylike(labels[0])
        assert len(images) == len(labels) == 10
        assert labels == [np.ones(shape=(1)) for _ in range(10)]


class TestValidateData:
    """
    Validates that all return data is ArrayLike

    MagicMock.__iter__ is used to mock `torch.utils.data.Dataset` __getitem__ behavior
    """

    def test_zero_tuple_return(self):
        """Tests function raises ValueError if minimum length < 1"""
        mock_dataset = MagicMock()

        with pytest.raises(ValueError):
            _validate_getitem(mock_dataset, 0)

    def test_one_return(self):
        """Tests function raises valueError if less than min_length"""
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = [(np.ones((1,)),) for _ in range(10)]

        _validate_getitem(mock_dataset, 1)  # 1 value tuple is acceptable

        with pytest.raises(ValueError):
            _validate_getitem(mock_dataset, 2)  # 1 value tuple is not acceptable

    def test_invalid_type_return(self):
        """Test function raises TypeError when return type isn't tuple"""
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = np.ones(shape=(1, 16, 16))

        with pytest.raises(TypeError):
            _validate_getitem(mock_dataset, 1)  # Non-tuple is not acceptable, even if > min_length

    def test_invalid_type_data_pos_0(self):
        """Tests function raises TypeError when data[0] is not ArrayLike"""
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = [({i: i}, np.ones((1,))) for i in range(10)]  # dict instead of ArrayLike

        with pytest.raises(TypeError):
            _validate_getitem(mock_dataset, 2)

    def test_invalid_type_data_pos_1(self):
        """Tests function raises TypeError when data[1] is not ArrayLike"""
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = [
            (np.ones((1, 5, 5)), {i: i}) for i in range(10)
        ]  # type int instead of ArrayLike

        with pytest.raises(TypeError):
            _validate_getitem(mock_dataset, 2)

    def test_valid_return(self):
        """Tests valid inputs raise no errors"""
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = [(np.ones(shape=(1, 16, 16)), np.ones(shape=(1))) for _ in range(10)]

        _validate_getitem(mock_dataset, 1)
