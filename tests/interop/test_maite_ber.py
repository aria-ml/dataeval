from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pytest
import torch

from dataeval._internal.maite.ber import BERArrayLike, BERDataset  # TODO: Move to public
from dataeval.metrics import BER


class TestBERArrayLike:
    """Tests MAITE compliant inputs for BER MAITE Wrapper"""

    def test_arraylike_equals_ber(self):
        """Checks maite wrapper does not affect data quality"""
        data, labels = np.random.randint(0, 100, (10, 10)), np.arange(10)

        mber = BERArrayLike(data=data, labels=labels)
        ber = BER(data=data, labels=labels)

        npt.assert_array_equal(mber.data, ber.data)
        npt.assert_array_equal(mber.labels, ber.labels)

    @pytest.mark.parametrize(
        "arr, larr",
        [
            (torch.randint(0, 100, (10, 10)), torch.arange(0, 10)),
            # (torch.randint(0, 100, (10, 10)).to("cuda"), torch.arange(0, 10).to("cuda")),
        ],
    )
    def test_valid_arraylike(self, arr, larr):
        """Test maite.protocols.ArrayLike objects pass evaluation for MaiteBER and not BER"""

        data, labels = arr, larr

        # MaiteBER handles ArrayLike conversion
        mber = BERArrayLike(data, labels)
        assert isinstance(mber.data, np.ndarray)
        assert isinstance(mber.labels, np.ndarray)

    @pytest.mark.parametrize(
        "arr, larr",
        [
            ([[], [1], [2, 2]], ValueError),
            ([np.arange(i) for i in range(5)], ValueError),
            ([torch.arange(i) for i in range(5)], ValueError),
        ],
    )
    def test_invalid_types(self, arr, larr):
        """Test non-arraylike objects fail evaluation"""

        with pytest.raises(ValueError):
            BERArrayLike(arr, larr)


class TestBERDataset:
    def test_valid_input(self):
        dataset = MagicMock()
        ber = BERDataset(dataset)

        assert ber.data is not None  # extract returns empty array
        assert ber.labels is not None  # extract returns empty array
        assert ber.method == "KNN"
        assert ber.k == 1
