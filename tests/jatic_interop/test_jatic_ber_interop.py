import numpy as np
import pytest

from daml._internal.interop.wrappers.jatic import JaticClassificationDatasetWrapper
from daml.metrics.ber import BER_FNN, BER_MST, BEROutput
from tests.utils.JaticUtils import (
    JaticImageClassificationDataset,
    check_jatic_classification,
)


@pytest.mark.interop
@pytest.mark.parametrize(
    "method, output",
    [
        (BER_MST, BEROutput(ber=0.137, ber_lower=0.07132636098401203)),
        (BER_FNN, BEROutput(ber=0.118, ber_lower=0.061072112753426215)),
    ],
)
class TestBERJaticInterop:
    def test_evaluate(self, method, output):
        """
        Confirm that a dataset that follows JATIC works with BER
        """

        path = "tests/datasets/mnist.npz"
        with np.load(path, allow_pickle=True) as fp:
            images, labels = fp["x_train"][:1000], fp["y_train"][:1000]

        # Create jatic compliant dataset
        jatic_ds = JaticImageClassificationDataset(images, labels)
        check_jatic_classification(jatic_ds)

        # Wrap into Daml dataset
        ber_ds = JaticClassificationDatasetWrapper(jatic_ds)
        method = method()

        assert len(ber_ds) == len(images)

        value = method.evaluate(ber_ds)
        assert value == output
