import numpy as np
import pytest

from daml._internal.interop.wrappers.jatic import JaticClassificationDatasetWrapper
from daml.metrics.outlier_detection import AE, AEGMM, LLR, VAE, VAEGMM
from tests.utils.JaticUtils import (
    MockJaticImageClassificationDataset,
    check_jatic_interop,
)


@pytest.mark.parametrize(
    "method", [AE, AEGMM, VAE, VAEGMM, pytest.param(LLR, marks=pytest.mark.functional)]
)
@pytest.mark.interop
class TestOutlierDetectionJaticInterop:
    """Tests if JATIC datasets can be correctly handled by BER methods"""

    def test_fit_and_eval(self, method):
        """Asserts that wrapped images are compatible with entire workflow"""
        # Create jatic dataset
        path = "tests/datasets/mnist.npz"
        with np.load(path, allow_pickle=True) as fp:
            images, labels = fp["x_train"][:1000], fp["y_train"][:1000]

        # Instantiate outlier detection method
        method = method()

        # Add the 3rd dimension to the images
        images = images[..., np.newaxis].astype(method._dataset_type)

        # Create jatic compliant dataset
        jatic_ds = MockJaticImageClassificationDataset(images, labels)
        check_jatic_interop(jatic_ds)
        # Wrap jatic dataset with daml dataset
        daml_ds = JaticClassificationDatasetWrapper(jatic_ds)

        # Initialize a detector
        method.initialize_detector(daml_ds.images[0].shape)

        # Test 1: Fit dataset
        method.fit_dataset(daml_ds, epochs=1, batch_size=32, verbose=0)

        # Test 2: Evaluate
        method.evaluate(daml_ds)
