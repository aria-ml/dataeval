import numpy as np
import pytest

from daml._internal.interop.wrappers.jatic import JaticClassificationDatasetWrapper
from daml.metrics.outlier_detection import OD_AE, OD_AEGMM, OD_LLR, OD_VAE, OD_VAEGMM
from tests.utils.JaticUtils import (
    JaticImageClassificationDataset,
    check_jatic_classification,
)


@pytest.mark.parametrize(
    "method",
    [
        OD_AE,
        pytest.param(OD_AEGMM, marks=pytest.mark.functional),
        pytest.param(OD_VAE, marks=pytest.mark.functional),
        pytest.param(OD_VAEGMM, marks=pytest.mark.functional),
        pytest.param(OD_LLR, marks=pytest.mark.functional),
    ],
)
@pytest.mark.interop
class TestOutlierDetectionJaticInterop:
    """Tests if JATIC datasets can be correctly handled by BER methods"""

    def test_fit_and_eval(self, method, mnist):
        """Asserts that wrapped images are compatible with entire workflow"""
        # Instantiate outlier detection method
        method = method()

        # Create jatic dataset
        images, labels = mnist(100)

        # Add the 3rd dimension to the images
        images = images[..., np.newaxis].astype(method._dataset_type)

        # Create jatic compliant dataset
        jatic_ds = JaticImageClassificationDataset(images, labels)
        check_jatic_classification(jatic_ds)

        # Wrap jatic dataset with daml dataset
        daml_ds = JaticClassificationDatasetWrapper(jatic_ds)

        # Test 1: Fit dataset
        method.fit_dataset(daml_ds, epochs=1, batch_size=32, verbose=0)

        # Test 2: Evaluate
        method.evaluate(daml_ds)
