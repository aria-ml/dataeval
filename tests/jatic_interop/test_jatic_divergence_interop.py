import numpy as np
import pytest

from daml._internal.interop.wrappers.jatic import JaticClassificationDatasetWrapper
from daml.metrics.divergence import HP_FNN, HP_MST, DivergenceOutput
from tests.utils.JaticUtils import (
    JaticImageClassificationDataset,
    check_jatic_classification,
)


@pytest.mark.interop
@pytest.mark.parametrize(
    "method, output",
    [
        (
            HP_MST,
            DivergenceOutput(
                dpdivergence=0.96875,
                error=1,
            ),
        ),
        (
            HP_FNN,
            DivergenceOutput(
                dpdivergence=1.0,
                error=0.0,
            ),
        ),
    ],
)
class TestDivergenceJaticInterop:
    def test_divergence_evaluate(self, method, output):
        """Jatic steps
        1. Load 2 jatic datasets
        2. Evaluate on both
        3. No errors
        """
        all_ones = np.ones(shape=(32, 32))
        all_fives = all_ones * 5

        dataset_a = JaticImageClassificationDataset(all_ones, all_ones)
        dataset_b = JaticImageClassificationDataset(all_fives, all_fives)

        check_jatic_classification(dataset_a)
        check_jatic_classification(dataset_b)

        # Initialize the autoencoder-based outlier detector from alibi-detect
        metric = method()
        dataset_a = JaticClassificationDatasetWrapper(dataset_a)
        dataset_b = JaticClassificationDatasetWrapper(dataset_b)
        result = metric.evaluate(dataset_a=dataset_a, dataset_b=dataset_b)
        assert result == output
