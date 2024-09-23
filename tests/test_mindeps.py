import numpy as np

from dataeval._internal.interop import to_numpy
from dataeval.detectors.drift import DriftCVM, DriftKS
from dataeval.detectors.linters import Duplicates, Outliers
from dataeval.metrics.bias import label_parity
from dataeval.metrics.estimators import ber, divergence, uap
from dataeval.metrics.stats import dimensionstats, hashstats, pixelstats, visualstats


class TestMinimalDependencies:
    images = np.zeros((100, 3, 16, 16))
    labels = np.random.randint(10, size=100)
    scores = np.zeros((100, 10))

    def testDrift(self):
        drift_cvm = DriftCVM(self.images)
        drift_cvm.predict(self.images)

        drift_ks = DriftKS(self.images)
        drift_ks.predict(self.images)

    def testOutliers(self):
        outliers = Outliers()
        outliers.evaluate(self.images)

        duplicates = Duplicates()
        duplicates.evaluate(self.images)

    def testFeasibility(self):
        ber(self.images, self.labels)
        uap(self.labels, self.scores)

    def testDivergence(self):
        divergence(self.images, self.images, "MST")

    def testStats(self):
        dimensionstats(self.images)
        hashstats(self.images)
        pixelstats(self.images)
        visualstats(self.images)

    def testBias(self):
        label_parity(self.labels, self.labels)

    def testToNumpy(self):
        actual = to_numpy([[1, 2], [3, 4]])  # type: ignore
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_equal(actual, expected)
