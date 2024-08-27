import numpy as np

from dataeval._internal.interop import to_numpy
from dataeval.detectors import DriftCVM, DriftKS, Duplicates, Linter
from dataeval.metrics import ChannelStats, ImageStats, ber, divergence, parity, uap


class TestMinimalDependencies:
    images = np.zeros((100, 3, 16, 16))
    labels = np.random.randint(10, size=100)
    scores = np.zeros((100, 10))

    def testDrift(self):
        drift_cvm = DriftCVM(self.images)
        drift_cvm.predict(self.images)

        drift_ks = DriftKS(self.images)
        drift_ks.predict(self.images)

    def testLinters(self):
        linter = Linter()
        linter.evaluate(self.images)

        duplicates = Duplicates()
        duplicates.evaluate(self.images)

    def testFeasibility(self):
        ber(self.images, self.labels)
        uap(self.labels, self.scores)

    def testDivergence(self):
        divergence(self.images, self.images, "MST")

    def testStats(self):
        imagestats = ImageStats()
        imagestats.update(self.images)
        imagestats.compute()

        channelstats = ChannelStats()
        channelstats.update(self.images)
        channelstats.compute()

    def testBias(self):
        parity(self.labels, self.labels)

    def testToNumpy(self):
        actual = to_numpy([[1, 2], [3, 4]])  # type: ignore
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_equal(actual, expected)
