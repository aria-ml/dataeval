import numpy as np

from dataeval.detectors import DriftCVM, DriftKS, Duplicates, Linter
from dataeval.metrics import BER, UAP, ChannelStats, Divergence, ImageStats, Parity


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
        linter = Linter(self.images)
        linter.evaluate()

        duplicates = Duplicates(self.images)
        duplicates.evaluate()

    def testFeasibility(self):
        ber = BER()
        ber.evaluate(self.images, self.labels)

        uap = UAP()
        uap.evaluate(self.labels, self.scores)

    def testDivergence(self):
        divergence = Divergence()
        divergence.evaluate(self.images, self.images)

    def testStats(self):
        imagestats = ImageStats()
        imagestats.update(self.images)
        imagestats.compute()

        channelstats = ChannelStats()
        channelstats.update(self.images)
        channelstats.compute()

    def testBias(self):
        parity = Parity()
        parity.evaluate(self.labels, self.labels)
