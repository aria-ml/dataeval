import numpy as np

from daml.detectors import DriftCVM, DriftKS, Duplicates, Linter
from daml.metrics import BER, UAP, ChannelStats, Divergence, ImageStats, Parity


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
        ber = BER(self.images, self.labels)
        ber.evaluate()

        uap = UAP(self.labels, self.scores)
        uap.evaluate()

    def testDivergence(self):
        divergence = Divergence(self.images, self.images)
        divergence.evaluate()

    def testStats(self):
        imagestats = ImageStats()
        imagestats.update(self.images)
        imagestats.compute()

        channelstats = ChannelStats()
        channelstats.update(self.images)
        channelstats.compute()

    def testBias(self):
        parity = Parity(self.labels, self.labels)
        parity.evaluate()
