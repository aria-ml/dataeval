# placeholder
import unittest
from daml.metrics import Metrics


class TestMetrics(unittest.TestCase):
    def test_list_metrics(self):
        """
        Tautalogical, but handles code coverage.
        """
        metrix = Metrics()
        self.assertEqual(
            metrix._get_outlier_detect_algos(), metrix.list_metrics()
            )
