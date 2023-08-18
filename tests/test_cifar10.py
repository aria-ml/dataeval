import unittest

from .utils.MockObjects import MockCifar10


class TestCifar10(unittest.TestCase):
    mc1 = MockCifar10()

    def test_cifar_train_size(self):
        self.assertEqual(
            len(self.mc1.train_dataset),
            50_000)

    def test_cifar_test_size(self):
        self.assertEqual(
            len(self.mc1.test_dataset),
            10_000)
