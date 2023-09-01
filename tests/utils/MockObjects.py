from .MockGenerators import (
    MockImageClassificationDataset,
    MockImageClassificationGenerator,
)


class MockCifar10:
    """
    A class containing pre-made mock train and test datasets in the shape of CIFAR10.

    :param train_dataset: A mock dataset containing 50,000 images and labels (0-9)
    :type train_dataset: :class:`MockImageClassificationDataset`
    :param test_dataset: A mock dataset containing 10,000 images and labels (0-9)
    :type test_dataset: :class:`MockImageClassificationDataset`
    :param MAKE_LABELS: A flag to determine whether labels should be 0-9 or all 1,
        defaults to False
    :type MAKE_LABELS: bool, optional
    """

    _train_size = 50_000
    _test_size = 10_000

    def __init__(self, MAKE_LABELS: bool = False) -> None:
        """Creates mock training and testing datasets based on CIFAR10

        :param MAKE_LABELS: A flag to determine whether labels should be 0-9 or all 1,
            defaults to False
        :type MAKE_LABELS: bool
        """

        labels = range(0, 10) if MAKE_LABELS else 1
        # Create train and test datasets
        train_generator = MockImageClassificationGenerator(
            limit=50_000, labels=labels, img_dims=32, channels=3
        )
        test_generator = MockImageClassificationGenerator(
            limit=10_000, labels=labels, img_dims=32, channels=3
        )

        self._train_dataset = train_generator.dataset
        self._test_dataset = test_generator.dataset

    @property
    def train_dataset(self) -> MockImageClassificationDataset:
        """Get the training dataset of the mock CIFAR10 dataset

        :return: A mock dataset with 50,000 images and labels
        :rtype: :class:`MockImageClassificationDataset`
        """
        return self._train_dataset

    @property
    def test_dataset(self) -> MockImageClassificationDataset:
        """Get the testing dataset of the mock CIFAR10 dataset

        :return: A mock dataset with 10,000 images and labels
        :rtype: :class:`MockImageClassificationDataset`
        """
        return self._test_dataset
