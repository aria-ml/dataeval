from jatic_toolbox import load_dataset

from daml._internal.datasets import DamlDataset
from daml._internal.interop.wrappers.jatic import JaticClassificationDatasetWrapper
from tests.utils.JaticUtils import check_jatic_interop


class TestJaticWrapper:
    def test_wrapper_interop(self):
        """Wrap a JATIC dataset and confirm it works with DAML"""
        jatic_ds = load_dataset(
            provider="torchvision",
            dataset_name="CIFAR10",
            task="image-classification",
            split="test",
            root="~/data",
            download=True,
        )
        check_jatic_interop(jatic_ds)

        # Show when wrapped, it is Daml compliant
        wrapped_ds = JaticClassificationDatasetWrapper(jatic_ds)
        assert isinstance(wrapped_ds, DamlDataset)
