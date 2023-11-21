import numpy as np
from jatic_toolbox.protocols import SupportsImageClassification, VisionDataset

from daml._internal.datasets import DamlDataset
from daml._internal.interop.wrappers.jatic import JaticClassificationDatasetWrapper
from tests.utils.JaticUtils import check_jatic_interop


class TestJaticWrapper:
    def test_wrapper_interop(self):
        """Wrap a JATIC dataset and confirm it works with DAML"""

        class FakeDataset:
            data: SupportsImageClassification = {
                "image": np.zeros((3, 224, 224)),
                "label": np.array([1]),
            }

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> SupportsImageClassification:
                return self.data

        dataset: VisionDataset = FakeDataset()
        check_jatic_interop(dataset)

        # Show when wrapped, it is Daml compliant
        wrapped_ds = JaticClassificationDatasetWrapper(dataset)
        assert isinstance(wrapped_ds, DamlDataset)
