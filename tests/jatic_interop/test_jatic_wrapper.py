from jatic_toolbox.protocols import ObjectDetectionDataset, VisionDataset

from daml._internal.datasets import DamlDataset
from daml._internal.interop.wrappers.jatic import (
    JaticClassificationDatasetWrapper,
    JaticObjectDetectionWrapper,
)
from tests.utils.JaticUtils import (
    MockJaticImageClassificationDataset,
    MockJaticObjectDetectionDataset,
    check_jatic_classification,
    check_jatic_object_detection,
)


class TestJaticWrapper:
    def test_classification_wrapper(self):
        """Wrap a JATIC classification dataset and confirm it works with DAML"""
        dataset: VisionDataset = MockJaticImageClassificationDataset()
        check_jatic_classification(dataset)

        # # Show when wrapped, it is Daml compliant
        wrapped_ds = JaticClassificationDatasetWrapper(dataset)
        assert isinstance(wrapped_ds, DamlDataset)

    def test_object_detection_wrapper(self):
        """Wrap a JATIC object detection dataset and confirm it works with DAML"""
        dataset: ObjectDetectionDataset = MockJaticObjectDetectionDataset()
        check_jatic_object_detection(dataset)

        # Show when wrapped, it is Daml compliant
        wrapped_ds = JaticObjectDetectionWrapper(dataset)
        assert isinstance(wrapped_ds, DamlDataset)
