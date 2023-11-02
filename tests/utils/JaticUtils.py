import jatic_toolbox.protocols as pr


def check_jatic_interop(data):
    """From JATIC toolbox notebook"""
    # Data has image?
    assert pr.is_typed_dict(data[0], pr.HasDataImage)
    # Data has label?
    assert pr.is_typed_dict(data[0], pr.HasDataLabel)
    # Data is ImageClassifierData?
    assert pr.is_typed_dict(data[0], pr.SupportsImageClassification)


class MockJaticImageClassificationDataset:
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> pr.SupportsImageClassification:
        return {"image": self._images[index], "label": self._labels[index]}
