from torchmetrics.classification.average_precision import MulticlassAveragePrecision


class UAPMetric(MulticlassAveragePrecision):
    def __init__(self, num_classes: int):
        super().__init__(num_classes=num_classes, average="weighted")
