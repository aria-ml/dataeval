import pytest

from daml import load_metric, list_metrics
from daml._internal.alibidetect.outlierdetectors import AlibiAE


class TestLoadListMetrics():

    def test_list_metrics(self):
        m = ["OutlierDetection"]
        x = list_metrics()
        assert (x == m)

    def test_load_metric_returns_documentation(self):
        # TODO When nothing is given to load_metric, it should return list_metrics
        assert (True)

    # Ensure that the program fails upon bad user input
    @pytest.mark.parametrize("metric, provider, method", [
        (None, None, None),
        (None, "Alibi-Detect", "Autoencoder"),
        ("NotOutlierDetection", "Alibi-Detect", "Autoencoder"),
        ("OutlierDetection", "NotAlibi-Detect", None),
        ("OutlierDetection", None, "NotAutoencoder"),
        ("OutlierDetection", "Alibi-Detect", "NotAutoencoder")])
    def test_load_metric_fails(self, metric, provider, method):
        with pytest.raises(ValueError):
            load_metric(metric=metric, provider=provider, method=method)

    @pytest.mark.parametrize("provider", ["Alibi-Detect", None])
    @pytest.mark.parametrize("method", ["Autoencoder", None])
    def test_load_metric_succeeds(self, provider, method):
        metric = load_metric(metric="OutlierDetection", provider=provider, method=method)
        assert (isinstance(metric, AlibiAE))
