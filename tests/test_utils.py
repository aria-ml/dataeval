import pytest

from daml import list_metrics, load_metric
from daml._internal.alibidetect.outlierdetectors import AlibiAE
from daml._internal.MetricClasses import Metrics
from daml._internal.utils import _get_supported_method


class TestLoadListMetrics:
    def test_list_metrics(self):
        m = [Metrics.OutlierDetection, Metrics.Divergence]
        x = list_metrics()
        assert x == m

    def test_load_metric_returns_documentation(self):
        # TODO When nothing is given to load_metric, it should return
        # list_metrics
        assert True

    # Ensure that the program fails upon bad user input
    @pytest.mark.parametrize(
        "metric, provider, method",
        [
            (None, None, None),
            (None, Metrics.Provider.AlibiDetect, Metrics.Method.AutoEncoder),
            ("NotAMetric", Metrics.Provider.AlibiDetect, Metrics.Method.AutoEncoder),
            (Metrics.OutlierDetection, "NotAProvider", None),
            (Metrics.OutlierDetection, None, "NotAnEncoder"),
            (Metrics.OutlierDetection, Metrics.Provider.AlibiDetect, "NotAnencoder"),
        ],
    )
    def test_load_metric_fails(self, metric, provider, method):
        with pytest.raises(ValueError):
            load_metric(metric=metric, provider=provider, method=method)

    @pytest.mark.parametrize("provider", [Metrics.Provider.AlibiDetect, None])
    @pytest.mark.parametrize("method", [Metrics.Method.AutoEncoder, None])
    def test_load_metric_succeeds(self, provider, method):
        metric = load_metric(
            metric=Metrics.OutlierDetection, provider=provider, method=method
        )
        assert isinstance(metric, AlibiAE)

    def test_get_support_metrics(self):
        assert _get_supported_method("not valid") is None
