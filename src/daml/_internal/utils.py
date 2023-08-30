from typing import Any, List, Optional

from daml._internal.alibidetect.outlierdetectors import AlibiAE
from daml._internal.divergence import DpDivergence
from daml._internal.MetricClasses import Metrics


def _get_supported_method(method):
    # TODO: develop a cleaner method for selecting the method class.
    if method == Metrics.Method.AutoEncoder:
        return AlibiAE(method)
    elif method == Metrics.Method.VariationalAutoEncoder:
        return AlibiAE(method)
    elif method == Metrics.Method.DpDivergence:
        return DpDivergence(method)
    return None


def list_metrics() -> List[str]:
    """Returns a list of metrics used in DAML

    :return: A list of metrics
    :rtype: List
    """
    return list(Metrics.metrics_providers_methods.keys())


def load_metric(
    metric: Optional[str] = None,
    provider: Optional[str] = None,
    method: Optional[str] = None,
) -> Any:
    """
    Function that returns a data metric algorithm

    :param metric: Group of algorithms based on what is being calculated
    :type metric: Optional[str]
    :param provider: Where to search for the dataset metrics
    :type provider: Optional[str]
    :param method: Name of a specific algorithm for a certain metric type
    :type method: Optional[str]

    :return: A metric method class
    :rtype: DataMetric
    """

    mpm = Metrics.metrics_providers_methods

    if metric is None:
        mpm_list = list(mpm.keys())
        raise ValueError(
            f"""
            No provider given. Choose one of the following: {mpm_list}
            """
        )

    # Gets providers for a specific metric
    providers = mpm.get(metric, {})
    if len(providers) == 0:
        raise ValueError(f"Metric, {metric}, is invalid")

    # Set a default provider
    if provider is None:
        provider = Metrics.Provider.AlibiDetect

    # Gets the methods a provider has for a specific metric
    supported_methods = providers.get(provider, [])

    if len(supported_methods) == 0:
        raise ValueError(
            f"""
            Provider, {provider}, is invalid for metric, {metric}
            """
        )

    # Set a default method, must be valid for the provider
    if method is None:
        method = supported_methods[0]

    # Check if the provider supports the method
    if method not in supported_methods:
        raise ValueError(
            f"""
            Method, {method}, is invalid for provider, {provider}
            """
        )

    return _get_supported_method(method)
