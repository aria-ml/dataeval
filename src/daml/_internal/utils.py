from typing import List, Optional

from daml._internal.alibidetect.outlierdetectors import AlibiAE


class Metrics:
    """A global dictionary to parse metrics, providers, and methods"""

    metrics_providers_methods = {"OutlierDetection": {"Alibi-Detect": ["Autoencoder"]}}


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
) -> AlibiAE:
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
        raise ValueError(f"No provider given. Choose one of the following: {mpm_list}")

    # Gets providers for a specific metric
    providers = mpm.get(metric, {})
    if len(providers) == 0:
        raise ValueError(f"Metric, {metric}, is invalid")

    # Set a default provider
    if provider is None:
        provider = "Alibi-Detect"

    # Gets the methods a provider has for a specific meric
    method_names = providers.get(provider, [])
    if len(method_names) == 0:
        raise ValueError(f"Provider, {provider}, is invalid for metric, {metric}")

    # Set a default method, must be valid for the provider
    if method is None:
        method = method_names[0]

    # Check if the provider supports the method
    if method not in method_names:
        raise ValueError(f"Method, {method}, is invalid for provider, {provider}")

    # TODO Add logic when more methods are developed
    return AlibiAE()
