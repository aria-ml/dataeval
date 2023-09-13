"""
This module contains utility functions to help
user workflows be simpler and more efficient
"""

from typing import Any, List, Optional

from daml._internal.alibidetect.outlierdetectors import (
    AlibiAE,
    AlibiAEGMM,
    AlibiLLR,
    AlibiVAE,
    AlibiVAEGMM,
)
from daml._internal.ber import MultiClassBER
from daml._internal.divergence import DpDivergence
from daml._internal.MetricClasses import Metrics

_methods = {
    Metrics.Method.AutoEncoder: AlibiAE,
    Metrics.Method.AutoEncoderGMM: AlibiAEGMM,
    Metrics.Method.LLR: AlibiLLR,
    Metrics.Method.VariationalAutoEncoder: AlibiVAE,
    Metrics.Method.VariationalAutoEncoderGMM: AlibiVAEGMM,
    Metrics.Method.DpDivergence: DpDivergence,
    Metrics.Method.MultiClassBER: MultiClassBER,
}


def list_metrics() -> List[str]:
    """Return a list of metrics used in DAML

    Returns
    -------
    List
        Names of all the metrics provided by DAML
    """

    return list(Metrics.metrics_providers_methods.keys())


def load_metric(
    metric: str,
    provider: Optional[str] = None,
    method: Optional[str] = None,
) -> Any:
    """
    Return a data metric algorithm

    Parameters
    ----------
    metric : str
        Group of algorithms based on what is being calculated
    provider : str, optional
        The parent library where the implementation is defined
    method : str, optional
        Name of the specific algorithm for a certain metric type

    Returns
    -------
    Metric
        A metric that performs data analysis with a specific method

    Raises
    ------
    ValueError
        If the metric, provider, or method are invalid. See docs for supported inputs
    """

    mpm = Metrics.metrics_providers_methods

    if metric is None:
        metric_list = list(mpm.keys())
        raise ValueError(
            f"""
            No metric given. Choose one of the following: {metric_list}
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

    if method in _methods:
        return _methods[method]()
    else:
        raise ValueError(
            f"""
            Method, {method}, is invalid for provider, {provider}
            """
        )
