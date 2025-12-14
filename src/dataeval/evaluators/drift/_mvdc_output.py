"""
Contains the results of the data reconstruction drift calculation and provides plotting functionality.

Source code derived from NannyML 0.13.0
https://github.com/NannyML/nannyml/blob/main/nannyml/base.py

Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = []

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal, NamedTuple

import polars as pl
from typing_extensions import Self

from dataeval.types import Output


class Metric(NamedTuple):
    display_name: str
    column_name: str


class AbstractResult(Output[pl.DataFrame]):
    def __init__(self, results_data: pl.DataFrame) -> None:
        self._data = results_data.clone()

    def data(self) -> pl.DataFrame:
        return self._data.clone()

    @property
    def empty(self) -> bool:
        return self._data is None or self._data.is_empty()

    def __len__(self) -> int:
        return 0 if self.empty else len(self._data)

    def filter(self, period: str = "all", metrics: str | Sequence[str] | None = None) -> Self:
        """Returns filtered result metric data."""
        if metrics and not isinstance(metrics, str | Sequence):
            raise ValueError("metrics value provided is not a valid metric or sequence of metrics")
        if isinstance(metrics, str):
            metrics = [metrics]
        return self._filter(period, metrics)

    @abstractmethod
    def _filter(self, period: str, metrics: Sequence[str] | None = None) -> Self: ...


class Abstract1DResult(AbstractResult, ABC):
    def __init__(self, results_data: pl.DataFrame) -> None:
        super().__init__(results_data)

    def _filter(self, period: str, metrics: Sequence[str] | None = None) -> Self:
        data = self._data
        if period != "all":
            data = self._data.filter(pl.col("chunk_period") == period)

        res = copy.deepcopy(self)
        res._data = data
        return res


class PerMetricResult(Abstract1DResult):
    def __init__(self, results_data: pl.DataFrame, metrics: Sequence[Metric] = []) -> None:
        super().__init__(results_data)
        self.metrics = metrics

    def _filter(self, period: str, metrics: Sequence[str] | None = None) -> Self:
        if metrics is None:
            metrics = [metric.column_name for metric in self.metrics]

        # Validate that all requested metrics exist
        available_metrics = {metric.column_name for metric in self.metrics}
        for m in metrics:
            if m not in available_metrics:
                raise KeyError(f"Metric '{m}' not found in available metrics: {available_metrics}")

        res = super()._filter(period)

        # Select chunk columns and requested metric columns
        chunk_cols = [col for col in res._data.columns if col.startswith("chunk_")]
        metric_cols = [col for col in res._data.columns if any(col.startswith(f"{m}_") for m in metrics)]
        data = res._data.select(chunk_cols + metric_cols)

        res._data = data
        res.metrics = [metric for metric in self.metrics if metric.column_name in metrics]

        return res


class DriftMVDCOutput(PerMetricResult):
    """Class wrapping the results of the classifier for drift detection and providing plotting functionality."""

    def __init__(self, results_data: pl.DataFrame) -> None:
        """Initialize a DomainClassifierCalculator results object.

        Parameters
        ----------
        results_data : pl.DataFrame
            Results data returned by a DomainClassifierCalculator.
        """
        metric = Metric(display_name="Domain Classifier", column_name="domain_classifier_auroc")
        super().__init__(results_data, [metric])

    @property
    def plot_type(self) -> Literal["drift_mvdc"]:
        return "drift_mvdc"
