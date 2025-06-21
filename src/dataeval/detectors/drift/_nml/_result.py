"""
Contains the results of the data reconstruction drift calculation and provides plotting functionality.

Source code derived from NannyML 0.13.0
https://github.com/NannyML/nannyml/blob/main/nannyml/base.py

Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import NamedTuple

import pandas as pd
from typing_extensions import Self

from dataeval.outputs._base import GenericOutput


class Metric(NamedTuple):
    display_name: str
    column_name: str


class AbstractResult(GenericOutput[pd.DataFrame]):
    def __init__(self, results_data: pd.DataFrame) -> None:
        self._data = results_data.copy(deep=True)

    def data(self) -> pd.DataFrame:
        return self.to_dataframe()

    @property
    def empty(self) -> bool:
        return self._data is None or self._data.empty

    def __len__(self) -> int:
        return 0 if self.empty else len(self._data)

    def to_dataframe(self, multilevel: bool = True) -> pd.DataFrame:
        """Export results to pandas dataframe."""
        if multilevel:
            return self._data
        column_names = [
            "_".join(col).replace("chunk_chunk_chunk", "chunk").replace("chunk_chunk", "chunk")
            for col in self._data.columns.values
        ]
        single_level_data = self._data.copy(deep=True)
        single_level_data.columns = column_names
        return single_level_data

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
    def __init__(self, results_data: pd.DataFrame) -> None:
        super().__init__(results_data)

    def _filter(self, period: str, metrics: Sequence[str] | None = None) -> Self:
        data = self._data
        if period != "all":
            data = self._data.loc[self._data.loc[:, ("chunk", "period")] == period, :]  # type: ignore | dataframe loc
            data = data.reset_index(drop=True)

        res = copy.deepcopy(self)
        res._data = data
        return res


class PerMetricResult(Abstract1DResult):
    def __init__(self, results_data: pd.DataFrame, metrics: Sequence[Metric] = []) -> None:
        super().__init__(results_data)
        self.metrics = metrics

    def _filter(self, period: str, metrics: Sequence[str] | None = None) -> Self:
        if metrics is None:
            metrics = [metric.column_name for metric in self.metrics]

        res = super()._filter(period)

        data = pd.concat([res._data.loc[:, (["chunk"])], res._data.loc[:, (metrics,)]], axis=1)  # type: ignore | dataframe loc
        data = data.reset_index(drop=True)

        res._data = data
        res.metrics = [metric for metric in self.metrics if metric.column_name in metrics]

        return res
