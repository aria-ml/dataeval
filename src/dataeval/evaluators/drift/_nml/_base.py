"""
Source code derived from NannyML 0.13.0
https://github.com/NannyML/nannyml/blob/main/nannyml/base.py

Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from logging import Logger

import pandas as pd
from typing_extensions import Self

from dataeval.evaluators.drift._mvdc import DriftMVDCOutput
from dataeval.evaluators.drift._nml._chunk import Chunker, CountBasedChunker


def _validate(data: pd.DataFrame, expected_features: int | None = None) -> int:
    if data.empty:
        raise ValueError("data contains no rows. Please provide a valid data set.")
    if expected_features is not None and data.shape[-1] != expected_features:
        raise ValueError(f"expected '{expected_features}' features in data set:\n\t{data}")
    return data.shape[-1]


class AbstractCalculator(ABC):
    """Base class for drift calculation."""

    def __init__(self, chunker: Chunker | None = None, logger: Logger | None = None) -> None:
        self.chunker = chunker if isinstance(chunker, Chunker) else CountBasedChunker(10)
        self.result: DriftMVDCOutput | None = None
        self.n_features: int | None = None
        self._logger = logger if isinstance(logger, Logger) else logging.getLogger(__name__)

    def fit(self, reference_data: pd.DataFrame) -> Self:
        """Trains the calculator using reference data."""
        self.n_features = _validate(reference_data)

        self._logger.debug(f"fitting {str(self)}")
        self.result = self._fit(reference_data)
        return self

    def calculate(self, data: pd.DataFrame) -> DriftMVDCOutput:
        """Performs a calculation on the provided data."""
        if self.result is None:
            raise RuntimeError("must run fit with reference data before running calculate")
        _validate(data, self.n_features)

        self._logger.debug(f"calculating {str(self)}")
        self.result = self._calculate(data)
        return self.result

    @abstractmethod
    def _fit(self, reference_data: pd.DataFrame) -> DriftMVDCOutput: ...

    @abstractmethod
    def _calculate(self, data: pd.DataFrame) -> DriftMVDCOutput: ...
