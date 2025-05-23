from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval.data._metadata import FactorInfo, Metadata


def mock_metadata(
    discrete_names: list[str] | None = None,
    discrete_data: NDArray[Any] | None = None,
    continuous_names: list[str] | None = None,
    continuous_data: NDArray[Any] | None = None,
) -> Metadata:
    """
    Creates a magic mock method that contains discrete and continuous data and factors
    but has no hard dependency on Metadata.
    """

    m = MagicMock(spec=Metadata)

    _factors = {}
    _factor_data = []

    if discrete_names and discrete_data is not None and discrete_data.size > 0:
        _factor_data.append(discrete_data)
        _factors |= dict.fromkeys(discrete_names, FactorInfo("discrete"))

    if continuous_names and continuous_data is not None and continuous_data.size > 0:
        _factor_data.append(continuous_data)
        _factors |= dict.fromkeys(continuous_names, FactorInfo("continuous"))

    m.factor_names = list(_factors)
    m.factor_data = np.hstack(_factor_data) if _factor_data else np.array([])
    m.factor_info = _factors
    m.dataframe = pl.DataFrame(m.factor_data, schema=m.factor_names)

    return m
