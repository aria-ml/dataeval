"""Validated settings for the new image statistics, on a base that refuses a misread number."""

__all__ = []

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, field_validator

from dataeval.types._config import UnitInterval


class StrictConfig(BaseModel):
    """
    Frozen, strict settings that refuse unknown fields, NaN and infinity, and a bool where a number is meant.

    A value must already be the type it is declared as: ``True`` is not a number, ``'1.5'`` is not a float, and ``3.0``
    is not a count, though an int is accepted for a float. A statistic's own function takes plain keywords and checks
    them through a section of these, so the checks are not part of any public signature. ``model_copy(update=...)`` and
    ``model_construct`` store a value without checking it, so a caller that reads a section from outside the library
    builds it through its constructor.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        # A nested section built by `model_construct` or `model_copy` skipped its checks, so check it again.
        revalidate_instances="always",
    )

    @field_validator("*", mode="before")
    @classmethod
    def _python_scalar(cls, value: Any) -> Any:
        # Strict mode refuses an `np.int64` for an int field and takes an `np.bool_` for a float field as 1.0. Turning
        # these two into the Python value first lets strict mode judge it. ponytail: containers are not unpacked, so a
        # tuple of `np.int64` is still refused; extend this when the first tuple field needs it.
        return value.item() if isinstance(value, (np.integer, np.bool_)) else value


class SurroundConfig(StrictConfig):
    """
    Settings for the surround of a box.

    Attributes
    ----------
    padding : float, default 0.25
        Greater than 0.
    min_pixels : int, default 32
        Greater than 0.
    min_target_valid_fraction : float, default 0.90
        In [0, 1].
    intensity_floor : float, default 1/255
        In [0, 1].
    """

    padding: PositiveFloat = 0.25
    min_pixels: PositiveInt = 32
    min_target_valid_fraction: UnitInterval = 0.90
    intensity_floor: UnitInterval = 1 / 255


class ImageStatsConfig(StrictConfig):
    """
    Settings for the statistics of :func:`~dataeval.core.image_stats` that take them.

    A section holds the settings of one group of statistics. This is a different thing from
    :class:`~dataeval.flags.ImageStats`, which selects which statistics to compute.

    Attributes
    ----------
    surround : SurroundConfig, default SurroundConfig()
        Settings for the surround of a box.
    """

    surround: SurroundConfig = SurroundConfig()
