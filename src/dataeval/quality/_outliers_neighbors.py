"""Judge metadata factors against their neighbors along a sequence, with :func:`dataeval.core.sequence_residuals`.

A frame-level factor is judged along its video and a detection-level factor along its track. The scores are stored as
score checks, so the evaluator's threshold is fitted to them like any other; a monotonic factor's order breaks are
stored as fixed findings, since a rule needs no threshold.
"""

__all__ = []

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval.core._sequence_residuals import SequenceResidualsResult, sequence_residuals
from dataeval.quality._factor_stats import factor_kind, factor_levels
from dataeval.quality._outliers_rows import FINDINGS_SCHEMA, ORDER_COLUMN, SEQUENCE_KEYS
from dataeval.quality._outliers_scores import LevelStrata, _ScoredCheck, _ScoreInput, stratum_codes
from dataeval.quality._shared import reported_level
from dataeval.types import FactorLevel

# What one sequence is called in a check's name, at each level a factor can be judged along one.
_ALONG: Mapping[FactorLevel, str] = {"unit": "sequence", "instance": "track"}
# The stream's own clocks, read as monotonic whatever their dtype.
_CLOCK_NAMES = frozenset({"time_s", "pts"})


def neighbor_name(name: str, level: FactorLevel) -> str:
    """Return the check name of a factor judged along its sequence, such as ``"brightness along sequence"``."""
    return f"{name} along {_ALONG[level]}"


def _is_monotonic(name: str, dtype: pl.DataType) -> bool:
    """Whether a factor never decreases along a sequence: a date, datetime or duration, or the stream's clocks."""
    return name in _CLOCK_NAMES or isinstance(dtype, pl.Datetime | pl.Duration) or dtype == pl.Date


def neighbor_checks(
    data: Any,
    neighbors: Sequence[str] | Literal["all"] | None,
    factors: Sequence[str] | None,
    judged: Sequence[FactorLevel],
) -> dict[FactorLevel, tuple[str, ...]]:
    """Place each factor `neighbors` names at its level; for ``"all"``, every ordered factor at the judged levels.

    Raises
    ------
    TypeError
        When `neighbors` is a string other than ``"all"``.
    ValueError
        When the metadata has no sequence level, a named factor cannot be judged along a sequence or lives at a level
        the call does not judge, or ``"all"`` finds nothing.
    KeyError
        When a named factor is not one of the metadata's.
    """
    if neighbors is None:
        return {}
    if isinstance(neighbors, str) and neighbors != "all":
        raise TypeError(f"Outliers.evaluate: neighbors= takes a list of factor names or 'all', not {neighbors!r}.")
    if "sequence" not in data.levels:
        raise ValueError(
            "Outliers.evaluate: neighbors= judges a factor along its video or track, and this metadata has no "
            "sequence level. Build the Metadata from a tracking dataset."
        )
    levels = factor_levels(data)
    if neighbors == "all":
        return _all_neighbors(data, levels, factors, judged)
    return _named_neighbors(data, levels, list(neighbors), judged)


def _all_neighbors(
    data: Any, levels: Mapping[str, FactorLevel], factors: Sequence[str] | None, judged: Sequence[FactorLevel]
) -> dict[FactorLevel, tuple[str, ...]]:
    """Return every ordered factor at each judged level that has sequences, within `factors`, refusing none at all."""
    found: dict[FactorLevel, tuple[str, ...]] = {}
    sequenced: list[FactorLevel] = [level for level in judged if level in _ALONG]
    for level in sequenced:
        names = tuple(
            name
            for name, own in levels.items()
            if own == level and (factors is None or name in factors) and factor_kind(data, name) == "ordered"
        )
        if names:
            found[level] = names
    if not found:
        raise ValueError(
            f"Outliers.evaluate: neighbors='all' finds no ordered factor at the frame or detection levels this call "
            f"judges ({list(judged)}). A factor is judged along its sequence at 'unit' or 'instance'; pass levels= "
            "that include one."
        )
    return found


def _named_neighbors(
    data: Any, levels: Mapping[str, FactorLevel], names: Sequence[str], judged: Sequence[FactorLevel]
) -> dict[FactorLevel, tuple[str, ...]]:
    """Group the named factors by the level each is judged along its sequence at."""
    found: dict[FactorLevel, list[str]] = {}
    for name in names:
        found.setdefault(_sequenced_level(data, levels, name, judged), []).append(name)
    return {level: tuple(dict.fromkeys(named)) for level, named in found.items()}


def _sequenced_level(
    data: Any, levels: Mapping[str, FactorLevel], name: str, judged: Sequence[FactorLevel]
) -> FactorLevel:
    """Return the level a named factor is judged along its sequence at, refusing one that cannot be."""
    if name not in levels:
        raise KeyError(
            f"Outliers.evaluate: neighbors= names {name!r}, which is not a factor of this metadata. "
            f"Its factors are {sorted(levels)}."
        )
    level = levels[name]
    if factor_kind(data, name) != "ordered":
        raise ValueError(
            f"Outliers.evaluate: neighbors= names {name!r}, which holds no numbers or times to compare with its "
            "neighbors. Judge a categorical factor within its video with strata=['item_index']."
        )
    if level not in _ALONG:
        raise ValueError(
            f"Outliers.evaluate: neighbors= names {name!r}, which lives at {level!r}; only frame ('unit') and "
            "detection ('instance') factors have neighbors along a sequence."
        )
    if level not in judged:
        raise ValueError(
            f"Outliers.evaluate: neighbors= names {name!r} at {level!r}, which this call does not judge; pass "
            f"levels= that include {level!r}."
        )
    return level


def score_neighbors(
    score_input: _ScoreInput, checks: Mapping[FactorLevel, Sequence[str]]
) -> tuple[list[_ScoredCheck], pl.DataFrame]:
    """Score each level's neighbor checks once, returning the stored checks and the order breaks as findings."""
    scored: list[_ScoredCheck] = []
    breaks: list[pl.DataFrame] = [pl.DataFrame(schema=FINDINGS_SCHEMA)]
    for level, names in checks.items():
        frame = _tracked(score_input.frames[level])
        result = sequence_residuals(
            frame,
            list(names),
            sequence=list(SEQUENCE_KEYS[level]),
            order=ORDER_COLUMN,
            monotonic=[name for name in names if _is_monotonic(name, frame.schema[name])],
        )
        scored.extend(_checks(result, level, score_input.level_kinds[level]))
        breaks.append(_order_findings(result, level, score_input))
    return scored, pl.concat(breaks)


def _tracked(frame: pl.DataFrame) -> pl.DataFrame:
    """Read an untracked detection's ``track_id`` of -1 as missing, so it forms no track with the others."""
    if "track_id" not in frame.columns:
        return frame
    return frame.with_columns(pl.when(pl.col("track_id") >= 0).then(pl.col("track_id")).alias("track_id"))


def _height(result: SequenceResidualsResult) -> int:
    """Return the rows of the scored frame: core scores each column once per row."""
    return result["row"].size // len(result["columns"])


def _checks(result: SequenceResidualsResult, level: FactorLevel, kind: str | None) -> list[_ScoredCheck]:
    """Cut core's result into one stored check per column, sharing the level's rows and whole-level codes."""
    height = _height(result)
    rows = np.arange(height, dtype=np.int64)
    stratum = np.full(height, -1, dtype=np.intp)
    checks: list[_ScoredCheck] = []
    for index, name in enumerate(result["columns"]):
        part = slice(index * height, (index + 1) * height)
        checks.append(
            _ScoredCheck(
                level=level,
                kind=kind,
                metric_name=neighbor_name(name, level),
                method="neighbors",
                dims=1,
                rows=rows,
                score=result["score"][part].copy(),
                reason=result["reason"][part].copy(),
                reasons=result["reasons"],
                stratum=stratum,
                labels={},
                value=result["value"][part].copy(),
                expected=result["expected"][part].copy(),
                spread=result["spread"][part].copy(),
            )
        )
    return checks


def _order_findings(result: SequenceResidualsResult, level: FactorLevel, score_input: _ScoreInput) -> pl.DataFrame:
    """Lay out one finding per row out of order, against the value it is out of order against."""
    entries = np.flatnonzero(result["out_of_order"])
    height = _height(result)
    addresses = [score_input.address(level, int(row)) for row in (entries % height).tolist()]
    value, bound = result["value"][entries], result["order_bound"][entries]
    count = entries.size
    return pl.DataFrame(
        {
            "item_index": [si.item for si in addresses],
            "target_index": [si.key for si in addresses],
            "level": [reported_level(si) for si in addresses],
            "metric_name": [neighbor_name(result["columns"][column], level) for column in (entries // height).tolist()],
            "metric_value": value.tolist(),
            "direction": np.where(value > bound, "upper", "lower").tolist(),
            "bound": bound.tolist(),
            "percentile": [None] * count,
            "population_mean": [None] * count,
            "population_std": [None] * count,
            "method": ["order"] * count,
            "population": [None] * count,
            "observed": [None] * count,
        },
        schema=FINDINGS_SCHEMA,
    )


def restratified(checks: Sequence[_ScoredCheck], strata: LevelStrata) -> list[_ScoredCheck]:
    """Re-code the stored neighbor checks into the populations `strata` give each level, without rescoring them.

    A neighbor's residual is taken within its own sequence, so a new stratification changes only the pool.
    """
    coded: dict[FactorLevel, tuple[NDArray[np.intp], Mapping[int, str]]] = {}
    found: list[_ScoredCheck] = []
    for check in (check for check in checks if check.method == "neighbors"):
        if check.level not in coded:
            coded[check.level] = stratum_codes(strata.get(check.level), check.rows.size)
        stratum, labels = coded[check.level]
        found.append(check._replace(stratum=stratum, labels=labels))
    return found
