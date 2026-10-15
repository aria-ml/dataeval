"""Judge metadata factors by scores from :func:`dataeval.core.combination_scores`.

Categorical factors and groups are judged by rarity within a stratum; numeric groups and residuals are judged by the
evaluator's threshold fitted to their scores in each population.
"""

__all__ = []

from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats

from dataeval._metadata._structurers._reserved import LEVEL_KEY_COLUMNS
from dataeval.core._combination_scores import ColumnKind, combination_scores
from dataeval.protocols import Threshold
from dataeval.quality._factor_stats import factor_kind
from dataeval.quality._outliers_rows import (
    _MIN_POPULATION,
    _MIN_ROWS_PER_DIMENSION,
    DEFAULT_OUTLIERS_OUTLIER_THRESHOLD,
    FINDINGS_SCHEMA,
    _Fit,
    _fit_outliers,
    _floor_reason,
    _key_columns,
    _Population,
    level_kind,
    level_strata,
)
from dataeval.quality._shared import LABEL_KIND, reported_level
from dataeval.types import FactorLevel, SourceIndex
from dataeval.utils.thresholds import ConstantThreshold, resolve_threshold

# A value is rare in its stratum when P(X <= k | Binomial(n, p)) <= alpha (#1521): at 95 % confidence it makes up
# less than 5 % of the stratum.
_RARITY_P = 0.05
_RARITY_ALPHA = 0.05
# The column Outliers' stratum labels are handed to core under.
_STRATUM = "__stratum__"
# The kinds core counts by rarity rather than measures.
_CATEGORICAL_KINDS = ("categorical", "tags")
# Core's reasons that hold for a whole population: none of its rows could be scored.
_POPULATION_REASONS = frozenset({"too few rows", "exact fit", "degenerate covariance"})
# Core's reasons that keep a row out of its population: it has no value to compare, or no spread to compare it with.
_ROW_REASONS = frozenset({"missing value", "missing stratum", "constant residual"})
# A small distance or a typical sample is not unusual, so only the upper bound flags these.
_UPPER_ONLY = frozenset({"mahalanobis", "distribution"})

# Each level's stratum codes (one per frame row, as text) and their labels; None where no stratum reaches it.
LevelStrata = Mapping[FactorLevel, tuple[pl.Series, Mapping[int, str]] | None]


def rarity_minimum(p: float = _RARITY_P, alpha: float = _RARITY_ALPHA) -> int:
    """Smallest stratum in which one occurrence can clear the exact binomial test: 93 at the defaults."""
    n = 1
    while stats.binom.cdf(1, n, p) > alpha:
        n += 1
    return n


class _ScoreInput(NamedTuple):
    """What a metadata result keeps so a re-stratification can rescore: one frame per level and the checks asked for."""

    frames: Mapping[FactorLevel, pl.DataFrame]  # factor columns read with store.select, plus item_index and the keys
    categorical: Mapping[FactorLevel, tuple[str, ...]]
    groups: Mapping[FactorLevel, tuple[tuple[str, ...], ...]]
    given: Mapping[FactorLevel, Mapping[str, str]]
    kinds: Mapping[FactorLevel, Mapping[str, ColumnKind]]
    level_kinds: Mapping[FactorLevel, str | None]  # the SourceIndex.kind of each level's rows

    def is_categorical(self, level: FactorLevel, group: Sequence[str]) -> bool:
        """Whether every column of a group is counted by rarity at `level`."""
        kinds = self.kinds.get(level, {})
        return all(kinds.get(name) in _CATEGORICAL_KINDS for name in group)

    def judged(self, strata: Sequence[str]) -> dict[FactorLevel, tuple[tuple[str, ...], ...]]:
        """Return the rarity checks at each level within `strata`: each categorical factor and group but the strata."""
        kept: dict[FactorLevel, tuple[tuple[str, ...], ...]] = {}
        for level in self.frames:
            singles = [(name,) for name in self.categorical.get(level, ())]
            groups = [group for group in self.groups.get(level, ()) if self.is_categorical(level, group)]
            checks = tuple(check for check in [*singles, *groups] if not set(check) & set(strata))
            if checks:
                kept[level] = checks
        return kept

    def scored_levels(self) -> list[FactorLevel]:
        """Return the levels with a numeric group or a residual to score."""
        return [level for level in self.frames if self.numeric_groups(level) or self.given.get(level)]

    def numeric_groups(self, level: FactorLevel) -> list[tuple[str, ...]]:
        """Return the groups at `level` scored by distance rather than counted by rarity."""
        return [group for group in self.groups.get(level, ()) if not self.is_categorical(level, group)]

    def address(self, level: FactorLevel, row: int) -> SourceIndex:
        """Return the address of a row of a stored frame, by its position, as ``_factor_stats._address`` spells it."""
        values = self.frames[level].row(row, named=True)
        kind = self.level_kinds[level]
        if kind is None:
            return SourceIndex(values["item_index"])
        if kind == LABEL_KIND:
            return SourceIndex(values["item_index"], values["target_index"])
        return SourceIndex(values["item_index"], values[str(LEVEL_KEY_COLUMNS[level])], level)


def _scored_columns(
    categorical: Mapping[FactorLevel, Sequence[str]],
    groups: Mapping[FactorLevel, Sequence[tuple[str, ...]]],
    given: Mapping[FactorLevel, Mapping[str, str]],
) -> dict[FactorLevel, list[str]]:
    """Return the columns each level's checks read, each once, in the order the checks name them."""
    columns: dict[FactorLevel, list[str]] = {level: list(names) for level, names in categorical.items()}
    for level, found in groups.items():
        columns.setdefault(level, []).extend(name for group in found for name in group)
    for level, pairs in given.items():
        columns.setdefault(level, []).extend(name for pair in pairs.items() for name in pair)
    return {level: list(dict.fromkeys(names)) for level, names in columns.items()}


def read_score_input(
    metadata: Any,
    categorical: Mapping[str, FactorLevel],
    groups: Mapping[FactorLevel, Sequence[tuple[str, ...]]] | None = None,
    given: Mapping[FactorLevel, Mapping[str, str]] | None = None,
    kinds: Mapping[str, ColumnKind] | None = None,
) -> _ScoreInput | None:
    """Read each level's scored columns once, with the columns that address its rows; None when nothing is scored.

    Each level's `kinds` are the caller's entries for its own columns, plus ``"categorical"`` for every column
    :func:`factor_kind` calls categorical, the caller's entry winning.
    """
    by_level: dict[FactorLevel, tuple[str, ...]] = {}
    for name, level in categorical.items():
        by_level[level] = (*by_level.get(level, ()), name)
    groups, given, kinds = groups or {}, given or {}, kinds or {}
    columns = _scored_columns(by_level, groups, given)
    if not columns:
        return None
    store = metadata._store
    frames: dict[FactorLevel, pl.DataFrame] = {
        level: store.select(level, [*found, *_key_columns(metadata, level)]) for level, found in columns.items()
    }
    level_column_kinds: dict[FactorLevel, Mapping[str, ColumnKind]] = {
        level: {
            **{name: "categorical" for name in names if factor_kind(metadata, name) == "categorical"},
            **{name: kinds[name] for name in names if name in kinds},
        }
        for level, names in columns.items()
    }
    return _ScoreInput(
        frames,
        by_level,
        {level: tuple(found) for level, found in groups.items()},
        {level: dict(pairs) for level, pairs in given.items()},
        level_column_kinds,
        {level: level_kind(metadata, level) for level in columns},
    )


def strata_by_level(metadata: Any, columns: Sequence[str], levels: Sequence[FactorLevel]) -> LevelStrata:
    """Each level's stratum codes and labels within the strata `columns` name, None where they do not reach it."""
    found: dict[FactorLevel, tuple[pl.Series, Mapping[int, str]] | None] = {}
    for level in levels:
        series, labels = level_strata(metadata, columns, level)
        found[level] = None if series is None else (series, labels)
    return found


def rarity_within(
    metadata: Any, columns: Sequence[str], score_input: _ScoreInput
) -> tuple[pl.DataFrame, list[_Population]]:
    """Judge the stored categorical factors and groups by rarity within the strata `columns` name, where they reach."""
    checks = score_input.judged(columns)
    return judge_rarity(score_input, checks, strata_by_level(metadata, columns, list(checks)))


def judge_rarity(
    score_input: _ScoreInput,
    checks: Mapping[FactorLevel, Sequence[tuple[str, ...]]],
    strata: LevelStrata,
) -> tuple[pl.DataFrame, list[_Population]]:
    """Rarity findings and populations for every categorical check, at every level a stratum reaches.

    Parameters
    ----------
    score_input : _ScoreInput
        The stored frames, the kind of each column and the address of each row.
    checks : Mapping[FactorLevel, Sequence[tuple[str, ...]]]
        The categorical factors (one column each) and groups to judge at each level.
    strata : Mapping[FactorLevel, tuple[pl.Series, Mapping[int, str]] or None]
        Each level's stratum codes (one per frame row) and their labels, as :func:`level_strata` returns them;
        None where no stratum reaches the level.

    Returns
    -------
    tuple[pl.DataFrame, list[_Population]]
        One finding per row of a flagged (stratum, value) cell, and one population per (check, stratum).
    """
    minimum = rarity_minimum()
    found: list[pl.DataFrame] = [pl.DataFrame(schema=FINDINGS_SCHEMA)]
    populations: list[_Population] = []
    for level, judged in checks.items():
        reached = strata.get(level)
        if not judged or reached is None:
            continue
        cells = _cells(score_input.frames[level], judged, *reached, score_input.kinds.get(level, {}))
        populations.extend(_populations(cells, score_input.level_kinds[level], minimum))
        found.append(_findings(_flagged(cells, minimum), level, score_input.address))
    return pl.concat(found), populations


def _cells(
    frame: pl.DataFrame,
    checks: Sequence[tuple[str, ...]],
    codes: pl.Series,
    labels: Mapping[int, str],
    kinds: Mapping[str, ColumnKind],
) -> pl.DataFrame:
    """Each (check, stratum, value) cell of one level: the value's count k, the stratum's n, and the rows holding it."""
    stratum = codes.replace_strict({str(code): label for code, label in labels.items()}, return_dtype=pl.Utf8)
    read_as: dict[str, ColumnKind] = {name: kinds.get(name, "categorical") for check in checks for name in check}
    scores = combination_scores(
        frame.with_columns(stratum.alias(_STRATUM)), groups=list(checks), strata=_STRATUM, kinds=read_as
    )
    names = np.array([", ".join(group) for group in scores["groups"]])
    return (
        pl
        .DataFrame({
            "check": names[scores["group"]],
            "stratum": stratum.to_numpy()[scores["row"]],
            "observed": np.array(scores["observed"])[scores["combination"]],
            "k": scores["score"],
            "n": scores["n"],
            "row": scores["row"],
            "reason": scores["reason"],
        })
        .filter(pl.col("reason") == 0)
        .group_by("check", "stratum", "observed", maintain_order=True)
        .agg(pl.col("k").first().cast(pl.Int64), pl.col("n").first(), pl.col("row"))
    )


def _populations(cells: pl.DataFrame, kind: str | None, minimum: int) -> list[_Population]:
    """One population per (check, stratum), judged when the stratum holds at least `minimum` rows."""
    strata = cells.group_by("check", "stratum", maintain_order=True).agg(pl.col("n").first())
    return [
        _Population(kind, check, "rarity", label, n, n >= minimum, None if n >= minimum else f"below {minimum} rows")
        for check, label, n in strata.iter_rows()
    ]


def _bh_survivors(p: NDArray[np.float64], alpha: float = _RARITY_ALPHA) -> NDArray[np.bool_]:
    """Benjamini-Hochberg at `alpha`: which of the p-values are discoveries."""
    ranked = np.sort(p)
    passed = np.flatnonzero(ranked <= alpha * np.arange(1, len(p) + 1) / max(len(p), 1))
    return p <= ranked[passed[-1]] if passed.size else np.zeros(len(p), dtype=np.bool_)


def _flagged(cells: pl.DataFrame, minimum: int) -> pl.DataFrame:
    """Return the cells to report: tested cells that survive Benjamini-Hochberg and whose value is not withheld.

    Every observed cell of a stratum with at least `minimum` rows is tested, across every check at the level. A value
    is withheld when it is rare in every tested stratum of its check, a stratum where it does not occur counting as
    rare, so it is kept only where some tested stratum holds it with ``P > alpha``. A withheld value is rare in the
    dataset, which is a balance finding rather than an outlier.
    """
    tested = cells.filter(pl.col("n") >= minimum)
    p = stats.binom.cdf(tested["k"].to_numpy(), tested["n"].to_numpy(), _RARITY_P)
    tested = tested.with_columns(p=pl.Series(p, dtype=pl.Float64), survives=pl.Series(_bh_survivors(p)))
    common = tested.group_by("check", "observed").agg((pl.col("p") > _RARITY_ALPHA).any().alias("common"))
    return tested.join(common, on=["check", "observed"]).filter(pl.col("survives") & pl.col("common"))


def _findings(
    flagged: pl.DataFrame, level: FactorLevel, address: Callable[[FactorLevel, int], SourceIndex]
) -> pl.DataFrame:
    """One finding per row of a flagged cell, reporting the one-sided Clopper-Pearson upper bound on ``k / n``."""
    rows = flagged.explode("row")
    k, n = rows["k"].to_numpy(), rows["n"].to_numpy()
    addresses = [address(level, row) for row in rows["row"].to_list()]
    count = rows.height
    return pl.DataFrame(
        {
            "item_index": [si.item for si in addresses],
            "target_index": [si.key for si in addresses],
            "level": [reported_level(si) for si in addresses],
            "metric_name": rows["check"].to_list(),
            # A flagged cell has P(X <= k) <= alpha, so k < n and the bound is defined.
            "metric_value": stats.beta.ppf(1 - _RARITY_ALPHA, k + 1, n - k).tolist(),
            "direction": ["lower"] * count,
            "bound": [_RARITY_P] * count,
            "percentile": [None] * count,
            "population_mean": [None] * count,
            "population_std": [None] * count,
            "method": ["rarity"] * count,
            "population": rows["stratum"].to_list(),
            "observed": rows["observed"].to_list(),
        },
        schema=FINDINGS_SCHEMA,
    )


class _ScoredCheck(NamedTuple):
    """One combination check's stored scores at one level, enough to refit thresholds without rescoring."""

    level: FactorLevel
    kind: str | None  # the SourceIndex.kind of the level's rows, which populations list it under
    metric_name: str  # "a, b" for a group, "y given x" for a residual, the column for a distribution
    method: str  # "mahalanobis" | "residual" | "distribution"
    dims: int
    rows: NDArray[np.int64]  # positions in the level frame, one array shared by every check at the level
    score: NDArray[np.float64]
    reason: NDArray[np.int8]
    reasons: tuple[str, ...]
    stratum: NDArray[np.intp]  # Outliers' code per entry, shared at the level; a code with no label is the whole level
    labels: Mapping[int, str]  # the population label of each code


def score_combinations(score_input: _ScoreInput, strata: LevelStrata) -> list[_ScoredCheck]:
    """Score every numeric group and residual with one :func:`combination_scores` call per level.

    Parameters
    ----------
    score_input : _ScoreInput
        The stored frames and the checks asked for at each level.
    strata : Mapping[FactorLevel, tuple[pl.Series, Mapping[int, str]] or None]
        Each level's stratum codes and labels; a level missing or None is scored whole.

    Returns
    -------
    list[_ScoredCheck]
        One entry per check per level, in the order core returns them.
    """
    checks: list[_ScoredCheck] = []
    for level in score_input.scored_levels():
        frame = score_input.frames[level]
        reached = strata.get(level)
        if reached is not None:
            frame = frame.with_columns(reached[0].alias(_STRATUM))
        result = combination_scores(
            frame,
            groups=score_input.numeric_groups(level),
            strata=None if reached is None else _STRATUM,
            given=dict(score_input.given.get(level, {})),
            kinds=dict(score_input.kinds.get(level, {})),
        )
        codes = np.full(frame.height, -1, np.intp) if reached is None else reached[0].cast(pl.Int64).to_numpy()
        labels = {} if reached is None else reached[1]
        checks.extend(_split(result, level, score_input.level_kinds[level], codes, labels))
    return checks


def _split(
    result: Mapping[str, Any], level: FactorLevel, kind: str | None, codes: NDArray[np.intp], labels: Mapping[int, str]
) -> list[_ScoredCheck]:
    """Cut core's result, sorted by group then row, into one stored check per test.

    Core scores every row once per test, in row order, so the checks at a level share one array of rows and one of
    stratum codes, and each stores only its own scores and reasons (copied, so core's result is not kept alive).
    """
    height = codes.size
    rows = np.arange(height, dtype=np.int64)
    checks: list[_ScoredCheck] = []
    for test, (group, method) in enumerate(zip(result["groups"], result["methods"], strict=True)):
        entries = slice(test * height, (test + 1) * height)
        checks.append(
            _ScoredCheck(
                level=level,
                kind=kind,
                metric_name=f"{group[0]} given {group[1]}" if method == "residual" else ", ".join(group),
                method=method,
                dims=result["dims"][test],
                rows=rows,
                score=result["score"][entries].copy(),
                reason=result["reason"][entries].copy(),
                reasons=result["reasons"],
                stratum=codes,
                labels=labels,
            )
        )
    return checks


def score_threshold(outlier_threshold: Any, metric_name: str) -> Threshold:
    """Return the threshold for a score check: a named entry, else a limit-free statistical scalar, else the default."""
    if isinstance(outlier_threshold, Mapping):
        named = outlier_threshold.get(metric_name)
        return resolve_threshold(named) if named is not None else DEFAULT_OUTLIERS_OUTLIER_THRESHOLD
    if outlier_threshold is None:
        return DEFAULT_OUTLIERS_OUTLIER_THRESHOLD
    threshold = resolve_threshold(outlier_threshold)
    limited = getattr(threshold, "lower_limit", None) is not None or getattr(threshold, "upper_limit", None) is not None
    if isinstance(threshold, ConstantThreshold) or limited:
        return DEFAULT_OUTLIERS_OUTLIER_THRESHOLD
    return threshold


def flag_scored(
    checks: Sequence[_ScoredCheck], outlier_threshold: Any, address: Callable[[FactorLevel, int], SourceIndex]
) -> tuple[pl.DataFrame, list[_Population]]:
    """Fit the threshold to each check's scores in each population and report the scores that cross it.

    Parameters
    ----------
    checks : Sequence[_ScoredCheck]
        The stored scores.
    outlier_threshold : ThresholdLike or Mapping[str, ThresholdLike] or None
        The evaluator's threshold, read for each check by :func:`score_threshold`.
    address : Callable[[FactorLevel, int], SourceIndex]
        The address of a row of a level's frame, by its position.

    Returns
    -------
    tuple[pl.DataFrame, list[_Population]]
        One finding per flagged entry, and one population per (check, stratum).
    """
    found: list[pl.DataFrame] = [pl.DataFrame(schema=FINDINGS_SCHEMA)]
    populations: list[_Population] = []
    for check in checks:
        threshold = score_threshold(outlier_threshold, check.metric_name)
        for code in np.unique(check.stratum).tolist():
            entries = np.flatnonzero(check.stratum == code)
            population, fit = _judge(check, entries, threshold)
            if population is not None:
                populations.append(population)
            if fit is not None and fit.mask.any():
                found.append(_score_findings(check, entries[fit.mask], fit, address))
    return pl.concat(found), populations


def _judge(
    check: _ScoredCheck, entries: NDArray[np.intp], threshold: Threshold
) -> tuple[_Population | None, _Fit | None]:
    """Judge one population of a check: its listing, and the fit over its entries when it was judged."""
    reasons = np.asarray(check.reasons)[check.reason[entries]]
    present = ~np.isin(reasons, list(_ROW_REASONS))
    n = int(present.sum())
    label = check.labels.get(int(check.stratum[entries[0]]))
    if n == 0:
        # A named check is listed even when no row has a value to score, so the user sees it was not judged.
        reason = f"below {_minimum(check)} rows"
        return _Population(check.kind, check.metric_name, check.method, label, 0, False, reason), None
    reason, fit = _fit_scores(check, entries, threshold, reasons[present])
    return _Population(check.kind, check.metric_name, check.method, label, n, reason is None, reason), fit


def _minimum(check: _ScoredCheck) -> int:
    """Return the fewest scored rows a population of a check is judged from: 10 per dimension for a distance."""
    if check.method == "mahalanobis":
        return max(_MIN_POPULATION, _MIN_ROWS_PER_DIMENSION * check.dims)
    return _MIN_POPULATION


def _fit_scores(
    check: _ScoredCheck, entries: NDArray[np.intp], threshold: Threshold, present: NDArray[np.str_]
) -> tuple[str | None, _Fit | None]:
    """Return why a population was not judged, or the fit of its scores; only the upper bound flags a distance."""
    minimum = _minimum(check)
    if set(present.tolist()) <= _POPULATION_REASONS:
        return _floor_reason(present.size, threshold, minimum) or str(present[0]), None
    values = np.where(check.reason[entries] == 0, check.score[entries], np.nan)
    real = np.isfinite(values)
    reason = _floor_reason(int(real.sum()), threshold, minimum)
    if reason is not None:
        return reason, None
    fit, _, reason = _fit_outliers(values, threshold)
    if reason is None and check.method in _UPPER_ONLY:
        if threshold(values)[1] is None:
            return "no upper bound", None
        fit = fit._replace(mask=fit.mask & fit.above)
    return reason, fit


def _score_findings(
    check: _ScoredCheck, flagged: NDArray[np.intp], fit: _Fit, address: Callable[[FactorLevel, int], SourceIndex]
) -> pl.DataFrame:
    """Lay out one finding per flagged entry of a check, described against the population it was judged in."""
    addresses = [address(check.level, row) for row in check.rows[flagged].tolist()]
    count = len(addresses)
    label = check.labels.get(int(check.stratum[flagged[0]]))
    return pl.DataFrame(
        {
            "item_index": [si.item for si in addresses],
            "target_index": [si.key for si in addresses],
            "level": [reported_level(si) for si in addresses],
            "metric_name": [check.metric_name] * count,
            "metric_value": check.score[flagged].tolist(),
            "direction": np.where(fit.above[fit.mask], "upper", "lower").tolist(),
            "bound": fit.bound[fit.mask].tolist(),
            "percentile": fit.percentile[fit.mask].tolist(),
            "population_mean": fit.mean[fit.mask].tolist(),
            "population_std": fit.std[fit.mask].tolist(),
            "method": [check.method] * count,
            "population": [label] * count,
            "observed": [None] * count,
        },
        schema=FINDINGS_SCHEMA,
    )
