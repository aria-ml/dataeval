"""What a statistic was measured with, recorded beside the values it produced."""

__all__ = []

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Literal


@dataclass(frozen=True)
class MeasurementSpec:
    """The settings one statistic was measured with.

    Two columns under one name are comparable only when they were measured the same way. A
    surround mean over a ring padded by a quarter of the box and one padded by half are
    different quantities, and nothing in the values says so. A producer records one of these
    per factor, beside its result, so the settings travel with the numbers rather than staying
    in the call that made them.

    Plain values only, so a spec serializes with :func:`json.dumps` and compares by value
    between runs. :meth:`to_dict` gives the mapping, and ``MeasurementSpec(**spec.to_dict())``
    rebuilds an equal spec, including from the JSON round trip.

    .. versionadded:: 1.3

    Attributes
    ----------
    algorithm : str
        Name of the method, such as ``"immerkaer"``.
    version : str
        Version of the method's implementation. It changes whenever the same input would give
        a different value.
    units : str
        What the values are measured in, such as ``"fraction"`` or ``"normalized intensity"``.
    region : str
        What the statistic reduces over, such as ``"box"`` or ``"surround"``.
    normalization : str or None, default None
        The domain values were mapped into before measuring, such as ``"[0, 1]"``. None when
        raw values are measured.
    config : Mapping[str, Any], default empty
        The effective settings, after defaults are resolved. Values must be JSON values:
        numbers, strings, booleans, None, and lists or mappings of them. Read back as a
        read-only mapping, with tuples turned into lists as JSON would.
    range_policy : {"declared", "inferred"} or None, default None
        Where the value range came from: stated by the caller, or inferred from each image.
        None for a statistic that reads no range.
    declared_range : tuple[float, float] or None, default None
        The range the caller stated. Given exactly when `range_policy` is ``"declared"``.
    rgb_channels : tuple[int, int, int] or None, default None
        The bands read as red, green and blue, for a statistic that interprets color.

    Raises
    ------
    TypeError
        If `config` holds a value JSON cannot represent, such as a callable or an array.
    ValueError
        If `range_policy` is not one of its values, `declared_range` disagrees with it, or
        `rgb_channels` does not name three bands.
    """

    algorithm: str
    version: str
    units: str
    region: str
    normalization: str | None = None
    # A factory, not a shared instance: a mappingproxy default is refused by 3.11's dataclasses.
    config: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    range_policy: Literal["declared", "inferred"] | None = None
    declared_range: tuple[float, float] | None = None
    rgb_channels: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        """Normalize to plain values and refuse what could not be written out and read back."""
        object.__setattr__(self, "config", MappingProxyType(_json_config(self.config)))
        if self.declared_range is not None:
            object.__setattr__(self, "declared_range", tuple(float(v) for v in self.declared_range))
        if self.rgb_channels is not None:
            object.__setattr__(self, "rgb_channels", tuple(int(c) for c in self.rgb_channels))
            if len(self.rgb_channels) != 3:
                raise ValueError(f"rgb_channels names the red, green and blue bands, got {self.rgb_channels}.")
        _check_range_policy(self.range_policy, self.declared_range)

    def __hash__(self) -> int:
        """Hash the serialized form, since ``config`` is a mapping the generated hash refuses."""
        return hash(json.dumps(self.to_dict(), sort_keys=True))

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as plain values, ready for :func:`json.dumps`."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        # A copy, so the caller can edit it without reaching the spec's own settings.
        values["config"] = json.loads(json.dumps(dict(self.config)))
        return values


def _json_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return `config` as JSON would read it back, or refuse it.

    A JSON round trip is the check and the normalization at once: it refuses a callable or an
    array, and turns tuples into the lists a deserialized spec would hold, so a spec read back
    from disk compares equal to the one written.
    """
    try:
        return json.loads(json.dumps(dict(config or {}), allow_nan=False))
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"MeasurementSpec config must hold only JSON values (numbers, strings, booleans, None, "
            f"and lists or mappings of them): {e}",
        ) from e


def _check_range_policy(policy: str | None, declared: tuple[float, float] | None) -> None:
    """Refuse a range policy that is not one of its values or disagrees with the declared range."""
    if policy not in (None, "declared", "inferred"):
        raise ValueError(f"range_policy must be 'declared', 'inferred' or None, got {policy!r}.")
    if (policy == "declared") != (declared is not None):
        raise ValueError(
            f"declared_range is given exactly when range_policy is 'declared'; got range_policy={policy!r} "
            f"with declared_range={declared!r}.",
        )
