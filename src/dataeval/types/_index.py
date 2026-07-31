"""Index types locating data within a source dataset."""

__all__ = [
    "SourceIndex",
]

from typing import NamedTuple

from typing_extensions import Self


class SourceIndex(NamedTuple):
    """
    The indices of the source item, target and channel.

    Attributes
    ----------
    item: int
        Index of the source item
    target : int | None
        Index of the box/target within the source item.
        - None: References item-level data
        - int: References a specific target/detection within the item (0-indexed per item)
        For Object Detection datasets, this maps to the target_index in Metadata.
    channel : int | None
        Index of the channel of the source image (if applicable)
    """

    item: int
    target: int | None = None
    channel: int | None = None

    def __repr__(self) -> str:
        """Compact representation showing only non-None fields."""
        parts = [f"{self.item}"]
        if self.target is not None:
            parts.append(f"{self.target}")
        if self.target is None and self.channel is not None:
            parts.append("None")
        if self.channel is not None:
            parts.append(f"{self.channel}")
        return f"SourceIndex({', '.join(parts)})"

    def __str__(self) -> str:
        """Human-readable string showing the full path."""
        parts = [str(self.item)]
        if self.target is not None:
            parts.append(str(self.target))
        if self.target is None and self.channel is not None:
            parts.append("-")
        if self.channel is not None:
            parts.append(str(self.channel))
        return "/".join(parts)

    @classmethod
    def from_string(cls, s: str) -> Self:
        """
        Construct a SourceIndex from a human-readable string.

        Parameters
        ----------
        s : str
            String in the format "item", "item/target", or "item/-/channel", "item/target/channel"
            Use "-" to represent None for the target field.

        Returns
        -------
        SourceIndex

        Examples
        --------
        >>> SourceIndex.from_string("0")
        SourceIndex(0)
        >>> SourceIndex.from_string("0/3")
        SourceIndex(0, 3)
        >>> SourceIndex.from_string("0/-/1")
        SourceIndex(0, None, 1)
        >>> SourceIndex.from_string("0/3/1")
        SourceIndex(0, 3, 1)
        """
        item, target, channel = None, None, None
        parts = s.split("/")

        if len(parts) > 0:
            item = int(parts[0])
        if len(parts) > 1:
            target = None if parts[1] == "-" else int(parts[1])
        if len(parts) > 2:
            channel = None if parts[2] == "-" else int(parts[2])
        if item is None or len(parts) > 3:
            raise ValueError(f"Invalid SourceIndex string format: {s}")
        return cls(item, target, channel)
