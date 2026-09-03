"""Writing a structured :class:`~dataeval.Metadata` to one file, and reading it back.

What a saved metadata file is **for** is skipping the walk over a dataset that produced
it. Structuring reads every item — decoding images, unpacking targets, accumulating
tracks — and everything downstream of it is arithmetic over the rows that walk produced.
Those rows are small next to the dataset and expensive next to nothing else, which is the
shape of a thing worth caching.

**A cache, not an interchange format.** What is written here is the library's own
per-level layout: one frame per level, the positional edges between them, and which
columns descendants inherit. That layout is what every feature in this package is free to
change, so a file is stamped with the format version and with the schema it was written
against, and :func:`restore` refuses anything it does not recognize rather than guessing.
Refusal is the designed outcome for a stale file — a caching caller catches
:class:`~dataeval.exceptions.MetadataFormatError` and recomputes.

**No executable payload.** The container is a zip of parquet frames and one JSON
manifest, all of it data. Nothing here pickles, and nothing here imports a name a file
asks it to: the task is matched against a fixed registry, not looked up. A saved file is
worth no more trust than the data it came from, and reading one grants it none.

Three things are deliberately *not* written:

- **The dataset.** It cannot be serialized and must not be. :func:`restore` takes a live
  one and binds it, which is why the level rows and the dataset can be checked against
  each other at load rather than diverging silently.
- **Binning.** Companion columns are stripped on the way out, so one file serves every
  ``continuous_factor_bins`` and ``auto_bin_method`` a caller might read it with, and
  binning re-runs lazily against the configuration in force. This is the reason
  ``exclude``, ``include`` and ``view`` are not written either: they are how a reader asks
  its question, not what the rows are.
- **The per-item metadata dicts.** :attr:`~dataeval.Metadata.raw` holds whatever the
  dataset put there, of unbounded size and arbitrary type, and writing it back would be
  the executable payload this format does not have. A restored instance reports that they
  are absent rather than answering as though the dataset had none.
"""

__all__ = []

import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval._metadata._columns import binned, digitized
from dataeval._metadata._encoding import (
    aggregations_from_list,
    aggregations_to_list,
    bins_from_mapping,
    bins_to_mapping,
    corrections_from_list,
    corrections_to_list,
    encoding_from_mapping,
    encoding_to_mapping,
)
from dataeval._metadata._links import LinkIndex
from dataeval._metadata._store import LevelStore
from dataeval._metadata._structurers import TASK_STRUCTURERS, FactorsStructurer, Structurer
from dataeval.exceptions import MetadataFormatError
from dataeval.types import FactorLevel, FactorLevelSchema
from dataeval.types._execution import __version__

if TYPE_CHECKING:
    from dataeval._metadata._metadata import Metadata

_logger = get_logger(__name__)

# Bumped whenever a change makes files written by an older dataeval unreadable — a
# renamed manifest key, a different member layout, a changed column convention. It is
# not the library version: most releases change nothing here, and a bump that did not
# have to happen throws away every cache in the field.
FORMAT_VERSION = 1

_MANIFEST = "manifest.json"
_FRAMES = "frames"
_LINKS = "links"


# ---------------------------------------------------------------------------- writing


def _without_companions(store: LevelStore) -> LevelStore:
    """Drop every binned/digitized column, leaving the values they were derived from.

    Binning is configuration, not data: the same rows binned two ways are the same rows.
    Stripping here is what lets one file serve every binning a reader might want, and
    costs nothing to undo — ``Metadata._bin`` rebuilds a companion column
    for any factor that has none.

    Both spellings are checked for every column, the same pairing
    ``Metadata._reset_bins`` uses, so a factor that changed type between
    bins cannot leave its older companion behind.
    """
    present = set(store.columns)
    companions = {name for column in store.columns for name in (binned(column), digitized(column)) if name in present}
    return store.without_columns(companions)


def _edge_frame(store: LevelStore, level: FactorLevel) -> pl.DataFrame | None:
    """One column of ancestor positions per parent of ``level``, or None where it has no parents.

    The general :meth:`LinkIndex.positions` form rather than either representation's
    internals, so the file says what an edge *means* and :meth:`LinkIndex.of` re-picks the
    tightest representation on the way back in — the same round trip
    :meth:`LevelStore.restrict` already performs on every filter.
    """
    parents = store.schema.parents_of(level)
    if not parents:
        return None
    return pl.DataFrame([pl.Series(parent, store.link(level, parent).positions()) for parent in parents])


def _applied_encodings(md: "Metadata") -> dict[str, Any]:
    """Merge what the caller declared with what binning resolved it to.

    So that the same object writes the same record whether or not anything has read
    ``factor_data`` from it yet: ``_factor_cache`` is empty until the binning pass runs, and
    reading it alone dropped a declared encoding entirely from an archive saved before the
    first read. The applied entry wins where both exist — it is the declared one plus any
    growth.
    """
    return {
        **md._encoding,
        **{name: info.encoding for name, info in md._factor_cache.items() if info.encoding is not None},
    }


def _plain(value: Any) -> Any:
    """Render one retained value as something JSON can hold.

    A metadata value may arrive as a NumPy scalar, which ``json`` cannot serialize. Only
    numbers and text reach here -- a column is set aside for mixing those two -- so
    unwrapping to the Python value it stands for is the whole conversion.
    """
    return value.item() if isinstance(value, np.generic) else value


def _manifest(md: "Metadata", store: LevelStore) -> dict[str, Any]:
    """Everything about the instance that is not a frame or an edge.

    ``frames`` is a list rather than a mapping because its **order** is load-bearing:
    :meth:`LevelStore.flat` concatenates the levels in the order the mapping holds them,
    and a JSON object offers no promise about that. Each entry carries its shape as well,
    so a truncated member is caught at the file it was truncated in rather than as a
    length disagreement somewhere downstream — and so that a frame with **no columns**,
    which an empty dataset's item level is, is described rather than written: a columnless
    frame is entirely described by saying so, and parquet on the supported polars floor
    cannot round-trip one.

    ``levels`` and ``edges`` are written even though the structurer determines both. They
    are the check with teeth: a future release that changes what a task's level graph
    looks like would otherwise restore old rows against a new graph and answer confidently
    with the wrong shape, whereas comparing the two makes that a refusal.
    """
    structurer = md._structurer
    applied = _applied_encodings(md)
    return {
        "format_version": FORMAT_VERSION,
        "dataeval_version": __version__,
        "task": structurer.task,
        "item_level": structurer.item_level,
        "label_level": structurer.label_level,
        "levels": list(store.schema),
        "edges": [[level, parent] for level in store.schema for parent in store.schema.parents_of(level)],
        "frames": [
            {"level": level, "height": frame.height, "columns": frame.width} for level, frame in store.frames.items()
        ],
        "propagating": {level: sorted(names) for level, names in store.propagating.items()},
        "column_order": list(store.column_order),
        "factors_by_level": {level: sorted(names) for level, names in md._factors_by_level.items()},
        "index2label": {str(index): label for index, label in md._index2label.items()},
        "count": int(md._count),
        "dropped_factors": {name: list(reasons) for name, reasons in md._dropped_factors.items()},
        # The roll-ups, keyed on the column each produced and in the order they ran. The
        # columns themselves are already in the store; this is what says *how* they were
        # reached, which is what `new()` needs to rebuild them over another dataset.
        "aggregations": aggregations_to_list(md._aggregations),
        # Written, unlike `raw`, which the format refuses for being unbounded and of
        # arbitrary type. These are bounded -- one value per row of one level -- and
        # scalar, because a column only reaches here by mixing numbers with text. Writing
        # them is what lets a repair be declared against a restored instance that has no
        # dataset to re-read, which is the whole point of recording repairs at all.
        # The declared repairs, in the order they apply. Written for the same reason the
        # held-back values are: a repair is a statement about what the rows are, and one
        # that did not survive the archive would leave a restored instance reading the
        # dataset differently from the instance that wrote it.
        "corrections": corrections_to_list(md._corrections),
        "unusable": {
            level: {name: [_plain(value) for value in values] for name, values in columns.items()}
            for level, columns in md._unusable_values.items()
        },
        # The pre-repair values of a factor that was already a column when a correction
        # named it. Without them a restored instance would read the corrected column back
        # as though it were what the dataset wrote, and `unrepair` would "restore" the
        # correction it was undoing.
        "pristine": {
            level: {name: [_plain(value) for value in values] for name, values in columns.items()}
            for level, columns in md._pristine_values.items()
        },
        "aggregated_from": dict(md._aggregated_from),
        # The applied encodings. Companion columns are stripped on the way in and rebuilt on
        # load, so without this a restored instance re-derives its cuts — which loses
        # `accept()` outright, and reproduces the old codes only for as long as the binning
        # heuristics do not change. Written as an optional key: a file from before this
        # existed simply has none, and restores the way it always did.
        #
        # What the caller declared underneath what binning applied, so that the same object
        # writes the same record whether or not anything has read `factor_data` from it
        # yet. `_factor_cache` is empty until the binning pass runs, and reading it alone
        # dropped a declared encoding entirely from an archive saved before the first read.
        # The applied entry wins where both exist: it is the declared one plus any growth.
        "encoding": encoding_to_mapping(applied),
        # The third spelling of a declaration, and the one every existing caller uses. It
        # does not resolve into an encoding until the binning pass runs, so an instance
        # saved before its first read carries the cut only here -- and without this member
        # a declared `continuous_factor_bins` was the one declaration a save could lose.
        #
        # Only where the record does not already cover the factor. Writing both put one
        # factor into two members, `_adopt_manifest` restored both, and the constructor
        # refuses that pair -- so `save(); load().new(...)` raised on every declared cut
        # that had been through a read. A resolved BinSpec carries `provenance="count"`,
        # so nothing is lost by omitting the count it came from.
        "continuous_factor_bins": bins_to_mapping({
            name: spec for name, spec in md._continuous_factor_bins.items() if name not in applied
        }),
        # Part of the declaration, not of the data: a closed vocabulary that silently
        # reopens across a round trip is the one failure `strict` exists to prevent, and it
        # fails open -- the next dataset's unseen value is appended rather than refused, and
        # nothing says the taxonomy widened.
        "strict": bool(md._strict),
        # The other structuring policy, written for the same reason. It decides what the
        # rows *are* -- which partly declared factors exist at all, and which rows of them
        # read as missing -- so a restored instance that reported the default would describe
        # a walk that did not happen, and `new()` from it would structure the next dataset
        # under the opposite rule.
        "partial_factors": bool(md._partial_factors),
        "is_filtered": bool(md._is_filtered),
        "cut_below_items": bool(md._cut_below_items),
    }


def _write_members(archive: zipfile.ZipFile, md: "Metadata", store: LevelStore) -> None:
    """Write the manifest and one parquet per level, streaming each frame straight in.

    Streamed rather than buffered so peak memory is whatever parquet needs for one frame
    rather than a second copy of the whole store.
    """
    archive.writestr(_MANIFEST, json.dumps(_manifest(md, store), indent=2, sort_keys=True))
    for level, frame in store.frames.items():
        # A frame with no columns holds nothing a parquet file could carry, and the
        # manifest entry already says both of its dimensions. Writing one anyway produces
        # a file polars 1.0.0 refuses to read back.
        if frame.width:
            with archive.open(f"{_FRAMES}/{level}.parquet", "w") as member:
                frame.write_parquet(member)
        edges = _edge_frame(store, level)
        if edges is not None:
            with archive.open(f"{_LINKS}/{level}.parquet", "w") as member:
                edges.write_parquet(member)


def save(md: "Metadata", path: Path | str) -> None:
    """Write the instance's rows, levels and edges to one file at ``path``.

    Structures first if it has to, so that saving a freshly constructed instance writes
    the rows a caller expects rather than an empty file — the same way
    :meth:`~dataeval.Embeddings.save` computes before it writes.

    Structures but does not **bin**: binning writes companion columns onto the instance,
    and saving must not change what the object it was handed reports. Every *declaration*
    is written regardless -- ``encoding`` and ``factor_levels`` through the manifest's
    encoding member, ``continuous_factor_bins`` through its own -- so no choice anybody
    made depends on whether something has read from the instance first. What an unbinned
    instance has not got is a *derived* cut, and losing one costs nothing: it is a function
    of the values, and the archive holds the same values to re-derive it from.

    Written to a temporary file in the destination directory and renamed over the target,
    so a reader either sees the previous file or the new one. A crash mid-write cannot
    leave a half-written archive in place, which for a cache would otherwise be a
    permanent one: every later read would fail, and nothing would ever overwrite it.

    Raises
    ------
    NotFittedError
        When the instance has neither a bound dataset nor structured rows, so there is
        nothing to write.
    """
    md._structure()
    store = _without_companions(md._store)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # ``mkstemp`` rather than ``NamedTemporaryFile``: the file outlives the handle here,
    # since the rename can only happen once the archive's central directory is flushed,
    # and a temporary that deletes itself on close would be gone by then. In the same
    # directory as the target so the rename stays within one filesystem and so is atomic.
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        # ``ZIP_STORED``: parquet already compresses its own pages, so deflating the
        # container again costs time to save almost nothing.
        with os.fdopen(descriptor, "wb") as handle, zipfile.ZipFile(handle, "w", zipfile.ZIP_STORED) as archive:
            _write_members(archive, md, store)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    _logger.info("Saved metadata to %s (%d levels, %s rows)", target, len(store.frames), store.counts)


# ---------------------------------------------------------------------------- reading


def _read_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    """Read the manifest and check it is a format this version knows how to restore.

    Raises
    ------
    MetadataFormatError
        When the member is missing, is not JSON, or names another format version.
    """
    try:
        manifest = json.loads(archive.read(_MANIFEST))
    except (KeyError, ValueError) as exc:
        raise MetadataFormatError(f"Not a dataeval metadata file: {exc}") from exc
    found = manifest.get("format_version") if isinstance(manifest, dict) else None
    if found != FORMAT_VERSION:
        wrote = manifest.get("dataeval_version") if isinstance(manifest, dict) else None
        raise MetadataFormatError(
            f"This file is metadata format {found!r} and this dataeval reads format "
            f"{FORMAT_VERSION} (the file was written by dataeval {wrote!r}, this is "
            f"{__version__}). Rebuild the metadata from its dataset and save it again.",
        )
    return manifest


def _structurer_from(manifest: Mapping[str, Any]) -> Structurer:
    """Rebuild the structurer that laid these rows out, without a dataset to select it by.

    Matched against the registry rather than imported by name: a file says which task it
    was, and a task it does not offer is a refusal rather than a lookup.

    Raises
    ------
    MetadataFormatError
        When the task is one this version does not produce, or the factors-only levels
        name a shape it does not lay out.
    """
    task = manifest.get("task")
    if isinstance(task, str) and task.upper() in TASK_STRUCTURERS:
        return TASK_STRUCTURERS[task.upper()]()
    if task != FactorsStructurer.task:
        raise MetadataFormatError(
            f"This file was written for task {task!r}, which this dataeval does not structure. "
            f"Tasks it reads are {[*sorted(TASK_STRUCTURERS), FactorsStructurer.task]}.",
        )
    try:
        return FactorsStructurer.for_shape(manifest["item_level"], manifest["label_level"])
    except (KeyError, ValueError) as exc:
        raise MetadataFormatError(f"This file names a level layout this dataeval does not lay out: {exc}") from exc


def _reject_schema_drift(manifest: Mapping[str, Any], schema: FactorLevelSchema) -> None:
    """Refuse rows laid out against a level graph this version no longer declares.

    The failure this prevents is silent rather than loud. Restoring a two-level file
    against a three-level graph produces a store whose every read succeeds and whose
    every gather is against an edge that was never written, so nothing raises and the
    numbers are wrong.

    Raises
    ------
    MetadataFormatError
        When the levels or the edges differ from what the restored structurer declares.
    """
    levels = [list(manifest.get("levels", ())), list(schema)]
    edges = [
        [list(edge) for edge in manifest.get("edges", ())],
        [[level, parent] for level in schema for parent in schema.parents_of(level)],
    ]
    if levels[0] != levels[1] or edges[0] != edges[1]:
        raise MetadataFormatError(
            f"This file was written against levels {levels[0]} with edges {edges[0]}, and this "
            f"dataeval declares levels {levels[1]} with edges {edges[1]} for task "
            f"{manifest.get('task')!r}. Rebuild the metadata from its dataset and save it again.",
        )


def _read_frame(archive: zipfile.ZipFile, name: str, height: int) -> pl.DataFrame:
    """Read one parquet member and check it holds the rows the manifest promised.

    Raises
    ------
    MetadataFormatError
        When the member is missing, unreadable, or the wrong height.
    """
    try:
        frame = pl.read_parquet(io.BytesIO(archive.read(name)))
    except (KeyError, OSError, pl.exceptions.PolarsError) as exc:
        raise MetadataFormatError(f"Metadata file is missing or cannot read {name!r}: {exc}") from exc
    if frame.height != height:
        raise MetadataFormatError(
            f"{name!r} holds {frame.height} row(s) and the manifest says {height}; the file is incomplete.",
        )
    return frame


def _level_frame(archive: zipfile.ZipFile, entry: Mapping[str, Any]) -> pl.DataFrame:
    """One level's own rows, or a frame with no columns where that is what it had.

    A columnless frame — an empty dataset's item level — is written as a manifest entry
    and no member, so it is rebuilt from the entry rather than read.

    Raises
    ------
    MetadataFormatError
        When the entry claims rows in a frame that has no columns to hold them.
    """
    if int(entry["columns"]):
        return _read_frame(archive, f"{_FRAMES}/{entry['level']}.parquet", int(entry["height"]))
    if int(entry["height"]):
        raise MetadataFormatError(
            f"Level {entry['level']!r} is recorded with {entry['height']} row(s) and no columns "
            "to hold them; the manifest is inconsistent.",
        )
    return pl.DataFrame()


def _read_edges(archive: zipfile.ZipFile, level: FactorLevel, height: int) -> dict[FactorLevel, NDArray[np.intp]]:
    """Read one level's ancestor positions, or nothing where it has no parents."""
    name = f"{_LINKS}/{level}.parquet"
    if name not in archive.namelist():
        return {}
    frame = _read_frame(archive, name, height)
    # The column names are level names, which the manifest's schema check has already
    # matched against the schema this store is being built for.
    return {
        cast("FactorLevel", parent): frame[parent].to_numpy().astype(np.intp, copy=False) for parent in frame.columns
    }


def _store_from(archive: zipfile.ZipFile, manifest: Mapping[str, Any], schema: FactorLevelSchema) -> LevelStore:
    """Rebuild the store from the members, in the frame order the manifest recorded.

    Every declared edge is rebuilt, including one whose level produced no frame — the same
    completeness :meth:`LevelStore.of` maintains, and for the same reason: a missing entry
    turns a composed link into a ``KeyError`` from deep inside :meth:`LevelStore.link`.

    Raises
    ------
    MetadataFormatError
        When a member is missing or inconsistent, or an edge names a row its parent level
        does not have.
    """
    frames: dict[FactorLevel, pl.DataFrame] = {
        entry["level"]: _level_frame(archive, entry) for entry in manifest["frames"]
    }
    positions = {level: _read_edges(archive, level, frame.height) for level, frame in frames.items()}
    empty = np.empty(0, dtype=np.intp)
    try:
        links: dict[tuple[FactorLevel, FactorLevel], LinkIndex] = {
            (level, parent): LinkIndex.of(positions.get(level, {}).get(parent, empty), frames[parent].height)
            if parent in frames
            else LinkIndex.of(empty, 0)
            for level in schema
            for parent in schema.parents_of(level)
        }
    except ValueError as exc:
        raise MetadataFormatError(f"Metadata file holds an edge that does not fit its levels: {exc}") from exc
    return LevelStore(
        schema=schema,
        frames=frames,
        links=links,
        propagating={level: frozenset(names) for level, names in manifest["propagating"].items()},
        column_order=tuple(manifest["column_order"]),
    )


def _adopt_manifest(md: "Metadata", manifest: Mapping[str, Any], structurer: Structurer, store: LevelStore) -> None:
    """Take on everything the manifest describes, mirroring ``Metadata._adopt``.

    Assigned field by field rather than routed through ``_adopt``, which takes a
    ``StructuredData`` — a richer bundle than what is worth writing, since the store is
    already what ``_adopt`` reduces it to. The cost is a second list of fields to keep in
    step, and ``test_round_trip_restores_every_field_adopt_sets`` is what keeps it there:
    it compares a restored instance against a freshly structured one attribute by
    attribute, so a field added to ``_adopt`` and forgotten here fails rather than
    silently restoring as its unstructured default.
    """
    md._structurer = structurer
    md._store = store
    md._factors_by_level = {level: set(names) for level, names in manifest["factors_by_level"].items()}
    for level in structurer.levels:
        md._factors_by_level.setdefault(level, set())
    md._index2label = {int(index): label for index, label in manifest["index2label"].items()}
    md._count = int(manifest["count"])
    md._dropped_factors = {name: list(reasons) for name, reasons in manifest["dropped_factors"].items()}
    md._aggregated_from = dict(manifest["aggregated_from"])
    md._aggregations = aggregations_from_list(manifest.get("aggregations", []))
    # Underneath whatever the caller passed, never over it. Binning is configuration rather
    # than data here — ``load(..., continuous_factor_bins=...)`` is meant to re-cut the
    # restored rows — so the archive's record fills in only the factors the reader said
    # nothing about. Both spellings of "the reader spoke" count, since ``_encoding`` is
    # consulted before ``continuous_factor_bins`` and would otherwise shadow it.
    spoken_for = set(md._encoding) | set(md._continuous_factor_bins)
    restored = {
        name: spec
        for name, spec in encoding_from_mapping(manifest.get("encoding", {})).items()
        if name not in spoken_for
    }
    md._encoding = {**restored, **md._encoding}
    # The same rule for the same reason, on the other member. Filtered against the reader's
    # `encoding` as well, so the archive cannot reintroduce a cut for a factor the reader
    # gave a vocabulary -- the pair is mutually exclusive per factor, and the constructor's
    # check has already run by the time this does.
    md._continuous_factor_bins = {
        **{
            name: spec
            for name, spec in bins_from_mapping(manifest.get("continuous_factor_bins", {})).items()
            if name not in spoken_for
        },
        **md._continuous_factor_bins,
    }
    # Underneath the reader's own, like every other declaration: `load(..., strict=True)`
    # closes a vocabulary the archive left open, and passing nothing keeps what was written.
    # Optional, so a file from before this existed restores as the permissive default.
    md._strict = md._strict or bool(manifest.get("strict", False))
    # Underneath the reader's own, like `strict`. Optional, so a file written before this
    # existed restores as the all-or-nothing default it was structured under.
    md._partial_factors = md._partial_factors or bool(manifest.get("partial_factors", False))
    # The level names come back as plain strings from JSON; the schema they belong to is
    # the one this manifest just declared, so they are the literals the rest of the code
    # reads them as.
    md._corrections = tuple(corrections_from_list(manifest.get("corrections", [])))
    md._unusable_values = {
        cast("FactorLevel", level): {name: list(values) for name, values in columns.items()}
        for level, columns in manifest.get("unusable", {}).items()
    }
    md._pristine_values = {
        cast("FactorLevel", level): {name: list(values) for name, values in columns.items()}
        for level, columns in manifest.get("pristine", {}).items()
    }
    # Derived rather than written: a held-back name can only be a column of the store if a
    # repair put it there, so the store already says which repairs landed. The other
    # repairable drop -- a column that named its rows -- is settled below, once the factor
    # set has been rebuilt and can say whether the reading gave it a vocabulary.
    held = {name for columns in md._unusable_values.values() for name in columns}
    md._repaired = held & set(md._store.columns)
    md._is_filtered = bool(manifest["is_filtered"])
    md._cut_below_items = bool(manifest["cut_below_items"])
    # Not written, and said so rather than answered as an empty dataset would be.
    md._raw = []
    md._raw_omitted = True

    # A view chosen at construction resolves here, at the first moment there is a schema to
    # resolve it against and before _build_factors reads it — the same point in the same
    # order _adopt resolves it.
    if md._view is not None:
        md._view = md._resolve_level(md._view)
    md._is_structured = True
    md._build_factors()
    # The second repairable drop, derived the same way: a name dropped for naming its rows
    # can only be a factor again if a reading gave it a vocabulary. Without this the
    # restored instance reports the same column in `factor_names` and in `dropped_factors`
    # at once, and `unusable` describes a factor that is being analysed.
    md._repaired |= {
        name for name, reasons in md._dropped_factors.items() if "cardinality_over_budget" in reasons
    } & md._factors


def _reject_dataset_mismatch(md: "Metadata", manifest: Mapping[str, Any]) -> None:
    """Refuse a dataset that is not the length the saved rows were built from.

    Not a checksum, and not meant to be: it costs nothing and catches the mistake that
    matters, which is loading one dataset's rows against another dataset. That
    misalignment is silent by nature — the rows are well-formed and the positions are in
    range, so every read succeeds and every answer is about the wrong items.

    Two kinds of file are exempt, both because their count is not a count of dataset
    items:

    - **Factors-only**, which counts distinct ``item_index`` values rather than dataset
      items, with nothing obliging those to agree with the length of a dataset someone
      later binds to it.
    - **Filtered**, whose count is the *surviving* items. A filtered instance goes on
      holding its whole dataset — that is what :attr:`~dataeval.Metadata.is_filtered`
      exists to warn about — so the two disagreeing is the normal state of affairs rather
      than the mistake this looks for.

    Raises
    ------
    ValueError
        When the bound dataset holds a different number of items.
    """
    dataset = md._dataset
    if dataset is None or manifest.get("task") not in TASK_STRUCTURERS or manifest.get("is_filtered"):
        return
    try:
        length = len(dataset)  # type: ignore[arg-type]
    except TypeError:
        return
    if length != int(manifest["count"]):
        raise ValueError(
            f"This metadata was saved for a dataset of {manifest['count']} item(s) and the one "
            f"given holds {length}. Loading rows against a different dataset misaligns every "
            "factor with the items it describes. Pass the dataset the metadata was built from.",
        )


def restore(md: "Metadata", path: Path | str) -> None:
    """Populate ``md`` from the file at ``path``, keeping the configuration it was built with.

    Everything the file describes is structure — rows, levels, edges, which factor sits
    where. Everything else the instance holds is the caller's question about that
    structure, and is left exactly as the constructor set it.

    Raises
    ------
    MetadataFormatError
        When the file is not readable as this version's metadata format.
    ValueError
        When a bound dataset holds a different number of items than the file was saved for.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = _read_manifest(archive)
            structurer = _structurer_from(manifest)
            _reject_schema_drift(manifest, structurer.levels)
            store = _store_from(archive, manifest, structurer.levels)
    except zipfile.BadZipFile as exc:
        raise MetadataFormatError(f"Not a dataeval metadata file: {exc}") from exc
    except KeyError as exc:
        raise MetadataFormatError(f"Metadata file is missing {exc}") from exc

    _reject_dataset_mismatch(md, manifest)
    _adopt_manifest(md, manifest, structurer, store)
    _logger.info(
        "Loaded metadata from %s written by dataeval %s (%s rows)",
        path,
        manifest.get("dataeval_version"),
        store.counts,
    )
