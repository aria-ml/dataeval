from __future__ import annotations

__all__ = []

from typing import Any, TypeVar

import polars as pl

from dataeval.data._metadata import Metadata
from dataeval.outputs import LabelStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import AnnotatedDataset

TValue = TypeVar("TValue")


@set_metadata
def labelstats(dataset: Metadata | AnnotatedDataset[Any]) -> LabelStatsOutput:
    """
    Calculates :term:`statistics<Statistics>` for data labels.

    This function computes counting metrics (e.g., total per class, total per image)
    on the labels.

    Parameters
    ----------
    dataset : Metadata or ImageClassificationDataset or ObjectDetect

    Returns
    -------
    LabelStatsOutput
        A dataclass containing the computed counting metrics for the labels.

    Examples
    --------
    Calculate basic :term:`statistics<Statistics>` on labels for a dataset.

    >>> from dataeval.data import Metadata
    >>> stats = labelstats(Metadata(dataset))
    >>> print(stats.to_table())
    Class Count: 5
    Label Count: 15
    Average # Labels per Image: 1.88
    --------------------------------------
      Label: Total Count - Image Count
      horse:      2      -      2
        cow:      4      -      3
      sheep:      2      -      2
        pig:      2      -      2
    chicken:      5      -      5
    """
    metadata = Metadata(dataset) if isinstance(dataset, AnnotatedDataset) else dataset
    metadata_df = metadata.dataframe

    # Count occurrences of each label across all images
    label_counts_df = metadata_df.group_by("class_label").len()
    label_counts = dict(zip(label_counts_df["class_label"], label_counts_df["len"]))

    # Count unique images per label (how many images contain each label)
    image_counts_df = metadata_df.select(["image_index", "class_label"]).unique().group_by("class_label").len()
    image_counts = dict(zip(image_counts_df["class_label"], image_counts_df["len"]))

    # Create index_location mapping (which images contain each label)
    index_location: dict[int, list[int]] = {}
    for row in metadata_df.group_by("class_label").agg(pl.col("image_index")).to_dicts():
        indices = row["image_index"]
        index_location[row["class_label"]] = sorted(dict.fromkeys(indices)) if isinstance(indices, list) else [indices]

    # Count labels per image
    label_per_image_df = metadata_df.group_by("image_index").agg(pl.len().alias("label_count"))

    # Join with all indices to include missing ones with 0 count
    all_indices = pl.DataFrame({"image_index": range(metadata.image_count)})
    complete_label_df = all_indices.join(label_per_image_df, on="image_index", how="left").fill_null(0)
    label_per_image = complete_label_df.sort("image_index")["label_count"].to_list()

    return LabelStatsOutput(
        label_counts_per_class=label_counts,
        label_counts_per_image=label_per_image,
        image_counts_per_class=image_counts,
        image_indices_per_class=index_location,
        image_count=len(label_per_image),
        class_count=len(metadata.class_names),
        label_count=sum(label_counts.values()),
        class_names=metadata.class_names,
    )
