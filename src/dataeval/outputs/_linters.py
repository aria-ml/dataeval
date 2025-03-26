from __future__ import annotations

__all__ = []

import contextlib
from dataclasses import dataclass
from typing import Generic, TypeVar, Union

from typing_extensions import TypeAlias

with contextlib.suppress(ImportError):
    import pandas as pd

from dataeval.outputs._base import Output
from dataeval.outputs._stats import DimensionStatsOutput, LabelStatsOutput, PixelStatsOutput, VisualStatsOutput

DuplicateGroup: TypeAlias = list[int]
DatasetDuplicateGroupMap: TypeAlias = dict[int, DuplicateGroup]
TIndexCollection = TypeVar("TIndexCollection", DuplicateGroup, DatasetDuplicateGroupMap)

IndexIssueMap: TypeAlias = dict[int, dict[str, float]]
OutlierStatsOutput: TypeAlias = Union[DimensionStatsOutput, PixelStatsOutput, VisualStatsOutput]
TIndexIssueMap = TypeVar("TIndexIssueMap", IndexIssueMap, list[IndexIssueMap])


@dataclass(frozen=True)
class DuplicatesOutput(Output, Generic[TIndexCollection]):
    """
    Output class for :class:`.Duplicates` lint detector.

    Attributes
    ----------
    exact : list[list[int] | dict[int, list[int]]]
        Indices of images that are exact matches
    near: list[list[int] | dict[int, list[int]]]
        Indices of images that are near matches

    Notes
    -----
    - For a single dataset, indices are returned as a list of index groups.
    - For multiple datasets, indices are returned as dictionaries where the key is the
      index of the dataset, and the value is the list index groups from that dataset.
    """

    exact: list[TIndexCollection]
    near: list[TIndexCollection]


def _reorganize_by_class_and_metric(result: IndexIssueMap, lstats: LabelStatsOutput):
    """Flip result from grouping by image to grouping by class and metric"""
    metrics = {}
    class_wise = {label: {} for label in lstats.class_names}

    # Group metrics and calculate class-wise counts
    for img, group in result.items():
        for extreme in group:
            metrics.setdefault(extreme, []).append(img)
            for i, images in enumerate(lstats.image_indices_per_class):
                if img in images:
                    class_wise[lstats.class_names[i]][extreme] = class_wise[lstats.class_names[i]].get(extreme, 0) + 1

    return metrics, class_wise


def _create_table(metrics, class_wise):
    """Create table for displaying the results"""
    max_class_length = max(len(str(label)) for label in class_wise) + 2
    max_total = max(len(metrics[group]) for group in metrics) + 2

    table_header = " | ".join(
        [f"{'Class':>{max_class_length}}"]
        + [f"{group:^{max(5, len(str(group))) + 2}}" for group in sorted(metrics.keys())]
        + [f"{'Total':<{max_total}}"]
    )
    table_rows = []

    for class_cat, results in class_wise.items():
        table_value = [f"{class_cat:>{max_class_length}}"]
        total = 0
        for group in sorted(metrics.keys()):
            count = results.get(group, 0)
            table_value.append(f"{count:^{max(5, len(str(group))) + 2}}")
            total += count
        table_value.append(f"{total:^{max_total}}")
        table_rows.append(" | ".join(table_value))

    table = [table_header] + table_rows
    return table


def _create_pandas_dataframe(class_wise):
    """Create data for pandas dataframe"""
    data = []
    for label, metrics_dict in class_wise.items():
        row = {"Class": label}
        total = sum(metrics_dict.values())
        row.update(metrics_dict)  # Add metric counts
        row["Total"] = total
        data.append(row)
    return data


@dataclass(frozen=True)
class OutliersOutput(Output, Generic[TIndexIssueMap]):
    """
    Output class for :class:`.Outliers` lint detector.

    Attributes
    ----------
    issues : dict[int, dict[str, float]] | list[dict[int, dict[str, float]]]
        Indices of image Outliers with their associated issue type and calculated values.

    - For a single dataset, a dictionary containing the indices of outliers and
      a dictionary showing the issues and calculated values for the given index.
    - For multiple stats outputs, a list of dictionaries containing the indices of
      outliers and their associated issues and calculated values.
    """

    issues: TIndexIssueMap

    def __len__(self) -> int:
        if isinstance(self.issues, dict):
            return len(self.issues)
        else:
            return sum(len(d) for d in self.issues)

    def to_table(self, labelstats: LabelStatsOutput) -> str:
        """
        Formats the outlier output results as a table.

        Parameters
        ----------
        labelstats : LabelStatsOutput
            Output of :func:`.labelstats`

        Returns
        -------
        str
        """
        if isinstance(self.issues, dict):
            metrics, classwise = _reorganize_by_class_and_metric(self.issues, labelstats)
            listed_table = _create_table(metrics, classwise)
            table = "\n".join(listed_table)
        else:
            outertable = []
            for d in self.issues:
                metrics, classwise = _reorganize_by_class_and_metric(d, labelstats)
                listed_table = _create_table(metrics, classwise)
                str_table = "\n".join(listed_table)
                outertable.append(str_table)
            table = "\n\n".join(outertable)
        return table

    def to_dataframe(self, labelstats: LabelStatsOutput) -> pd.DataFrame:
        """
        Exports the outliers output results to a pandas DataFrame.

        Parameters
        ----------
        labelstats : LabelStatsOutput
            Output of :func:`.labelstats`

        Returns
        -------
        pd.DataFrame

        Notes
        -----
        This method requires `pandas <https://pandas.pydata.org/>`_ to be installed.
        """
        import pandas as pd

        if isinstance(self.issues, dict):
            _, classwise = _reorganize_by_class_and_metric(self.issues, labelstats)
            data = _create_pandas_dataframe(classwise)
            df = pd.DataFrame(data)
        else:
            df_list = []
            for i, d in enumerate(self.issues):
                _, classwise = _reorganize_by_class_and_metric(d, labelstats)
                data = _create_pandas_dataframe(classwise)
                single_df = pd.DataFrame(data)
                single_df["Dataset"] = i
                df_list.append(single_df)
            df = pd.concat(df_list)
        return df
