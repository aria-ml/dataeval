from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from dataeval.data._metadata import Metadata
from dataeval.metrics.stats._labelstats import labelstats


def get_metadata(label_array: list[list[int]]) -> Metadata:
    mock = MagicMock(spec=Metadata)
    index2label = {}
    class_labels = []
    item_indices = []
    for i, labels in enumerate(label_array):
        class_labels.extend(labels)
        item_indices.extend([i] * len(labels))
        for label in labels:
            if label not in index2label:
                index2label[label] = str(label)
    mock.dataframe = pl.from_dict({"item_index": item_indices, "class_label": class_labels})
    mock.class_labels = np.asarray(class_labels)
    mock.item_indices = np.asarray(item_indices)
    mock.item_count = len(label_array)
    mock.class_names = sorted(index2label.values())
    return mock


@pytest.mark.required
class TestLabelStats:
    def test_labelstats_list_int(self):
        label_array = [[0, 0, 0, 0, 0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        metadata = get_metadata(label_array)
        stats = labelstats(metadata)

        assert stats.label_counts_per_class == {0: 8, 1: 3, 2: 2, 3: 1}
        assert stats.class_count == 4
        assert stats.label_count == 14
        assert stats.image_indices_per_class == {0: [0, 1, 2, 3], 1: [1, 2, 3], 2: [2, 3], 3: [3]}
        assert stats.image_counts_per_class == {0: 4, 1: 3, 2: 2, 3: 1}
        assert stats.label_counts_per_image == [5, 2, 3, 4]
        assert stats.image_count == 4

    def test_labelstats_empty_target(self):
        label_array = [[0, 0, 0, 0, 0], [], [0, 1, 2, 3], []]
        metadata = get_metadata(label_array)
        stats = labelstats(metadata)

        assert stats.label_counts_per_class == {0: 6, 1: 1, 2: 1, 3: 1}
        assert stats.class_count == 4
        assert stats.label_count == 9
        assert stats.image_indices_per_class == {0: [0, 2], 1: [2], 2: [2], 3: [2]}
        assert stats.image_counts_per_class == {0: 2, 1: 1, 2: 1, 3: 1}
        assert stats.label_counts_per_image == [5, 0, 4, 0]
        assert stats.image_count == 4

    def test_labelstats_to_dataframe(self):
        label_array = [[0, 0, 0, 0, 0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        metadata = get_metadata(label_array)
        stats = labelstats(metadata)
        stats_df = stats.to_dataframe()
        assert stats_df.shape == (4, 3)

    def test_labelstats_to_table(self):
        label_array = [[0, 0, 0, 0, 0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        stats = labelstats(get_metadata(label_array))
        assert stats is not None
        table_result = stats.to_table()
        assert isinstance(table_result, str)
