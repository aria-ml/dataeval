import logging

import numpy as np
import pytest

from dataeval.utils._merge import flatten_metadata, merge_metadata


@pytest.mark.required
class TestUtilsMetadata:
    duplicate_keys = {
        "a": 1,
        "b": {
            "b1": "b1",
            "b2": "b2",
        },
        "c": {
            "d": [
                {"e": 1, "f": 2, "g": 3},
                {"e": 4, "f": 5, "g": 6},
                {"e": 7, "f": 8, "g": 9, "z": 0},
            ],
            "h": [1.1, 1.2, 1.3],
        },
        "d": {
            "d": {"e": 4, "f": 5, "g": 6},
            "h": 1,
        },
    }

    inconsistent_keys = [
        {"a": 1, "b": [1], "c": [1, 2]},
        {"a": 2},
        {"a": 3, "d": [{"e": {"f": [{"g": 1, "h": 2}]}}]},
    ]

    numpy_value = [{"time": np.array([1.2, 3.4, 5.6]), "altitude": [235, 6789, 101112], "point": 4}]

    voc_test = [
        {
            "annotation": {
                "folder": "VOC2011",
                "filename": "2008_000009.jpg",
                "source": {"database": "The VOC2008 Database", "annotation": "PASCAL VOC2008", "image": "flickr"},
                "size": {"width": "600", "height": "300", "depth": "3"},
                "segmented": "0",
                "object": [
                    {
                        "name": "cat",
                        "pose": "Unspecified",
                        "truncated": "0",
                        "occluded": "1",
                        "bndbox": {"xmin": "53", "ymin": "87", "xmax": "471", "ymax": "420"},
                        "difficult": "0",
                    },
                    {
                        "name": "dog",
                        "pose": "Unspecified",
                        "truncated": "1",
                        "occluded": "0",
                        "bndbox": {"xmin": "158", "ymin": "44", "xmax": "289", "ymax": "167"},
                        "difficult": "0",
                    },
                    {
                        "name": "person",
                        "pose": "Right",
                        "truncated": "1",
                        "occluded": "0",
                        "bndbox": {"xmin": "158", "ymin": "44", "xmax": "289", "ymax": "167"},
                        "difficult": "0",
                    },
                ],
            },
        },
        {
            "annotation": {
                "folder": "VOC2011",
                "filename": "2008_000036.jpg",
                "source": {"database": "The VOC2008 Database", "annotation": "PASCAL VOC2008", "image": "flickr"},
                "size": {"width": "500", "height": "375", "depth": "3"},
                "segmented": "0",
                "object": [
                    {
                        "name": "bicycle",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "0",
                        "bndbox": {"xmin": "120", "ymin": "1", "xmax": "203", "ymax": "35"},
                        "difficult": "0",
                    },
                    {
                        "name": "bicycle",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "117", "ymin": "38", "xmax": "273", "ymax": "121"},
                        "difficult": "0",
                    },
                    {
                        "name": "person",
                        "pose": "Left",
                        "truncated": "0",
                        "occluded": "0",
                        "bndbox": {"xmin": "206", "ymin": "74", "xmax": "395", "ymax": "237"},
                        "difficult": "0",
                        "part": [
                            {"name": "head", "bndbox": {"xmin": "321", "ymin": "75", "xmax": "359", "ymax": "122"}},
                            {"name": "foot", "bndbox": {"xmin": "205", "ymin": "183", "xmax": "240", "ymax": "222"}},
                            {"name": "foot", "bndbox": {"xmin": "209", "ymin": "208", "xmax": "250", "ymax": "237"}},
                            {"name": "hand", "bndbox": {"xmin": "371", "ymin": "204", "xmax": "396", "ymax": "219"}},
                        ],
                    },
                    {
                        "name": "boat",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "24", "ymin": "2", "xmax": "500", "ymax": "188"},
                        "difficult": "0",
                    },
                    {
                        "name": "boat",
                        "pose": "Left",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "1", "ymin": "187", "xmax": "500", "ymax": "282"},
                        "difficult": "0",
                    },
                ],
            },
        },
        {
            "annotation": {
                "folder": "VOC2011",
                "filename": "2008_000128.jpg",
                "source": {"database": "The VOC2008 Database", "annotation": "PASCAL VOC2008", "image": "flickr"},
                "size": {"width": "500", "height": "375", "depth": "3"},
                "segmented": "0",
                "object": [
                    {
                        "name": "sofa",
                        "pose": "Left",
                        "truncated": "0",
                        "occluded": "1",
                        "bndbox": {"xmin": "11", "ymin": "29", "xmax": "500", "ymax": "375"},
                        "difficult": "0",
                    },
                    {
                        "name": "person",
                        "pose": "Unspecified",
                        "truncated": "1",
                        "occluded": "1",
                        "bndbox": {"xmin": "1", "ymin": "85", "xmax": "361", "ymax": "375"},
                        "difficult": "0",
                        "part": [
                            {"name": "head", "bndbox": {"xmin": "243", "ymin": "88", "xmax": "358", "ymax": "225"}},
                            {"name": "hand", "bndbox": {"xmin": "168", "ymin": "209", "xmax": "216", "ymax": "257"}},
                            {"name": "hand", "bndbox": {"xmin": "94", "ymin": "252", "xmax": "128", "ymax": "308"}},
                        ],
                    },
                    {
                        "name": "person",
                        "pose": "Unspecified",
                        "truncated": "0",
                        "occluded": "1",
                        "bndbox": {"xmin": "92", "ymin": "173", "xmax": "212", "ymax": "357"},
                        "difficult": "0",
                    },
                ],
            },
        },
    ]

    def test_ignore_lists(self):
        a, d = merge_metadata([self.duplicate_keys], return_dropped=True, ignore_lists=True)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1],
            "b1": ["b1"],
            "b2": ["b2"],
            "e": [4],
            "f": [5],
            "g": [6],
            "h": [1],
        }
        assert d == {"c_d": ["nested_list"], "c_h": ["nested_list"]}

    def test_fully_qualified_keys(self):
        a, d = merge_metadata([self.duplicate_keys], return_dropped=True, fully_qualified=True)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1, 1, 1],
            "b_b1": ["b1", "b1", "b1"],
            "b_b2": ["b2", "b2", "b2"],
            "c_d_e": [1, 4, 7],
            "c_d_f": [2, 5, 8],
            "c_d_g": [3, 6, 9],
            "c_h": [1.1, 1.2, 1.3],
            "d_d_e": [4, 4, 4],
            "d_d_f": [5, 5, 5],
            "d_d_g": [6, 6, 6],
            "d_h": [1, 1, 1],
        }
        assert d == {"c_d_z": ["inconsistent_key"]}

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_duplicate_keys(self, return_numpy):
        a = merge_metadata([self.duplicate_keys], return_numpy=return_numpy)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1, 1, 1],
            "b1": ["b1", "b1", "b1"],
            "b2": ["b2", "b2", "b2"],
            "c_d_e": [1, 4, 7],
            "c_d_f": [2, 5, 8],
            "c_d_g": [3, 6, 9],
            "c_h": [1.1, 1.2, 1.3],
            "d_d_e": [4, 4, 4],
            "d_d_f": [5, 5, 5],
            "d_d_g": [6, 6, 6],
            "d_h": [1, 1, 1],
        }

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_inconsistent_keys(self, return_numpy):
        a, d = merge_metadata(self.inconsistent_keys, return_dropped=True, return_numpy=return_numpy)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1, 2, 3],
        }
        assert d == {"b": ["inconsistent_key"], "c": ["inconsistent_size"], "d_e_f": ["nested_list"]}

    def test_inconsistent_key(self):
        list_metadata = [{"common": 1, "target": [{"a": 1, "b": 3, "c": 5}, {"a": 2, "b": 4}], "source": "example"}]
        reorganized_metadata, dropped_keys = merge_metadata(list_metadata, return_dropped=True)
        assert reorganized_metadata == {
            "common": [1, 1],
            "a": [1, 2],
            "b": [3, 4],
            "source": ["example", "example"],
        }
        assert dropped_keys == {"target_c": ["inconsistent_key"]}

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_voc_test(self, return_numpy):
        a = merge_metadata(self.voc_test, return_numpy=return_numpy)
        assert {k: list(v) for k, v in a.items()} == {
            "folder": [
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
                "VOC2011",
            ],
            "filename": [
                "2008_000009.jpg",
                "2008_000009.jpg",
                "2008_000009.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000036.jpg",
                "2008_000128.jpg",
                "2008_000128.jpg",
                "2008_000128.jpg",
            ],
            "database": [
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
                "The VOC2008 Database",
            ],
            "annotation": [
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
                "PASCAL VOC2008",
            ],
            "image": [
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
                "flickr",
            ],
            "width": [600, 600, 600, 500, 500, 500, 500, 500, 500, 500, 500],
            "height": [300, 300, 300, 375, 375, 375, 375, 375, 375, 375, 375],
            "depth": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "segmented": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "name": [
                "cat",
                "dog",
                "person",
                "bicycle",
                "bicycle",
                "person",
                "boat",
                "boat",
                "sofa",
                "person",
                "person",
            ],
            "pose": [
                "Unspecified",
                "Unspecified",
                "Right",
                "Left",
                "Left",
                "Left",
                "Left",
                "Left",
                "Left",
                "Unspecified",
                "Unspecified",
            ],
            "truncated": [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
            "occluded": [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
            "xmin": [53, 158, 158, 120, 117, 206, 24, 1, 11, 1, 92],
            "ymin": [87, 44, 44, 1, 38, 74, 2, 187, 29, 85, 173],
            "xmax": [471, 289, 289, 203, 273, 395, 500, 500, 500, 361, 212],
            "ymax": [420, 167, 167, 35, 121, 237, 188, 282, 375, 375, 357],
            "difficult": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }

    @pytest.mark.filterwarnings("error")
    def test_flatten_metadata_no_dropped_no_warn(self):
        flatten_metadata({"a": {"b": 1, "c": 2}}, return_dropped=False)

    def test_flatten_metadata_no_dropped_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            flatten_metadata(self.inconsistent_keys[0], return_dropped=False)
        assert "Metadata entries were dropped" in caplog.text

    @pytest.mark.filterwarnings("error")
    def test_merge_metadata_no_dropped_no_warn(self):
        merge_metadata([{"a": {"b": 1, "c": 2}}], return_dropped=False)

    def test_merge_metadata_no_dropped_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            merge_metadata(self.inconsistent_keys, return_dropped=False)
        assert "Metadata entries were dropped" in caplog.text

    def test_handle_numpy(self):
        output, dropped = merge_metadata(self.numpy_value, return_dropped=True)
        assert output == {
            "time": [1.2, 3.4, 5.6],
            "altitude": [235, 6789, 101112],
            "point": [4, 4, 4],
        }
        assert dropped == {}

    def test_targets_per_image_mismatch(self):
        targets_per_image = [1]
        with pytest.raises(ValueError, match="Number of targets per image must be equal"):
            merge_metadata([{"a": 1}, {"a": 2}], targets_per_image=targets_per_image)

    def test_merge_metadata_drop_no_targets(self):
        merge_metadatad = merge_metadata([{"a": 1}, {"a": 2}, {"a": 3}], targets_per_image=[1, 0, 1])
        assert merge_metadatad["a"] == [1, 3]


@pytest.mark.required
class TestKeepPartialKeys:
    """A key some entries do not declare: dropped by default, kept with missing values on request."""

    def test_a_key_only_some_entries_declare_is_dropped_by_default(self):
        merged, dropped = merge_metadata([{"w": "sun"}, {"w": "rain"}, {}], return_dropped=True)
        assert "w" not in merged
        assert dropped == {"w": ["inconsistent_key"]}

    def test_keeping_it_gives_the_silent_entries_a_missing_value(self):
        merged, dropped = merge_metadata([{"w": "sun"}, {"w": "rain"}, {}], return_dropped=True, keep_partial=True)
        assert merged["w"] == ["sun", "rain", None]
        assert dropped == {}

    def test_a_key_first_seen_late_is_padded_backwards(self):
        """Otherwise it lines up against the wrong entries from the row it appears on."""
        merged = merge_metadata([{"a": 1}, {"a": 2, "b": 9}], keep_partial=True)
        assert merged["b"] == [None, 9]

    def test_the_values_that_were_recorded_keep_their_type(self):
        """A missing value reaching ``simplify_type`` came back as the string 'None', which
        then made the whole column a string column."""
        merged = merge_metadata([{"a": 1, "c": 9}, {"a": 2}], keep_partial=True)
        assert merged["c"] == [9, None]

    def test_padding_follows_the_targets_each_entry_contributes(self):
        merged = merge_metadata(
            [{"a": [1, 2], "c": 9}, {"a": [3, 4]}],
            targets_per_image=[2, 2],
            keep_partial=True,
        )
        assert merged["c"] == [9, 9, None, None]
        assert merged["a"] == [1, 2, 3, 4]

    def test_a_key_dropped_for_a_reason_of_its_own_stays_dropped(self):
        """Only absence is forgiven. A nested list has no usable values to keep."""
        merged, dropped = merge_metadata([{"n": [{"x": [[1, 2]]}], "a": 1}], return_dropped=True, keep_partial=True)
        assert "n_x" not in merged
        assert dropped

    def test_a_key_inconsistent_within_one_entry_is_still_dropped(self):
        """`dropped` names the full path and `merged` the shortened column, so `y` went
        looking for itself under `objs_y`, did not find it, and was rebuilt from padding —
        destroying the value the one target that recorded it actually held."""
        merged, dropped = merge_metadata(
            [{"objs": [{"x": 1, "y": 2}, {"x": 3}]}, {"objs": [{"x": 5, "y": 6}, {"x": 7, "y": 8}]}],
            return_dropped=True,
            keep_partial=True,
        )
        assert "y" not in merged
        assert dropped == {"objs_y": ["inconsistent_key"]}

    def test_a_key_whose_name_merely_ends_in_a_dropped_one_survives(self):
        """Matching the trailing segment rather than a bare substring keeps `y` from
        answering for `entropy`."""
        merged = merge_metadata([{"entropy": 1, "y": 2}, {"entropy": 3, "y": 4}], keep_partial=True)
        assert merged["entropy"] == [1, 3]

    def test_an_entry_contributing_no_rows_contributes_no_values(self):
        """Its scalars have nothing to attach to: appending them anyway advanced one column
        past the row count while `_image_index` stayed behind, so the merged columns
        described different numbers of rows."""
        merged = merge_metadata([{"a": [], "b": 1}, {"a": [7], "b": 2}], keep_partial=True)
        assert {len(v) for v in merged.values()} == {1}


@pytest.mark.required
class TestAColumnWhoseValuesDisagreeAboutTheirType:
    """Promoting numbers to text to unify a column is a loss, not a widening.

    ``simplify_type`` gives a column one type by promoting to the widest one present. Where
    that is text the promotion turns ``1.0`` into the *category* ``"1"``, so the column can
    no longer be binned, ordered or read as continuous and every bias evaluator scores it as
    a category set. It is dropped instead, and offered for repair.
    """

    def test_numbers_beside_text_are_dropped_rather_than_stringified(self):
        merged, dropped = merge_metadata(
            [{"direction": 1.0}, {"direction": "N"}, {"direction": 2.0}], return_dropped=True
        )
        assert "direction" not in merged
        assert dropped == {"direction": ["mixed_types"]}

    def test_a_column_of_numerals_reads_as_numbers(self):
        """Metadata through JSON is all text, so a column of counts is a column of numerals
        and has to keep working."""
        merged = merge_metadata([{"grade": "1"}, {"grade": "2"}, {"grade": "3"}])
        assert merged["grade"] == [1, 2, 3]

    def test_numerals_beside_a_word_are_the_same_problem_in_another_spelling(self):
        """``["1", "2", "many"]`` and ``[1.0, "N", 2.0]`` are one case: only some of the
        values read as numbers, and neither reading is one the library can pick."""
        merged, dropped = merge_metadata([{"grade": "1"}, {"grade": "2"}, {"grade": "many"}], return_dropped=True)
        assert "grade" not in merged
        assert dropped == {"grade": ["mixed_types"]}

    def test_a_numeral_beside_a_number_resolves_to_a_number(self):
        """Both read as numbers, so nothing is read as a category and nothing is lost."""
        merged = merge_metadata([{"n": "1"}, {"n": 2.0}])
        assert merged["n"] == [1, 2]

    @pytest.mark.parametrize("value", [np.int64(1), np.float64(1.0), True])
    def test_a_value_numpy_or_bool_still_counts_as_a_number(self, value):
        merged, dropped = merge_metadata([{"d": value}, {"d": "N"}], return_dropped=True)
        assert dropped == {"d": ["mixed_types"]}

    @pytest.mark.parametrize(
        "entries",
        [
            [{"d": 1}, {"d": 2.0}, {"d": 3}],
            [{"d": "sun"}, {"d": "rain"}],
            [{"d": "1"}, {"d": "2"}, {"d": "3"}],
            [{"objs": [{"x": 1}, {"x": 2}]}, {"objs": [{"x": 3}, {"x": 4}]}],
        ],
    )
    def test_a_column_that_agrees_with_itself_is_untouched(self, entries):
        _, dropped = merge_metadata(entries, return_dropped=True)
        assert dropped == {}
