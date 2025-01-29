import pytest

from dataeval.utils.metadata import _convert_type, _try_cast, flatten, merge


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
            }
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
            }
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
            }
        },
    ]

    dict_of_dicts = {
        "sample1": {
            "a": {
                "that": 37,
                "this": 43,
            },
            "b": {
                "that": 37,
                "this": 43,
            },
            "c": "today",
            "d": [1, 2, 3, 4],
            "e": {
                "when": 0.4,
                "what": 23,
            },
        },
        "sample2": {
            "a": {
                "that": 3,
                "this": 3,
            },
            "b": {
                "that": 7,
                "this": 4,
            },
            "c": "yesterday",
            "d": [2, 3, 4],
            "e": {
                "when": 14.4,
                "what": 2.3,
            },
        },
        "sample3": {
            "a": {
                "that": 3.7,
                "this": 4.3,
            },
            "b": {
                "that": 0.37,
                "this": 0.43,
            },
            "c": "tomorrow",
            "d": [0.1, 0.2, 0.3, 4],
            "e": {
                "when": 4,
                "what": 0.23,
            },
        },
        "sample4": {
            "a": {
                "that": 137,
                "this": 143,
            },
            "b": {
                "that": 100,
            },
            "c": "today",
            "d": 4,
            "e": {
                "when": "75",
                "what": "14",
            },
        },
    }

    def test_ignore_lists(self):
        a, d = merge([self.duplicate_keys], return_dropped=True, ignore_lists=True)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1],
            "b1": ["b1"],
            "b2": ["b2"],
            "e": [4],
            "f": [5],
            "g": [6],
            "h": [1],
            "_image_index": [0],
        }
        assert d == {"c_d": ["nested_list"], "c_h": ["nested_list"]}

    def test_fully_qualified_keys(self):
        a, d = merge([self.duplicate_keys], return_dropped=True, fully_qualified=True)
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
            "_image_index": [0, 0, 0],
        }
        assert d == {"c_d_z": ["inconsistent_key"]}

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_duplicate_keys(self, return_numpy):
        a = merge([self.duplicate_keys], return_numpy=return_numpy)
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
            "_image_index": [0, 0, 0],
        }

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_inconsistent_keys(self, return_numpy):
        a, d = merge(self.inconsistent_keys, return_dropped=True, return_numpy=return_numpy)
        assert {k: list(v) for k, v in a.items()} == {
            "a": [1, 2, 3],
            "_image_index": [0, 1, 2],
        }
        assert d == {"b": ["inconsistent_key"], "c": ["inconsistent_size"], "d_e_f": ["nested_list"]}

    def test_inconsistent_key(self):
        list_metadata = [{"common": 1, "target": [{"a": 1, "b": 3, "c": 5}, {"a": 2, "b": 4}], "source": "example"}]
        reorganized_metadata, dropped_keys = merge(list_metadata, return_dropped=True)
        assert reorganized_metadata == {
            "common": [1, 1],
            "a": [1, 2],
            "b": [3, 4],
            "source": ["example", "example"],
            "_image_index": [0, 0],
        }
        assert dropped_keys == {"target_c": ["inconsistent_key"]}

    @pytest.mark.parametrize("return_numpy", [False, True])
    def test_voc_test(self, return_numpy):
        a = merge(self.voc_test, return_numpy=return_numpy)
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
            "_image_index": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
        }

    def test_dict_of_dicts(self):
        output, dropped = merge(self.dict_of_dicts, return_dropped=True)  # type: ignore
        assert output == {
            "keys": ["sample1", "sample2", "sample3", "sample4"],
            "a_that": [37.0, 3.0, 3.7, 137.0],
            "b_that": [37.0, 7.0, 0.37, 100.0],
            "c": ["today", "yesterday", "tomorrow", "today"],
            "when": [0.4, 14.4, 4.0, 75.0],
            "what": [23.0, 2.3, 0.23, 14.0],
            "_image_index": [0, 1, 2, 3],
        }
        assert dropped == {
            "a_this": ["inconsistent_key"],
            "b_this": ["inconsistent_key"],
            "d": ["inconsistent_key", "nested_list"],
            "this": ["inconsistent_key"],
        }

    @pytest.mark.parametrize(
        "value, target, output",
        (
            ("123", int, 123),
            ("123", float, 123.0),
            ("123", str, "123"),
            ("12.3", int, None),
            ("12.3", float, 12.3),
            ("12.3", str, "12.3"),
            ("foo", int, None),
            ("foo", float, None),
            ("foo", str, "foo"),
        ),
    )
    def test_try_cast_(self, value, target, output):
        assert output == _try_cast(value, target)

    @pytest.mark.parametrize(
        "value, output",
        (
            ("123", 123),
            ("12.3", 12.3),
            ("foo", "foo"),
            ([123, "12.3"], [123.0, 12.3]),
            ([123, "foo"], ["123", "foo"]),
            (["123", "456"], [123, 456]),
        ),
    )
    def test_convert_type(self, value, output):
        assert output == _convert_type(value)

    @pytest.mark.filterwarnings("error")
    def test_flatten_no_dropped_no_warn(self):
        flatten({"a": {"b": 1, "c": 2}}, return_dropped=False)

    def test_flatten_no_dropped_warns(self):
        with pytest.warns(UserWarning, match=r"Metadata entries were dropped"):
            flatten(self.inconsistent_keys[0], return_dropped=False)

    @pytest.mark.filterwarnings("error")
    def test_merge_no_dropped_no_warn(self):
        merge([{"a": {"b": 1, "c": 2}}], return_dropped=False)

    def test_merge_no_dropped_warns(self):
        with pytest.warns(UserWarning, match=r"Metadata entries were dropped"):
            merge(self.inconsistent_keys, return_dropped=False)
