import json
import warnings

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.exceptions import ShapeMismatchError
from dataeval.shift._drift._domain_spec import DomainSpec, item_values
from tests.conftest import MockODDataset

LOCALE = {"locale": {"western": ["France", "Germany", "Spain"], "russia": "Russia", "other": None}}


def _metadata(**factors):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Metadata.from_factors(factors)


@pytest.mark.required
class TestParse:
    def test_str_is_one_domain_per_value(self):
        spec = DomainSpec.parse("locale")
        assert spec.factor == "locale"
        assert spec.groups is None
        assert spec.canonical() == "locale"

    def test_named_form_keeps_names_and_order(self):
        spec = DomainSpec.parse(LOCALE)
        assert spec.groups is not None
        assert [name for name, _ in spec.groups] == ["western", "russia", "other"]
        assert spec.has_catch_all

    def test_shorthand_names(self):
        spec = DomainSpec.parse({"f": [["France", "Germany"], "Russia", (0, 10), (None, -5.5), None]})
        assert spec.groups is not None
        assert [name for name, _ in spec.groups] == ["France|Germany", "Russia", "[0, 10)", "[-inf, -5.5)", "other"]

    def test_numpy_values_become_plain_python(self):
        spec = DomainSpec.parse({"cam": {"a": np.int64(1), "b": np.array([2, 3]), "c": (np.float64(4.0), None)}})
        assert spec.groups is not None
        groups = dict(spec.groups)
        assert type(groups["a"]) is int
        assert groups["b"] == [2, 3]
        assert all(type(v) is int for v in groups["b"])
        assert type(groups["c"][0]) is float
        assert spec.assign([1, 3, 4.5, 3.0]) == ["a", "b", "c", None]

    def test_canonical_round_trips(self):
        spec = DomainSpec.parse({"alt": [(None, 100), (100, 1000), None]})
        assert DomainSpec.parse(spec.canonical()) == spec

    @pytest.mark.parametrize(
        ("spec", "match"),
        [
            ({"a": "x", "b": "y"}, "one factor"),
            ({"locale": {}}, "no groups"),
            ({"locale": []}, "no groups"),
            ({"locale": {"a": []}}, "empty list"),
            ({"locale": {"a": [["x"]]}}, "value or a"),
            ({"locale": {"a": {"x": 1}}}, "value or a"),
            ({"locale": {"a": ("France", "Germany", "Spain")}}, "use a list"),
            ({"locale": {"a": ("France", "Spain")}}, "use a list"),
            ({"locale": {"a": 1, "b": True}}, "equal in Python"),
            ({"locale": {"a": [1, 1.0]}}, "equal in Python"),
            ({"locale": {"a": "x", "b": ["x"]}}, "equal in Python"),
            ({"alt": {"a": [5], "b": (0, 10)}}, "falls in the range"),
            ({"alt": {"a": (0, 100), "b": (50, 200)}}, "overlap"),
            ({"locale": {"a": None, "b": None}}, "catch-all"),
            ({"locale": ["a", ["a"]]}, "named 'a'"),
            ({"locale": {"": "x"}}, "non-empty strings"),
            ({"locale": ("x", "y")}, "list"),
            ("", "factor name"),
            (3, "one factor"),
        ],
    )
    def test_invalid_specs_raise(self, spec, match):
        with pytest.raises(ValueError, match=match):
            DomainSpec.parse(spec)

    def test_adjacent_ranges_do_not_overlap(self):
        DomainSpec.parse({"alt": {"low": (0, 100), "high": (100, 200)}})


@pytest.mark.required
class TestAssign:
    def test_values_ranges_catch_all_and_missing(self):
        spec = DomainSpec.parse({"f": {"a": ["x", (0, 10)], "b": "y", "rest": None}})
        assert spec.assign(["x", 5, 10, "y", "z", None]) == ["a", "a", "rest", "b", "rest", None]

    def test_unmatched_without_catch_all_is_none(self):
        assert DomainSpec.parse({"f": {"a": "x"}}).assign(["x", "y"]) == ["a", None]

    def test_exact_match_is_type_sensitive(self):
        assert DomainSpec.parse({"f": {"a": 1}}).assign([1, 1.0]) == ["a", None]

    def test_str_form_names_values(self):
        assert DomainSpec.parse("f").assign([1, None, 2]) == ["1", None, "2"]

    def test_domain_order(self):
        assert DomainSpec.parse(LOCALE).domain_order(["Russia"]) == ("western", "russia", "other")
        assert DomainSpec.parse("f").domain_order(["c", None, "a", "c"]) == ("a", "c")


@pytest.mark.required
class TestJson:
    @pytest.mark.parametrize(
        "spec", [LOCALE, "locale", {"alt": {"low": (None, 100), "mid": [(100, 1000), 5000], "rest": None}}]
    )
    def test_round_trip_through_json_text(self, spec):
        parsed = DomainSpec.parse(spec)
        assert DomainSpec.from_json(json.loads(json.dumps(parsed.to_json()))) == parsed


@pytest.mark.required
class TestItemValues:
    def test_values_in_item_order(self):
        assert item_values(_metadata(t=["a", "b", "c"]), "t", 3, "caller") == ["a", "b", "c"]

    def test_rows_out_of_dataset_order_are_reordered(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = Metadata.from_factors({"t": ["b", "a", "c"]}, class_labels=[0, 0, 0], item_indices=[1, 0, 2])
        assert item_values(md, "t", 3, "caller") == ["a", "b", "c"]

    def test_misaligned_rows_raise(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = Metadata.from_factors({"cam": [1, 2, 3, 4]}, class_labels=[0, 0, 0, 0], item_indices=[2, 0, 1, 0])
        with pytest.raises(ShapeMismatchError):
            item_values(md, "cam", 4, "caller")

    def test_sample_count_mismatch_raises(self):
        with pytest.raises(ShapeMismatchError):
            item_values(_metadata(t=["a", "b", "c"]), "t", 4, "caller")

    def test_filtered_metadata_raises(self):
        md = _metadata(t=["a", "b", "c"]).where(pl.col("t") == "a")
        with pytest.raises(ValueError, match="filtered"):
            item_values(md, "t", 1, "caller")

    def test_missing_factor_raises(self):
        with pytest.raises(ValueError, match="no factor 'place'"):
            item_values(_metadata(t=["a", "b", "c"]), "place", 3, "caller")

    def test_factor_recorded_per_target_raises(self):
        images = list(np.zeros((4, 1, 4, 4), dtype=np.float32))
        boxes = [[[0, 0, 1, 1], [0, 0, 2, 2]], [[0, 0, 1, 1]], [[0, 0, 1, 1], [1, 1, 2, 2]], [[0, 0, 1, 1]]]
        meta = [{"size": [1, 2]}, {"size": [3]}, {"size": [4, 5]}, {"size": [6]}]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = Metadata(MockODDataset(images, [[0, 1], [0], [1, 1], [0]], boxes, metadata=meta))  # type: ignore
        with pytest.raises(ValueError, match="recorded per"):
            item_values(md, "size", 4, "caller")
