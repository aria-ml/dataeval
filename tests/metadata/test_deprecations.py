"""Level-spelling fallout: retired names raise rather than deprecate, and what the
compatibility surface keeps."""

import copy

import numpy as np
import pytest

from dataeval import Metadata
from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget
from tests.metadata.test_structurers import _SHAPES, _mot_dataset


def _od_metadata() -> Metadata:
    """Object detection over 3 images with 2, 1 and 2 detections."""
    counts = (2, 1, 2)
    targets = [
        ObjectDetectionTarget(
            np.tile(np.array([[1.0, 1.0, 2.0, 2.0]]), (count, 1)),
            np.arange(count),
            np.full(count, 0.5),
        )
        for count in counts
    ]
    dataset = MockDataset(
        np.zeros((3, 3, 16, 16)),
        targets,
        [{"weather": value} for value in ("sun", "rain", "sun")],
    )
    return Metadata(dataset)


def _ic_metadata() -> Metadata:
    dataset = MockDataset(
        np.zeros((4, 3, 16, 16)),
        np.eye(4, 2)[[0, 1, 0, 1]],
        [{"weather": value} for value in ("sun", "rain", "sun", "rain")],
    )
    return Metadata(dataset)


@pytest.mark.required
class TestRetiredSpellingsNoLongerResolve:
    """The v1.1 level spellings stopped resolving in v1.2: they raise, they do not deprecate."""

    def test_target_is_unknown_on_od(self, recwarn):
        md = _od_metadata()
        with pytest.raises(ValueError, match="Unknown level 'target'"):
            md.rows_at("target")  # type: ignore[arg-type]
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_target_is_unknown_as_a_factor_destination(self, recwarn):
        md = _od_metadata()
        with pytest.raises(ValueError, match="Unknown level 'target'"):
            md.add_factors({"a": np.arange(5.0)}, level="target")  # type: ignore[arg-type]
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_image_is_unknown_on_od(self, recwarn):
        md = _od_metadata()
        with pytest.raises(ValueError, match="Unknown level 'image'"):
            md.rows_at("image")  # type: ignore[arg-type]
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_image_is_unknown_as_a_view(self, recwarn):
        md = _ic_metadata()
        with pytest.raises(ValueError, match="Unknown level 'image'"):
            md.view = "image"  # type: ignore[arg-type]
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_add_factors_requires_a_stated_destination(self, recwarn):
        """Inference from array length is gone; `level` and `source_index` are the two ways."""
        md = _od_metadata()
        with pytest.raises(ValueError, match="destination"):
            md.add_factors({"a": np.arange(md.level_counts["unit"], dtype=np.float64)})
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_from_factors_rejects_a_retired_spelling(self):
        with pytest.raises(ValueError, match="Unknown level.*'image'"):
            Metadata.from_factors({"a": np.array([0, 1, 0])}, level="image")  # type: ignore[arg-type]


@pytest.mark.required
class TestDeepCopy:
    """A Metadata carries a schema and a layout, and must stay copyable."""

    @pytest.mark.parametrize("build", [_ic_metadata, _od_metadata], ids=["IC", "OD"])
    def test_deepcopy_structures_identically(self, build):
        md = build()
        md._structure()

        clone = copy.deepcopy(md)

        assert clone.levels == md.levels
        assert clone.level_counts == md.level_counts
        assert clone.dataframe.equals(md.dataframe)
        np.testing.assert_array_equal(clone.class_labels, md.class_labels)
        np.testing.assert_array_equal(clone.item_indices, md.item_indices)

    @pytest.mark.parametrize("build", [_ic_metadata, _od_metadata], ids=["IC", "OD"])
    def test_copy_is_independent(self, build):
        """The regression this guards: deep-copy, then mutate, without touching the original."""
        md = build()
        clone = copy.deepcopy(md)
        clone.add_factors({"brightness": np.arange(clone.level_counts["unit"], dtype=float)}, level="unit")

        assert "brightness" in clone.dataframe.columns
        assert "brightness" not in md.dataframe.columns

    def test_deepcopy_before_structuring(self):
        md = _od_metadata()
        clone = copy.deepcopy(md)
        assert clone.levels == ("unit", "instance")
        assert clone.dataframe.height == md.dataframe.height


@pytest.mark.required
class TestLevelMessagesNameTheUnitType:
    """An unknown level is explained in the dataset's own vocabulary."""

    def test_unknown_level_names_the_unit_type(self):
        md = _od_metadata()
        with pytest.raises(ValueError, match="this dataset's units are images"):
            md.rows_at("frame")  # type: ignore[arg-type]

    def test_unknown_level_on_mot_names_frames(self):
        md = Metadata(_mot_dataset(_SHAPES))
        with pytest.raises(ValueError, match="this dataset's units are frames"):
            md.rows_at("clip")  # type: ignore[arg-type]

    def test_unknown_level_still_lists_the_levels(self):
        md = _od_metadata()
        with pytest.raises(ValueError, match="Available levels are"):
            md.rows_at("frame")  # type: ignore[arg-type]

    def test_unknown_level_on_factors_only_names_items(self):
        factors = {"weather": np.array([0, 1, 0, 1])}
        md = Metadata.from_factors(factors)
        with pytest.raises(ValueError, match="this dataset's units are items"):
            md.rows_at("bogus")  # type: ignore[arg-type]
