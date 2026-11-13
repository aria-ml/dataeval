"""An image's own `value_range`, in its datum metadata, declares the interval its values occupy."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import ChannelGroup

pytestmark = pytest.mark.required

# Float data above 255 has no encoding to decode, so every statistic that reads a range is NaN until one is declared.
WIDE = np.random.default_rng(0).uniform(0, 1000, (3, 16, 16)).astype(np.float32)
READERS = ImageStats.VISUAL_BRIGHTNESS | ImageStats.PIXEL_ENTROPY | ImageStats.PIXEL_HISTOGRAM


class Items:
    """A dataset of images with one metadata mapping each, which counts how often each item is read."""

    metadata = {"id": "items"}

    def __init__(self, images: list[np.ndarray], metadata: Sequence[Mapping[str, Any]] | None = None) -> None:
        self.images = images
        self.meta = metadata or [{} for _ in images]
        self.reads: list[int] = []

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self.reads.append(i)
        return self.images[i], np.array([1.0, 0.0]), {"id": i, **self.meta[i]}


def stats_of(data: Any, **kw: Any) -> dict[str, list[Any]]:
    result = image_stats(data, stats=kw.pop("stats", READERS), **kw)
    return {name: np.asarray(values).tolist() for name, values in result["stats"].items()}


class TestDeclaredInTheMetadata:
    def test_a_datum_range_reads_like_the_keyword(self):
        from_metadata = stats_of(Items([WIDE], [{"value_range": (0, 1000)}]))
        from_keyword = stats_of([WIDE], value_range=(0, 1000))
        assert from_metadata == from_keyword
        assert not np.isnan(from_metadata["brightness"][0])

    def test_without_the_key_the_range_is_unknown_as_before(self):
        assert np.isnan(stats_of(Items([WIDE]), stats=ImageStats.VISUAL_BRIGHTNESS)["brightness"][0])

    def test_each_image_is_read_against_its_own_range(self):
        data = Items([WIDE, WIDE], [{"value_range": (0, 1000)}, {"value_range": (0, 2000)}])
        brightness = stats_of(data, stats=ImageStats.VISUAL_BRIGHTNESS)["brightness"]
        assert brightness[0] != brightness[1]
        for got, declared in zip(brightness, [(0, 1000), (0, 2000)], strict=True):
            assert got == stats_of([WIDE], value_range=declared, stats=ImageStats.VISUAL_BRIGHTNESS)["brightness"][0]

    def test_a_list_is_as_good_as_a_tuple(self):
        assert stats_of(Items([WIDE], [{"value_range": [0, 1000]}])) == stats_of([WIDE], value_range=(0, 1000))

    def test_an_iterable_of_arrays_has_no_metadata_to_read(self):
        assert np.isnan(stats_of([WIDE], stats=ImageStats.VISUAL_BRIGHTNESS)["brightness"][0])


class TestPrecedence:
    def test_the_call_range_beats_the_metadata(self):
        data = Items([WIDE], [{"value_range": (0, 2000)}])
        assert stats_of(data, value_range=(0, 1000)) == stats_of([WIDE], value_range=(0, 1000))

    def test_a_group_range_beats_both(self):
        image = np.concatenate([WIDE, WIDE])
        data = Items([image], [{"value_range": (0, 2000)}])
        brightness = ImageStats.VISUAL_BRIGHTNESS
        stats = {None: brightness, "a": brightness, "b": brightness}
        groups = {"a": ChannelGroup([0, 1, 2], value_range=(0, 1000)), "b": [3, 4, 5]}
        got = stats_of(data, stats=stats, channels=groups)
        keyword = stats_of([image], stats=stats, channels=groups, value_range=(0, 2000))
        assert got == keyword
        assert got["a_brightness"] != got["b_brightness"]


class TestMalformed:
    @pytest.mark.parametrize("bad", [5, (0,), (0, 1, 2), (0, float("nan")), (1000, 0), "0-1000", (0, float("inf"))])
    def test_a_malformed_range_raises_naming_the_image_and_the_key(self, bad: Any):
        data = Items([WIDE, WIDE], [{}, {"value_range": bad}])
        with pytest.raises(ValueError, match=r"image 1.*value_range|value_range.*image 1"):
            stats_of(data)

    def test_a_statistic_that_reads_no_range_does_not_check_it(self):
        data = Items([WIDE], [{"value_range": "nonsense"}])
        assert stats_of(data, stats=ImageStats.HASH | ImageStats.DIMENSION_WIDTH)["width"] == [16]

    def test_the_call_range_means_the_metadata_is_never_read(self):
        data = Items([WIDE], [{"value_range": "nonsense"}])
        assert stats_of(data, value_range=(0, 1000)) == stats_of([WIDE], value_range=(0, 1000))


class TestNothingElseChanges:
    def test_a_dataset_without_the_key_matches_the_same_images_as_a_list(self):
        images = [np.random.default_rng(i).integers(0, 255, (3, 12, 12)).astype(np.uint8) for i in range(4)]
        stats = ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.DIMENSION
        assert stats_of(Items(images), stats=stats) == stats_of(images, stats=stats)

    def test_the_key_does_not_add_reads_of_the_dataset(self):
        images = [WIDE, WIDE, WIDE]
        plain, keyed = Items(images), Items(images, [{"value_range": (0, 1000)}] * 3)
        stats_of(plain)
        stats_of(keyed)
        assert keyed.reads == plain.reads


def depth_of(image: np.ndarray, **kw: Any) -> float:
    return stats_of([image], stats=ImageStats.DIMENSION_DEPTH, **kw)["depth"][0]


def zeros(dtype: Any) -> np.ndarray:
    return np.zeros((1, 4, 4), dtype=dtype)


class TestDepthFromADeclaredRange:
    @pytest.mark.parametrize(
        ("dtype", "declared", "depth"),
        [
            (np.uint16, (0, 4095), 12),
            (np.uint16, (0, 65535), 16),
            (np.uint8, (0, 255), 8),
            (np.uint16, (0, 255), 8),
            (np.uint32, (0, 4294967295), 32),
            (np.bool_, (0, 1), 1),
        ],
    )
    def test_a_power_of_two_less_one_names_its_depth(self, dtype: Any, declared: Any, depth: int):
        assert depth_of(zeros(dtype), value_range=declared) == depth

    @pytest.mark.parametrize(
        ("dtype", "declared"),
        [
            (np.uint8, (0, 4095)),  # the dtype cannot hold it
            (np.uint16, (0, 1000)),  # not a power of two less one
            (np.uint16, (0, 1023)),  # a depth the decoder never reports
            (np.uint16, (1, 4095)),  # does not start at 0
            (np.float32, (0, 255)),  # floats carry no encoding
            (np.float32, (0, 1)),
        ],
    )
    def test_anything_else_stays_nan_as_before(self, dtype: Any, declared: Any):
        assert np.isnan(depth_of(zeros(dtype), value_range=declared))

    def test_an_undeclared_image_is_decoded_as_before(self):
        assert depth_of(np.full((1, 4, 4), 3000, dtype=np.uint16)) == 12

    def test_the_metadata_range_names_it_too(self):
        # Dark enough that the decoder alone would read 8 bits.
        data = Items([np.full((1, 4, 4), 100, dtype=np.uint16)], [{"value_range": (0, 4095)}])
        assert stats_of(data, stats=ImageStats.DIMENSION_DEPTH)["depth"] == [12]

    def test_no_other_statistic_changes_for_declaring_what_the_data_already_is(self):
        image = np.random.default_rng(1).integers(0, 255, (3, 12, 12)).astype(np.uint8)
        stats = ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.DIMENSION & ~ImageStats.DIMENSION_DEPTH
        assert stats_of([image], stats=stats, value_range=(0, 255)) == stats_of([image], stats=stats)
