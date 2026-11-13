"""SATURATION reads the endpoint from the range an image declares in its own datum metadata."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats
from dataeval.quality import Outliers
from dataeval.utils.preprocessing import ChannelGroup

pytestmark = pytest.mark.required

TWELVE = np.full((1, 4, 4), 4095, dtype=np.uint16)
SIXTEEN = np.full((1, 4, 4), 65535, dtype=np.uint16)


class Sensors:
    """A dataset of images, each with its own metadata."""

    metadata = {"id": "sensors"}

    def __init__(self, images: list[np.ndarray], metadata: Sequence[Mapping[str, Any]] | None = None) -> None:
        self.images = images
        self.meta = metadata or [{} for _ in images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return self.images[i], np.array([1.0, 0.0]), {"id": i, **self.meta[i]}


def read(data: Any, **kw: Any) -> tuple[list[float], list[float], list[str]]:
    result = image_stats(data, stats=kw.pop("stats", ImageStats.SATURATION), **kw)
    details = result.get("details", {})
    return (
        np.asarray(result["stats"]["saturation"]).tolist(),
        np.asarray(details["saturation_upper"]).tolist(),
        np.asarray(details["saturation_upper_rule"]).tolist(),
    )


class TestTheEndpointComesFromTheImage:
    def test_a_declared_range_is_the_endpoint_and_without_one_the_dtype_is(self):
        assert read(Sensors([TWELVE], [{"value_range": (0, 4095)}])) == ([1.0], [4095.0], ["declared"])
        assert read(Sensors([TWELVE])) == ([0.0], [65535.0], ["dtype"])

    def test_each_image_is_read_against_its_own_endpoint(self):
        declared = [{"value_range": (0, 4095)}, {"value_range": (0, 65535)}]
        assert read(Sensors([TWELVE, SIXTEEN], declared)) == ([1.0, 1.0], [4095.0, 65535.0], ["declared", "declared"])
        assert read(Sensors([TWELVE, SIXTEEN])) == ([0.0, 1.0], [65535.0, 65535.0], ["dtype", "dtype"])

    def test_an_image_that_declares_none_beside_one_that_does_uses_its_dtype(self):
        got = read(Sensors([TWELVE, SIXTEEN], [{"value_range": (0, 4095)}, {}]))
        assert got == ([1.0, 1.0], [4095.0, 65535.0], ["declared", "dtype"])

    def test_the_default_request_reads_it_too(self):
        result = image_stats(Sensors([TWELVE], [{"value_range": (0, 4095)}]))
        assert result["stats"]["saturation"].tolist() == [1.0]

    def test_the_call_range_beats_the_metadata(self):
        data = Sensors([TWELVE], [{"value_range": (0, 65535)}])
        assert read(data, value_range=(0, 4095)) == ([1.0], [4095.0], ["declared"])

    def test_a_channel_groups_own_range_beats_both(self):
        data = Sensors([np.full((2, 4, 4), 4095, dtype=np.uint16)], [{"value_range": (0, 65535)}])
        stats = {None: ImageStats.SATURATION, "nir": ImageStats.SATURATION}
        result = image_stats(data, stats=stats, channels={"nir": ChannelGroup(1, value_range=(0, 4095))})
        assert result["stats"]["saturation"].tolist() == [0.0]
        assert result["stats"]["nir_saturation"].tolist() == [1.0]

    def test_a_malformed_range_raises_naming_the_image(self):
        with pytest.raises(ValueError, match=r"image 1.*value_range"):
            read(Sensors([TWELVE, TWELVE], [{}, {"value_range": (4095, 0)}]))


class TestInThePixelFamily:
    def test_normalizing_without_a_range_does_not_list_saturation_as_unmeasurable(
        self, caplog: pytest.LogCaptureFixture
    ):
        wide = np.random.default_rng(0).uniform(0, 1000, (3, 8, 8)).astype(np.float32)
        with caplog.at_level(logging.WARNING, logger="dataeval"):
            image_stats([wide], stats=ImageStats.PIXEL, normalize_pixel_values=True)
        said = [r.getMessage() for r in caplog.records if "no value range" in r.getMessage()]
        assert said
        assert all("PIXEL_SATURATION" not in m for m in said)

    def test_a_float_image_without_a_range_is_nan_and_says_why(self):
        got = image_stats([np.zeros((3, 4, 4), dtype=np.float32)], stats=ImageStats.PIXEL)
        assert np.isnan(got["stats"]["saturation"][0])
        assert got.get("details", {})["saturation_status"].tolist() == ["unknown_upper"]


# Each kind of image: how to make n of them, and the range to declare for them.
KINDS: dict[str, tuple[Any, tuple[float, float]]] = {
    "float32": (lambda rng: rng.random((3, 8, 8)).astype(np.float32), (0, 1)),
    "uint8": (lambda rng: rng.integers(0, 256, (3, 8, 8)).astype(np.uint8), (0, 255)),
    "twelve": (lambda rng: rng.integers(0, 4096, (3, 8, 8)).astype(np.uint16), (0, 4095)),
}


def pictures(kind: str, n: int = 24) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    return [KINDS[kind][0](rng) for _ in range(n)]


class TestDefaultOutliers:
    @pytest.mark.parametrize("kind", list(KINDS))
    @pytest.mark.parametrize("declare", [False, True])
    def test_runs_on_float_8_bit_and_12_bit_data_with_or_without_a_declared_range(self, kind: str, declare: bool):
        images = pictures(kind)
        declaration = {"value_range": KINDS[kind][1]} if declare else {}
        result = Outliers().evaluate(Sensors(images, [declaration for _ in images]))
        assert result.data().height >= 0

    def test_an_image_clipped_all_over_is_flagged_on_the_saturation_column(self):
        images = [*pictures("uint8"), np.full((3, 8, 8), 255, dtype=np.uint8)]
        flagged = Outliers().evaluate(images).data()
        assert (flagged["metric_name"] == "saturation").any()
