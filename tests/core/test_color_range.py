"""Color is read against a declared range, or 255 for an unsigned 8-bit datum, and is unknown for anything else."""

import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.core._color import srgb_to_lab
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import ChannelGroup

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

STATS = ["color_lightness_mean", "color_chroma_mean", "color_achromatic_fraction", "color_hue_cos", "color_hue_sin"]


def details_of(result: Any) -> Any:
    """The result's details, which a result carries only when a requested statistic reports some."""
    assert "details" in result
    return result["details"]


def read(data: Any, **kw: Any) -> dict[str, Any]:
    result = image_stats(data, stats=ImageStats.COLOR, color_rgb_channels=(0, 1, 2), **kw)
    return {
        **{name: result["stats"][name].tolist() for name in STATS},
        "status": details_of(result)["color_status"].tolist(),
    }


def picture(dtype: Any = np.uint8) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (3, 16, 16)).astype(dtype)


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


@pytest.mark.parametrize("maximum", [1, 100, 255])
def test_an_undeclared_uint8_reads_against_255_whatever_its_maximum(maximum: int):
    # The shared decode would read a maximum of 1 as a 1-bit image and make this red full red.
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    image[0] = maximum
    got = read([image])
    assert got["status"] == ["ok"]
    expected = srgb_to_lab(np.array([maximum / 255, 0.0, 0.0]))
    assert got["color_lightness_mean"][0] == pytest.approx(expected[0], rel=1e-5)
    assert got["color_chroma_mean"][0] == pytest.approx(np.hypot(expected[1], expected[2]), rel=1e-5)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("maximum", [1, 255, 3000])
def test_an_undeclared_other_dtype_is_unknown_range_and_the_log_says_so(
    dtype: Any, maximum: int, caplog: pytest.LogCaptureFixture
):
    image = np.zeros((3, 8, 8), dtype=dtype)
    image[0] = maximum
    with caplog.at_level(logging.WARNING, logger="dataeval"):
        got = read([image])
    assert got["status"] == ["unknown_range"]
    assert all(np.isnan(got[name][0]) for name in STATS)
    assert [r for r in caplog.records if "no value range was declared" in r.getMessage()]


def test_the_same_image_stored_four_ways_reads_the_same_once_declared():
    base = picture()
    four = [
        (base, {}),  # uint8: undeclared
        (base.astype(np.float64) / 255, {"value_range": (0, 1)}),
        (base.astype(np.float32), {"value_range": (0, 255)}),
        (base.astype(np.uint16) * 257, {"value_range": (0, 65535)}),
    ]
    results = [read([image], **kw) for image, kw in four]
    for result in results[1:]:
        for name in STATS:
            np.testing.assert_allclose(result[name], results[0][name], rtol=1e-6, err_msg=name)
        assert result["status"] == ["ok"]


def test_a_value_range_in_datum_metadata_is_read_and_the_call_beats_it():
    floats = (picture().astype(np.float32) / 255)[:, :, :]
    from_metadata = read(Sensors([floats], [{"value_range": (0, 1)}]))
    from_call = read([floats], value_range=(0, 1))
    assert from_metadata == from_call
    assert from_metadata["status"] == ["ok"]
    beaten = read(Sensors([floats], [{"value_range": (0, 65535)}]), value_range=(0, 1))
    assert beaten == from_call


def test_a_datum_with_no_range_beside_one_that_declares_one_reads_each_its_own_way():
    floats = picture().astype(np.float32) / 255
    got = read(Sensors([floats, floats], [{"value_range": (0, 1)}, {}]))
    assert got["status"] == ["ok", "unknown_range"]


def test_normalize_pixel_values_changes_no_color_column():
    base = read([picture()])
    assert read([picture()], normalize_pixel_values=True) == base


def test_a_datum_without_the_declared_channels_is_said_once_with_the_images(caplog: pytest.LogCaptureFixture):
    gray = np.full((1, 8, 8), 100, dtype=np.uint8)
    with caplog.at_level(logging.WARNING, logger="dataeval"):
        got = read([gray, picture(), gray])
    assert got["status"] == ["missing_band", "ok", "missing_band"]
    (message,) = [r.getMessage() for r in caplog.records if "channels color_rgb_channels names" in r.getMessage()]
    assert message.endswith("indices: 0, 2")


def test_a_declared_range_and_an_8_bit_datum_say_nothing(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="dataeval"), warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        read([picture()])
        read([picture().astype(np.float32)], value_range=(0, 255))
    assert not [r for r in caplog.records if "color" in r.getMessage()]


def test_a_channel_groups_range_does_not_set_colors_range(caplog: pytest.LogCaptureFixture):
    # Color reads the datum's own channels, so the range a group declares for its band view is not its range.
    floats = picture().astype(np.float32) / 255
    with caplog.at_level(logging.WARNING, logger="dataeval"):
        result = image_stats(
            [floats],
            stats={None: ImageStats.COLOR, "g": ImageStats.PIXEL_MEAN},
            channels={"g": ChannelGroup((0, 1, 2), value_range=(0, 1))},
            color_rgb_channels=(0, 1, 2),
        )
    assert details_of(result)["color_status"].tolist() == ["unknown_range"]
