"""The surround is read against one interval for the box and its ring, from every place a range can be declared."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import ChannelGroup

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

CONTEXT = ImageStats.CONTEXT
BOX = (10, 10, 40, 40)  # a 30 x 30 box: its window is (2, 2, 48, 48), so 1,216 candidate pixels with no peer
CANDIDATE = 1216
SHAPE = (3, 60, 80)
WIDE_NAMES = "CONTEXT_SURROUND_MEAN, CONTEXT_SURROUND_STD"


def details_of(result: Any) -> Any:
    """The result's details, which a result carries only when a requested statistic reports some."""
    assert "details" in result
    return result["details"]


def run(image: Any, **kw: Any) -> dict[str, Any]:
    return image_stats([image], boxes=[[BOX]], stats=CONTEXT, per_image=False, **kw)  # type: ignore[arg-type]


def pattern(high: int = 191, offset: int = 10) -> np.ndarray:
    """Values that vary across the picture and across channels, with extremes that no encoding's own range has."""
    y, x = np.mgrid[0:60, 0:80]
    return np.stack([offset + (3 * y + 7 * x + 11 * c) % high for c in range(3)])


def messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records]


class TestDeclaredRange:
    def test_a_value_above_the_range_and_a_nan_are_not_measurable_and_nothing_is_clipped(self):
        image = np.full(SHAPE, 0.5)
        image[0, 2, 2] = 1.5  # in the ring, above the declared range, in one channel only
        image[1, 2, 3] = np.nan
        result = run(image, value_range=(0.0, 1.0))
        details = details_of(result)
        assert result["stats"]["surround_mean"][0] == pytest.approx(
            0.5, rel=1e-6
        )  # a clipped 1.5 would move it; float32 column
        assert details["surround_measurable_pixels"][0] == CANDIDATE - 2
        assert details["surround_valid_fraction"][0] == pytest.approx((CANDIDATE - 2) / CANDIDATE, abs=1e-12)
        assert details["context_range_high"][0] == 1.0

    def test_an_infinite_pixel_with_no_declared_range_is_unknown_and_the_warning_names_the_flags(self, caplog):
        image = np.full(SHAPE, 0.5)
        image[0, 55, 75] = np.inf  # outside the window: the range is read off the whole datum
        with caplog.at_level(logging.WARNING, logger="dataeval"):
            result = run(image)
        assert details_of(result)["context_status"][0] == "unknown_range"
        assert np.isnan(result["stats"]["surround_mean"][0])
        assert any(f"no value range could be established, so {WIDE_NAMES} are NaN" in m for m in messages(caplog))

    def test_an_all_nan_image_reads_unknown_range_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="dataeval"):
            result = run(np.full(SHAPE, np.nan))
        assert details_of(result)["context_status"][0] == "unknown_range"
        assert any("no value range could be established" in m for m in messages(caplog))


class TestEncodings:
    def test_uint8_declared_uint16_and_float_agree_whatever_the_pixels_own_extremes_are(self):
        # The pixels run from 10 to 200, never 0 or 255, so a range read off the view or the box would read differently.
        # The columns are float32, as every statistic's is, so the agreement here is to its resolution; the calculator's
        # own float64 agreement to 1e-12 is tested in `test_context_surround.py`.
        u8 = pattern().astype(np.uint8)
        assert (u8.min(), u8.max()) == (10, 200)
        means = {
            "uint8": run(u8),
            "uint16": run(u8.astype(np.uint16), value_range=(0, 255)),
            "float": run(u8 / 255.0),
        }
        reference = means["uint8"]
        for name, result in means.items():
            for stat in ("surround_mean", "surround_std"):
                assert result["stats"][stat][0] == pytest.approx(reference["stats"][stat][0], rel=1e-6), (name, stat)
        assert details_of(reference)["context_range_high"][0] == 255.0

    def test_undeclared_uint16_pictures_record_the_range_each_decoded_to_and_differ_by_that_ratio(self):
        a, b = pattern(2800, 100).astype(np.uint16), pattern(2800, 100).astype(np.uint16)
        a[0, 0, 0], b[0, 0, 0] = 3000, 5000  # one pixel outside the window: it only decides the decoded range
        result = image_stats([a, b], boxes=[[BOX], [BOX]], stats=CONTEXT, per_image=False)  # type: ignore[arg-type]
        assert details_of(result)["context_range_high"].tolist() == [4095.0, 65535.0]
        mean_a, mean_b = result["stats"]["surround_mean"]
        assert mean_a / mean_b == pytest.approx(65535 / 4095, rel=1e-6)  # float32 columns


class Sensors:
    """A dataset of images, each with its own metadata, for the range an image declares for itself."""

    metadata = {"id": "sensors"}

    def __init__(self, images: list[np.ndarray], metadata: Sequence[Mapping[str, Any]]) -> None:
        self.images, self.meta = images, metadata

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return self.images[i], np.array([1.0, 0.0]), {"id": i, **self.meta[i]}


def test_a_range_declared_in_the_datum_metadata_is_read_for_this_request():
    # The values peak at 2,900, which decodes to 4,095. Declared as 16 bits, the same pixels read 65,535, so only a
    # metadata range that was read can say so, and `_reads_a_range` must hold these flags for it to be.
    image = pattern(2800, 100).astype(np.uint16)
    result = image_stats(Sensors([image], [{"value_range": (0, 65535)}]), boxes=[[BOX]], stats=CONTEXT, per_image=False)
    assert details_of(result)["context_range_high"].tolist() == [65535.0]


def cube(bands: int = 4) -> np.ndarray:
    image = np.full((bands, 60, 80), 1000, dtype=np.uint16)
    image[3:] = 3000
    return image


class TestBands:
    GROUP = {"nir": ChannelGroup(3, value_range=(0, 4095))}

    def test_a_band_view_has_its_own_columns_and_no_geometry_details(self):
        result = image_stats(
            [cube()], boxes=[[BOX]], stats=CONTEXT, per_image=False, channels=self.GROUP, value_range=(0, 65535)
        )
        stats, details = result["stats"], details_of(result)
        assert stats["nir_surround_mean"][0] == pytest.approx(3000 / 4095, rel=1e-6)
        assert stats["surround_mean"][0] == pytest.approx((3 * 1000 + 3000) / 4 / 65535, rel=1e-6)  # the whole cube
        assert details["nir_context_status"].tolist() == ["ok"]
        for name in (
            "surround_measurable_pixels",
            "surround_valid_fraction",
            "target_valid_fraction",
            "context_range_low",
            "context_range_high",
        ):
            assert f"nir_{name}" in details, name
        assert details["nir_context_range_high"][0] == 4095.0
        for name in ("surround_candidate_pixels", "surround_retained_pixels", "ring_peer_excluded_fraction"):
            assert name in details
            assert f"nir_{name}" not in details, name

    def test_a_group_naming_a_band_the_datum_lacks_is_insufficient_support_and_the_warning_fires(self, caplog):
        group = {"nir": ChannelGroup(7, value_range=(0, 4095))}
        with caplog.at_level(logging.WARNING, logger="dataeval"):
            result = image_stats(
                [cube()], boxes=[[BOX]], stats=CONTEXT, per_image=False, channels=group, value_range=(0, 65535)
            )
        assert details_of(result)["nir_context_status"].tolist() == ["insufficient_support"]
        assert details_of(result)["nir_surround_measurable_pixels"].tolist() == [0]
        assert np.isnan(result["stats"]["nir_surround_mean"][0])
        assert any("names band 7 but the datum carries 4" in m for m in messages(caplog))

    def test_a_group_naming_a_band_the_datum_lacks_is_insufficient_support_with_no_declared_range_too(self, caplog):
        # Nothing was read for want of the band, which is the reason to give, and not that no interval could be found
        # for the all-NaN stand-in that stands in for it.
        cube = np.random.default_rng(1605).random((5, 60, 80))
        with caplog.at_level(logging.WARNING, logger="dataeval"):
            result = image_stats(
                [cube], boxes=[[BOX]], stats=CONTEXT, per_image=False, channels={"bad": ChannelGroup(7)}
            )
        assert details_of(result)["bad_context_status"].tolist() == ["insufficient_support"]
        assert details_of(result)["bad_surround_measurable_pixels"].tolist() == [0]
        assert details_of(result)["context_status"].tolist() == ["ok"]  # the unnamed view reads the cube
        assert any("names band 7 but the datum carries 5" in m for m in messages(caplog))
        assert not any("no value range could be established for channel group" in m for m in messages(caplog))

    def test_more_than_three_bands_in_the_unnamed_view_warn_and_a_group_that_covers_it_silences_it(self, caplog):
        with caplog.at_level(logging.WARNING, logger="dataeval"):
            run(cube(), value_range=(0, 65535))
        assert any(f"4 bands measured as a single picture for {WIDE_NAMES}" in m for m in messages(caplog))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="dataeval"):
            run(cube(), value_range=(0, 65535), channels=self.GROUP)
        assert not any("measured as a single picture" in m for m in messages(caplog))


class TestShapes:
    def test_a_two_dimensional_datum_measures_as_one_channel(self):
        flat = np.random.default_rng(1605).random((60, 80))
        assert run(flat)["stats"] == run(flat[None])["stats"]

    def test_a_four_dimensional_datum_is_read_over_its_last_three_dimensions(self):
        stack = np.random.default_rng(1605).random((2, *SHAPE))
        assert run(stack)["stats"] == run(stack[0])["stats"]

    def test_a_datum_with_no_picture_has_only_an_image_row_and_it_is_not_applicable(self):
        # Boxes on data with fewer than two dimensions are refused upstream, so such a datum has none.
        result = image_stats([np.arange(10.0)], boxes=[[]], stats=CONTEXT)
        assert details_of(result)["context_status"].tolist() == ["not_applicable"]
        assert np.isnan(result["stats"]["surround_mean"][0])

    def test_a_boolean_image_has_intensities_of_zero_and_one(self):
        image = np.broadcast_to(np.arange(80) % 2 == 0, SHAPE).copy()
        window = np.zeros((60, 80), dtype=bool)
        window[2:48, 2:48] = True
        window[10:40, 10:40] = False  # the window less the box
        result = run(image)
        assert result["stats"]["surround_mean"][0] == pytest.approx(image[0][window].mean(), rel=1e-6)
        assert details_of(result)["context_range_high"][0] == 1.0
