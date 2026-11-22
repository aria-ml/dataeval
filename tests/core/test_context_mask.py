"""The context statistics under `image_stats(mask=)`: a masked pixel is unreadable, as a NaN already is (#1593).

The fixtures are synthetic. A small float image of known pixels carries a box with a peer in its surround, and the
mask cuts the surround and the target by counts worked out here. A second frame of random texture carries a hood and a
HUD and four boxes, for the comparison with a frame whose masked pixels are written as NaN, which the context
statistics have always read as unreadable. The guard at the end lists the statistics that no masked view covers.
"""

import logging
import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.core._calculators._base import ViewKind
from dataeval.core._calculators._contextstats import read_surround, surround_regions
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import normalize_image_shape

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

CONTEXT = ImageStats.CONTEXT
CONTEXT_NAMES = (
    "surround_intensity_mean",
    "surround_intensity_std",
    "target_surround_contrast",
    "signal_clutter_ratio",
)
# The details the mask moves, and the ones that count boxes alone.
MOVED = ("surround_measurable_pixels", "surround_valid_fraction", "target_valid_fraction")
BOXES_ONLY = ("surround_candidate_pixels", "surround_retained_pixels", "surround_peer_excluded_fraction")

# A 20x20 ramp. The focal box's window is rows and columns 2 to 17 (padding 0.25 of its 10 px), its target is rows and
# columns 5 to 14, and the peer overlaps the window's top right corner: 12 of the window's pixels, none of the target's.
SIZE = 20
RAMP = (np.arange(SIZE * SIZE).reshape(1, SIZE, SIZE) / (SIZE * SIZE - 1)).astype(np.float64)
FOCAL = (5.0, 5.0, 15.0, 15.0)
PEER = (16.0, 2.0, 19.0, 8.0)
LOW_LEFT = np.zeros((SIZE, SIZE), dtype=bool)
LOW_LEFT[14:, :8] = True  # the window's bottom left: 21 surround pixels and 3 of the target's 100


def read(
    image: np.ndarray, boxes: list[Any], mask: np.ndarray | None = None, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The context statistics and details of every box row, from a declared range so that no range is inferred."""
    kwargs.setdefault("value_range", (0.0, 1.0))
    logged: list[logging.LogRecord] = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = logged.append  # type: ignore[method-assign]
    logger = logging.getLogger("dataeval")
    logger.addHandler(handler)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a masked read must not warn, through `warnings` or the log
            result = image_stats(
                [image],
                boxes=[boxes],
                labels=[[0] * len(boxes)],
                stats=CONTEXT,
                per_image=False,
                mask=mask,
                **kwargs,
            )
    finally:
        logger.removeHandler(handler)
    assert not logged, [record.getMessage() for record in logged]
    return dict(result["stats"]), dict(result.get("details", {}))


def masked_by(mask: np.ndarray, *, target_pixels: int = 0, surround_pixels: int = 0) -> np.ndarray:
    """The mask, after checking that it covers the counts this test was written for within the focal box's window."""
    assert np.count_nonzero(mask[5:15, 5:15]) == target_pixels
    assert np.count_nonzero(mask[2:18, 2:18]) - target_pixels == surround_pixels
    return mask


class TestTheCountsMoveAsTheDesignSays:
    """AT3: the pixels the mask removes are unreadable, and only the counts that read pixels show it."""

    def test_boxes_alone_set_the_retained_count_and_the_measurable_count_loses_the_masked_pixels(self):
        mask = masked_by(LOW_LEFT, target_pixels=3, surround_pixels=21)
        loose_stats, loose = read(RAMP, [FOCAL, PEER])
        masked_stats, masked = read(RAMP, [FOCAL, PEER], mask)
        for name in BOXES_ONLY:
            np.testing.assert_array_equal(masked[name], loose[name], err_msg=name)
        assert list(masked["surround_retained_pixels"]) == [144, 32]
        assert list(loose["surround_measurable_pixels"]) == [144, 32]
        # 21 of the focal box's 144 retained surround pixels are under the mask; the peer's own window is clear of it.
        assert list(masked["surround_measurable_pixels"]) == [144 - 21, 32]
        np.testing.assert_allclose(masked["surround_valid_fraction"], [(144 - 21) / 144, 1.0])
        np.testing.assert_allclose(masked["target_valid_fraction"], [97 / 100, 1.0])
        assert list(masked["context_status"]) == ["ok", "ok"]
        # Not vacuous: the surround's pixels changed, so the surround's values did.
        assert masked_stats["surround_intensity_mean"][0] != loose_stats["surround_intensity_mean"][0]
        assert masked_stats["surround_intensity_mean"][1] == loose_stats["surround_intensity_mean"][1]

    def test_the_value_is_read_from_the_pixels_the_mask_leaves(self):
        mask = masked_by(LOW_LEFT, target_pixels=3, surround_pixels=21)
        stats, _ = read(RAMP, [FOCAL, PEER], mask)
        window = np.zeros((SIZE, SIZE), dtype=bool)
        window[2:18, 2:18] = True
        target = np.zeros_like(window)
        target[5:15, 5:15] = True
        peer = np.zeros_like(window)
        peer[2:8, 16:19] = True
        surround = RAMP[0][window & ~target & ~peer & ~mask]
        inside = RAMP[0][target & ~mask]
        assert surround.size == 144 - 21
        assert stats["surround_intensity_mean"][0] == pytest.approx(surround.mean(), abs=1e-6)
        assert stats["surround_intensity_std"][0] == pytest.approx(surround.std(), abs=1e-6)
        difference = inside.mean() - surround.mean()
        assert stats["target_surround_contrast"][0] == pytest.approx(difference / surround.mean(), abs=1e-6)
        assert stats["signal_clutter_ratio"][0] == pytest.approx(difference / surround.std(), abs=1e-6)

    @pytest.mark.parametrize("pixels", ["window", "hole"])
    def test_a_surround_the_mask_covers_reads_insufficient_support(self, pixels):
        mask = np.ones((SIZE, SIZE), dtype=bool)
        mask[5:15, 5:15] = False  # the target is clear, every pixel around it is under the mask
        if pixels == "hole":
            mask[2, 2:6] = False  # four readable surround pixels are still under the 32 the surround needs
        stats, details = read(RAMP, [FOCAL], mask)
        assert list(details["context_status"]) == ["insufficient_support"]
        assert list(details["surround_candidate_pixels"]) == [156]
        assert list(details["surround_retained_pixels"]) == [156]
        assert list(details["surround_measurable_pixels"]) == [0 if pixels == "window" else 4]
        assert details["target_valid_fraction"][0] == 1.0
        for name in CONTEXT_NAMES:
            assert np.isnan(stats[name]).all(), name
        # The surround, and not the target, is why the ratios are NaN, so no target reason is given.
        assert list(details["context_ratio_reason"]) == [""]

    @pytest.mark.parametrize(("under", "reason"), [(10, ""), (11, "insufficient_target_support")])
    def test_a_target_over_a_tenth_under_the_mask_loses_its_ratios(self, under, reason):
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        mask[14, 5:15] = True  # a tenth of the target's 100 pixels, which the 0.90 floor still allows
        if under == 11:
            mask[13, 5] = True
        masked_by(mask, target_pixels=under, surround_pixels=0)
        stats, details = read(RAMP, [FOCAL], mask)
        assert details["target_valid_fraction"][0] == pytest.approx((100 - under) / 100)
        assert list(details["context_status"]) == ["ok"]
        assert list(details["context_ratio_reason"]) == [reason]
        assert np.isfinite(stats["surround_intensity_mean"]).all()
        assert np.isfinite(stats["surround_intensity_std"]).all()
        ratios = [stats["target_surround_contrast"][0], stats["signal_clutter_ratio"][0]]
        assert (np.isnan(ratios) == bool(reason)).all()

    def test_a_target_wholly_under_the_mask_has_no_mean_and_loses_its_ratios(self):
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        mask[5:15, 5:15] = True
        stats, details = read(RAMP, [FOCAL], mask)
        assert details["target_valid_fraction"][0] == 0.0
        assert list(details["context_ratio_reason"]) == ["insufficient_target_support"]
        assert np.isfinite(stats["surround_intensity_mean"]).all()
        assert np.isnan(stats["target_surround_contrast"]).all()


# A frame of random texture with a hood across the bottom and a HUD near the top left, and four boxes: one clear of the
# mask, one straddling the hood, one wholly under it and one that leaves the image over the HUD.
H, W = 48, 64
HOOD = np.zeros((H, W), dtype=bool)
HOOD[36:] = True
HOOD[2:8, 2:20] = True
BOXES = [(30.0, 10.0, 50.0, 30.0), (10.0, 30.0, 40.0, 44.0), (40.0, 38.0, 60.0, 46.0), (-5.0, -4.0, 10.0, 6.0)]
GROUP = {"rg": [0, 1]}


def texture(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((3, H, W))


def nan_under(image: np.ndarray, mask: np.ndarray, fill: float = np.nan) -> np.ndarray:
    out = image.copy()
    out[:, mask] = fill
    return out


def assert_same_columns(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert list(left) == list(right)
    for name in left:
        assert left[name].dtype == right[name].dtype, name
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


class TestAMaskedPixelIsAsUnreadableAsANaN:
    """A masked pixel reads as the frame reads the same pixel written as NaN, which was always unreadable."""

    @pytest.mark.parametrize("groups", [None, GROUP])
    def test_every_column_equals_the_frame_with_nan_under_the_mask(self, groups):
        image = texture()
        kwargs: dict[str, Any] = {} if groups is None else {"channels": groups}
        stats, details = read(image, BOXES, HOOD, **kwargs)
        oracle_stats, oracle_details = read(nan_under(image, HOOD), BOXES, **kwargs)
        assert_same_columns(stats, oracle_stats)
        assert_same_columns({k: v for k, v in details.items() if k != "masked_fraction"}, oracle_details)
        # Not vacuous: the boxes the mask cuts read differently from the frame read whole.
        loose_stats, loose = read(image, BOXES, **kwargs)
        assert not np.array_equal(stats["surround_intensity_mean"], loose_stats["surround_intensity_mean"])
        assert not np.array_equal(details["surround_valid_fraction"], loose["surround_valid_fraction"])

    def test_the_masked_pixels_hold_nothing_that_reaches_a_value(self):
        image = texture()
        for fill in (0.0, 1.0, np.nan):
            stats, details = read(nan_under(image, HOOD, fill), BOXES, HOOD)
            reference, reference_details = read(nan_under(image, HOOD, 0.5), BOXES, HOOD)
            assert_same_columns(stats, reference)
            assert_same_columns(details, reference_details)

    def test_the_box_under_the_mask_and_the_box_past_the_image_are_read_without_raising(self):
        _, details = read(texture(), BOXES, HOOD)
        # The last box leaves the image over the HUD, which leaves its clipped window fewer than 32 readable pixels.
        assert list(details["context_status"]) == ["ok", "ok", "insufficient_support", "insufficient_support"]
        np.testing.assert_allclose(details["target_valid_fraction"][:3], [1.0, 6 / 14, 0.0])
        assert details["surround_measurable_pixels"][2] == 0
        assert 0 < details["surround_measurable_pixels"][3] < 32

    def test_a_mask_nothing_is_under_changes_nothing(self):
        image = texture()
        empty = np.zeros((H, W), dtype=bool)
        stats, details = read(image, BOXES, empty)
        plain, plain_details = read(image, BOXES)
        assert_same_columns(stats, plain)
        assert_same_columns(details, plain_details)


def surround(image: np.ndarray = RAMP, **kwargs: Any) -> Any:
    """`read_surround` over the focal box's window, its retained pixels all of the window, and the range [0, 1]."""
    regions = surround_regions(np.asarray(FOCAL), (SIZE, SIZE), 0.25)
    wx0, wy0, wx1, wy1 = regions.window
    tx0, ty0, tx1, ty1 = regions.target
    retained = np.ones((wy1 - wy0, wx1 - wx0), dtype=bool)
    retained[ty0 - wy0 : ty1 - wy0, tx0 - wx0 : tx1 - wx0] = False  # the focal box is in the coverage
    return read_surround(normalize_image_shape(image), (0,), regions, retained, 0.0, 1.0, **kwargs)


class TestReadSurround:
    """`read_surround`'s `excluded` is the whole-frame mask over the surround's window, and None changes nothing."""

    def test_without_excluded_it_reads_as_it_always_did(self):
        none = surround()
        assert surround(excluded=None) == none
        assert surround(excluded=np.zeros((16, 16), dtype=bool)) == none
        assert none.measurable == 256 - 100
        assert none.target_valid_fraction == 1.0

    def test_excluded_leaves_a_pixel_out_of_the_surround_and_the_target(self):
        excluded = np.zeros((16, 16), dtype=bool)
        excluded[0, 0] = True  # a surround pixel
        excluded[3, 3] = True  # a target pixel: the target starts at 3, 3 of the window
        reading = surround(excluded=excluded)
        assert reading.measurable == 156 - 1
        assert reading.target_valid_fraction == 99 / 100
        target = RAMP[0, 5:15, 5:15].copy()
        target[0, 0] = np.nan
        assert reading.target_mean == pytest.approx(float(np.nanmean(target)), abs=1e-12)

    def test_a_masked_pixel_that_is_in_range_and_finite_is_still_unreadable(self):
        excluded = np.zeros((16, 16), dtype=bool)
        excluded[:, :3] = True
        moved = RAMP.copy()
        moved[0, 2:18, 2:5] = 0.25
        assert surround(moved, excluded=excluded) == surround(excluded=excluded)
        assert surround(moved) != surround()


class TestTheGuard:
    """AT12: a statistic of a masked class that answers `WHOLE` and not `MASK` has a mask rule, or it is not added."""

    GUARDED = {"blur_edge_directionality", *CONTEXT_NAMES}

    @staticmethod
    def classes() -> list[tuple[Any, Any]]:
        found = CalculatorRegistry.get_calculators(ImageStats.ALL)
        assert {cls for cls, _ in found} == set(CalculatorRegistry._registry[ImageStats])
        return found

    def test_the_statistics_no_masked_view_covers_are_directionality_and_the_four_context_statistics(self):
        guarded = {
            handler.name
            for cls, flags in self.classes()
            if cls.masks_pixels
            for handler in cls._handlers_for(flags).values()
            if ViewKind.WHOLE in handler.views and ViewKind.MASK not in handler.views
        }
        assert guarded == self.GUARDED, (
            "a statistic that answers the whole view and not the masked one runs over a NaN-holed region under a "
            "mask: give it a mask rule, test it here, and add it to GUARDED"
        )

    def test_a_class_that_keeps_the_whole_frame_has_no_masked_or_surround_view(self):
        # The whole cache those classes get carries no box mask, and the surround pass hands them no whole range.
        kept = [cls for cls, _ in self.classes() if not cls.masks_pixels]
        assert {cls.__name__ for cls in kept} == {
            "DimensionStatCalculator",
            "HashStatCalculator",
            "SpatialStatCalculator",
        }
        for cls, flags in self.classes():
            if cls.masks_pixels:
                continue
            for handler in cls._handlers_for(flags).values():
                assert not handler.views & (ViewKind.MASK | ViewKind.SURROUND), handler.name

    def test_each_guarded_statistic_has_its_defined_masked_behavior(self):
        image = texture()
        request = ImageStats.BLUR | CONTEXT
        masked = image_stats([image], boxes=[BOXES], labels=[[0] * 4], stats=request, mask=HOOD, channels=GROUP)
        loose = image_stats([image], boxes=[BOXES], labels=[[0] * 4], stats=request, channels=GROUP)
        oracle = image_stats([nan_under(image, HOOD)], boxes=[BOXES], labels=[[0] * 4], stats=request, channels=GROUP)

        def named(name: str) -> list[str]:
            """The statistic's columns on the whole view and on the group, whose surround ones put the group second."""
            ways = [name, f"rg_{name}", name.replace("surround_", "surround_rg_", 1)]
            found = [way for way in dict.fromkeys(ways) if way in masked["stats"]]
            assert len(found) == 2, name
            return found

        def directionality(name: str) -> None:
            for column in named(name):
                assert np.isnan(masked["stats"][column]).all()
                assert list(masked.get("details", {})[f"{column}_reason"]) == ["masked"] * 5

        def context(name: str) -> None:
            for column in named(name):
                np.testing.assert_array_equal(masked["stats"][column], oracle["stats"][column], column)
                assert not np.array_equal(masked["stats"][column], loose["stats"][column], equal_nan=True), column

        behavior = {"blur_edge_directionality": directionality, **dict.fromkeys(CONTEXT_NAMES, context)}
        assert set(behavior) == self.GUARDED
        for name, check in behavior.items():
            check(name)
