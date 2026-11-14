"""The surround against a per-pixel oracle, with the boxes reordered, and run twice.

The oracle is written from the definition, one pixel at a time, and shares it with the calculator. It checks the
arithmetic, the rounding and the bookkeeping over many awkward boxes, and not the definition itself, which the
hand-worked fixtures F2 and F4 in `test_context_surround.py` and `test_context_surround_geometry.py` check.
"""

import math
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

CONTEXT = ImageStats.CONTEXT
HEIGHT, WIDTH = 41, 57
NAN = float("nan")


def details_of(result: Any) -> Any:
    """The result's details, which a result carries only when a requested statistic reports some."""
    assert "details" in result
    return result["details"]


def rounded(box: tuple[float, ...], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return max(0, math.floor(x0)), max(0, math.floor(y0)), min(width, math.ceil(x1)), min(height, math.ceil(y1))


def oracle(image: np.ndarray, boxes: list[Any], focal: int, valid: list[bool], low: float, high: float, padding: float):
    """The surround of one box by looping over its window: counts, mean and population standard deviation."""
    _, height, width = image.shape
    x0, y0, x1, y1 = boxes[focal]
    pad_x, pad_y = padding * (x1 - x0), padding * (y1 - y0)
    window = rounded((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), width, height)
    target = rounded(boxes[focal], width, height)
    covered = np.zeros((height, width), dtype=bool)
    for box, ok in zip(boxes, valid, strict=True):
        if ok:
            bx0, by0, bx1, by1 = rounded(box, width, height)
            covered[by0:by1, bx0:bx1] = True
    candidate = retained = 0
    intensities: list[float] = []
    for y in range(window[1], window[3]):
        for x in range(window[0], window[2]):
            if target[0] <= x < target[2] and target[1] <= y < target[3]:
                continue
            candidate += 1
            if covered[y, x]:
                continue
            retained += 1
            pixel = image[:, y, x]
            if np.isfinite(pixel).all() and (pixel >= low).all() and (pixel <= high).all():
                intensities.append(float(np.mean((pixel - low) / (high - low))))
    values = np.array(intensities)
    mean = float(values.mean()) if values.size else NAN
    std = float(np.sqrt(np.mean((values - mean) ** 2))) if values.size else NAN
    excluded = (candidate - retained) / candidate if candidate else NAN
    return candidate, retained, len(intensities), mean, std, excluded


def target_oracle(image: np.ndarray, box: Any, low: float, high: float) -> tuple[float, float]:
    """A box's target by looping over its pixels: the share that is valid in every channel, and their mean intensity."""
    _, height, width = image.shape
    x0, y0, x1, y1 = rounded(box, width, height)
    intensities: list[float] = []
    count = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            count += 1
            pixel = image[:, y, x]
            if np.isfinite(pixel).all() and (pixel >= low).all() and (pixel <= high).all():
                intensities.append(float(np.mean((pixel - low) / (high - low))))
    return len(intensities) / count, float(np.mean(intensities)) if intensities else NAN


def ratio_oracle(fraction: float, target: float, mean: float, std: float, measured: bool) -> tuple[Any, ...]:
    """The two ratios, their floor flags and the reason, from the definition, at the default floor and minimum."""
    floor, minimum = 1 / 255, 0.90
    if not measured:
        return NAN, NAN, False, False, ""
    if not fraction >= minimum or math.isnan(target):
        return NAN, NAN, False, False, "insufficient_target_support"
    return (target - mean) / max(mean, floor), (target - mean) / max(std, floor), mean < floor, std < floor, ""


def picture(seed: int) -> np.ndarray:
    """Floats in [0, 1], with a few NaN and a few values above the declared range, so the exclusions matter."""
    rng = np.random.default_rng(seed)
    image = rng.random((3, HEIGHT, WIDTH))
    odd = rng.random(image.shape)
    image[odd < 0.05] = np.nan
    image[(odd >= 0.05) & (odd < 0.08)] = 1.5
    return image


def random_boxes(seed: int, count: int) -> list[tuple[float, float, float, float]]:
    """Valid boxes of every kind: inside, half outside, fractional, overlapping."""
    rng = np.random.default_rng(seed)
    boxes: list[tuple[float, float, float, float]] = []
    while len(boxes) < count:
        x0, y0 = rng.uniform(-8, WIDTH), rng.uniform(-8, HEIGHT)
        x1, y1 = x0 + rng.uniform(1, 25), y0 + rng.uniform(1, 25)
        if x1 > 0 and y1 > 0:  # x0 < WIDTH and y0 < HEIGHT by construction
            boxes.append((float(x0), float(y0), float(x1), float(y1)))
    return boxes


def measure(image_boxes: list[tuple[np.ndarray, list[Any]]], **kw: Any) -> dict[str, Any]:
    result = image_stats(
        [image for image, _ in image_boxes],
        boxes=[boxes for _, boxes in image_boxes],
        stats=CONTEXT,
        per_image=False,
        surround_min_pixels=1,
        value_range=(0.0, 1.0),
        **kw,
    )
    return {**result["stats"], **details_of(result)}


def test_the_calculator_agrees_with_a_per_pixel_oracle_on_200_awkward_boxes():
    dense, sparse = random_boxes(1, 140), random_boxes(2, 60)
    # Three boxes that are not valid, in the sparse image: they are not removed from any ring, and they say so.
    bad = [(NAN, 0, 5, 5), (30, 10, 20, 15), (5, 5, 5, 9)]
    cases = [(picture(1605), dense), (picture(1606), sparse + bad)]
    validity = [[True] * len(dense), [True] * len(sparse) + [False] * len(bad)]
    frames = ["ok", "incomplete_geometry"]  # a frame holding a nonfinite or reversed box says so on every other row
    columns = measure(cases)

    row = 0
    measured = 0
    for (image, boxes), valid, frame in zip(cases, validity, frames, strict=True):
        for focal in range(len(boxes)):
            if not valid[focal]:
                assert columns["context_status"][row] == "invalid_box"
                row += 1
                continue
            candidate, retained, measurable, mean, std, excluded = oracle(image, boxes, focal, valid, 0.0, 1.0, 0.25)
            assert columns["surround_candidate_pixels"][row] == candidate, row
            assert columns["surround_retained_pixels"][row] == retained, row
            assert columns["surround_measurable_pixels"][row] == measurable, row
            np.testing.assert_allclose(columns["ring_peer_excluded_fraction"][row], excluded, rtol=0, atol=1e-12)
            # The statistic columns are float32, as every statistic's is.
            np.testing.assert_allclose(columns["surround_mean"][row], mean, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(columns["surround_std"][row], std, rtol=1e-6, atol=1e-7)
            expected = "empty_ring" if candidate == 0 else (frame if measurable > 0 else "insufficient_support")
            assert columns["context_status"][row] == expected, row
            measured += measurable > 0
            row += 1
    assert row == len(columns["context_status"]) == 203
    # Most rings are crowded out by 200 boxes in a small image, so the rows with a value are the minority that is
    # compared as values; the rest are compared as counts and as NaN.
    assert measured >= 30


def light_picture(seed: int) -> np.ndarray:
    """Floats in [0, 1] with 3% of the values missing or above the range in each channel, so a pixel is valid in all
    three at 91%, and the targets straddle the 0.90 that the ratios need: some reach it and some fall short."""
    rng = np.random.default_rng(seed)
    image = rng.random((3, HEIGHT, WIDTH))
    odd = rng.random(image.shape)
    image[odd < 0.015] = np.nan
    image[(odd >= 0.015) & (odd < 0.03)] = 1.5
    return image


# Four-by-four targets of 0.5 in a field of 0, far apart: each ring is 0, so both denominators sit on the floor.
LIT_BOXES = [(5, 5, 9, 9), (20, 5, 24, 9), (40, 5, 44, 9), (5, 25, 9, 29), (40, 30, 44, 34)]


def lit_picture() -> np.ndarray:
    image = np.zeros((3, HEIGHT, WIDTH))
    for x0, y0, x1, y1 in LIT_BOXES:
        image[:, y0:y1, x0:x1] = 0.5
    return image


def test_the_ratios_agree_with_a_per_pixel_oracle_on_awkward_boxes():
    dense, sparse = random_boxes(1, 30), random_boxes(2, 15)
    bad = [(NAN, 0, 5, 5), (30, 10, 20, 15), (5, 5, 5, 9)]  # not removed from any ring, and they say so
    cases = [(light_picture(1605), dense), (light_picture(1606), sparse + bad), (lit_picture(), LIT_BOXES)]
    validity = [[True] * len(dense), [True] * len(sparse) + [False] * len(bad), [True] * len(LIT_BOXES)]
    frames = ["ok", "incomplete_geometry", "ok"]
    columns = measure(cases)

    row = valued = unsupported = floored = 0
    for (image, boxes), valid, frame in zip(cases, validity, frames, strict=True):
        for focal in range(len(boxes)):
            if not valid[focal]:
                assert math.isnan(columns["target_surround_contrast"][row]), row
                assert columns["contrast_reason"][row] == "", row
                row += 1
                continue
            candidate, _, measurable, mean, std, _ = oracle(image, boxes, focal, valid, 0.0, 1.0, 0.25)
            fraction, target = target_oracle(image, boxes[focal], 0.0, 1.0)
            assert columns["target_valid_fraction"][row] == pytest.approx(fraction, abs=1e-12), row
            status = "empty_ring" if candidate == 0 else (frame if measurable > 0 else "insufficient_support")
            expected = ratio_oracle(fraction, target, mean, std, status in ("ok", "incomplete_geometry"))
            # The statistic columns are float32, as every statistic's is.
            for name, value in zip(("target_surround_contrast", "signal_clutter_ratio"), expected[:2], strict=True):
                np.testing.assert_allclose(columns[name][row], value, rtol=1e-6, atol=1e-6, err_msg=f"{name} {row}")
            assert columns["target_surround_contrast_floored"][row] == expected[2], row
            assert columns["signal_clutter_ratio_floored"][row] == expected[3], row
            assert columns["contrast_reason"][row] == expected[4], row
            valued += not math.isnan(expected[0])
            unsupported += expected[4] == "insufficient_target_support"
            floored += bool(expected[2] or expected[3])
            row += 1
    assert row == len(columns["context_status"]) == 53
    # The gate and the mean over valid pixels are both exercised: rows with a value, rows the target's own validity
    # removed, and rows whose denominator is the floor. A test of a gate that no row reaches would pass without it.
    assert valued >= 20
    assert unsupported >= 8
    assert floored >= len(LIT_BOXES)


def test_the_boxes_were_awkward_enough_to_matter():
    # The oracle test is only as good as its boxes: some are cut by the image, and peers took part of many rings.
    boxes = random_boxes(1, 140)
    cut = sum(1 for x0, y0, x1, y1 in boxes if x0 < 0 or y0 < 0 or x1 > WIDTH or y1 > HEIGHT)
    assert cut > 20  # boxes whose own coordinates leave the image, so their windows are clipped
    columns = measure([(picture(1605), boxes)])
    assert (columns["ring_peer_excluded_fraction"] > 0).sum() > 40  # peers took part of many rings


def test_reordering_the_boxes_permutes_the_rows_and_changes_nothing_else():
    image, boxes = picture(1605), random_boxes(3, 24)
    order = np.random.default_rng(7).permutation(len(boxes))
    original = measure([(image, boxes)])
    shuffled = measure([(image, [boxes[i] for i in order])])
    assert original.keys() == shuffled.keys()
    for name, column in original.items():
        np.testing.assert_array_equal(shuffled[name], column[order], err_msg=name)


def test_reordering_the_boxes_permutes_the_ratio_rows_and_changes_nothing_else():
    image, boxes = light_picture(1605), random_boxes(3, 24)
    order = np.random.default_rng(7).permutation(len(boxes))
    original = measure([(image, boxes)])
    shuffled = measure([(image, [boxes[i] for i in order])])
    # Most of the 24 rows hold a value, so the permutation is checked on numbers and not on NaN alone.
    assert np.isfinite(original["target_surround_contrast"]).sum() >= 5
    assert original.keys() == shuffled.keys()
    for name, column in original.items():
        np.testing.assert_array_equal(shuffled[name], column[order], err_msg=name)


def test_two_runs_are_identical():
    cases = [(picture(1605), random_boxes(1, 30)), (light_picture(1606), random_boxes(2, 30))]
    first, second = measure(cases), measure(cases)
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)
