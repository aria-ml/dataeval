"""Tests for the ImageStats blur flags: the calculator wrapped around `core._blur.blur_stat`.

Fixtures come from `tests.core.test_blur.textured`, a seeded 256x256 image of overlapping hard-edged blobs on
a coarse texture, which reads a finite blur at 8 bits. Every other generator is local and seeded.
"""

import inspect
import logging
import warnings
from typing import Any, cast

import numpy as np
import pytest
from scipy import ndimage

import dataeval._experimental as experimental_module
import dataeval.core._calculators._blurstats as blurstats_module
import dataeval.core._image_stats as image_stats_module
from dataeval.config import use_max_processes
from dataeval.core import compute_stats, image_stats
from dataeval.core._blur import blur_stat
from dataeval.core._calculators._base import ViewKind
from dataeval.core._calculators._blurstats import BlurStatCalculator
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.data import QualityGate
from dataeval.exceptions import ExperimentalWarning
from dataeval.flags import ImageStats
from dataeval.quality import Outliers
from tests.core.test_blur import textured

# The blur flags are marked experimental where `image_stats` resolves flags, which warns once per process. That is
# not what most of these tests are about, and where it falls in the run depends on test order. The tests of the
# warning itself reset the once-per-process record first (`fresh_process`) and catch the warning explicitly.
pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

COLUMNS = ("blur", "blur_directionality")

# The sentence both warnings end on, which names the way to keep the v1.2 columns.
WAY_OUT = "`ImageStats.ALL` includes the blur statistics, and `ImageStats.ALL & ~ImageStats.BLUR` leaves them out."

# Recorded from `main` before the blur flags, in the order a bare `image_stats` call returns them, and with the
# `saturation` that v1.3 added to `PIXEL` after the histogram.
DEFAULT_COLUMNS = """
    offset_x offset_y width height channels size aspect_ratio depth center distance_center distance_edge invalid_box
    rel_offset_x rel_offset_y rel_width rel_height rel_size rel_distance_center rel_distance_edge
    xxhash phash dhash phash_d4 dhash_d4 mean std var skew kurtosis entropy missing zeros histogram saturation
    brightness contrast darkness sharpness percentiles
"""
# The ten columns of `ImageStats.BASIC`, which the docs promise to change only after a deprecation and a migration note.
BASIC_COLUMNS = "aspect_ratio brightness channels contrast height mean sharpness std var width"
OUTLIERS_COLUMNS = "item_index metric_name metric_value direction bound percentile population_mean population_std"
POPULATION_COLUMNS = "level metric_name method population n judged reason"
# The 34 columns of `ImageStats.STANDARD` (PIXEL | VISUAL | DIMENSION), which a default `Outliers` run computes: the
# ones it can judge and the ones it cannot. v1.3 added `saturation` to the 33 of v1.2.
STANDARD_COLUMNS = """
    aspect_ratio brightness center channels contrast darkness depth distance_center distance_edge entropy height
    histogram invalid_box kurtosis mean missing offset_x offset_y percentiles rel_distance_center rel_distance_edge
    rel_height rel_offset_x rel_offset_y rel_size rel_width saturation sharpness size skew std var width zeros
"""


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def picture() -> np.ndarray:
    """A 3x256x256 8-bit image: `textured` in band 0, shifted 3 pixels along x in band 1 and 5 along y in band 2.

    The bands differ, so the average over bands is a real picture of its own that no single band reproduces.
    """
    t = textured()
    return to_uint8(np.stack([t, np.roll(t, 3, axis=1), np.roll(t, 5, axis=0)]))


def gray_of(image: np.ndarray) -> np.ndarray:
    """The picture the calculator is meant to read: the unweighted mean over bands, in the 0-255 display range."""
    return image.astype(np.float64).mean(axis=0)


def stats_of(images, **kwargs):
    return image_stats(images, stats=ImageStats.BLUR, **kwargs)["stats"]


def test_columns_and_values():
    image = picture()
    assert ImageStats.BLUR == ImageStats.VISUAL_BLUR | ImageStats.VISUAL_BLUR_DIRECTIONALITY

    stats = stats_of([image])

    assert set(stats) == set(COLUMNS)
    expected = blur_stat(gray_of(image))
    assert np.isfinite(expected).all()
    for name, value in zip(COLUMNS, expected, strict=True):
        assert stats[name][0] == pytest.approx(value, rel=1e-6)


def test_blur_is_in_all_and_in_no_smaller_group():
    """`ALL` holds blur and every other statistic, `SPATIAL` and `CONTEXT` too; the smaller groups hold no blur."""
    assert ImageStats.BLUR in ImageStats.ALL
    assert ImageStats.ALL == (
        ImageStats.STANDARD | ImageStats.HASH | ImageStats.BLUR | ImageStats.SPATIAL | ImageStats.CONTEXT
    )
    v12 = ImageStats.ALL & ~ImageStats.BLUR & ~ImageStats.SPATIAL & ~ImageStats.PIXEL_SATURATION & ~ImageStats.CONTEXT
    assert v12 == (ImageStats.STANDARD | ImageStats.HASH) & ~ImageStats.PIXEL_SATURATION
    assert ~ImageStats.ALL == ImageStats.NONE
    for group in (ImageStats.VISUAL, ImageStats.VISUAL_BASIC, ImageStats.STANDARD, ImageStats.BASIC):
        assert not ImageStats.BLUR & group
    assert ImageStats.BASIC == ImageStats.PIXEL_BASIC | ImageStats.VISUAL_BASIC | ImageStats.DIMENSION_BASIC


def test_standard_is_the_three_families_and_no_hash_is_its_v12_name():
    """`STANDARD` names the pixel, visual and dimension statistics, with no hashes and no blur.

    `NO_HASH` shipped in v1.2 for the same set and stays as another name for it, so Python prints `STANDARD` for both.
    """
    assert ImageStats.STANDARD == ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.DIMENSION
    assert ImageStats.NO_HASH is ImageStats.STANDARD
    assert ImageStats.NO_HASH.name == "STANDARD"
    assert "STANDARD" in repr(ImageStats.NO_HASH)
    assert "NO_HASH" not in repr(ImageStats.NO_HASH)
    assert ImageStats["NO_HASH"] is ImageStats.STANDARD
    assert ImageStats.BASIC in ImageStats.STANDARD
    assert not ImageStats.STANDARD & (ImageStats.HASH | ImageStats.BLUR)


def test_all_is_spelled_from_the_families():
    """`ALL` is every family, so a later change to what `STANDARD` holds cannot take a statistic out of it."""
    families = (
        ImageStats.PIXEL
        | ImageStats.VISUAL
        | ImageStats.DIMENSION
        | ImageStats.HASH
        | ImageStats.BLUR
        | ImageStats.SPATIAL
        | ImageStats.PIXEL_SATURATION
        | ImageStats.CONTEXT
    )
    assert families == ImageStats.ALL


def test_standard_columns_are_a_promise():
    """The 34 names of `ImageStats.STANDARD` are a promise: they change only after a deprecation and a migration note.

    They are the 33 of v1.2 and `saturation`, which v1.3 added to `PIXEL` before `STANDARD` was released.

    A statistic added to `PIXEL`, `VISUAL` or `DIMENSION` makes this test fail. An enum group cannot warn at import, so
    the person who trips it decides: keep the new flag out of `STANDARD` by redefining it, for example
    `(PIXEL | VISUAL | DIMENSION) & ~NEW_FLAG`, or grow it only in a major release, with a migration note. Change the
    names here on purpose, in the same change as the notes.
    """
    stats = image_stats([picture()], stats=ImageStats.STANDARD)["stats"]

    assert sorted(stats) == STANDARD_COLUMNS.split()


def test_basic_columns_are_a_promise():
    """The ten names of `ImageStats.BASIC` are a promise: they change only after a deprecation and a migration note.

    This test is the pin, so changing the set means changing it here on purpose, in the same change as the notes.
    """
    stats = image_stats([picture()], stats=ImageStats.BASIC)["stats"]

    assert set(stats) == set(BASIC_COLUMNS.split())


@pytest.mark.parametrize("function", [image_stats, compute_stats])
def test_signature_default_is_the_v12_set_without_blur(function):
    assert inspect.signature(function).parameters["stats"].default == ImageStats.STANDARD | ImageStats.HASH


def test_a_bare_call_equals_the_v12_set_with_no_blur_and_no_warning(fresh_process):
    """A call with no `stats` returns what it returned before the blur flags, and does not warn about blur."""
    images = [picture(), picture()[:, ::-1]]

    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        bare = image_stats(images)
        explicit = image_stats(images, stats=ImageStats.NO_HASH | ImageStats.HASH)

    assert list(bare["stats"]) == list(explicit["stats"]) == DEFAULT_COLUMNS.split()
    assert not set(COLUMNS) & set(bare["stats"])
    for name, values in bare["stats"].items():
        np.testing.assert_array_equal(values, explicit["stats"][name], err_msg=name)


def test_all_returns_the_blur_columns_and_warns(fresh_process):
    """The default leaves blur out, but a request for `ALL` is a request for it, so it warns like any blur flag.

    The spatial and context statistics need boxes and have their own tests, so they are left out here. `ALL` holds the
    saturation statistic too, so its column comes with the rest.
    """
    with pytest.warns(ExperimentalWarning, match=r"'ImageStats\.BLUR' is experimental"):
        stats = image_stats([picture()], stats=ImageStats.ALL & ~ImageStats.SPATIAL & ~ImageStats.CONTEXT)["stats"]

    assert set(stats) == set(DEFAULT_COLUMNS.split()) | set(COLUMNS) | {"saturation"}


def test_the_experimental_warning_names_the_complement_as_the_way_out(fresh_process):
    with pytest.warns(ExperimentalWarning, match=r"'ImageStats\.BLUR' is experimental") as record:
        image_stats([picture()], stats=ImageStats.ALL & ~ImageStats.SPATIAL & ~ImageStats.CONTEXT)

    message = str(record[0].message)
    assert WAY_OUT in message


def test_existing_flag_values_unchanged():
    """Recorded from `main` before the blur flags: the new members are appended, so nothing before them moves."""
    assert ImageStats.VISUAL_SHARPNESS.value == 4096
    assert ImageStats.HASH_DHASH_D4.value == 137438953472


def test_one_detection_serves_both_columns(monkeypatch):
    image = picture()
    calls = []

    def counting(gray):
        calls.append(gray.shape)
        return blur_stat(gray)

    monkeypatch.setattr(blurstats_module, "blur_stat", counting)

    computed = BlurStatCalculator(image, CalculatorCache(image)).compute(ImageStats.BLUR)

    assert set(computed) == set(COLUMNS)
    assert calls == [(256, 256)]


def test_nan_rules_through_the_calculator():
    """An image with no edges and one too small to hold 100 edge points give NaN, never an exception."""
    flat = np.full((3, 256, 256), 128, dtype=np.uint8)
    ramp = np.linspace(0.0, 1.0, 3 * 16 * 16).reshape(3, 16, 16)  # the probe image QualityGate reads statistics off

    for image in (flat, ramp):
        stats = stats_of([image], normalize_pixel_values=False)
        assert all(np.isnan(stats[name][0]) for name in COLUMNS)

    assert QualityGate("blur", min=0.0).two_pass is False


def test_unmeasurable_band_reports_nan():
    """Signed float data has no value range, so how sharp it is has no answer, as for sharpness.

    The same picture, 256x256 and with finite blur once an interval is declared, so the NaN is the missing range
    and not the edge-point floor.
    """
    signed = ((textured().astype(np.float64) - 128.0) * 4.0)[np.newaxis]  # (1, 256, 256), -512 to 508
    stats = ImageStats.BLUR | ImageStats.VISUAL_SHARPNESS

    undeclared = image_stats([signed], stats=stats, normalize_pixel_values=False)["stats"]
    declared = image_stats([signed], stats=stats, normalize_pixel_values=False, value_range=(-512.0, 512.0))["stats"]

    for name in ("sharpness", *COLUMNS):
        assert np.isnan(undeclared[name][0]), name
        assert np.isfinite(declared[name][0]), name


def test_nan_pixel_float_image():
    """One NaN pixel in a float image makes every column NaN, and nothing raises.

    The NaN spreads through the smoothing and makes the mean gradient NaN, so no pixel passes the threshold: the
    answer is withheld rather than computed over the part that is left.
    """
    clean = picture().astype(np.float64)
    holed = clean.copy()
    holed[:, 100, 100] = np.nan

    assert np.isfinite(stats_of([clean], normalize_pixel_values=False)["blur"][0]), "the picture is measurable"
    stats = stats_of([holed], normalize_pixel_values=False)

    assert all(np.isnan(stats[name][0]) for name in COLUMNS)


def test_tiny_box_and_1x1():
    """Boxes of 3x3 and smaller, and images of 1x1, are below the edge-point floor."""
    image = picture()

    result = image_stats(
        [image],
        boxes=[[(0, 0, 3, 3), (10, 10, 11, 11), (20, 20, 60, 60)]],
        stats=ImageStats.BLUR,
        per_target=True,
    )["stats"]

    # One row for the image, then one per box in order
    assert np.isfinite(result["blur"][0])
    assert np.isnan(result["blur"][1:3]).all()
    assert len(result["blur"]) == 4
    for tiny in (np.full((3, 1, 1), 128, dtype=np.uint8), np.zeros((1, 1)), np.full((1, 3), 7.0)):
        stats = stats_of([tiny], normalize_pixel_values=False)
        assert all(np.isnan(stats[name][0]) for name in COLUMNS)


def test_uint8_equals_uint16():
    """The picture at 8 bits and at 16 bits (`x * 257`) is one picture, within 1%.

    The 8-bit image reads off the window; the 16-bit one is rescaled to the display range first. On this fixture
    the two came out identical, so the 1% is room for rounding on other platforms, not a measured difference.
    """
    image = picture()
    wide = image.astype(np.uint16) * 257

    narrow_stats = stats_of([image])
    wide_stats = stats_of([wide])

    assert np.isfinite(narrow_stats["blur"][0])
    for name in COLUMNS:
        assert wide_stats[name][0] == pytest.approx(narrow_stats[name][0], rel=0.01)


def test_2d_is_one_gray_band_and_1d_has_no_picture():
    """A 2D array reads like the same picture as a one-band cube; 1D data has no picture to measure."""
    gray = gray_of(picture())
    as_cube = stats_of([gray[np.newaxis]], normalize_pixel_values=False)

    flat = stats_of([gray], normalize_pixel_values=False)
    line = stats_of([np.random.default_rng(0).random(10000)], normalize_pixel_values=False)

    assert np.isfinite([flat[name][0] for name in COLUMNS]).all()
    for name in COLUMNS:
        assert flat[name][0] == as_cube[name][0]
        assert np.isnan(line[name][0])


def test_bands_and_background():
    """Named band groups get prefixed blur columns; the background view, which holds NaN holes, gets none."""
    t = textured()
    cube = to_uint8(np.stack([np.roll(t, 3 * band, axis=1) for band in range(6)]))
    groups = {"vis": [0, 1, 2], "nir": [3, 4, 5]}

    stats = stats_of([cube], channels=groups)

    assert set(stats) == {name for prefix in ("", "vis_", "nir_") for name in (prefix + c for c in COLUMNS)}
    assert np.isfinite([stats[name][0] for name in stats]).all()
    for prefix, bands in (("", slice(0, 6)), ("vis_", slice(0, 3)), ("nir_", slice(3, 6))):
        assert stats[f"{prefix}blur"][0] == pytest.approx(blur_stat(gray_of(cube[bands])).blur, rel=1e-6)

    # Sharpness answers over the background; blur does not, so it is the sharpness columns that show the view ran.
    background = image_stats(
        [cube],
        boxes=[[(20, 20, 90, 90)]],
        stats=ImageStats.BLUR | ImageStats.VISUAL_SHARPNESS,
        per_background=True,
        channels=groups,
    )["stats"]

    assert {"background_sharpness", "background_vis_sharpness", "background_nir_sharpness"} <= set(background)
    assert not [name for name in background if "background" in name and "blur" in name]
    assert set(COLUMNS) <= set(background), "the unprefixed whole-image columns are still there"
    assert BlurStatCalculator.flags_for_view(ImageStats.BLUR, ViewKind.MASK) == ImageStats(0)
    assert BlurStatCalculator.flags_for_view(ImageStats.BLUR, ViewKind.BAND) == ImageStats.BLUR


def test_background_with_only_blur_says_why_it_does_not_apply():
    """Blur alone has nothing to compute over a background, and the warning names it among the reasons."""
    with pytest.warns(
        UserWarning,
        match=r"none of the requested statistics apply to a background region.*the filters "
        r"of the blur statistics spread the masked holes",
    ):
        result = image_stats([picture()], boxes=[[(20, 20, 90, 90)]], stats=ImageStats.BLUR, per_background=True)[
            "stats"
        ]

    assert set(result) == set(COLUMNS) | {"background_fraction"}


def test_deterministic_permutes():
    """Reversing the image list reverses the rows and changes nothing else, to the bit."""
    base = textured()
    images = [to_uint8(np.stack([ndimage.gaussian_filter(base, sigma)] * 3)) for sigma in (0, 1, 2)]

    forward = stats_of(images)
    backward = stats_of(images[::-1])
    again = stats_of(images)

    assert set(forward) == set(backward) == set(COLUMNS)
    for name in COLUMNS:
        assert np.isfinite(forward[name]).all()
        assert np.array_equal(backward[name], forward[name][::-1])
        assert np.array_equal(again[name], forward[name])
    assert forward["blur"][0] < forward["blur"][1] < forward["blur"][2], "the rows differ, so the reversal shows"


@pytest.fixture
def fresh_process(monkeypatch):
    """Start the test with no experimental warning given yet, as in a new process."""
    monkeypatch.setattr(experimental_module, "_WARNED", set())


def test_experimental_warning_once(fresh_process):
    """The first call asking for a blur flag warns, naming it, at the caller; later calls are silent."""
    with pytest.warns(ExperimentalWarning, match=r"'ImageStats\.BLUR' is experimental") as first:
        stats_of([picture()])

    assert len(first) == 1
    assert first[0].filename == __file__, "the warning points at the caller of image_stats"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        stats_of([picture()])


def test_experimental_warning_for_a_band_group_alone(fresh_process):
    """The flags resolve per view: a mapping that asks for blur only of a band group still warns."""
    cube = np.zeros((6, 8, 8), np.uint8)

    with pytest.warns(ExperimentalWarning, match=r"'ImageStats\.BLUR' is experimental"):
        image_stats(
            [cube],
            stats={None: ImageStats.PIXEL_MEAN, "ir": ImageStats.VISUAL_BLUR_DIRECTIONALITY},
            channels={"ir": 5},
        )


def test_only_a_request_that_holds_blur_warns(fresh_process):
    """The bare call and `VISUAL` do not hold the blur flags, so they are silent. `ALL` holds them, so it warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        image_stats([picture()])
        image_stats([picture()], stats=ImageStats.NO_HASH | ImageStats.HASH)
        image_stats([picture()], stats=ImageStats.VISUAL)
        image_stats([picture()], stats=ImageStats.BASIC)

    with pytest.warns(ExperimentalWarning, match=r"'ImageStats\.BLUR' is experimental"):
        image_stats([picture()], stats=ImageStats.ALL)


@pytest.mark.parametrize("stat", COLUMNS)
def test_quality_gate_warns_for_a_blur_statistic(fresh_process, stat):
    """`QualityGate` reads the statistic without `image_stats`, so it warns itself, at the caller, once."""
    with pytest.warns(ExperimentalWarning, match=r"'ImageStats\.BLUR' is experimental") as first:
        QualityGate(stat, max=5.0)

    assert len(first) == 1
    assert first[0].filename == __file__, "the warning points at the caller of QualityGate"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        QualityGate(stat, max=5.0)


def test_quality_gate_on_another_statistic_stays_silent(fresh_process):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        QualityGate("sharpness", min=0.0)


def test_wide_cube_warns_and_names_blur(caplog):
    """A cube wider than three bands measured as one picture gets the 'name band groups' warning."""
    cube = to_uint8(np.stack([np.roll(textured(), 3 * band, axis=1) for band in range(5)]))

    with caplog.at_level(logging.WARNING, logger="dataeval.core"):
        stats_of([cube])

    messages = [r.getMessage() for r in caplog.records if "measured as a single picture" in r.getMessage()]
    assert len(messages) == 1
    assert "5 bands measured as a single picture for" in messages[0]
    assert "VISUAL_BLUR" in messages[0]


def test_unmeasurable_names_include_blur(caplog):
    """The 'no value range' warning names the blur flags among the statistics that go NaN."""
    signed = ((textured().astype(np.float64) - 128.0) * 4.0)[np.newaxis]

    with caplog.at_level(logging.WARNING, logger="dataeval.core"):
        stats_of([signed], normalize_pixel_values=False)

    messages = [r.getMessage() for r in caplog.records if "no value range could be established" in r.getMessage()]
    assert len(messages) == 1
    assert "VISUAL_BLUR" in messages[0]


def sized(height: int, width: int, bands: int = 3) -> np.ndarray:
    return np.zeros((bands, height, width), dtype=np.uint8)


def mixed_size_warnings(record) -> list[str]:
    return [str(w.message) for w in record if "different image sizes" in str(w.message)]


def test_mixed_sizes_warn_once_with_the_count_of_sizes():
    """Edge widths are in pixels at each image's own size, so a request over several sizes says so, once."""
    images = [sized(32, 32), sized(48, 40), sized(32, 32), sized(20, 64)]

    with pytest.warns(UserWarning, match="different image sizes") as record:
        stats_of(images)

    (message,) = mixed_size_warnings(record)
    assert message.startswith("blur statistics were requested for images of 3 different image sizes")
    assert "compare only within one size" in message
    assert message.endswith(WAY_OUT)
    assert [w.filename for w in record if "different image sizes" in str(w.message)] == [__file__]


def test_mixed_sizes_warn_under_multiple_processes():
    """The main process iterates the images on the pool path too, so the sizes are all seen there."""
    images = [sized(32, 32), sized(48, 40)] * 3

    with use_max_processes(2), pytest.warns(UserWarning, match="different image sizes") as record:
        stats_of(images)

    (message,) = mixed_size_warnings(record)
    assert "of 2 different image sizes" in message


def test_mixed_sizes_warn_for_a_band_group_alone():
    """The blur flag of a band group is a blur request too."""
    images = [sized(32, 32, bands=6), sized(48, 40, bands=6)]

    with pytest.warns(UserWarning, match="of 2 different image sizes"):
        image_stats(
            images,
            stats={None: ImageStats.PIXEL_MEAN, "ir": ImageStats.VISUAL_BLUR_DIRECTIONALITY},
            channels={"ir": 5},
        )


def test_mixed_sizes_warn_through_outliers():
    """The documented way to ask, `Outliers(flags=VISUAL | BLUR)`, thresholds across every row, so it warns too."""
    rng = np.random.default_rng(20261107)
    images = [rng.integers(0, 256, (3, 32 + 8 * (i % 2), 32), dtype=np.uint8) for i in range(24)]

    with pytest.warns(UserWarning, match="of 2 different image sizes"):
        Outliers(flags=ImageStats.VISUAL | ImageStats.BLUR).evaluate(images)


def test_one_size_does_not_warn():
    """Bands, a 2D array and boxes of other sizes do not make a second size: only the (H, W) of each image counts."""
    images = [sized(32, 32), np.zeros((32, 32), dtype=np.uint8), sized(32, 32, bands=5)]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", ExperimentalWarning)
        stats_of(images)
        image_stats(
            [sized(64, 64)],
            boxes=[[(0, 0, 10, 10), (5, 5, 30, 40)]],
            stats=ImageStats.BLUR,
            per_target=True,
        )


def test_mixed_sizes_without_a_blur_flag_stay_silent_and_unnoted(monkeypatch):
    """A request with no blur flag neither warns nor records sizes: the recording is a wrapper that is never built."""
    images = [sized(32, 32), sized(48, 40)]

    def refuse(datum, sizes):
        raise AssertionError("sizes were recorded for a request with no blur flag")

    monkeypatch.setattr(image_stats_module, "_noting_sizes", refuse)

    with pytest.raises(AssertionError, match="sizes were recorded"):  # control: the patch reaches a blur request
        stats_of(images)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", ExperimentalWarning)
        image_stats(images)
        image_stats(images, stats=ImageStats.NO_HASH | ImageStats.HASH)
        image_stats(images, stats=ImageStats.VISUAL, per_target=False)


def test_default_columns_unchanged():
    """With the blur flags not requested, the columns of a bare call and of a default Outliers run are the ones
    recorded from `main` before the blur flags.
    """
    rng = np.random.default_rng(20261107)
    images = [rng.integers(0, 256, (3, 32, 32), dtype=np.uint8) for _ in range(24)]  # Outliers judges from 20 rows

    bare_stats = image_stats(images)["stats"]
    outliers = Outliers().evaluate(images)

    assert list(bare_stats) == DEFAULT_COLUMNS.split()
    assert outliers.data().columns == OUTLIERS_COLUMNS.split()
    assert outliers.populations.columns == POPULATION_COLUMNS.split()
    assert sorted(cast(Any, outliers.calculation_results)["stats"]) == STANDARD_COLUMNS.split()


def test_default_paths_never_run_the_blur_measurement(monkeypatch):
    """A bare call and a default Outliers run never call `blur_stat`: the column names alone do not show that."""
    rng = np.random.default_rng(20261107)
    images = [rng.integers(0, 256, (3, 32, 32), dtype=np.uint8) for _ in range(24)]  # Outliers judges from 20 rows

    def refuse(gray):
        raise AssertionError("blur_stat ran for a request with no blur flag")

    monkeypatch.setattr(blurstats_module, "blur_stat", refuse)

    with pytest.raises(AssertionError, match="blur_stat ran"):  # control: the patch reaches a request that asks
        image_stats(images, stats=ImageStats.BLUR)
    image_stats(images)
    image_stats(images, stats=ImageStats.NO_HASH | ImageStats.HASH)
    Outliers().evaluate(images)


def test_0d_datum_is_nan_not_an_error():
    """A scalar has no picture: both columns are NaN and nothing raises (pins the `ndim < 2` guard)."""
    stats = stats_of([np.float64(5.0)], normalize_pixel_values=False)

    assert set(stats) == set(COLUMNS)
    assert all(np.isnan(stats[name][0]) for name in COLUMNS)


def test_unmeasurable_view_never_runs_the_detection(monkeypatch):
    """An image with no value range is NaN without the cost of looking for edges (pins the `is_unmeasurable` guard)."""
    signed = ((textured().astype(np.float64) - 128.0) * 4.0)[np.newaxis]
    calls = []

    def counting(gray):
        calls.append(gray.shape)
        return blur_stat(gray)

    monkeypatch.setattr(blurstats_module, "blur_stat", counting)
    cache = CalculatorCache(signed)

    computed = BlurStatCalculator(signed, cache).compute(ImageStats.BLUR)

    assert cache.is_unmeasurable
    assert all(np.isnan(computed[name][0]) for name in COLUMNS)
    assert calls == []


def test_box_past_the_border_is_nan():
    """A box extending past the image reads fill (NaN) outside it, so every column is NaN, whichever edge it crosses."""
    result = image_stats(
        [picture()],
        boxes=[[(20, 20, 90, 90), (200, 200, 300, 300), (-30, -30, 60, 60)]],
        stats=ImageStats.BLUR,
        per_target=True,
    )["stats"]

    for name in COLUMNS:
        assert np.isfinite(result[name][:2]).all(), "the image and the box inside it are measured"
        assert np.isnan(result[name][2:]).all(), "a box past the right/bottom border and one past the left/top"
