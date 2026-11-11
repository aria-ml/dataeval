"""``VideoStats.BLUR``: the calculator behind ``blur_level``.

Every video here is a sequence of in-memory frames built from seeded blurred noise, so no test needs ffmpeg. A video
steps through 7 textures of rising softness, one after the other (a `drift`) or in a cycle, so the median over a
subset of frames depends on which frames the subset holds.
"""

import math
import warnings
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("cv2", reason="VideoStats.BLUR requires opencv")

import cv2

import dataeval._experimental as experimental_module
import dataeval.core._calculators._videoblurstats as blur_module
from dataeval.config import use_max_processes
from dataeval.core import image_stats, video_stats
from dataeval.core._blur import BlurValues, blur_stat
from dataeval.core._persistence import STEP, StartSampler
from dataeval.exceptions import ExperimentalWarning
from dataeval.flags import ImageStats, VideoStats
from tests.core.test_video_stats import _Frame, _OneShot

pytestmark = [
    pytest.mark.optional,
    pytest.mark.filterwarnings("ignore:.*(PERSISTENCE|BLUR).* experimental:dataeval.exceptions.ExperimentalWarning"),
]

CYCLE = 7


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch):
    """Start each test with no experimental warning given, and leave the process as it was found."""
    monkeypatch.setattr(experimental_module, "_WARNED", set())


def _texture(seed: int, size: tuple[int, int], softness: float) -> np.ndarray:
    """Blurred noise as a uint8 (H, W) picture."""
    noise = np.random.default_rng(seed).integers(0, 256, size, dtype=np.uint8)
    return cast("np.ndarray", cv2.GaussianBlur(noise, (0, 0), sigmaX=softness, sigmaY=0.8))


def _video(n: int, size: tuple[int, int], channels: int = 1, seed: int = 0, drift: bool = False) -> list[_Frame]:
    """`n` frames of (`channels`, H, W) uint8 pixels through `CYCLE` textures of rising softness.

    The textures cycle frame by frame, or with `drift` each holds a seventh of the video in turn.

    A color frame is the texture at a different gain and offset per band, so its bands agree on where the edges
    are, as the bands of a photograph do, while the luma and the band mean differ.
    """
    gains = [(1.0, 0.0), (0.8, 25.0), (0.6, 60.0)]
    cycle = []
    for i in range(CYCLE):
        base = _texture(seed * CYCLE + i, size, 0.8 + 0.3 * i)
        bands = [np.clip(base.astype(np.float32) * g + o, 0, 255).astype(np.uint8) for g, o in gains[:channels]]
        cycle.append(np.stack(bands))
    which = [min(k * CYCLE // n, CYCLE - 1) if drift else k % CYCLE for k in range(n)]
    return [_Frame(cycle[i], k / 30, k) for k, i in enumerate(which)]


def _replay(n: int, ceiling: int) -> tuple[StartSampler, list[int]]:
    """The sampler a video of `n` frames ends with, and every index it took on, kept or since dropped."""
    sampler = StartSampler(ceiling)
    taken = [k for k in range(n) if sampler.offer(k)]
    return sampler, taken


def _median(values: Any) -> float:
    """The median of the finite values, NaN when there is none."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else math.nan


def _run(videos: list[Any], stats: VideoStats = VideoStats.BLUR, **kwargs: Any) -> Any:
    return video_stats(cast(Any, videos), stats=stats, **kwargs)["stats"]


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize(("short_side", "n"), [(256, 700), (720, 60)])
def test_level_equals_median_of_image_values(short_side, n, channels):
    frames = _video(n, (short_side, short_side * 4 // 3), channels, drift=True)
    sampler, taken = _replay(n, 64)
    kept = sampler.kept
    per_image = image_stats([frames[k].pixels for k in kept], stats=ImageStats.BLUR)["stats"]
    if short_side == 256:
        # Long enough for the sampler to double its stride and drop starts it had taken. The video drifts, so the
        # median over every frame it took on differs from the median over the frames it kept, and a calculator
        # that kept the dropped ones would show.
        assert sampler.stride > STEP
        dropped = image_stats([frames[k].pixels for k in taken if k not in kept], stats=ImageStats.BLUR)["stats"]
        assert _median([*per_image["blur"], *dropped["blur"]]) != _median(per_image["blur"])

    got = _run([frames])
    assert set(got) == {"blur_level"}
    # image_stats stores float32, so its values carry that rounding: 6e-8 relative at most.
    assert per_image["blur"].dtype == np.float32
    # Single-band frames: the video's gray is the image, so the two paths measure the same picture.
    tolerance = 1e-6 if channels == 1 else 0.03
    want = _median(per_image["blur"])
    assert math.isfinite(want)
    assert got["blur_level"][0] == pytest.approx(want, rel=tolerance)
    if channels == 1:
        # Exactly the median of the same function over the same frames, before any float32 storage.
        values = [blur_stat(frames[k].pixels[0].astype(np.float32)) for k in kept]
        assert got["blur_level"][0] == _median([v.blur for v in values])


@pytest.mark.parametrize("n", [100, 400])
def test_kept_frames_bounded(monkeypatch, n):
    calls = []

    def spy(gray):
        calls.append(gray.shape)
        return blur_stat(gray)

    monkeypatch.setattr(blur_module, "blur_stat", spy)
    frames = _video(n, (200, 266), drift=True)
    sampler, _ = _replay(n, 8)
    doublings = round(math.log2(sampler.stride / STEP))
    assert doublings >= 2  # the stride really doubled

    got = _run([frames], blur_samples=8)

    assert len(sampler.kept) <= 8
    # The level is the median over the replayed sampler's kept frames, so no other frame feeds it.
    values = [blur_stat(frames[k].pixels[0].astype(np.float32)) for k in sampler.kept]
    assert got["blur_level"][0] == _median([v.blur for v in values])
    # A frame is computed when offered, and dropped when the stride doubles: half the ceiling per doubling.
    assert len(sampler.kept) <= len(calls) <= 8 + 4 * doublings


# The three videos have three sizes, which is what makes their rows differ, so the size warning is expected here.
@pytest.mark.filterwarnings("ignore:blur statistics were requested:UserWarning")
def test_reorder_permutes_rows():
    videos = [_video(30, (200, 266)), _video(18, (160, 214), seed=1), _video(40, (180, 240), seed=2)]
    with use_max_processes(1):
        forward = video_stats(cast(Any, videos), stats=VideoStats.BLUR | VideoStats.DESCRIPTIVE_FRAME_COUNT)
    assert np.isfinite(forward["stats"]["blur_level"]).all()
    assert len(set(forward["stats"]["blur_level"])) == 3  # the rows differ, so a swap would show
    for processes in (1, 2):
        with use_max_processes(processes):
            backward = video_stats(cast(Any, videos[::-1]), stats=VideoStats.BLUR | VideoStats.DESCRIPTIVE_FRAME_COUNT)
        assert list(backward["stats"]) == list(forward["stats"])
        for name, column in forward["stats"].items():
            np.testing.assert_array_equal(backward["stats"][name], column[::-1], err_msg=name)
        assert backward["source_index"] == forward["source_index"]


def test_decode_once_with_persistence(monkeypatch):
    import dataeval.core._video_runner as runner

    real = runner.to_canonical_grayscale
    base = _video(60, (360, 640))
    # Each frame its own array, so a second conversion of one frame shows as the same array seen twice.
    streams = [[_Frame(f.pixels.copy(), f.time_s, f.frame_index) for f in base] for _ in range(2)]

    def converted(stats):
        seen: list[Any] = []
        monkeypatch.setattr(runner, "to_canonical_grayscale", lambda p: seen.append(p) or real(p))
        got = _run([_OneShot(cast(Any, streams.pop()))], stats)
        return seen, got

    seen, got = converted(VideoStats.PERSISTENCE | VideoStats.BLUR)
    assert np.isfinite(got["blur_level"]).all()
    assert len({id(p) for p in seen}) == len(seen)  # no frame converted twice
    # While blur_samples equals persistence_samples (both default to 64), the blur starts are persistence starts,
    # so asking for blur as well reads no frame more.
    persistence_only, _ = converted(VideoStats.PERSISTENCE)
    assert len(seen) == len(persistence_only) > 0


@pytest.mark.parametrize("shape", [(1080, 1920), (1920, 1080), (1080, 1440)])
def test_large_frames_shrunk(monkeypatch, shape):
    calls = []

    def spy(gray):
        calls.append((gray.shape, gray.dtype))
        return BlurValues(2.0, 0.5)

    monkeypatch.setattr(blur_module, "blur_stat", spy)
    frame = _Frame(np.zeros((1, *shape), np.uint8), 0.0, 0)
    got = _run([[frame]])

    factor = 720 / min(shape)
    assert calls == [((round(shape[0] * factor), round(shape[1] * factor)), np.dtype(np.float32))]
    assert min(calls[0][0]) == 720
    assert got["blur_level"][0] == 2.0


@pytest.mark.parametrize("blur_samples", [64, 4])
def test_frames_that_cannot_be_starts_are_never_read(blur_samples):
    textured = _video(60, (200, 266))
    # `_Frame` fails the test on any read of pixels=None, and starts fall on every STEP-th frame at the least.
    frames = [f if k % STEP == 0 else _Frame(None, f.time_s, k) for k, f in enumerate(textured)]
    got = _run([frames], blur_samples=blur_samples)
    assert np.isfinite(got["blur_level"]).all()


def test_empty_and_flat_video():
    flat_pixels = np.full((1, 200, 266), 128, np.uint8)
    flat = [_Frame(flat_pixels, k / 30, k) for k in range(12)]
    # Starts at 0, 5, 10 and 15: textured at 0 and 10, flat at 5 and 15, so only two frames have a value.
    textured = _video(20, (200, 266))
    mixed = [_Frame(f.pixels if k % 10 == 0 else flat_pixels, k / 30, k) for k, f in enumerate(textured)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a median of nothing would warn here
        warnings.simplefilter("ignore", ExperimentalWarning)
        got = _run([[], flat, mixed])
    assert np.isnan(got["blur_level"][:2]).all()
    values = [blur_stat(textured[k].pixels[0].astype(np.float32)) for k in (0, 10)]
    assert got["blur_level"][2] == _median([v.blur for v in values])


def test_blur_returns_only_the_level_column():
    got = _run([_video(10, (200, 266))])
    assert list(got) == ["blur_level"]
    assert np.isfinite(got["blur_level"]).all()


def _left_out(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "left out of the BLUR median" in r.getMessage()]


def test_mixed_sizes_measure_only_the_first_size(monkeypatch, caplog):
    caplog.set_level("WARNING", logger="dataeval.core")
    shapes = []

    def spy(gray):
        shapes.append(gray.shape)
        return blur_stat(gray)

    monkeypatch.setattr(blur_module, "blur_stat", spy)
    # Starts at 0, 5, 10 and 15, alternating a small, a large and a smaller frame. The large one is not shrunk
    # first, so a frame that was measured would show at (720, 900).
    small, large, smaller = (_video(1, size)[0].pixels for size in ((200, 260), (800, 1000), (160, 220)))
    pixels = [[small, large, smaller, small][k // 5] if k % 5 == 0 else small for k in range(20)]
    frames = [_Frame(p, k / 30, k) for k, p in enumerate(pixels)]
    got = _run([frames])
    assert set(shapes) == {(200, 260)}
    assert np.isfinite(got["blur_level"]).all()
    # One message per other size, each saying which size the median reads.
    tail = "are left out of the BLUR median, which reads only frames of the first frame's size, 260x200. — indices: 0"
    assert sorted(_left_out(caplog)) == [f"Frames of 1000x800 {tail}", f"Frames of 220x160 {tail}"]


def test_a_size_change_keeps_the_first_size(caplog):
    caplog.set_level("WARNING", logger="dataeval.core")
    first = _video(30, (200, 266), drift=True)
    # Frames 30 to 59 are another size, so a start among them would move the median of a drifting video.
    later = [_Frame(f.pixels, f.time_s + 1, 30 + k) for k, f in enumerate(_video(30, (160, 214), seed=1, drift=True))]
    got = _run([first + later])
    want = _run([first])
    assert got["blur_level"][0] == want["blur_level"][0]
    assert len(_left_out(caplog)) == 1


def test_a_frame_of_another_size_does_not_take_a_start_place():
    # Starts every 5 frames, at most 4 kept. Only frames 0-4, 15-19, 30-34 and 45-49 have the first size, so the
    # starts 0, 15, 30 and 45 fit the 4 places, but a start of another size that took a place and forced the stride
    # to double would leave 0 alone.
    def first_size(k: int) -> bool:
        return (k // 5) % 3 == 0

    big, small = _video(30, (200, 266), drift=True), _video(30, (160, 214), seed=1, drift=True)
    frames = [_Frame((big if first_size(k) else small)[k % 30].pixels, k / 30, k) for k in range(60)]
    sampler = StartSampler(4)
    for k in filter(first_size, range(60)):
        sampler.offer(k)
    assert sampler.kept == [0, 15, 30, 45]
    values = [blur_stat(frames[k].pixels[0].astype(np.float32)) for k in sampler.kept]
    assert _run([frames], blur_samples=4)["blur_level"][0] == _median([v.blur for v in values])


def _short_side_video(size: int | None, n: int = 1) -> list[_Frame]:
    """`n` flat frames of a 4:3 landscape picture of short side `size`, or no frames at all."""
    pixels = np.zeros((1, size, size * 4 // 3), np.uint8) if size else None
    return [_Frame(pixels, k / 30, k) for k in range(n if size else 0)]


def _size_warnings(videos: list[Any], stats: VideoStats = VideoStats.BLUR) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(videos, stats)
    return [w for w in caught if "different short sides" in str(w.message)]


@pytest.mark.parametrize(
    ("sides", "listed"),
    [((360, 480), "2 different short sides (360, 480)"), ((480, 1080), "2 different short sides (480, 720)")],
)
def test_videos_measured_at_different_short_sides_warn_once(sides, listed):
    (caught,) = _size_warnings([_short_side_video(side) for side in sides])
    assert caught.category is UserWarning
    # Raised from the call, so the warning points at the caller's line and not at the library.
    assert (caught.filename, caught.lineno) == (__file__, _run.__code__.co_firstlineno + 1)
    assert f"videos measured at {listed}. An edge width is in pixels at the measured size" in str(caught.message)
    assert "`VideoStats.ALL & ~VideoStats.BLUR`" in str(caught.message)


def test_the_sides_are_listed_sorted_and_counted_once_each():
    (caught,) = _size_warnings([_short_side_video(side) for side in (480, 360, 480, 240)])
    assert "3 different short sides (240, 360, 480)" in str(caught.message)


@pytest.mark.parametrize(
    "sides",
    [
        (360, 360),
        # Both measure at 720, the cap.
        (1080, 2160),
        # An empty video measured nothing, so it reports no side.
        (None, 480),
        (480, None, 480),
    ],
)
def test_videos_measured_at_one_short_side_do_not_warn(sides):
    assert not _size_warnings([_short_side_video(side) for side in sides])


def test_the_side_is_the_first_frames_not_a_later_one():
    # Both videos begin at 200, so both report 200, whatever size a later frame has.
    plain = _short_side_video(200, 3)
    changing = [*_short_side_video(200, 1), *_short_side_video(480, 9)]
    changing = [_Frame(f.pixels, k / 30, k) for k, f in enumerate(changing)]
    assert not _size_warnings([plain, changing])


@pytest.mark.parametrize("stats", [VideoStats.BASIC, VideoStats.ALL & ~VideoStats.BLUR])
def test_the_size_warning_needs_a_blur_flag(stats):
    assert not _size_warnings([_short_side_video(360), _short_side_video(480)], stats)


def test_the_size_warning_is_given_with_the_threads_that_fan_videos_out():
    videos = [_short_side_video(side) for side in (360, 480, 480)]
    with use_max_processes(2):
        assert len(_size_warnings(videos)) == 1
