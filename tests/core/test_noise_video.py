"""The noise statistics on decoded video whose added noise is known.

The scene is the synthetic videos' (`tests/fixtures/video_generation_scripts/scenes.py`): moving shapes over a
background that carries its own per-frame texture, here at mid-gray and converted to one gray channel. Seeded white
Gaussian noise of a known sigma is added to the same frames in float, rounded and clipped to uint8, and each clip goes
through ffmpeg's lossless FFV1 codec, so the decoded frames hold the noise unchanged. The clean frames' levels lie more
than 4 sigma from 0 and 255 at the largest sigma, so clipping is rare. The clean clip's reading is the scene's own
floor, from its texture and its shapes' edges, measured here and not assumed.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats

pytestmark = pytest.mark.ffmpeg

SIGMAS = (0, 4, 8, 16)
N_FRAMES = 30


def _scene(n_frames: int) -> np.ndarray:
    """The scene of `scenes.render_clip` with its background at mid-gray, in gray, as ``(n_frames, H, W)`` uint8."""
    import cv2

    from tests.fixtures.video_generation_scripts.scenes import COLORS, _draw_shape, make_clip

    clip = make_clip("noise_scene", num_frames=n_frames)
    rng = np.random.RandomState(clip.bg_noise_seed)
    frames = []
    for f in range(n_frames):
        canvas = np.full((clip.height, clip.width, 3), 128, dtype=np.int16)
        canvas = (canvas + rng.randint(-4, 4, canvas.shape, dtype=np.int16)).astype(np.uint8)
        for track in clip.tracks:
            _draw_shape(canvas, track.class_name, track.boxes[f], COLORS[track.class_name])
        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY))
    return np.stack(frames)


def _round_trip(frames: np.ndarray, path: Path) -> np.ndarray:
    """Write gray frames with FFV1 and decode them back."""
    n, height, width = frames.shape
    raw = ["-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}"]
    encode = ["ffmpeg", "-loglevel", "error", "-y", *raw, "-r", "30", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "gray"]
    subprocess.run([*encode, str(path)], input=frames.tobytes(), check=True)
    decode = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    out = subprocess.run(decode, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(-1, height, width)


def build_clips(directory: Path, n_frames: int = N_FRAMES, seed: int = 0) -> dict[int, np.ndarray]:
    """Each sigma's clip, written and decoded, as ``(n_frames, H, W)`` uint8."""
    clean = _scene(n_frames).astype(np.float64)
    assert clean.min() - 4 * max(SIGMAS) > 0, "levels too near 0"
    assert clean.max() + 4 * max(SIGMAS) < 255, "levels too near 255"
    clips = {}
    for sigma in SIGMAS:
        noise = np.random.default_rng([seed, sigma]).normal(0.0, sigma, clean.shape) if sigma else 0.0
        frames = np.clip(np.round(clean + noise), 0, 255).astype(np.uint8)
        clips[sigma] = _round_trip(frames, directory / f"sigma_{sigma}.mkv")
        np.testing.assert_array_equal(clips[sigma], frames, err_msg="FFV1 should be lossless")
    return clips


@pytest.fixture(scope="module")
def readings(tmp_path_factory: pytest.TempPathFactory) -> dict[int, dict[str, np.ndarray]]:
    """Each clip's per-frame `noise_std` and `noise_var`, read by `image_stats` on the decoded frames."""
    pytest.importorskip("cv2", reason="synthetic video fixtures need opencv")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            pytest.skip(f"synthetic video fixtures need {tool} on PATH")
    clips = build_clips(tmp_path_factory.mktemp("noise_videos"))
    out = {}
    for sigma, frames in clips.items():
        result = image_stats(list(frames[:, None]), stats=ImageStats.NOISE)
        assert "details" in result
        assert set(result["details"]["noise_status"]) == {"ok"}
        out[sigma] = {name: np.asarray(result["stats"][name], dtype=np.float64) for name in ("noise_std", "noise_var")}
    return out


def test_the_reading_rises_with_the_noise(readings: dict[int, dict[str, np.ndarray]]):
    medians = [np.median(readings[sigma]["noise_std"]) for sigma in SIGMAS]
    assert np.all(np.diff(medians) > 0), medians


@pytest.mark.parametrize("sigma", [8, 16])
def test_the_variance_above_the_clean_clip_is_the_noise_added(readings: dict[int, dict[str, np.ndarray]], sigma: int):
    """Frame by frame, against the same frame of the clean clip. Measured between 0.99 and 1.01 over ten noise seeds."""
    added = (readings[sigma]["noise_var"] - readings[0]["noise_var"]) / sigma**2
    assert np.median(added) == pytest.approx(1.0, rel=0.1)
