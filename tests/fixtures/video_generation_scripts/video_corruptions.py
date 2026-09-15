"""
video_corruptions.py
---------------------
Turns a canonical clean clip into content-duplicate variants:

  - near_duplicate : same codec family, heavier compression + tiny resize.
                      Byte-identical hash will differ; perceptual hash should
                      still consider it "the same video".
  - reencode        : different codec/container entirely (H.264 mp4 -> VP9 webm).
  - reframed        : cropped, rescaled, and retimed (fps change) - simulates
                      someone re-cutting/re-exporting the same footage.

All are produced with ffmpeg so the pixel content is genuinely transformed,
not just copied with a new filename.
"""

import shutil
import subprocess


def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_canonical_mp4(src_avi: str, dst_mp4: str, fps: int = 30):
    _run([
        "ffmpeg",
        "-y",
        "-i",
        src_avi,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        dst_mp4,
    ])


def make_lossless_mp4(src_avi: str, dst_mp4: str, fps: int = 30):
    """crf 0 = mathematically lossless x264. Needed whenever a downstream
    consumer needs to verify BIT-EXACT frame equality (e.g. planted exact-
    duplicate frames) rather than 'visually identical' - ordinary crf>0
    encoding does not guarantee identical decoded output for identical
    input frames, since rate control/quantization can still introduce tiny
    per-frame drift even at zero motion."""
    _run([
        "ffmpeg",
        "-y",
        "-i",
        src_avi,
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-preset",
        "veryslow",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        dst_mp4,
    ])


def make_exact_duplicate(src_mp4: str, dst_mp4: str):
    """Creates an exact, bit-for-bit file copy."""
    shutil.copyfile(src_mp4, dst_mp4)


def make_near_duplicate(src_mp4: str, dst_mp4: str):
    """Same codec, heavier compression (crf 32) + 2% downscale. Visually near-identical."""
    _run([
        "ffmpeg",
        "-y",
        "-i",
        src_mp4,
        "-vf",
        "scale=trunc(iw*0.98/2)*2:trunc(ih*0.98/2)*2",
        "-c:v",
        "libx264",
        "-crf",
        "32",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        dst_mp4,
    ])


def make_reencode(src_mp4: str, dst_webm: str):
    """Different codec + container entirely: H.264/mp4 -> VP9/webm."""
    _run([
        "ffmpeg",
        "-y",
        "-i",
        src_mp4,
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "400k",
        dst_webm,
    ])


def make_reframed(src_mp4: str, dst_mp4: str, crop_frac: float = 0.85, new_fps: int = 24):
    """Crop borders, rescale back to a standard size, and change frame rate."""
    _run([
        "ffmpeg",
        "-y",
        "-i",
        src_mp4,
        "-vf",
        f"crop=trunc(iw*{crop_frac}/2)*2:trunc(ih*{crop_frac}/2)*2,scale=320:240,fps={new_fps}",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        dst_mp4,
    ])
