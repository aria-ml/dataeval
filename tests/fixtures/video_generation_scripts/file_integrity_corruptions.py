"""
file_integrity_corruptions.py
------------------------------
Corruptions that break the FILE itself, as opposed to transforming its
visual content (that's video_corruptions.py). These are the errors a real
ingestion pipeline actually produces: an interrupted upload, a disk-full
mid-write, someone renaming a file, a transcode step that drops timestamps,
a sensor dropout.

Several of these (truncated, corrupted_header, zero_byte) produce files that
ffprobe cannot parse at all — that's the point, and build_dataset.py handles
the probe failure gracefully rather than crashing.
"""

import os
import random
import shutil
import subprocess


def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_truncated(src: str, dst: str, keep_fraction: float = 0.4):
    """Copy only the first `keep_fraction` of the file's bytes - an
    interrupted write/download/copy."""
    size = os.path.getsize(src)
    keep = int(size * keep_fraction)
    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        f_out.write(f_in.read(keep))


def make_corrupted_header(src: str, dst: str, header_bytes: int = 256):
    """Overwrite the first N bytes with deterministic garbage. File size is
    unchanged, but the container header (ftyp/moov box, etc.) is destroyed,
    so most players/parsers will refuse to open it."""
    shutil.copyfile(src, dst)
    size = os.path.getsize(dst)
    n = min(header_bytes, size)
    with open(dst, "r+b") as f:
        f.write(random.Random(dst).randbytes(n))


def make_zero_byte(dst: str):
    """An empty file at the expected path - e.g. a failed export that still
    registered as 'done' in some tracking system."""
    open(dst, "wb").close()


def make_wrong_codec_container(src_avi_xvid: str, dst_mp4: str):
    """Take a raw XVID/AVI-encoded stream and save it with a .mp4 extension
    WITHOUT transcoding - the classic 'someone renamed the file' bug. Strict
    MP4 parsers will reject it; lenient ones may silently misdecode it."""
    shutil.copyfile(src_avi_xvid, dst_mp4)


def make_variable_frame_rate(src_mp4: str, dst_mp4: str):
    """Re-time frames with irregular (non-constant) intervals and mux with
    vsync=vfr, so per-frame durations genuinely vary instead of the stream
    being a clean constant frame rate. A tool should catch avg_frame_rate
    vs. actual per-frame timing mismatches on this file."""
    _run([
        "ffmpeg",
        "-y",
        "-i",
        src_mp4,
        "-vf",
        "setpts=PTS+0.5*sin(N/4)/TB",
        "-vsync",
        "vfr",
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


def make_black_frames(src_mp4: str, dst_mp4: str, start_sec: float, end_sec: float):
    """Blank every pixel to black between start_sec and end_sec while leaving
    the rest of the clip untouched - simulates a sensor dropout, a corrupted
    GOP, or an ingestion glitch that produced blank footage mid-clip."""
    _run([
        "ffmpeg",
        "-y",
        "-i",
        src_mp4,
        "-vf",
        (f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,{start_sec},{end_sec})'"),
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
