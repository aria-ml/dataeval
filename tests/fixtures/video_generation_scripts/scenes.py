"""
scenes.py
---------
Generates small synthetic video clips with moving colored shapes, where every
bounding box is known exactly (we drew it), so ground truth is never a guess.

Each clip gets a deterministic RNG seed derived from its clip_id, so the whole
dataset is reproducible.
"""

import hashlib
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

WIDTH, HEIGHT = 320, 240
FPS = 30
CLASSES = ["square", "circle", "triangle"]
COLORS = {
    "square": (60, 120, 220),  # BGR
    "circle": (60, 200, 90),
    "triangle": (210, 90, 200),
}


def seed_for(clip_id: str) -> int:
    return int(hashlib.sha256(clip_id.encode()).hexdigest(), 16) % (2**32)


@dataclass
class Track:
    track_id: int
    class_name: str
    boxes: dict  # frame_idx -> (x, y, w, h)
    first_frame: int
    last_frame: int


@dataclass
class Clip:
    clip_id: str
    num_frames: int
    fps: int = FPS
    width: int = WIDTH
    height: int = HEIGHT
    tracks: list = field(default_factory=list)
    bg_noise_seed: int = 0


def _draw_shape(canvas, class_name, box, color):
    x, y, w, h = box
    if class_name == "square":
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
    elif class_name == "circle":
        cv2.circle(canvas, (x + w // 2, y + h // 2), w // 2, color, -1)
    elif class_name == "triangle":
        pts = np.array([
            [x + w // 2, y],
            [x, y + h],
            [x + w, y + h],
        ])
        cv2.fillPoly(canvas, [pts], color)


def make_clip(clip_id: str, num_frames: int = 90, n_objects: int = 2) -> Clip:
    """Build a clip with n_objects bouncing shapes and exact per-frame boxes."""
    rng = random.Random(seed_for(clip_id))
    tracks = []
    for i in range(n_objects):
        class_name = rng.choice(CLASSES)
        size = rng.randint(24, 36)
        x = rng.randint(0, WIDTH - size)
        y = rng.randint(0, HEIGHT - size)
        vx = rng.choice([-1, 1]) * rng.randint(2, 5)
        vy = rng.choice([-1, 1]) * rng.randint(2, 5)
        boxes = {}
        for f in range(num_frames):
            if x <= 0 or x + size >= WIDTH:
                vx = -vx
            if y <= 0 or y + size >= HEIGHT:
                vy = -vy
            x = max(0, min(WIDTH - size, x + vx))
            y = max(0, min(HEIGHT - size, y + vy))
            boxes[f] = (int(x), int(y), size, size)
        tracks.append(
            Track(
                track_id=i + 1,
                class_name=class_name,
                boxes=boxes,
                first_frame=0,
                last_frame=num_frames - 1,
            )
        )
    return Clip(clip_id=clip_id, num_frames=num_frames, tracks=tracks, bg_noise_seed=seed_for(clip_id + "_bg"))


def render_clip(clip: Clip, out_path_avi: str):
    """Render a Clip to a raw .avi (later transcoded to canonical mp4 via ffmpeg)."""
    rng = np.random.RandomState(clip.bg_noise_seed % (2**32))
    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path_avi, fourcc, clip.fps, (clip.width, clip.height))
    base_bg = rng.randint(20, 60)
    for f in range(clip.num_frames):
        canvas = np.full((clip.height, clip.width, 3), base_bg, dtype=np.uint8)
        noise = rng.randint(-4, 4, canvas.shape, dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        for t in clip.tracks:
            if f in t.boxes:
                _draw_shape(canvas, t.class_name, t.boxes[f], COLORS[t.class_name])
        writer.write(canvas)
    writer.release()


def clip_to_annotation_records(clip: Clip):
    """Ground-truth annotation records: one dict per (track, frame)."""
    records = []
    for t in clip.tracks:
        for f, (x, y, w, h) in sorted(t.boxes.items()):
            records.append({
                "frame": f,
                "track_id": t.track_id,
                "class": t.class_name,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            })
    return records
