"""
frame_stutter.py
-----------------
Builds one clip that contains BOTH exact-duplicate frames (a real frozen-
capture / dropped-frame bug: frame N is byte-identical to frame N-1) and
near-duplicate frames (frame N is visually indistinguishable from frame N-1
but not byte-identical - e.g. sensor noise during a stall, or a
duplicate-frame insertion that went through a lossy re-encode).

Ground truth is MEASURED after the final H.264 encode/decode round-trip,
not just assumed pre-encode - so the manifest reflects what a reader of the
actual delivered file will see, including any codec-introduced drift.
"""

import cv2
import numpy as np
from scenes import COLORS, _draw_shape, make_clip

WIDTH, HEIGHT = 320, 240
FPS = 30


def render_stutter_clip(clip_id: str, out_path_avi: str, num_frames: int, exact_dup_groups, near_dup_groups):
    """
    exact_dup_groups: list of (source_frame_idx, [duplicate_frame_idx, ...])
    near_dup_groups:  list of (source_frame_idx, [duplicate_frame_idx, ...], noise_amplitude)

    Frames not covered by either group render normally from the clip's
    object trajectories. Returns the Clip object (for trajectory reuse) and
    a list of *intended* frame events (exact pixel verification happens
    separately, after encoding, in verify_frame_events).
    """
    clip = make_clip(clip_id, num_frames=num_frames, n_objects=2)
    rng = np.random.RandomState(clip.bg_noise_seed % (2**32))
    base_bg = rng.randint(20, 60)

    dup_map = {}
    for source, dups in exact_dup_groups:
        for f in dups:
            dup_map[f] = (source, "exact_duplicate", 0)
    for source, dups, amp in near_dup_groups:
        for f in dups:
            dup_map[f] = (source, "near_duplicate", amp)

    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path_avi, fourcc, clip.fps, (clip.width, clip.height))
    rendered = {}
    intended_events = []

    for f in range(num_frames):
        if f in dup_map:
            source, kind, amp = dup_map[f]
            base_canvas = rendered[source]
            if kind == "exact_duplicate":
                canvas = base_canvas.copy()
            else:
                noise = rng.randint(-amp, amp + 1, base_canvas.shape, dtype=np.int16)
                canvas = np.clip(base_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            intended_events.append({"frame": f, "type": kind, "duplicate_of_frame": source})
        else:
            canvas = np.full((clip.height, clip.width, 3), base_bg, dtype=np.uint8)
            noise = rng.randint(-4, 4, canvas.shape, dtype=np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            for t in clip.tracks:
                if f in t.boxes:
                    _draw_shape(canvas, t.class_name, t.boxes[f], COLORS[t.class_name])
        rendered[f] = canvas
        writer.write(canvas)
    writer.release()
    return clip, intended_events


def verify_frame_events(final_mp4_path: str, intended_events: list):
    """
    Re-read the ACTUAL encoded file and measure, per planted event, the real
    pixel difference between the duplicate frame and its claimed source
    frame. This is what goes in the manifest - measured, not assumed.
    """
    cap = cv2.VideoCapture(final_mp4_path)
    frames = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames[idx] = frame
        idx += 1
    cap.release()

    verified = []
    for ev in intended_events:
        f, source = ev["frame"], ev["duplicate_of_frame"]
        if f not in frames or source not in frames:
            continue
        diff = np.abs(frames[f].astype(np.int16) - frames[source].astype(np.int16))
        max_abs_diff = int(diff.max())
        mean_abs_diff = round(float(diff.mean()), 4)
        verified.append({
            "frame": f,
            "duplicate_of_frame": source,
            "claimed_type": ev["type"],
            "measured_max_abs_pixel_diff": max_abs_diff,
            "measured_mean_abs_pixel_diff": mean_abs_diff,
            "measured_exact": max_abs_diff == 0,
        })
    return verified
