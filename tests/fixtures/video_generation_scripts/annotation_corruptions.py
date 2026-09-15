"""
annotation_corruptions.py
--------------------------
Takes the *exact* ground-truth track records for a synthetic clip and plants
one specific, well-defined annotation error at a time. Each injector returns
the corrupted records PLUS a small "descriptor" dict that states precisely
what was done and where — this descriptor is what goes into the manifest and
is what a test asserts against (e.g. "tool must report an id_switch on
track 1/2 at frame 41", not "tool must report *something* is wrong").

All injectors operate on a deep copy and never mutate their input.
"""

import copy


def _by_track(records):
    tracks = {}
    for r in records:
        tracks.setdefault(r["track_id"], []).append(r)
    for k in tracks:
        tracks[k].sort(key=lambda r: r["frame"])
    return tracks


def inject_id_switch(records, rng):
    """Two tracks swap identities from a frame onward (classic MOT ID switch)."""
    tracks = _by_track(records)
    ids = list(tracks.keys())
    if len(ids) < 2:
        return records, None
    a, b = rng.sample(ids, 2)
    frames_a = {r["frame"] for r in tracks[a]}
    frames_b = {r["frame"] for r in tracks[b]}
    overlap = sorted(frames_a & frames_b)
    if len(overlap) < 4:
        return records, None
    swap_frame = overlap[len(overlap) // 2]
    out = copy.deepcopy(records)
    for r in out:
        if r["track_id"] == a and r["frame"] >= swap_frame:
            r["track_id"] = b
        elif r["track_id"] == b and r["frame"] >= swap_frame:
            r["track_id"] = a
    descriptor = {
        "type": "id_switch",
        "track_a": a,
        "track_b": b,
        "swap_frame": swap_frame,
    }
    return out, descriptor


def inject_box_drift(records, rng, rate_px_per_frame=0.8):
    """One track's box slowly drifts away from the true object position."""
    tracks = _by_track(records)
    ids = list(tracks.keys())
    track_id = rng.choice(ids)
    frames = [r["frame"] for r in tracks[track_id]]
    start_frame = frames[len(frames) // 3]
    out = copy.deepcopy(records)
    for r in out:
        if r["track_id"] == track_id and r["frame"] >= start_frame:
            delta = int(round(rate_px_per_frame * (r["frame"] - start_frame)))
            r["x"] += delta
            r["y"] += delta // 2
    descriptor = {
        "type": "box_drift",
        "track_id": track_id,
        "start_frame": start_frame,
        "rate_px_per_frame": rate_px_per_frame,
    }
    return out, descriptor


def inject_temporal_offset(records, rng, num_frames, offset_frames=5):
    """One track's timestamps are shifted relative to the true video frames."""
    tracks = _by_track(records)
    ids = list(tracks.keys())
    track_id = rng.choice(ids)
    offset = rng.choice([1, -1]) * offset_frames
    out = []
    for r in records:
        if r["track_id"] != track_id:
            out.append(copy.deepcopy(r))
            continue
        new_frame = r["frame"] + offset
        if 0 <= new_frame < num_frames:
            r2 = copy.deepcopy(r)
            r2["frame"] = new_frame
            out.append(r2)
    descriptor = {
        "type": "temporal_offset",
        "track_id": track_id,
        "offset_frames": offset,
    }
    return out, descriptor


def inject_fragmentation(records, rng, next_track_id, gap_frames=3):
    """One continuous track is split into two track_ids with a dropped-frame gap."""
    tracks = _by_track(records)
    ids = list(tracks.keys())
    track_id = rng.choice(ids)
    frames = [r["frame"] for r in tracks[track_id]]
    if len(frames) < 10:
        return records, None, next_track_id
    split_idx = len(frames) // 2
    split_frame = frames[split_idx]
    gap_end = split_frame + gap_frames
    new_id = next_track_id
    out = []
    for r in records:
        if r["track_id"] != track_id:
            out.append(copy.deepcopy(r))
            continue
        if r["frame"] < split_frame:
            out.append(copy.deepcopy(r))
        elif r["frame"] < gap_end:
            continue  # dropped -> the gap
        else:
            r2 = copy.deepcopy(r)
            r2["track_id"] = new_id
            out.append(r2)
    descriptor = {
        "type": "fragmentation",
        "original_track_id": track_id,
        "new_track_id": new_id,
        "split_frame": split_frame,
        "gap_frames": gap_frames,
    }
    return out, descriptor, new_id + 1


def inject_ghost_track(records, rng, next_track_id, num_frames, width=320, height=240):
    """A fabricated track with no corresponding real object, mid-clip."""
    start = num_frames // 4
    end = min(num_frames - 1, start + num_frames // 3)
    x0, y0 = rng.randint(10, width - 60), rng.randint(10, height - 60)
    vx, vy = rng.choice([-2, 2]), rng.choice([-1, 1])
    new_id = next_track_id
    out = copy.deepcopy(records)
    for f in range(start, end + 1):
        out.append({
            "frame": f,
            "track_id": new_id,
            "class": "ghost",
            "x": max(0, x0 + vx * (f - start)),
            "y": max(0, y0 + vy * (f - start)),
            "w": 20,
            "h": 20,
        })
    descriptor = {
        "type": "ghost_track",
        "track_id": new_id,
        "start_frame": start,
        "end_frame": end,
    }
    return out, descriptor, new_id + 1


def inject_global_off_by_one(records, num_frames, delta=1):
    """
    Shift EVERY track's frame index by exactly `delta`, everywhere in the
    clip - a systemic indexing bug (e.g. 0-indexed vs 1-indexed frame
    numbering at export/import), not a per-track drift.

    This is deliberately a different shape of error than inject_temporal_offset:
      - temporal_offset: ONE arbitrarily chosen track, an arbitrary (usually
        larger) delta - simulates a single track's timestamps coming from a
        different clock/source.
      - off_by_one_global: ALL tracks, delta is always exactly +/-1 - simulates
        a fencepost bug in the pipeline itself, and is much harder to catch
        because a 1-frame shift on a slow-moving object still looks like a
        plausible box under naive IoU/overlap checks.
    """
    out = []
    for r in records:
        new_frame = r["frame"] + delta
        if 0 <= new_frame < num_frames:
            r2 = copy.deepcopy(r)
            r2["frame"] = new_frame
            out.append(r2)
    descriptor = {
        "type": "off_by_one_global",
        "delta_frames": delta,
        "scope": "all_tracks",
        "note": (
            "Contrast with 'temporal_offset': that shifts one chosen "
            "track by an arbitrary delta; this shifts every track by "
            "exactly one frame, everywhere - a systemic indexing bug, "
            "not a per-track timing drift."
        ),
    }
    return out, descriptor


def apply_errors(records, error_types, clip, rng):
    """
    Apply a list of error type names in sequence, chaining outputs, and return
    (corrupted_records, [descriptors]) — descriptors omit any error that
    couldn't be planted (e.g. not enough tracks to swap).
    """
    current = copy.deepcopy(records)
    descriptors = []
    next_id = max(r["track_id"] for r in records) + 1

    for etype in error_types:
        if etype == "id_switch":
            current, d = inject_id_switch(current, rng)
        elif etype == "box_drift":
            current, d = inject_box_drift(current, rng)
        elif etype == "temporal_offset":
            current, d = inject_temporal_offset(current, rng, clip.num_frames)
        elif etype == "fragmentation":
            current, d, next_id = inject_fragmentation(current, rng, next_id)
        elif etype == "ghost_track":
            current, d, next_id = inject_ghost_track(current, rng, next_id, clip.num_frames, clip.width, clip.height)
        elif etype == "off_by_one_global":
            current, d = inject_global_off_by_one(current, clip.num_frames)
        else:
            raise ValueError(f"Unknown error type: {etype}")
        if d is not None:
            descriptors.append(d)

    return current, descriptors
