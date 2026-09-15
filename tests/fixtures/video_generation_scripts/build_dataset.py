"""
build_dataset.py
-----------------
Builds the full synthetic test dataset for a video dataset-cleaning tool:

  videos/           - all clip files (base clips + planted duplicates/leaks)
  annotations/       - clean + corrupted annotation JSON per base clip
  manifest.json      - the single source of ground truth for every planted
                        error: near-duplicates, re-encodes, reframed footage,
                        cross-split leakage (with exact containment ratios),
                        and annotation errors (id_switch, box_drift,
                        temporal_offset, fragmentation, ghost_track).

Run: python3 build_dataset.py
Everything is deterministic (seeded from clip_id), so re-running regenerates
byte-for-byte identical ground truth (video pixel bytes may vary slightly
across ffmpeg versions, but structure/labels never do).
"""

import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from . import file_integrity_corruptions as fic
from . import frame_stutter
from .annotation_corruptions import apply_errors, inject_global_off_by_one
from .scenes import clip_to_annotation_records, make_clip, render_clip
from .video_corruptions import (
    make_canonical_mp4,
    make_exact_duplicate,
    make_lossless_mp4,
    make_near_duplicate,
    make_reencode,
    make_reframed,
)


def ffprobe_info(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_frames,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        "-count_frames",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    fmt = data.get("format", {})
    num, den = stream["r_frame_rate"].split("/")
    fps = round(float(num) / float(den), 3)
    return {
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "num_frames": int(stream.get("nb_read_frames", 0)),
        "duration_sec": round(float(fmt.get("duration", 0)), 3),
    }


def safe_probe(path: Path) -> dict:
    """Like ffprobe_info, but for files that are deliberately broken
    (truncated / corrupted header / zero-byte) and can't be parsed at all.
    Returns {'readable': False, ...} instead of raising."""
    try:
        info = ffprobe_info(path)
        info["readable"] = True
        return info
    except subprocess.CalledProcessError:
        return {
            "readable": False,
            "actual_size_bytes": path.stat().st_size if path.exists() else 0,
            "probe_error": "ffprobe could not parse this file (expected for this corruption type)",
        }


def build_synthetic_dataset(output_dir: Path) -> Path:
    """Generates the full synthetic test dataset in the specified directory."""
    videos_dir = output_dir / "videos"
    annotations_dir = output_dir / "annotations"
    videos_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    base_clip_ids = [f"clip_{i:03d}" for i in range(1, 10)]  # clip_001..clip_009
    num_frames = 90

    base_clips = {}
    base_records = {}
    wrong_codec_source = "clip_003"

    for cid in base_clip_ids:
        clip = make_clip(cid, num_frames=num_frames, n_objects=2)
        avi_path = videos_dir / f"_tmp_{cid}.avi"
        mp4_path = videos_dir / f"{cid}.mp4"
        render_clip(clip, str(avi_path))
        make_canonical_mp4(str(avi_path), str(mp4_path))
        if cid != wrong_codec_source:
            avi_path.unlink()
        base_clips[cid] = clip
        base_records[cid] = clip_to_annotation_records(clip)

    # Pre-leakage splits
    splits = {
        "train": ["clip_001", "clip_002", "clip_003", "clip_004", "clip_005", "clip_009"],
        "val": ["clip_006", "clip_007"],
        "test": ["clip_008"],
    }

    duplicate_plants = []
    # -- within-split duplicates (same split as their source; not leakage) --
    within_split_plan = [
        ("clip_004", "near_duplicate", "train_dup_001", "train"),
        ("clip_005", "reencode", "train_reenc_001", "train"),
        ("clip_001", "reframed", "train_reframed_001", "train"),
        ("clip_008", "exact_duplicate", "train_dup_002", "train"),
    ]

    # -- cross-split leakage: source lives in train, duplicate planted in val/test --
    leakage_plan = [
        ("clip_001", "near_duplicate", "val_leak_001", "val"),
        ("clip_003", "reframed", "val_leak_002", "val"),
        ("clip_002", "reencode", "test_leak_001", "test"),
    ]

    def make_derivative(source_id, method, new_id, target_split):
        src = videos_dir / f"{source_id}.mp4"
        if method == "near_duplicate":
            dst = videos_dir / f"{new_id}.mp4"
            make_near_duplicate(str(src), str(dst))
        elif method == "reencode":
            dst = videos_dir / f"{new_id}.webm"
            make_reencode(str(src), str(dst))
        elif method == "reframed":
            dst = videos_dir / f"{new_id}.mp4"
            make_reframed(str(src), str(dst))
        elif method == "exact_duplicate":
            dst = videos_dir / f"{new_id}.mp4"
            make_exact_duplicate(str(src), str(dst))
        else:
            raise ValueError(method)
        splits[target_split].append(new_id)
        return {
            "clip_id": new_id,
            "source_clip_id": source_id,
            "relationship": method,
            "split": target_split,
            "source_split": [s for s, members in splits.items() if source_id in members][0],
            "is_cross_split_leak": None,  # filled below
            "file": dst.name,
        }

    for source_id, method, new_id, target_split in within_split_plan:
        d = make_derivative(source_id, method, new_id, target_split)
        d["is_cross_split_leak"] = False
        duplicate_plants.append(d)

    for source_id, method, new_id, target_split in leakage_plan:
        d = make_derivative(source_id, method, new_id, target_split)
        d["is_cross_split_leak"] = True
        duplicate_plants.append(d)

    # -- leakage summary with exact, assertable containment ratios --
    leakage_summary = {}
    for split_name, members in splits.items():
        leaked = [d for d in duplicate_plants if d["split"] == split_name and d["is_cross_split_leak"]]
        total = len(members)
        leakage_summary[split_name] = {
            "total_clips": total,
            "leaked_clip_ids": [d["clip_id"] for d in leaked],
            "leaked_from_split": sorted({d["source_split"] for d in leaked}) if leaked else [],
            "num_leaked": len(leaked),
            "containment_ratio": round(len(leaked) / total, 4) if total else 0.0,
        }

    # File-integrity errors: corruptions to the FILE itself, not its content.
    file_integrity_plants = []

    # (source_clip_id, corruption_type, new_clip_id, target_split, extra_kwargs)
    file_integrity_plan = [
        ("clip_006", "truncated", "truncated_001", "val", {"keep_fraction": 0.35}),
        # ("clip_007", "corrupted_header", "corrupted_header_001", "val", {}),  # uncomment if testing corrupted files
        # ("clip_002", "zero_byte", "zero_byte_001", "train", {}),  # uncomment if testing corrupted files
        (wrong_codec_source, "wrong_codec_container", "wrong_codec_001", "train", {}),
        ("clip_004", "variable_frame_rate", "vfr_001", "train", {}),
        ("clip_005", "black_frames", "black_frames_001", "train", {"start_sec": 1.0, "end_sec": 2.0}),
    ]

    expected_behavior = {
        "truncated": "File is incomplete/unreadable past truncation point.",
        "corrupted_header": "Container header is garbage.",
        "zero_byte": "File exists but has zero bytes.",
        "wrong_codec_container": "File has .mp4 extension but contains raw XVID/AVI stream.",
        "variable_frame_rate": "Per-frame timestamps are irregular.",
        "black_frames": "A contiguous time window is fully black.",
    }

    for source_id, ctype, new_id, target_split, kwargs in file_integrity_plan:
        src_mp4 = videos_dir / f"{source_id}.mp4"
        if ctype == "truncated":
            dst = videos_dir / f"{new_id}.mp4"
            fic.make_truncated(str(src_mp4), str(dst), **kwargs)
        elif ctype == "corrupted_header":
            dst = videos_dir / f"{new_id}.mp4"
            fic.make_corrupted_header(str(src_mp4), str(dst), **kwargs)
        elif ctype == "zero_byte":
            dst = videos_dir / f"{new_id}.mp4"
            fic.make_zero_byte(str(dst))
        elif ctype == "wrong_codec_container":
            src_avi = videos_dir / f"_tmp_{source_id}.avi"
            dst = videos_dir / f"{new_id}.mp4"
            fic.make_wrong_codec_container(str(src_avi), str(dst))
        elif ctype == "variable_frame_rate":
            dst = videos_dir / f"{new_id}.mp4"
            fic.make_variable_frame_rate(str(src_mp4), str(dst))
        elif ctype == "black_frames":
            dst = videos_dir / f"{new_id}.mp4"
            fic.make_black_frames(str(src_mp4), str(dst), **kwargs)
        else:
            raise ValueError(ctype)

        splits[target_split].append(new_id)
        file_integrity_plants.append({
            "clip_id": new_id,
            "source_clip_id": source_id,
            "corruption_type": ctype,
            "split": target_split,
            "file": dst.name,
            "params": kwargs,
            "expected_behavior": expected_behavior[ctype],
        })

    # now safe to remove the wrong-codec source's raw avi
    (videos_dir / f"_tmp_{wrong_codec_source}.avi").unlink(missing_ok=True)

    # Internal frame duplicates and near duplicates
    exact_dup_groups = [(20, [21]), (60, [61, 62])]  # a couple of exact-duplicate events
    near_dup_groups = [(35, [36], 2), (75, [76, 77], 4)]  # a couple of near-duplicate events

    stutter_avi = videos_dir / "_tmp_frame_stutter_001.avi"
    stutter_mp4 = videos_dir / "frame_stutter_001.mp4"
    _stutter_clip, intended_events = frame_stutter.render_stutter_clip(
        "frame_stutter_001",
        str(stutter_avi),
        num_frames,
        exact_dup_groups,
        near_dup_groups,
    )
    make_lossless_mp4(str(stutter_avi), str(stutter_mp4))
    stutter_avi.unlink()

    verified_frame_events = frame_stutter.verify_frame_events(str(stutter_mp4), intended_events)
    splits["train"].append("frame_stutter_001")
    for ev in verified_frame_events:
        tag = "OK" if ev["measured_exact"] == (ev["claimed_type"] == "exact_duplicate") else "MISMATCH"
        print(
            f"  frame {ev['frame']} dup-of {ev['duplicate_of_frame']}: "
            f"claimed={ev['claimed_type']} measured_exact={ev['measured_exact']} [{tag}]"
        )

    # Annotation errors, planted on the base clips only
    annotation_error_plan = {
        "clip_001": ["id_switch"],
        "clip_002": ["box_drift"],
        "clip_003": ["temporal_offset"],
        "clip_004": ["fragmentation"],
        "clip_005": ["ghost_track"],
        "clip_006": ["id_switch", "box_drift"],
        "clip_007": ["fragmentation", "ghost_track"],
        "clip_008": [],  # control clip: must stay clean, no planted errors
        "clip_009": ["off_by_one_global"],  # systemic, not per-track - see annotation_corruptions.py
    }

    annotation_manifest = {}
    for cid, error_types in annotation_error_plan.items():
        rng = random.Random(f"anno-{cid}")
        clean_records = base_records[cid]
        clean_path = annotations_dir / f"{cid}-clean.json"
        clean_path.write_text(json.dumps(clean_records, indent=2))

        if error_types == ["off_by_one_global"]:
            corrupted_records, descriptor = inject_global_off_by_one(clean_records, base_clips[cid].num_frames, delta=1)
            descriptors = [descriptor]
        elif error_types:
            corrupted_records, descriptors = apply_errors(clean_records, error_types, base_clips[cid], rng)
        else:
            corrupted_records, descriptors = clean_records, []

        corrupted_path = annotations_dir / f"{cid}-corrupted.json"
        corrupted_path.write_text(json.dumps(corrupted_records, indent=2))

        annotation_manifest[cid] = {
            "clean_annotations": clean_path.name,
            "corrupted_annotations": corrupted_path.name,
            "planted_errors": descriptors,
            "is_control_clean_clip": len(error_types) == 0,
        }

    # Probe every generated file so the manifest has real technical metadata
    file_info = {}
    for cid in base_clip_ids:
        file_info[cid] = safe_probe(videos_dir / f"{cid}.mp4")
    for d in duplicate_plants:
        file_info[d["clip_id"]] = safe_probe(videos_dir / d["file"])
    for d in file_integrity_plants:
        file_info[d["clip_id"]] = safe_probe(videos_dir / d["file"])
    file_info["frame_stutter_001"] = safe_probe(stutter_mp4)

    # Create manifest.json
    clips_section = {}
    for cid in base_clip_ids:
        clips_section[cid] = {
            "file": f"{cid}.mp4",
            "kind": "base",
            "split": [s for s, m in splits.items() if cid in m][0],
            **file_info[cid],
            "annotations": annotation_manifest[cid],
        }
    for d in duplicate_plants:
        clips_section[d["clip_id"]] = {
            "file": d["file"],
            "kind": "derivative",
            "split": d["split"],
            "derived_from": d["source_clip_id"],
            "relationship": d["relationship"],
            "is_cross_split_leak": d["is_cross_split_leak"],
            **file_info[d["clip_id"]],
            "annotations": annotation_manifest[d["source_clip_id"]],
        }
    for d in file_integrity_plants:
        clips_section[d["clip_id"]] = {
            "file": d["file"],
            "kind": "corrupted",
            "split": d["split"],
            "derived_from": d["source_clip_id"],
            "corruption_type": d["corruption_type"],
            "expected_behavior": d["expected_behavior"],
            **file_info[d["clip_id"]],
        }
    clips_section["frame_stutter_001"] = {
        "file": stutter_mp4.name,
        "kind": "frame_stutter",
        "split": "train",
        "note": "Contains both exact-duplicate and near-duplicate frames; see "
        "frame_level_issues for the measured, per-event ground truth.",
        **file_info["frame_stutter_001"],
    }

    manifest = {
        "dataset_name": "synthetic_video_cleaning_testset",
        "version": "1.1",
        "generation": "fully synthetic; all ground truth is exact by construction",
        "splits": splits,
        "clips": clips_section,
        "duplicate_relationships": duplicate_plants,
        "leakage_summary": leakage_summary,
        "file_integrity_issues": file_integrity_plants,
        "frame_level_issues": {
            "clip_id": "frame_stutter_001",
            "events": verified_frame_events,
            "note": "measured_exact / measured_*_pixel_diff are computed by re-reading the "
            "final encoded file, not assumed from the pre-encode generation step.",
        },
        "error_type_catalog": {
            "video_content_level": ["near_duplicate", "reencode", "reframed"],
            "file_integrity_level": [
                "truncated",
                "corrupted_header",
                "zero_byte",
                "wrong_codec_container",
                "variable_frame_rate",
                "black_frames",
            ],
            "frame_level": ["exact_duplicate_frame", "near_duplicate_frame"],
            "annotation_level": [
                "id_switch",
                "box_drift",
                "temporal_offset",
                "fragmentation",
                "ghost_track",
                "off_by_one_global",
            ],
            "referential_level": ["orphaned_annotation", "missing_annotation_file"],
        },
        "control_clips": {
            "clean_clip_no_planted_errors": "clip_008",
            "note": "Use this to check your tool's false-positive rate.",
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return output_dir


if __name__ == "__main__":
    default_dir = Path(__file__).resolve().parent.parent / "test_videos"
    build_synthetic_dataset(default_dir)
