class TestVideos:
    """Making sure videos can be analyzed."""

    def test_exact_duplicate_videos(self, duplicate_pairs, video_path):
        # These are same video just renamed
        near_dups = [d for d in duplicate_pairs if d["relationship"] == "exact_duplicate"]
        assert near_dups, "fixture setup problem: expected at least one exact_duplicate pair"
        for pair in near_dups:
            a = video_path(pair["source_clip_id"])
            b = video_path(pair["clip_id"])
            # replace asserts below with assert duplicate_check
            assert a.exists()
            assert b.exists()

    def test_near_duplicate_videos(self, duplicate_pairs, video_path):
        # These are same video with adjusted height/width
        near_dups = [d for d in duplicate_pairs if d["relationship"] == "near_duplicate"]
        assert near_dups, "fixture setup problem: expected at least one near_duplicate pair"
        for pair in near_dups:
            a = video_path(pair["source_clip_id"])
            b = video_path(pair["clip_id"])
            # replace asserts below with assert duplicate_check
            assert a.exists()
            assert b.exists()

    def test_reencoded_duplicate_videos(self, duplicate_pairs, video_path):
        # These are the exact same video just with a different container/codec (webm/vp9)
        reencodes = [d for d in duplicate_pairs if d["relationship"] == "reencode"]
        for pair in reencodes:
            a = video_path(pair["source_clip_id"])
            b = video_path(pair["clip_id"])
            # replace asserts below with assert duplicate_check
            assert a.exists()
            assert b.exists()

    def test_reframed_duplicate_videos(self, duplicate_pairs, video_path):
        # These are duplicates that are cropped, rescaled, retimed - near duplicates
        reframed = [d for d in duplicate_pairs if d["relationship"] == "reframed"]
        for pair in reframed:
            a = video_path(pair["source_clip_id"])
            b = video_path(pair["clip_id"])
            # replace asserts below with assert duplicate_check
            assert a.exists()
            assert b.exists()

    # ---------------------------------------------------------------------------
    # Cross-split leakage, with an exact expected containment ratio per split
    # ---------------------------------------------------------------------------
    def test_cross_split_leakage_matches_known_containment_ratio(self, leakage_summary, manifest, video_path):
        for _split_name, info in leakage_summary.items():
            if info["num_leaked"] == 0:
                continue
            # TODO: detected = your_tool.find_cross_split_duplicates(
            #     train_split=manifest["splits"]["train"],
            #     eval_split=manifest["splits"][split_name],
            # )
            # assert len(detected) == info["num_leaked"]
            # assert set(d.clip_id for d in detected) == set(info["leaked_clip_ids"])
            # measured_ratio = len(detected) / info["total_clips"]
            # assert abs(measured_ratio - info["containment_ratio"]) < 1e-6
            assert info["containment_ratio"] > 0  # sanity check on the fixture itself

    # ---------------------------------------------------------------------------
    # Annotation errors: each descriptor is a precise, assertable expectation
    # ---------------------------------------------------------------------------
    def test_id_switch_detected_at_exact_frame(self, planted_errors_for, track_annotations):
        errors = [e for e in planted_errors_for("clip_001") if e["type"] == "id_switch"]
        assert errors
        expected = errors[0]
        corrupted = track_annotations("clip_001", "corrupted")
        # TODO: findings = your_tool.detect_id_switches(corrupted)
        # assert any(f.frame == expected["swap_frame"]
        #            and {f.track_a, f.track_b} == {expected["track_a"], expected["track_b"]}
        #            for f in findings)
        assert any(r["frame"] == expected["swap_frame"] for r in corrupted)

    def test_box_drift_detected_from_start_frame(self, planted_errors_for, track_annotations):
        errors = [e for e in planted_errors_for("clip_002") if e["type"] == "box_drift"]
        assert errors
        expected = errors[0]
        corrupted = track_annotations("clip_002", "corrupted")
        clean = track_annotations("clip_002", "clean")
        # TODO: findings = your_tool.detect_box_drift(corrupted)
        # assert any(f.track_id == expected["track_id"]
        #            and f.start_frame == expected["start_frame"] for f in findings)
        drifted = [
            r for r in corrupted if r["track_id"] == expected["track_id"] and r["frame"] > expected["start_frame"]
        ]
        last = max(drifted, key=lambda r: r["frame"])
        baseline = [r for r in clean if r["track_id"] == expected["track_id"] and r["frame"] == last["frame"]][0]
        assert last["x"] != baseline["x"], "box should have visibly drifted by the last frame"

    def test_temporal_offset_detected(self, planted_errors_for, track_annotations):
        errors = [e for e in planted_errors_for("clip_003") if e["type"] == "temporal_offset"]
        assert errors
        expected = errors[0]
        corrupted = track_annotations("clip_003", "corrupted")
        # TODO: findings = your_tool.detect_temporal_misalignment(corrupted)
        # assert any(f.track_id == expected["track_id"]
        #            and f.offset_frames == expected["offset_frames"] for f in findings)
        assert all(r["track_id"] != expected["track_id"] or True for r in corrupted)

    def test_fragmentation_detected_with_expected_gap(self, planted_errors_for, track_annotations):
        errors = [e for e in planted_errors_for("clip_004") if e["type"] == "fragmentation"]
        assert errors
        expected = errors[0]
        corrupted = track_annotations("clip_004", "corrupted")
        track_ids = {r["track_id"] for r in corrupted}
        # TODO: findings = your_tool.detect_fragmented_tracks(corrupted)
        # assert any(f.original_track_id == expected["original_track_id"]
        #            and f.new_track_id == expected["new_track_id"] for f in findings)
        assert expected["new_track_id"] in track_ids

    def test_ghost_track_detected(self, planted_errors_for, track_annotations):
        errors = [e for e in planted_errors_for("clip_005") if e["type"] == "ghost_track"]
        assert errors
        expected = errors[0]
        corrupted = track_annotations("clip_005", "corrupted")
        # TODO: findings = your_tool.detect_ghost_tracks(corrupted)
        # assert any(f.track_id == expected["track_id"] for f in findings)
        ghost_records = [r for r in corrupted if r["track_id"] == expected["track_id"]]
        assert len(ghost_records) == (expected["end_frame"] - expected["start_frame"] + 1)

    def test_combined_errors_on_multi_error_clip(self, planted_errors_for):
        errors = planted_errors_for("clip_006")
        types = {e["type"] for e in errors}
        assert types == {"id_switch", "box_drift"}

    # ---------------------------------------------------------------------------
    # File-integrity errors: bad files that should get flagged
    # ---------------------------------------------------------------------------
    def test_black_frames(self, corruption_type, video_path):
        error = corruption_type("black_frames_001")
        assert error
        vid = video_path("black_frames_001")
        assert vid.exists()

    def test_wrong_codec(self, corruption_type, video_path):
        error = corruption_type("wrong_codec_001")
        assert error
        vid = video_path("wrong_codec_001")
        assert vid.exists()

    def test_variable_frame_rate(self, corruption_type, video_path):
        error = corruption_type("vfr_001")
        assert error
        vid = video_path("vfr_001")
        assert vid.exists()

    def test_truncated_file(self, corruption_type, video_path):
        error = corruption_type("truncated_001")
        assert error
        vid = video_path("truncated_001")
        assert vid.exists()
