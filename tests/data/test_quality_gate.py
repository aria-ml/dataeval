"""Tests for keeping the frames whose value on one image_stats statistic lies within declared bounds."""

import logging

import numpy as np
import pytest

import dataeval.core._image_stats as image_stats_module
from dataeval.core import image_stats
from dataeval.core._calculators._base import Calculator, Handler
from dataeval.data import Chain, QualityGate, Redundancy, SequenceFrames, _quality_gate
from dataeval.data._quality_gate import _numeric_scalar_stats
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import normalize_image_shape
from tests.data.test_frames import _FakeDataset, constant_frames, drifting_frames, emit, image_dataset, make_dataset


class _Scripted(QualityGate):
    """A sharpness gate whose reading of each frame is scripted: ``values[frame_index]``.

    Every frame of :func:`make_dataset` is filled with its own frame index, so that is read back off the pixels.
    """

    def __init__(self, values, **bounds):
        super().__init__("sharpness", **bounds)
        self.values = values

    def _measure(self, pixels):
        return self.values[int(pixels.flat[0])]


class _Exploding(Calculator[ImageStats]):
    """A statistic that cannot be read off the probe image -- a model-based blur score whose model is absent, say."""

    def __init__(self, datum, cache):
        pass

    def get_applicable_flags(self):
        return ImageStats.VISUAL_SHARPNESS

    def get_handlers(self):
        return {ImageStats.VISUAL_SHARPNESS: Handler("blur", self._blur)}

    def _blur(self):
        raise RuntimeError("blur model unavailable")


def _dataset(n):
    """A one-sequence tracking dataset of ``n`` frames, frame ``i`` filled with ``i``."""
    dataset, _ = make_dataset((n,))
    return dataset


def _two_videos(fills_a, fills_b):
    """A two-sequence tracking dataset of constant frames, each filled with the value it is scripted by."""
    a, b = image_dataset(constant_frames(fills_a)), image_dataset(constant_frames(fills_b))
    stream, target, _ = b[0]
    return _FakeDataset([a[0], (stream, target, {"id": "vid1"})], a.metadata)


def _kept(selector, n):
    """The positions ``selector`` keeps of a one-sequence dataset of ``n`` frames."""
    return [entry["position"] for entry in SequenceFrames(_dataset(n), selector)._record_of(0).entries]


@pytest.mark.required
class TestQualityGate:
    def test_the_value_is_the_one_image_stats_reports(self):
        dataset = image_dataset(drifting_frames())
        gate = QualityGate("sharpness", min=0.0)
        frames = list(dataset[0][0])
        expected = image_stats(
            [np.asarray(f.pixels) for f in frames], stats=ImageStats.VISUAL_SHARPNESS, per_target=False
        )["stats"]["sharpness"]
        measured = [gate._measure(normalize_image_shape(np.asarray(f.pixels))) for f in frames]
        assert measured == expected.tolist()

    def test_a_bound_at_a_frames_own_reported_value_keeps_that_frame(self):
        """image_stats reports float32, so a bound copied from its output must meet the gate's reading exactly."""
        dataset = image_dataset(drifting_frames())
        images = [np.asarray(f.pixels) for f in dataset[0][0]]
        reported = image_stats(images, stats=ImageStats.VISUAL_SHARPNESS, per_target=False)["stats"]["sharpness"]
        calculator = QualityGate("sharpness", min=0.0)._calculator
        unrounded = [
            image_stats_module._collect_calculator_stats([calculator], normalize_image_shape(image), None)[0][0][
                "sharpness"
            ][0]
            for image in images
        ]
        assert unrounded != reported.tolist()  # so float32 rounding is what this tests
        for position, value in enumerate(reported.tolist()):
            for bound in ({"min": value}, {"max": value}):
                kept = SequenceFrames(dataset, QualityGate("sharpness", **bound)).frame_map[:, 1].tolist()
                assert position in kept, (position, bound)

    def test_absolute_bounds_keep_the_inclusive_range(self):
        kept = _kept(_Scripted([0.1, 0.5, 0.9, 0.5], min=0.5, max=0.9), 4)
        assert kept == [1, 2, 3]

    def test_a_quantile_drops_strictly_below_the_cutoff(self):
        # inverted_cdf at 0.15 over 1..10 is 2.0, so 1.0 (position 1) is the only value strictly below it
        kept = _kept(_Scripted([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 10.0], min_quantile=0.15), 10)
        assert kept == [0, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_an_all_equal_video_drops_nothing(self):
        assert _kept(_Scripted([3.0] * 6, min_quantile=0.5), 6) == list(range(6))

    def test_nan_is_kept_marked_and_left_out_of_the_quantile(self):
        frames = SequenceFrames(_dataset(5), _Scripted([np.nan, 1.0, 2.0, 3.0, np.nan], min_quantile=0.34))
        entries = frames._record_of(0).entries
        assert [e["position"] for e in entries] == [0, 2, 3, 4]
        assert [e["unmeasured"] for e in entries] == [True, False, False, True]

    def test_dropped_frames_are_excluded_not_represented(self):
        frames = SequenceFrames(_dataset(4), _Scripted([0.1, 0.5, 0.9, 0.5], min=0.5))
        rows = [m[2]["frames_represented"] for m in emit(frames)]
        assert rows == [1.0, 1.0, 1.0]

    def test_quantile_bounds_make_it_two_pass_and_absolute_ones_do_not(self):
        assert QualityGate("sharpness", min_quantile=0.1).two_pass
        assert not QualityGate("sharpness", min=0.0).two_pass

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "at least one bound"),
            ({"min": 0.0, "min_quantile": 0.1}, "min"),
            ({"min_quantile": 1.5}, r"\[0, 1\]"),
            ({"min": 2.0, "max": 1.0}, "min"),
            ({"min": float("nan")}, "min must be a number, not NaN"),
            ({"max": float("nan")}, "max must be a number, not NaN"),
            ({"min_quantile": 0.9, "max_quantile": 0.1}, "min_quantile.*max_quantile"),
        ],
    )
    def test_refusals(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            QualityGate("sharpness", **kwargs)

    def test_an_infinite_bound_is_no_bound_on_that_side(self):
        assert _kept(_Scripted([0.1, 0.5, 0.9], min=-np.inf, max=np.inf), 3) == [0, 1, 2]

    def test_an_unknown_statistic_lists_the_ones_it_could_be(self):
        with pytest.raises(ValueError, match="sharpness"):
            QualityGate("sharpnes", min=0.0)

    @pytest.mark.parametrize("stat", ["histogram", "percentiles", "center", "xxhash", "phash"])
    def test_a_statistic_that_is_not_one_number_is_refused_at_construction(self, stat):
        """Each has one output name under one flag, but its reading is a list, tuple or string."""
        with pytest.raises(ValueError, match="numeric"):
            QualityGate(stat, min=0.0)

    def test_the_supported_set_holds_the_visual_scalars(self):
        assert {"brightness", "contrast", "darkness", "sharpness"} <= set(_numeric_scalar_stats())

    def test_construction_probes_only_the_named_statistic(self, monkeypatch):
        """Resolving a name computes nothing else, so a costly calculator elsewhere in the registry never runs."""
        real = image_stats_module._collect_calculator_stats
        probed = []

        def counting(calculators, *args, **kwargs):
            probed.append(calculators)
            return real(calculators, *args, **kwargs)

        monkeypatch.setattr(image_stats_module, "_collect_calculator_stats", counting)
        _numeric_scalar_stats.cache_clear()  # so building the supported set would show up as probes
        QualityGate("sharpness", min=0.0)
        assert len(probed) == 1

    def test_a_statistic_whose_calculator_raises_on_the_probe_is_refused_as_unsupported(self, monkeypatch, caplog):
        real = _quality_gate._single_statistics
        monkeypatch.setattr(
            _quality_gate,
            "_single_statistics",
            lambda: {**real(), "blur": (_Exploding, ImageStats.VISUAL_SHARPNESS)},
        )
        # Cleared on both sides, so the supported set is built with the fake present and never outlives this test.
        _numeric_scalar_stats.cache_clear()
        try:
            with caplog.at_level(logging.DEBUG, logger="dataeval"), pytest.raises(ValueError, match="numeric"):
                QualityGate("blur", min=0.0)
            assert "blur" not in _numeric_scalar_stats()
            # The cause is left findable rather than swallowed.
            assert any("blur" in r.getMessage() and r.exc_info for r in caplog.records)
            assert "sharpness" in _numeric_scalar_stats()
        finally:
            _numeric_scalar_stats.cache_clear()

    @pytest.mark.parametrize("error", [AttributeError, TypeError, ValueError])
    def test_unexpected_calculator_errors_propagate(self, monkeypatch, error):
        def broken(calculators, datum, box):
            raise error("calculator bug")

        monkeypatch.setattr(image_stats_module, "_collect_calculator_stats", broken)
        with pytest.raises(error, match="calculator bug"):
            QualityGate("sharpness", min=0.0)

    def test_it_chains_first_as_the_whole_input_stage(self):
        # inverted_cdf at 0.5 is 9.0, so the gate drops frames 0-2, and Redundancy folds the constant 3-9 into frame 3
        chain = Chain(_Scripted([1.0] * 3 + [9.0] * 7, min_quantile=0.5), Redundancy())
        assert chain.two_pass
        rows = emit(SequenceFrames(_dataset(10), chain))
        assert [m["frame"] for _, _, m in rows] == [3]
        assert [m["frames_represented"] for _, _, m in rows] == [7.0]

    def test_quantiles_are_per_video_and_nan_is_logged_once_per_video(self, caplog):
        # Per video the 0.5 cutoffs are 2 and 12; pooled over both it would be 4, keeping only 4 of the first video.
        values = {0: np.nan, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 11: 11.0, 12: 12.0, 13: np.nan, 14: 14.0, 15: 15.0}
        frames = SequenceFrames(_two_videos([0, 1, 2, 3, 4], [11, 12, 13, 14, 15]), _Scripted(values, min_quantile=0.5))
        with caplog.at_level(logging.INFO, logger="dataeval"):
            frames._recorded()
        assert [e["position"] for e in frames._record_of(0).entries] == [0, 2, 3, 4]
        assert [e["position"] for e in frames._record_of(1).entries] == [1, 2, 3, 4]
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("QualityGate:")]
        assert len(lines) == 2
        assert "sequence 0" in lines[0]
        assert "sequence 1" in lines[1]

    def test_a_video_with_no_finite_value_keeps_every_frame_marked(self):
        frames = SequenceFrames(_dataset(3), _Scripted([np.inf, -np.inf, np.nan], min_quantile=0.5))
        entries = frames._record_of(0).entries
        assert [e["position"] for e in entries] == [0, 1, 2]
        assert [e["unmeasured"] for e in entries] == [True, True, True]

    def test_infinities_beside_finite_values_are_judged_against_the_cutoff(self):
        frames = SequenceFrames(_dataset(4), _Scripted([np.inf, -np.inf, 1.0, 2.0], min_quantile=0.5))
        entries = frames._record_of(0).entries
        assert [e["position"] for e in entries] == [0, 2, 3]
        assert not any(e["unmeasured"] for e in entries)

    def test_frames_of_different_sizes_are_each_measured(self):
        """Frames of different shapes in one video are each measured on their own, as image_stats measures them."""
        rng = np.random.default_rng(0)
        images = [rng.integers(0, 256, shape, dtype=np.uint8) for shape in [(3, 12, 14), (1, 20, 10)]]
        expected = image_stats(images, stats=ImageStats.VISUAL_SHARPNESS, per_target=False)["stats"]["sharpness"]
        gate = QualityGate("sharpness", min=0.0)
        assert [gate._measure(image) for image in images] == pytest.approx(list(expected))
        assert len(emit(SequenceFrames(image_dataset(images), gate))) == 2
