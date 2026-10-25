import warnings

import numpy as np
import polars as pl
import pytest

import dataeval.shift
import dataeval.shift._drift._known_domains as known_domains_module
from dataeval import Embeddings, Metadata
from dataeval.exceptions import NotFittedError, ShapeMismatchError
from dataeval.extractors import FlattenExtractor
from dataeval.shift import DriftOutput
from dataeval.shift._drift._base import BaseDrift, ChunkableMixin
from dataeval.shift._drift._known_domains import DriftKnownDomains, _greater, _select_threshold
from dataeval.utils.thresholds import ConstantThreshold, ZScoreThreshold
from tests.conftest import MockICDataset

# This module deliberately exercises the experimental detector.
pytestmark = [
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

N_FEATURES = 6
CENTERS = {"day": 0.0, "dusk": 4.0, "night": 8.0}


def _domain(center: float, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(center, 1.0, size=(n, N_FEATURES)).astype(np.float32)


@pytest.fixture(scope="module")
def reference():
    return {name: _domain(c, 150, seed) for seed, (name, c) in enumerate(CENTERS.items())}


@pytest.fixture(scope="module")
def detector(reference):
    return DriftKnownDomains(n_folds=3).fit(reference)


SPEC = {"time": {"day": ["morning", "noon"], "dusk": "dusk", "night": "night"}}


def _metadata(**factors):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Metadata.from_factors(factors)


@pytest.fixture(scope="module")
def labeled(reference):
    """The reference as one shuffled array with a 'time' factor; 'day' spans two values."""
    x = np.concatenate(list(reference.values()))
    time = np.array(["morning"] * 75 + ["noon"] * 75 + ["dusk"] * 150 + ["night"] * 150)
    order = np.random.default_rng(5).permutation(len(x))
    return x[order], time[order].tolist()


@pytest.fixture(scope="module")
def fitted(labeled):
    x, time = labeled
    return DriftKnownDomains(n_folds=3, domains=SPEC).fit(x, metadata=_metadata(time=time))


def _dataset(x, **factors):
    """An annotated dataset whose images flatten back to ``x``."""
    return MockICDataset(x.reshape(len(x), 1, 2, 3), [0] * len(x), metadata=factors)  # type: ignore


@pytest.mark.required
class TestNotYetPublic:
    def test_is_marked_experimental(self):
        assert "DriftKnownDomains" in getattr(DriftKnownDomains, "__experimental__", "")

    def test_is_not_exported_from_the_public_package(self):
        assert "DriftKnownDomains" not in dataeval.shift.__all__
        assert not hasattr(dataeval.shift, "DriftKnownDomains")


@pytest.mark.required
class TestSelectThreshold:
    # Known data scores low, novel data scores high, with overlap in the middle.
    scores = np.concatenate([np.linspace(0.0, 0.5, 100), np.linspace(0.3, 0.9, 100)])
    novel = np.concatenate([np.zeros(100), np.ones(100)])

    def test_conservative_bounds_the_false_positive_rate(self):
        _, false_pos, _ = _select_threshold(self.scores, self.novel, "conservative", 0.05)
        assert false_pos <= 0.05

    def test_sensitive_bounds_the_true_positive_rate(self):
        _, _, tpr = _select_threshold(self.scores, self.novel, "sensitive", 0.05)
        assert tpr >= 0.95

    def test_balanced_maximizes_youdens_j(self):
        def j(mode):
            _, false_pos, true_pos = _select_threshold(self.scores, self.novel, mode, 0.05)
            return true_pos - false_pos

        assert j("balanced") >= max(j("conservative"), j("sensitive"))

    def test_threshold_is_always_a_usable_score(self):
        """Non-separable scores give J = 0 at sklearn's infinite threshold; it must stay a score."""
        scores = np.full(20, 0.4)
        threshold, _, _ = _select_threshold(scores, np.r_[np.zeros(10), np.ones(10)], "balanced", 0.05)
        assert np.isfinite(threshold)
        assert threshold <= 1.0


@pytest.mark.required
class TestDriftKnownDomains:
    def test_predict_before_fit(self):
        with pytest.raises(NotFittedError):
            DriftKnownDomains().predict(np.zeros((5, N_FEATURES)))

    def test_fit_needs_three_domains(self, reference):
        two = dict(list(reference.items())[:2])
        with pytest.raises(ValueError, match="at least three"):
            DriftKnownDomains().fit(two)

    def test_fit_needs_matching_feature_counts(self, reference):
        bad = {**reference, "odd": np.zeros((50, N_FEATURES + 1), dtype=np.float32)}
        with pytest.raises(ShapeMismatchError):
            DriftKnownDomains().fit(bad)

    def test_fit_needs_a_sample_per_fold(self, reference):
        tiny = {**reference, "tiny": _domain(12.0, 2, 99)}
        with pytest.raises(ValueError, match="n_folds"):
            DriftKnownDomains(n_folds=3).fit(tiny)

    def test_predict_wrong_feature_count(self, detector):
        with pytest.raises(ShapeMismatchError):
            detector.predict(np.zeros((5, N_FEATURES + 1)))

    def test_known_domain_is_not_drift(self, detector):
        result = detector.predict(_domain(CENTERS["dusk"], 100, 123))
        assert isinstance(result, DriftOutput)
        assert not result.drifted
        assert result.details["closest_domain"] == "dusk"
        assert result.details["scores"].shape == (100,)

    @pytest.mark.parametrize("center", [2.0, 6.0])
    def test_domain_between_known_domains_is_drift(self, detector, center):
        result = detector.predict(_domain(center, 100, 321))
        assert result.drifted
        assert result.distance > 0
        assert result.details["p_val"] < result.threshold

    @pytest.mark.parametrize("center", list(CENTERS.values()))
    def test_every_known_domain_passes_including_the_middle_one(self, detector, center):
        """The middle domain is the least certain by position; its own null rate keeps it from alarming."""
        assert not detector.predict(_domain(center, 200, 555)).drifted

    def test_domain_probabilities_cover_every_domain(self, detector):
        result = detector.predict(_domain(CENTERS["night"], 50, 7))
        probabilities = result.details["domain_probabilities"]
        assert set(probabilities) == set(CENTERS)
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-5)

    def test_refit_replaces_the_domains(self, reference):
        d = DriftKnownDomains(n_folds=3).fit(reference)
        d.fit({"a": _domain(0.0, 60, 1), "b": _domain(5.0, 60, 2), "c": _domain(10.0, 60, 3)})
        assert d.domain_names == ("a", "b", "c")

    def test_hyperparameters_reach_the_classifier(self, reference):
        stump = {"n_estimators": 1, "max_depth": 1, "verbosity": -1}
        d = DriftKnownDomains(n_folds=3, hyperparameters=stump).fit(reference)
        assert d.config.hyperparameters == stump

    def test_repr(self, reference):
        assert "fitted=False" in repr(DriftKnownDomains())
        assert "fitted=True" in repr(DriftKnownDomains(n_folds=3).fit(reference))


@pytest.mark.required
class TestWarmStart:
    @pytest.fixture(scope="class")
    def onnxmltools(self):
        return pytest.importorskip("onnxmltools")

    @pytest.fixture(scope="class")
    def saved(self, onnxmltools, detector, tmp_path_factory):
        path = tmp_path_factory.mktemp("known_domains") / "detector.onnx"
        detector.save(path)
        return path

    def test_save_before_fit(self, onnxmltools, tmp_path):
        with pytest.raises(NotFittedError):
            DriftKnownDomains().save(tmp_path / "detector.onnx")

    def test_loaded_detector_predicts_like_the_original(self, detector, saved):
        loaded = DriftKnownDomains.load(saved)
        for center, seed in ((CENTERS["day"], 11), (2.0, 12)):
            data = _domain(center, 80, seed)
            original, restored = detector.predict(data), loaded.predict(data)
            np.testing.assert_allclose(restored.details["scores"], original.details["scores"], atol=1e-4)
            assert restored.drifted == original.drifted
            assert restored.details["closest_domain"] == original.details["closest_domain"]

    def test_loaded_detector_keeps_its_calibration(self, detector, saved):
        loaded = DriftKnownDomains.load(saved)
        assert loaded.domain_names == detector.domain_names
        assert loaded.score_threshold == pytest.approx(detector.score_threshold)
        assert loaded.config == detector.config
        assert "fitted=True" in repr(loaded)

    def test_loaded_detector_checks_feature_count(self, saved):
        with pytest.raises(ShapeMismatchError):
            DriftKnownDomains.load(saved).predict(np.zeros((5, N_FEATURES + 1)))

    def test_loaded_detector_can_be_refit(self, saved, reference):
        loaded = DriftKnownDomains.load(saved)
        loaded.fit({"a": reference["day"], "b": reference["dusk"], "c": reference["night"]})
        assert loaded.domain_names == ("a", "b", "c")

    @pytest.fixture(scope="class")
    def ranged(self, labeled):
        x, time = labeled
        altitude = {"morning": 1.0, "noon": 1.5, "dusk": 3.0, "night": 12.0}
        # night's 12.0 matches no range, so it lands in the catch-all; every group has samples.
        spec = {"altitude": {"low": (None, 2), "mid": [(2, 5)], "other": None}}
        md = _metadata(altitude=[altitude[t] for t in time])
        return DriftKnownDomains(n_folds=3, domains=spec).fit(x, metadata=md)

    def test_spec_with_ranges_round_trips(self, onnxmltools, ranged, tmp_path):
        ranged.save(tmp_path / "ranged.onnx")
        loaded = DriftKnownDomains.load(tmp_path / "ranged.onnx")
        assert loaded.config == ranged.config
        assert loaded.domain_names == ("low", "mid", "other")
        md = _metadata(altitude=[1.0] * 20 + [12.0] * 20)
        data = np.concatenate([_domain(0.0, 20, 40), _domain(8.0, 20, 41)])
        original = ranged.predict(data, metadata=md).details["domains"]
        restored = loaded.predict(data, metadata=md).details["domains"]
        assert restored["domain"].to_list() == original["domain"].to_list() == ["low", "other"]
        np.testing.assert_allclose(restored["agreement"].to_numpy(), original["agreement"].to_numpy(), atol=1e-6)

    def test_load_takes_an_extractor(self, saved):
        extractor = FlattenExtractor()
        loaded = DriftKnownDomains.load(saved, extractor=extractor)
        assert loaded.config.extractor is extractor
        assert loaded.predict(_domain(0.0, 10, 42).reshape(10, 1, 2, 3)).details["scores"].shape == (10,)

    def test_save_leaves_the_extractor_out(self, onnxmltools, labeled, tmp_path):
        x, time = labeled
        d = DriftKnownDomains(n_folds=3, domains=SPEC, extractor=FlattenExtractor())
        d.fit(_dataset(x, time=time)).save(tmp_path / "with_extractor.onnx")
        assert DriftKnownDomains.load(tmp_path / "with_extractor.onnx").config.extractor is None

    def test_format_one_files_are_refused(self, onnxmltools, detector, tmp_path, monkeypatch):
        monkeypatch.setattr(known_domains_module, "_FORMAT", 1)
        detector.save(tmp_path / "old.onnx")
        monkeypatch.undo()
        with pytest.raises(ValueError, match="format 1"):
            DriftKnownDomains.load(tmp_path / "old.onnx")

    def test_loaded_detector_holds_no_reference_data(self, saved):
        with pytest.raises(RuntimeError, match="load"):
            _ = DriftKnownDomains.load(saved).reference_data

    def test_load_rejects_a_foreign_onnx_model(self, onnxmltools, tmp_path):
        import lightgbm as lgb
        from onnxmltools.convert.common.data_types import FloatTensorType

        x = np.random.default_rng(0).normal(size=(60, 3)).astype(np.float32)
        model = lgb.LGBMClassifier(n_estimators=2, verbosity=-1).fit(x, np.arange(60) % 3)
        foreign = onnxmltools.convert_lightgbm(
            model, initial_types=[("features", FloatTensorType([None, 3]))], zipmap=False
        )
        path = tmp_path / "foreign.onnx"
        path.write_bytes(foreign.SerializeToString())
        with pytest.raises(ValueError, match="DriftKnownDomains"):
            DriftKnownDomains.load(path)


@pytest.mark.required
class TestFitFromMetadata:
    def test_metadata_fit_matches_mapping_fit(self, labeled, fitted):
        x, time = labeled
        groups = {"day": ["morning", "noon"], "dusk": ["dusk"], "night": ["night"]}
        mapping = {name: x[np.isin(time, values)] for name, values in groups.items()}
        by_mapping = DriftKnownDomains(n_folds=3).fit(mapping)
        assert fitted.domain_names == by_mapping.domain_names == ("day", "dusk", "night")
        assert fitted.score_threshold == pytest.approx(by_mapping.score_threshold)
        np.testing.assert_array_equal(fitted._null_counts, by_mapping._null_counts)

    def test_null_counts_hold_held_out_flagged_and_agreed(self, fitted):
        counts = fitted._null_counts
        assert counts.shape == (3, 3)
        np.testing.assert_array_equal(counts[:, 0], [150, 150, 150])
        assert (counts[:, 1] <= counts[:, 0]).all()
        assert (counts[:, 2] <= counts[:, 0]).all()

    def test_fit_from_an_annotated_dataset(self, labeled, fitted):
        x, time = labeled
        d = DriftKnownDomains(n_folds=3, domains=SPEC, extractor=FlattenExtractor()).fit(_dataset(x, time=time))
        assert d.domain_names == fitted.domain_names
        assert d.score_threshold == pytest.approx(fitted.score_threshold)

    def test_fit_from_embeddings_with_metadata(self, labeled, fitted):
        x, time = labeled
        ds = _dataset(x, time=time)
        embeddings = Embeddings(ds, extractor=FlattenExtractor(), batch_size=64)
        d = DriftKnownDomains(n_folds=3, domains=SPEC).fit(embeddings, metadata=Metadata(ds))  # type: ignore
        assert d.score_threshold == pytest.approx(fitted.score_threshold)

    def test_str_spec_is_one_domain_per_value_in_sorted_order(self, labeled):
        x, time = labeled
        d = DriftKnownDomains(n_folds=3, domains="time").fit(x, metadata=_metadata(time=time))
        assert d.domain_names == ("dusk", "morning", "night", "noon")

    def test_unmatched_reference_values_are_dropped_with_a_warning(self, labeled):
        x, time = labeled
        spec = {"time": {"day": "morning", "dusk": "dusk", "night": "night"}}
        with pytest.warns(UserWarning, match="noon"):
            d = DriftKnownDomains(n_folds=3, domains=spec).fit(x, metadata=_metadata(time=time))
        assert d._null_counts[0, 0] == 75

    def test_empty_catch_all_raises(self, labeled):
        x, time = labeled
        spec = {"time": {**SPEC["time"], "other": None}}
        with pytest.raises(ValueError, match="catch-all"):
            DriftKnownDomains(n_folds=3, domains=spec).fit(x, metadata=_metadata(time=time))

    def test_empty_named_group_raises(self, labeled):
        x, time = labeled
        spec = {"time": {**SPEC["time"], "dawn": "dawn"}}
        with pytest.raises(ValueError, match="'dawn'"):
            DriftKnownDomains(n_folds=3, domains=spec).fit(x, metadata=_metadata(time=time))

    def test_mapping_with_domains_raises(self, reference):
        with pytest.raises(ValueError, match="already defines the domains"):
            DriftKnownDomains(domains=SPEC).fit(reference)

    def test_mapping_with_metadata_raises(self, reference, labeled):
        with pytest.raises(ValueError, match="metadata="):
            DriftKnownDomains().fit(reference, metadata=_metadata(time=labeled[1]))

    def test_one_array_without_domains_raises(self, labeled):
        with pytest.raises(ValueError, match="domains="):
            DriftKnownDomains().fit(labeled[0])

    def test_no_metadata_source_raises(self, labeled):
        with pytest.raises(ValueError, match="needs metadata"):
            DriftKnownDomains(domains=SPEC).fit(labeled[0])

    def test_dataset_without_extractor_raises(self, labeled):
        x, time = labeled
        with pytest.raises(ValueError, match="Array-like"):
            DriftKnownDomains(domains=SPEC).fit(_dataset(x, time=time))

    def test_invalid_spec_raises_at_construction(self):
        with pytest.raises(ValueError, match="use a list"):
            DriftKnownDomains(domains={"time": {"day": ("morning", "noon")}})

    def test_invalid_extractor_raises(self):
        with pytest.raises(ValueError, match="extractor"):
            DriftKnownDomains(extractor="not an extractor")  # type: ignore

    def test_config_holds_the_canonical_spec(self):
        d = DriftKnownDomains(domains={"time": [["morning", "noon"], "dusk"]})
        assert d.config.domains == {"time": {"morning|noon": ["morning", "noon"], "dusk": "dusk"}}


def _batch(*parts):
    """Concatenate (center, n, seed, label) parts into data and labels."""
    x = np.concatenate([_domain(center, n, seed) for center, n, seed, _ in parts])
    labels = [label for _, n, _, label in parts for _ in range(n)]
    return x, labels


@pytest.mark.required
class TestPredictDeclaredDomains:
    def test_without_declared_domains_matches_the_single_closest_domain_test(self, fitted):
        data = _domain(2.0, 100, 321)
        result = fitted.predict(data)
        table = result.details["domains"]
        assert table.height == 1
        assert table["domain"][0] is None
        closest = fitted.domain_names.index(result.details["closest_domain"])
        held_out, ref_flagged, _ = fitted._null_counts[closest]
        flagged = int((result.details["scores"] >= fitted.score_threshold).sum())
        expected = _greater(flagged, 100, int(ref_flagged), int(held_out))
        assert result.details["p_val"] == pytest.approx(expected)
        assert result.threshold == pytest.approx(0.05)
        assert result.drifted == (expected < 0.05)

    def test_known_domains_declared_correctly_do_not_drift(self, fitted):
        x, labels = _batch((0.0, 60, 1, "day"), (4.0, 60, 2, "dusk"), (8.0, 60, 3, "night"))
        result = fitted.predict(x, domain_labels=labels)
        table = result.details["domains"]
        assert table["domain"].to_list() == ["day", "dusk", "night"]
        assert not result.drifted
        assert (table["agreement"] > 0.8).all()
        assert result.threshold == pytest.approx(0.05 / 6)

    def test_a_domain_that_moved_onto_another_fails_agreement_not_novelty(self, fitted):
        x, labels = _batch((0.0, 60, 4, "day"), (8.0, 60, 5, "dusk"))  # "dusk" data now looks like night
        result = fitted.predict(x, domain_labels=labels)
        dusk = result.details["domains"].filter(pl.col("domain") == "dusk").row(0, named=True)
        assert dusk["closest_domain"] == "night"
        assert dusk["agreement_p_val"] < result.threshold
        assert dusk["novelty_p_val"] >= result.threshold
        assert dusk["drifted"]
        assert result.drifted

    def test_a_blend_fails_novelty(self, fitted):
        x, labels = _batch((2.0, 80, 6, "day"))
        row = fitted.predict(x, domain_labels=labels).details["domains"].row(0, named=True)
        assert row["novelty_p_val"] < 0.05 / 2

    def test_unknown_labels_are_undeclared_with_a_warning(self, fitted):
        x, labels = _batch((0.0, 30, 7, "day"), (4.0, 30, 8, "dawn"))
        with pytest.warns(UserWarning, match="dawn"):
            table = fitted.predict(x, domain_labels=labels).details["domains"]
        assert table["domain"].to_list() == ["day", None]
        assert table["agreement"][1] is None

    def test_labels_as_a_numpy_array_with_missing_entries(self, fitted):
        x, labels = _batch((0.0, 30, 9, "day"), (4.0, 30, 10, "dusk"))
        array = np.array(labels, dtype=object)
        array[:5] = None
        table = fitted.predict(x, domain_labels=array).details["domains"]
        assert table["domain"].to_list() == ["day", "dusk", None]
        assert table["n"].to_list() == [25, 30, 5]

    def test_one_sample_row_beside_undeclared(self, fitted):
        x, labels = _batch((0.0, 1, 11, "day"), (4.0, 40, 12, "unknown"))
        with pytest.warns(UserWarning, match="unknown"):
            table = fitted.predict(x, domain_labels=labels).details["domains"]
        assert table["n"].to_list() == [1, 40]
        assert np.isfinite(table["novelty_p_val"].to_numpy()).all()

    def test_test_metadata_is_read_through_the_spec(self, fitted):
        x, labels = _batch((0.0, 30, 13, "morning"), (0.0, 30, 14, "noon"), (8.0, 30, 15, "night"))
        table = fitted.predict(x, metadata=_metadata(time=labels)).details["domains"]
        assert table["domain"].to_list() == ["day", "night"]
        assert table["n"].to_list() == [60, 30]

    def test_unmatched_test_values_are_undeclared(self, fitted):
        x, labels = _batch((0.0, 30, 16, "morning"), (4.0, 30, 17, "twilight"))
        table = fitted.predict(x, metadata=_metadata(time=labels)).details["domains"]
        assert table["domain"].to_list() == ["day", None]

    def test_annotated_dataset_supplies_its_own_metadata(self, labeled):
        x, time = labeled
        d = DriftKnownDomains(n_folds=3, domains=SPEC, extractor=FlattenExtractor()).fit(_dataset(x, time=time))
        tx, tlabels = _batch((4.0, 40, 18, "dusk"))
        table = d.predict(_dataset(tx, time=tlabels)).details["domains"]
        assert table["domain"].to_list() == ["dusk"]

    def test_annotated_dataset_without_the_factor_warns_and_falls_back(self, labeled):
        x, time = labeled
        d = DriftKnownDomains(n_folds=3, domains=SPEC, extractor=FlattenExtractor()).fit(_dataset(x, time=time))
        tx, tlabels = _batch((4.0, 40, 19, "dusk"))
        with pytest.warns(UserWarning, match="has no 'time'"):
            table = d.predict(_dataset(tx, place=tlabels)).details["domains"]
        assert table["domain"].to_list() == [None]

    def test_explicit_metadata_without_the_factor_raises(self, fitted):
        with pytest.raises(ValueError, match="no factor 'time'"):
            fitted.predict(_domain(0.0, 10, 20), metadata=_metadata(place=["a"] * 10))

    def test_metadata_matching_nothing_warns(self, fitted):
        with pytest.warns(UserWarning, match="matched no sample"):
            fitted.predict(_domain(0.0, 10, 21), metadata=_metadata(time=[1, 2, 3, 4, 5] * 2))

    def test_int_spec_matches_a_float_column(self, labeled):
        x, time = labeled
        cam = {"morning": 1, "noon": 1, "dusk": 2, "night": 3}
        spec = {"cam": {"one": 1, "two": 2, "three": 3}}
        d = DriftKnownDomains(n_folds=3, domains=spec).fit(x, metadata=_metadata(cam=[cam[t] for t in time]))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            table = d.predict(_domain(4.0, 10, 33), metadata=_metadata(cam=[2.0] * 10)).details["domains"]
        assert table["domain"].to_list() == ["two"]

    def test_metadata_and_labels_together_raise(self, fitted):
        with pytest.raises(ValueError, match="not both"):
            fitted.predict(_domain(0.0, 10, 22), metadata=_metadata(time=["day"] * 10), domain_labels=["day"] * 10)

    def test_label_count_mismatch_raises(self, fitted):
        with pytest.raises(ShapeMismatchError):
            fitted.predict(_domain(0.0, 10, 23), domain_labels=["day"] * 9)

    def test_mapping_fit_detector_rejects_metadata(self, detector):
        with pytest.raises(ValueError, match="domains="):
            detector.predict(_domain(0.0, 10, 24), metadata=_metadata(time=["day"] * 10))

    def test_mapping_fit_detector_accepts_labels(self, detector):
        table = detector.predict(_domain(0.0, 20, 25), domain_labels=["day"] * 20).details["domains"]
        assert table["domain"].to_list() == ["day"]

    def test_str_spec_maps_numeric_test_values_back(self, labeled):
        x, time = labeled
        cam = {"morning": 1, "noon": 1, "dusk": 2, "night": 3}
        d = DriftKnownDomains(n_folds=3, domains="cam").fit(x, metadata=_metadata(cam=[cam[t] for t in time]))
        assert d.domain_names == ("1", "2", "3")
        table = d.predict(_domain(4.0, 30, 26), metadata=_metadata(cam=[2] * 30)).details["domains"]
        assert table["domain"].to_list() == ["2"]

    def test_raw_values_as_labels_get_a_hint_that_labels_are_names(self, labeled):
        x, time = labeled
        cam = {"morning": 1, "noon": 1, "dusk": 2, "night": 3}
        d = DriftKnownDomains(n_folds=3, domains="cam").fit(x, metadata=_metadata(cam=[cam[t] for t in time]))
        with pytest.warns(UserWarning, match=r"\[1\].*domain names"):
            d.predict(_domain(0.0, 10, 34), domain_labels=[1] * 10)

    def test_fdr_threshold_is_less_strict(self, reference):
        # dusk and night both declared on day-like data: two agreement tests reject, so BH's level rises
        x, labels = _batch((0.0, 40, 27, "day"), (0.0, 40, 28, "dusk"), (0.0, 40, 29, "night"))
        bonferroni = DriftKnownDomains(n_folds=3).fit(reference).predict(x, domain_labels=labels)
        fdr = DriftKnownDomains(n_folds=3, correction="fdr").fit(reference).predict(x, domain_labels=labels)
        assert fdr.threshold > bonferroni.threshold

    def test_predicted_domains_name_each_samples_argmax(self, fitted):
        x, _ = _batch((0.0, 20, 30, "day"), (8.0, 20, 31, "night"))
        predicted = fitted.predict(x).details["predicted_domains"]
        assert predicted.shape == (40,)
        assert (predicted[:20] == "day").mean() > 0.8
        assert (predicted[20:] == "night").mean() > 0.8

    def test_one_dimensional_input_is_one_sample(self, fitted):
        assert fitted.predict(_domain(0.0, 1, 32)[0]).details["scores"].shape == (1,)


@pytest.mark.required
class TestChunked:
    def test_is_a_chunkable_drift_detector(self):
        assert isinstance(DriftKnownDomains(), BaseDrift)
        assert isinstance(DriftKnownDomains(), ChunkableMixin)

    def test_reference_data_is_the_stacked_reference(self, detector):
        assert detector.reference_data.shape == (450, N_FEATURES)

    def test_chunk_value_is_the_whole_batch_p_value(self, reference, detector):
        data = _domain(2.0, 100, 50)
        chunked = DriftKnownDomains(n_folds=3).chunked(chunk_size=100).fit(reference)
        value = chunked.predict(data).details["value"][0]
        assert value == pytest.approx(detector.predict(data).details["p_val"])

    def test_default_threshold_is_p_val_as_a_lower_bound(self, reference):
        chunked = DriftKnownDomains(n_folds=3, p_val=0.01).chunked(chunk_size=100).fit(reference)
        assert chunked._threshold_bounds == (0.01, None)

    def test_a_blended_chunk_drifts_and_scores_below_the_known_one(self, reference):
        # 200 per chunk: with 100, a blend's p-value ranges up to ~0.08 across seeds; with 200, up to ~0.01
        data = np.concatenate([_domain(4.0, 200, 51), _domain(2.0, 200, 52)])
        result = DriftKnownDomains(n_folds=3).chunked(chunk_size=200).fit(reference).predict(data)
        values = result.details["value"].to_list()
        assert result.details["drifted"].to_list()[1]
        assert values[1] < values[0]

    def test_a_non_constant_threshold_override_raises(self):
        with pytest.raises(ValueError, match="ConstantThreshold"):
            DriftKnownDomains().chunked(chunk_size=100, threshold=ZScoreThreshold())

    def test_a_number_resolves_to_a_non_constant_threshold_and_raises(self):
        with pytest.raises(ValueError, match="ConstantThreshold"):
            DriftKnownDomains().chunked(chunk_size=100, threshold=2.0)

    def test_a_constant_threshold_override_is_used(self, reference):
        chunked = DriftKnownDomains(n_folds=3).chunked(chunk_size=100, threshold=ConstantThreshold(lower=0.001))
        assert chunked.fit(reference)._threshold_bounds == (0.001, None)

    def test_chunked_fit_forwards_metadata(self, labeled):
        x, time = labeled
        chunked = DriftKnownDomains(n_folds=3, domains=SPEC).chunked(chunk_size=100)
        chunked.fit(x, metadata=_metadata(time=time))
        assert chunked.predict(_domain(4.0, 100, 53)).details.height == 1

    def test_chunked_predict_checks_feature_count(self, reference):
        chunked = DriftKnownDomains(n_folds=3).chunked(chunk_size=100).fit(reference)
        with pytest.raises(ShapeMismatchError):
            chunked.predict(np.zeros((100, N_FEATURES + 1)))
