"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4.

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
import torch

from dataeval.exceptions import NotFittedError
from dataeval.protocols import Dataset
from dataeval.shift._drift._base import (
    ChunkedDrift,
    ChunkResult,
    DriftOutput,
    _chunk_results_to_dataframe,
)
from dataeval.shift._drift._chunk import (
    CountChunker,
    IndexChunker,
    SizeChunker,
    resolve_chunker,
)
from dataeval.shift._drift._domain_classifier import DriftDomainClassifier
from dataeval.shift._drift._kneighbors import DriftKNeighbors
from dataeval.shift._drift._mmd import DriftMMD
from dataeval.shift._drift._reconstruction import DriftReconstruction
from dataeval.shift._drift._univariate import DriftUnivariate
from dataeval.utils.models import AE
from dataeval.utils.thresholds import resolve_threshold


@pytest.mark.required
class TestBaseDrift:
    data = np.random.random((1, 10))
    model = torch.nn.Identity()
    batch_size = 10
    device = torch.device("cpu")

    def get_dataset(self, n: int = 100, n_features: int = 10) -> Dataset:
        mock = MagicMock(spec=Dataset)
        mock._selection = list(range(n))
        mock.__len__.return_value = n
        mock.__getitem__.return_value = np.random.random(n_features), np.zeros(10), {}
        return mock

    def test_base_init_update_x_ref_valueerror(self):
        with pytest.raises(ValueError, match="update_strategy"):
            DriftUnivariate(update_strategy="invalid")  # type: ignore

    def test_base_init_correction_valueerror(self):
        with pytest.raises(ValueError, match="correction"):
            DriftUnivariate(n_features=2, correction="invalid")  # type: ignore

    def test_base_init_extractor_valueerror(self):
        with pytest.raises(ValueError, match="extractor"):
            DriftUnivariate(extractor="invalid")  # type: ignore

    def test_base_init_infer_n_features(self):
        base = DriftUnivariate()
        base.fit(self.data)
        assert base.n_features == 10

    def test_base_init_set_n_features(self):
        base = DriftUnivariate(n_features=1)
        base.fit(self.data)
        assert base.n_features == 1

    def test_base_predict_correction_valueerror(self):
        base = DriftUnivariate()
        base.fit(self.data)
        mock_score = MagicMock()
        mock_score.return_value = (np.array(0.5), np.array(0.5))
        base.score = mock_score
        base.correction = "invalid"  # type: ignore
        with pytest.raises(ValueError, match="needs to be either `bonferroni` or `fdr`"):
            base.predict(np.empty([]))

    def test_base_fit_non_array_data_raises(self):
        base = DriftUnivariate()
        with pytest.raises(ValueError, match="Array-like"):
            base.fit("not an array")  # type: ignore

    def test_base_reference_data_before_fit_raises(self):
        base = DriftUnivariate()
        with pytest.raises(NotFittedError, match="Must call fit"):
            _ = base.reference_data

    def test_base_n_features_before_fit_raises(self):
        base = DriftUnivariate()
        with pytest.raises(NotFittedError, match="Must call fit"):
            _ = base.n_features

    def test_base_predict_before_fit_raises(self):
        base = DriftUnivariate()
        with pytest.raises(NotFittedError, match="Must call fit"):
            base.predict(np.zeros((10, 5)))


@pytest.mark.required
class TestDriftChunkedOutput:
    """Tests for DriftOutput with chunked (pl.DataFrame) details."""

    @pytest.fixture
    def sample_output(self):
        chunks = [
            ChunkResult(
                key="[0:9]",
                index=0,
                start_index=0,
                end_index=9,
                value=0.3,
                upper_threshold=0.5,
                lower_threshold=0.1,
                drifted=False,
            ),
            ChunkResult(
                key="[10:19]",
                index=1,
                start_index=10,
                end_index=19,
                value=0.7,
                upper_threshold=0.5,
                lower_threshold=0.1,
                drifted=True,
            ),
        ]
        df = _chunk_results_to_dataframe(chunks)
        return DriftOutput(
            drifted=bool(df["drifted"].any()),
            threshold=0.5,
            distance=float(df["value"].cast(pl.Float64).mean() or 0.0),  # type: ignore
            metric_name="test_metric",
            details=df,
        )

    def test_details_is_dataframe(self, sample_output):
        assert isinstance(sample_output.details, pl.DataFrame)
        assert len(sample_output.details) == 2

    def test_drifted_true(self, sample_output):
        assert sample_output.drifted is True

    def test_threshold(self, sample_output):
        assert sample_output.threshold == 0.5

    def test_distance(self, sample_output):
        assert sample_output.distance == pytest.approx(0.5, abs=1e-6)


@pytest.mark.required
class TestChunkers:
    """Tests for chunker validation and behavior."""

    def test_count_chunker_invalid(self):
        with pytest.raises(ValueError, match="invalid"):
            CountChunker(0)
        with pytest.raises(ValueError, match="invalid"):
            CountChunker(-1)

    def test_size_chunker_invalid_size(self):
        with pytest.raises(ValueError, match="invalid"):
            SizeChunker(0)

    def test_size_chunker_invalid_incomplete(self):
        with pytest.raises(ValueError, match="invalid"):
            SizeChunker(10, incomplete="invalid")  # type: ignore

    def test_size_chunker_keep(self):
        chunker = SizeChunker(3, incomplete="keep")
        groups = chunker.split(10)
        assert len(groups) == 4  # 3+3+3+1
        assert len(groups[-1]) == 1

    def test_size_chunker_drop(self):
        chunker = SizeChunker(3, incomplete="drop")
        groups = chunker.split(10)
        assert len(groups) == 3  # 3+3+3, drops last 1
        total = sum(len(g) for g in groups)
        assert total == 9

    def test_size_chunker_append(self):
        chunker = SizeChunker(3, incomplete="append")
        groups = chunker.split(10)
        assert len(groups) == 3  # 3+3+4
        assert len(groups[-1]) == 4

    def test_index_chunker_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            IndexChunker([])

    def test_index_chunker_split(self):
        chunker = IndexChunker([[0, 2, 4], [1, 3, 5]])
        groups = chunker.split(6)
        assert len(groups) == 2
        np.testing.assert_array_equal(groups[0], [0, 2, 4])
        np.testing.assert_array_equal(groups[1], [1, 3, 5])

    def test_base_chunker_callable(self):
        chunker = CountChunker(3)
        result = chunker(9)
        assert len(result) == 3

    def test_resolve_chunker_passthrough(self):
        chunker = CountChunker(3)
        assert resolve_chunker(chunker=chunker) is chunker

    def test_resolve_chunker_none(self):
        assert resolve_chunker() is None

    def test_resolve_chunker_indices(self):
        result = resolve_chunker(chunk_indices=[[0, 1], [2, 3]])
        assert isinstance(result, IndexChunker)


@pytest.mark.required
class TestDriftConfigRepr:
    """Tests that constructor params override config defaults and are reflected in repr."""

    def test_univariate_params_override_config(self):
        det = DriftUnivariate(method="cvm", p_val=0.1, correction="fdr")
        assert det.config.method == "cvm"
        assert det.config.p_val == 0.1
        assert det.config.correction == "fdr"
        assert "method='cvm'" in repr(det)
        assert "p_val=0.1" in repr(det)

    def test_univariate_config_object(self):
        cfg = DriftUnivariate.Config(method="mwu", p_val=0.01)
        det = DriftUnivariate(config=cfg)
        assert det.config.method == "mwu"
        assert det.config.p_val == 0.01

    def test_univariate_param_overrides_config_object(self):
        cfg = DriftUnivariate.Config(method="ks", p_val=0.05)
        det = DriftUnivariate(method="cvm", config=cfg)
        assert det.config.method == "cvm"
        assert det.config.p_val == 0.05  # not overridden, stays from config

    def test_domain_classifier_tuple_threshold_in_config(self):
        det = DriftDomainClassifier(threshold=(0.45, 0.65))
        assert det.config.threshold == (0.45, 0.65)
        assert "threshold=(0.45, 0.65)" in repr(det)

    def test_domain_classifier_default_config(self):
        det = DriftDomainClassifier()
        assert det.config.threshold == 0.55
        assert det.config.n_folds == 5

    def test_kneighbors_params_override_config(self):
        det = DriftKNeighbors(k=3, distance_metric="cosine", p_val=0.01)
        assert det.config.k == 3
        assert det.config.distance_metric == "cosine"
        assert det.config.p_val == 0.01
        assert "k=3" in repr(det)
        assert "distance_metric='cosine'" in repr(det)

    def test_mmd_params_override_config(self):
        det = DriftMMD(p_val=0.1, n_permutations=50, device="cpu")
        assert det.config.p_val == 0.1
        assert det.config.n_permutations == 50
        assert "p_val=0.1" in repr(det)
        assert "n_permutations=50" in repr(det)

    def test_reconstruction_param_override_config(self):
        det = DriftReconstruction(model=torch.nn.Identity(), p_val=0.01)
        assert det.config.p_val == 0.01
        assert "p_val=0.01" in repr(det)


@pytest.mark.required
class TestDriftFitPreconditions:
    """Every accessor that reads fitted state says so rather than answering with None."""

    def test_reference_data_before_fit_is_an_error(self):
        # DriftReconstruction inherits the base accessor; the adaptive detectors override it.
        detector = DriftReconstruction(AE(input_shape=(1, 8, 8)))
        with pytest.raises(NotFittedError, match="Must call fit\\(\\) before accessing reference_data"):
            detector.reference_data  # noqa: B018

    def test_chunked_predict_before_fit_is_an_error(self):
        chunked = ChunkedDrift(DriftUnivariate(), chunk_size=4)
        with pytest.raises(NotFittedError, match="Must call fit\\(\\) before predict\\(\\)"):
            chunked.predict(np.zeros((8, 3), dtype=np.float32))


@pytest.mark.required
class TestChunkedDriftConstruction:
    def test_a_detector_that_cannot_chunk_is_rejected(self):
        """Chunked mode needs the per-chunk metric hook, which ChunkableMixin supplies."""

        class _NotChunkable:
            pass

        with pytest.raises(TypeError, match="does not support chunked mode"):
            ChunkedDrift(_NotChunkable())  # type: ignore[arg-type]

    def test_some_chunking_specification_is_required(self):
        with pytest.raises(ValueError, match="Must provide chunker, chunk_size, or chunk_count"):
            ChunkedDrift(DriftUnivariate())

    def test_any_chunker_the_protocol_describes_is_accepted(self):
        """A plain callable satisfying :class:`~dataeval.protocols.Chunker` needs none of the private base class."""

        class _EqualChunker:
            def __call__(self, n: int) -> list[np.ndarray]:
                return [idx.astype(np.intp) for idx in np.array_split(np.arange(n), 4)]

        reference = np.random.default_rng(0).random((80, 3)).astype(np.float32)
        chunked = DriftUnivariate().chunked(chunker=_EqualChunker()).fit(reference)
        assert len(chunked.predict(reference).details) == 4


@pytest.mark.required
class TestChunkedDriftReferenceChunks:
    """The thresholds are a spread of reference-chunk scores, so fit needs enough chunks to have one."""

    reference = np.random.default_rng(0).random((261, 8)).astype(np.float32)

    @pytest.mark.parametrize(
        "detector",
        [DriftMMD(n_permutations=10, device="cpu"), DriftUnivariate(method="ks")],
        ids=["mmd", "ks"],
    )
    def test_two_reference_chunks_are_rejected(self, detector):
        """Two chunks scored against each other are one pair measured twice, so the baseline has no spread.

        MMD's two scores then differ only by float rounding, which pins both bounds to one value and flags
        every chunk; KS's are exactly equal, which leaves no bounds at all and flags none.
        """
        with pytest.raises(ValueError, match=r"261 reference samples into 2 chunks.*at least 3"):
            detector.chunked(chunk_size=200).fit(self.reference)

    def test_one_reference_chunk_is_rejected(self):
        with pytest.raises(ValueError, match=r"into 1 chunk.*at least 3"):
            DriftUnivariate().chunked(chunk_count=1).fit(self.reference)

    def test_a_constant_threshold_needs_no_spread(self):
        """ConstantThreshold ignores the baseline, so two reference chunks are enough for it."""
        from dataeval.utils.thresholds import ConstantThreshold

        chunked = DriftUnivariate().chunked(chunk_size=200, threshold=ConstantThreshold(upper=0.5)).fit(self.reference)
        assert chunked._threshold_bounds == (None, 0.5)

    def test_three_reference_chunks_are_enough(self):
        chunked = DriftUnivariate().chunked(chunk_count=3).fit(self.reference)
        assert chunked._baseline_values is not None
        assert len(chunked._baseline_values) == 3


@pytest.mark.required
class TestChunkedDriftThresholdLike:
    """chunked() takes the threshold spellings Outliers takes, resolved as resolve_threshold resolves them."""

    reference = np.random.default_rng(0).random((240, 8)).astype(np.float32)
    test = np.random.default_rng(1).random((120, 8)).astype(np.float32) + 0.2

    @pytest.mark.parametrize("spelling", ["zscore", 2.5, ("zscore", 2.5), ["modzscore", 3.0]], ids=str)
    def test_a_spelling_gives_the_result_of_its_resolved_threshold(self, spelling):
        spelled = DriftUnivariate().chunked(chunk_count=6, threshold=spelling).fit(self.reference)
        resolved = DriftUnivariate().chunked(chunk_count=6, threshold=resolve_threshold(spelling)).fit(self.reference)

        assert spelled._threshold_bounds == resolved._threshold_bounds
        assert spelled.predict(self.test).details.equals(resolved.predict(self.test).details)

    def test_a_constant_spelling_needs_no_spread(self):
        chunked = DriftUnivariate().chunked(chunk_size=200, threshold=("constant", (None, 0.5))).fit(self.reference)
        assert chunked._threshold_bounds == (None, 0.5)

    def test_unset_keeps_the_detector_default(self):
        default = DriftUnivariate()._default_chunk_threshold()
        unset = DriftUnivariate().chunked(chunk_count=6).fit(self.reference)
        explicit = DriftUnivariate().chunked(chunk_count=6, threshold=default).fit(self.reference)
        assert unset._threshold_bounds == explicit._threshold_bounds

    def test_chunked_drift_takes_a_spelling_directly(self):
        assert ChunkedDrift(DriftUnivariate(), chunk_count=6, threshold="iqr")._threshold_override == resolve_threshold(
            "iqr"
        )


@pytest.mark.required
class TestChunkedDriftIncomplete:
    """``incomplete`` decides what becomes of the reference's short final chunk."""

    reference = np.random.default_rng(0).random((261, 8)).astype(np.float32)

    @pytest.mark.parametrize(
        ("incomplete", "sizes"),
        [(None, [50] * 5 + [11]), ("keep", [50] * 5 + [11]), ("drop", [50] * 5), ("append", [50] * 4 + [61])],
    )
    def test_the_reference_is_split_as_asked(self, incomplete, sizes):
        detector = DriftUnivariate()
        seen: list[int] = []
        baselines = detector._compute_chunk_baselines

        def spy(chunks):
            seen.extend(len(chunk) for chunk in chunks)
            return baselines(chunks)

        detector._compute_chunk_baselines = spy
        detector.chunked(chunk_size=50, incomplete=incomplete).fit(self.reference)
        assert seen == sizes

    def test_test_data_always_merges_its_remainder(self):
        """Only the reference is split as asked; predict appends so every test chunk is full size."""
        chunked = DriftUnivariate().chunked(chunk_size=50, incomplete="keep").fit(self.reference)
        output = chunked.predict(self.reference)
        assert output.details["key"].to_list() == ["[0:49]", "[50:99]", "[100:149]", "[150:199]", "[200:260]"]

    @pytest.mark.parametrize(
        "spec", [{"chunk_count": 5}, {"chunker": SizeChunker(50)}, {"chunker": SizeChunker(50), "chunk_size": 50}]
    )
    def test_incomplete_without_a_chunk_size_is_rejected(self, spec):
        with pytest.raises(ValueError, match="incomplete applies only to chunk_size"):
            DriftUnivariate().chunked(incomplete="drop", **spec)


@pytest.mark.required
class TestChunkedDriftPredictEdges:
    """Predict needs a chunking rule at call time, and answers an empty run without one."""

    def test_a_detector_whose_chunker_was_cleared_is_rejected(self):
        chunked = ChunkedDrift(DriftUnivariate(), chunk_size=4)
        chunked.fit(np.random.default_rng(0).random((16, 3)).astype(np.float32))
        chunked._chunker = None  # the resolved chunker is what predict falls back to
        with pytest.raises(ValueError, match="No chunking specification provided"):
            chunked.predict(np.random.default_rng(1).random((16, 3)).astype(np.float32))

    def test_no_chunks_yields_an_undrifted_result(self):
        """An empty chunk list produces an empty frame, which reads as "no drift"."""
        chunked = ChunkedDrift(DriftUnivariate(), chunk_size=4)
        chunked.fit(np.random.default_rng(0).random((16, 3)).astype(np.float32))
        output = chunked.predict(np.random.default_rng(1).random((16, 3)).astype(np.float32), chunk_indices=[])
        assert output.drifted is False


class TestDriftOutputFeatureNames:
    """Test that per-feature statistics can be read by name.

    Everything ``details`` reports per feature is positional. When the extractor knows
    what its columns are, the result carries the names so the caller does not have to
    rebuild the column order by hand.
    """

    @pytest.fixture
    def metadata_dataset(self):
        from tests.embeddings.test_embeddings import MockDataset

        return MockDataset(
            np.ones((20, 3, 3)),
            np.ones((20, 3)),
            [{"altitude": float(i), "sensor": f"s_{i % 2}"} for i in range(20)],
        )

    def test_names_absent_without_an_extractor(self):
        """Array input has no names to report."""
        from dataeval.shift import DriftUnivariate

        rng = np.random.default_rng(0)
        detector = DriftUnivariate().fit(rng.standard_normal((50, 4)).astype(np.float32))

        assert detector.predict(rng.standard_normal((30, 4)).astype(np.float32)).feature_names == ()

    def test_names_absent_for_an_anonymous_extractor(self):
        """An extractor with no ``feature_names`` leaves the field empty rather than raising."""
        from dataeval.extractors import FlattenExtractor
        from dataeval.shift import DriftUnivariate

        rng = np.random.default_rng(0)
        detector = DriftUnivariate(extractor=FlattenExtractor()).fit(rng.standard_normal((50, 2, 2)).astype(np.float32))

        assert detector.predict(rng.standard_normal((30, 2, 2)).astype(np.float32)).feature_names == ()

    def test_names_match_metadata_factors(self, metadata_dataset):
        """A Metadata extractor labels the axis its own factor order defines."""
        from dataeval import Metadata
        from dataeval.shift import DriftUnivariate

        extractor = Metadata()
        result = DriftUnivariate(extractor=extractor).fit(metadata_dataset).predict(metadata_dataset)

        assert list(result.feature_names) == list(extractor.factor_names)
        assert len(result.feature_names) == len(result.details["p_vals"])

    def test_unfitted_named_extractor_does_not_raise(self):
        """Resolving names must answer empty, not propagate NotFittedError.

        On Python 3.11 a runtime-checkable protocol's instance check calls
        ``hasattr``, which would invoke the property and raise. The lookup is duck-typed
        to avoid that.
        """
        from dataeval.shift import DriftUnivariate

        class Unfitted:
            def __call__(self, data, /):
                return np.asarray(data, dtype=np.float32)

            @property
            def feature_names(self):
                raise NotFittedError("not fitted")

        detector = DriftUnivariate(extractor=Unfitted())
        assert detector._feature_names == ()

    def test_chunked_result_carries_names(self, metadata_dataset):
        """The chunked wrapper reports the wrapped detector's names."""
        from dataeval import Metadata
        from dataeval.shift import DriftUnivariate

        extractor = Metadata()
        chunked = DriftUnivariate(extractor=extractor).chunked(chunk_size=5)
        chunked.fit(metadata_dataset)
        feature_names = chunked.predict(metadata_dataset).feature_names

        assert feature_names
        assert list(feature_names) == list(extractor.factor_names)


@pytest.mark.required
class TestFeatureWiseScoringRefusesMismatchedWidths:
    """Feature f of one side is compared against feature f of the other, so an unequal
    count means the columns past the first difference each meet a different feature -- and
    every p-value from there on would be reported under a name that did not produce it."""

    def test_a_wider_test_set_raises(self):
        detector = DriftUnivariate(method="ks").fit(np.random.default_rng(0).normal(size=(40, 3)))
        with pytest.raises(ValueError, match="Reference data has 3 features"):
            detector.predict(np.random.default_rng(1).normal(size=(40, 5)))

    def test_a_narrower_test_set_raises(self):
        """The direction the loop would have read straight through, silently."""
        detector = DriftUnivariate(method="ks").fit(np.random.default_rng(0).normal(size=(40, 5)))
        with pytest.raises(ValueError, match="Reference data has 5 features"):
            detector.predict(np.random.default_rng(1).normal(size=(40, 3)))

    def test_the_message_names_both_counts(self):
        detector = DriftUnivariate(method="ks").fit(np.random.default_rng(0).normal(size=(40, 3)))
        with pytest.raises(ValueError, match="the data to score has 7"):
            detector.predict(np.random.default_rng(1).normal(size=(40, 7)))

    def test_matching_widths_are_scored(self):
        detector = DriftUnivariate(method="ks").fit(np.random.default_rng(0).normal(size=(40, 3)))
        assert len(detector.predict(np.random.default_rng(1).normal(size=(40, 3))).details["p_vals"]) == 3
