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
        with pytest.raises(ValueError, match="not a valid UpdateStrategy"):
            DriftUnivariate(update_strategy="invalid")  # type: ignore

    def test_base_init_correction_valueerror(self):
        with pytest.raises(ValueError, match="must be `bonferroni` or `fdr`"):
            DriftUnivariate(n_features=2, correction="invalid")  # type: ignore

    def test_base_init_extractor_valueerror(self):
        with pytest.raises(ValueError, match="not a valid FeatureExtractor"):
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
