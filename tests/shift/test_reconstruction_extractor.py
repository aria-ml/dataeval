"""Tests for the optional ``extractor`` bridge on the reconstruction detectors.

Both :class:`DriftReconstruction` and :class:`OODReconstruction` gained an
optional ``extractor: FeatureExtractor`` constructor parameter so that a caller
can feed a full dataset / raw images (anything the extractor accepts) instead of
having to pre-materialize a float array.  These tests exercise that bridge and
also assert the legacy array-in path is unchanged when no extractor is given.
"""

import numpy as np
import pytest

from dataeval.protocols import FeatureExtractor
from dataeval.shift._drift._base import DriftOutput
from dataeval.shift._drift._reconstruction import DriftReconstruction
from dataeval.shift._ood._base import OODScoreOutput
from dataeval.shift._ood._reconstruction import OODReconstruction
from dataeval.utils.models import AE

input_shape = (1, 8, 8)


class StackingExtractor:
    """Trivial FeatureExtractor: stack a list of small images into ``(N, C, H, W)`` float32 in [0, 1].

    Stands in for a MAITE-dataset -> array bridge.  Clamps to the unit interval
    so the resulting array satisfies the OOD ``[0, 1]`` requirement.
    """

    def __call__(self, data):
        arr = np.stack([np.asarray(img, dtype=np.float32) for img in data])
        return np.clip(arr, 0.0, 1.0)


@pytest.fixture
def image_list():
    """A plain Python list of small image arrays (stand-in for a dataset)."""
    rng = np.random.default_rng(0)
    return [rng.uniform(0.0, 1.0, input_shape).astype(np.float32) for _ in range(20)]


@pytest.fixture
def raw_array():
    rng = np.random.default_rng(1)
    return rng.uniform(0.0, 1.0, (20, *input_shape)).astype(np.float32)


def test_stacking_extractor_is_feature_extractor():
    """Sanity: the trivial extractor satisfies the FeatureExtractor protocol."""
    assert isinstance(StackingExtractor(), FeatureExtractor)


@pytest.mark.required
def test_ood_reconstruction_extractor_accepts_list(image_list):
    """OODReconstruction with an extractor can fit/score/predict on a plain list."""
    ood = OODReconstruction(
        AE(input_shape=input_shape),
        extractor=StackingExtractor(),
        threshold_perc=90,
        config=OODReconstruction.Config(epochs=1),
    )
    # No caller-side conversion to array:
    ood.fit(image_list)
    assert hasattr(ood, "_ref_score")

    scores = ood.score(image_list)
    assert isinstance(scores, OODScoreOutput)
    assert scores.instance_score.shape == (len(image_list),)

    preds = ood.predict(image_list)
    assert preds.is_ood.shape == (len(image_list),)


@pytest.mark.required
def test_ood_reconstruction_range_check_runs_post_extraction():
    """The [0, 1] range validation must run on the POST-extraction array."""

    class OutOfRangeExtractor:
        def __call__(self, data):
            return np.stack([np.asarray(img, dtype=np.float32) for img in data]) + 5.0

    imgs = [np.random.rand(*input_shape).astype(np.float32) for _ in range(5)]
    ood = OODReconstruction(
        AE(input_shape=input_shape),
        extractor=OutOfRangeExtractor(),
        config=OODReconstruction.Config(epochs=1),
    )
    with pytest.raises(ValueError, match="unit interval"):
        ood.fit(imgs)


@pytest.mark.required
def test_ood_reconstruction_no_extractor_raw_array(raw_array):
    """Without an extractor, OODReconstruction still works on a raw array."""
    ood = OODReconstruction(
        AE(input_shape=input_shape),
        threshold_perc=90,
        config=OODReconstruction.Config(epochs=1),
    )
    ood.fit(raw_array)
    scores = ood.score(raw_array)
    assert scores.instance_score.shape == (raw_array.shape[0],)
    assert ood._extractor is None


@pytest.mark.required
def test_drift_reconstruction_extractor_accepts_list(image_list):
    """DriftReconstruction with an extractor can fit/predict on a plain list."""
    det = DriftReconstruction(
        AE(input_shape=input_shape),
        extractor=StackingExtractor(),
        config=DriftReconstruction.Config(epochs=1, batch_size=10),
    ).fit(image_list)

    result = det.predict(image_list)
    assert isinstance(result, DriftOutput)
    assert result.metric_name == "reconstruction_error"
    assert "p_val" in result.details
    assert "mean_ref_error" in result.details
    assert "mean_test_error" in result.details


@pytest.mark.required
def test_drift_reconstruction_extractor_chunked(image_list):
    """The extractor is also applied on the chunked path."""
    import polars as pl

    det = (
        DriftReconstruction(
            AE(input_shape=input_shape),
            extractor=StackingExtractor(),
            config=DriftReconstruction.Config(epochs=1, batch_size=10),
        )
        .chunked(chunk_size=10)
        .fit(image_list)
    )
    result = det.predict(image_list)
    assert isinstance(result, DriftOutput)
    assert isinstance(result.details, pl.DataFrame)


@pytest.mark.required
def test_drift_reconstruction_no_extractor_raw_array(raw_array):
    """Without an extractor, DriftReconstruction still works on a raw array."""
    det = DriftReconstruction(
        AE(input_shape=input_shape),
        config=DriftReconstruction.Config(epochs=1, batch_size=10),
    ).fit(raw_array)
    result = det.predict(raw_array)
    assert isinstance(result, DriftOutput)
    assert det.extractor is None
