"""The stock image extractors refuse a video stream, which holds frames rather than pixels."""

import warnings
from typing import Any

import numpy as np
import pytest
import torch

from dataeval import Embeddings
from dataeval.exceptions import MaiteShapeError
from dataeval.extractors import FlattenExtractor, TorchExtractor


class _Stream:
    """An iterable of frame objects, as a tracking dataset holds in the image slot."""

    def __iter__(self):
        return iter([object(), object()])


class _TrackingDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        return _Stream(), None, {"id": index}


@pytest.mark.required
class TestImageExtractorsRefuseVideo:
    def test_flatten(self):
        with pytest.raises(MaiteShapeError, match="SequenceFrames"):
            FlattenExtractor()([_Stream(), _Stream()])

    def test_torch(self):
        with pytest.raises(MaiteShapeError, match="SequenceFrames"):
            TorchExtractor(torch.nn.Flatten(), device="cpu")([_Stream(), _Stream()])

    def test_bovw(self):
        pytest.importorskip("cv2", reason="opencv-python not installed")
        from dataeval.extractors import BoVWExtractor

        extractor = BoVWExtractor(vocab_size=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(MaiteShapeError, match="SequenceFrames"):
                extractor([_Stream(), _Stream()])

    def test_embeddings_of_a_tracking_dataset(self):
        with pytest.raises(MaiteShapeError, match="SequenceFrames"):
            np.asarray(Embeddings(_TrackingDataset(), extractor=FlattenExtractor(), batch_size=2))

    def test_an_extractor_that_reads_video_is_not_refused(self):
        class VideoExtractor:
            def __call__(self, data):
                return np.zeros((len(list(data)), 3), dtype=np.float32)

        embeddings = np.asarray(Embeddings(_TrackingDataset(), extractor=VideoExtractor(), batch_size=2))

        assert embeddings.shape == (4, 3)
