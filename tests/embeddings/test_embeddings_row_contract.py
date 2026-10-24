"""``Embeddings`` stores one row per dataset item, so an extractor that returns anything else is refused.

Before the check, a detection model wrapped in ``ScoresExtractor`` and handed to ``Embeddings`` returned
half the rows it should have, with no error, because each image took the row at its own position.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dataeval import Embeddings
from dataeval.exceptions import ShapeMismatchError
from dataeval.extractors import ClasswiseUncertaintyExtractor, FlattenExtractor, ScoresExtractor

IMAGES = [np.full((1, 4, 4), i, dtype=np.float32) for i in range(8)]


class RowsPerImage:
    """Returns ``k`` rows for every image, as an extractor that emits one row per detection does."""

    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, data: Any) -> np.ndarray:
        return np.repeat(np.arange(len(data), dtype=np.float32)[:, None], self.k, axis=0)


class WrongOnTheSecondBatch:
    """Correct for the first call and short by one row after that."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, data: Any) -> np.ndarray:
        self.calls += 1
        rows = len(data) if self.calls == 1 else len(data) - 1
        return np.zeros((rows, 3), dtype=np.float32)


class Detector:
    """A fake detection model returning ``counts[i % len(counts)]`` detections for the i-th image it sees.

    Every score is ``1/3``, a valid distribution over three classes. With ``tag_rows`` each score is instead the
    index of the image that produced it, so a test can see which image a stored row came from.
    """

    def __init__(self, counts: list[int], *, tag_rows: bool = False) -> None:
        self.counts = counts
        self.tag_rows = tag_rows
        self.seen = 0

    def __call__(self, images: list[Any]) -> list[Any]:
        predictions = []
        for _ in images:
            count = self.counts[self.seen % len(self.counts)]
            fill = float(self.seen) if self.tag_rows else 1 / 3
            predictions.append(SimpleNamespace(scores=np.full((count, 3), fill, dtype=np.float32)))
            self.seen += 1
        return predictions


@pytest.mark.required
class TestRowContract:
    def test_an_extractor_with_too_many_rows_raises_and_names_itself(self):
        with pytest.raises(ShapeMismatchError, match=r"RowsPerImage returned 8 rows for a batch of 4 images"):
            Embeddings(IMAGES, extractor=RowsPerImage(2), batch_size=4)[:]

    def test_the_message_points_at_the_drift_detector_route(self):
        with pytest.raises(ShapeMismatchError, match=r"ScoresExtractor.*drift detector"):
            Embeddings(IMAGES, extractor=RowsPerImage(2), batch_size=4)[:]

    def test_a_detection_model_through_scores_extractor_raises(self):
        extractor = ScoresExtractor(Detector([2]))
        with pytest.raises(ShapeMismatchError, match=r"ScoresExtractor returned 8 rows for a batch of 4 images"):
            Embeddings(IMAGES, extractor=extractor, batch_size=4)[:]

    def test_a_mismatch_in_a_later_batch_raises(self):
        with pytest.raises(ShapeMismatchError, match=r"WrongOnTheSecondBatch returned 3 rows for a batch of 4"):
            Embeddings(IMAGES, extractor=WrongOnTheSecondBatch(), batch_size=4)[:]

    def test_a_single_value_raises(self):
        with pytest.raises(ShapeMismatchError, match=r"returned a single value for a batch of 4 images"):
            Embeddings(IMAGES, extractor=lambda data: np.float32(1.0), batch_size=4)[:]  # pyright: ignore[reportArgumentType]

    def test_a_dict_raises_a_type_error(self):
        scorer = ScoresExtractor(Detector([2]))
        extractor = ClasswiseUncertaintyExtractor(scorer, preds_type="probs")
        with pytest.raises(TypeError, match=r"ClasswiseUncertaintyExtractor returned a dict, not an array"):
            Embeddings(IMAGES, extractor=extractor, batch_size=4)[:]  # pyright: ignore[reportArgumentType]

    def test_one_row_per_image_still_works(self):
        embeddings = Embeddings(IMAGES, extractor=FlattenExtractor(), batch_size=3)
        assert np.asarray(embeddings).shape == (8, 16)

    def test_a_matching_count_is_not_proof_of_alignment(self):
        """Documented limit of the check, locked so that tightening it later is a decision.

        Detection counts ``[0, 2, 0, 2]`` give four rows for four images, so the row count matches. The rows
        were produced by images 1, 1, 3 and 3, yet they are stored against images 0, 1, 2 and 3. Nothing in a
        shape distinguishes this from a correct result, which is why the class docstring tells users not to
        wrap per-detection extractors in ``Embeddings`` instead of relying on this check.
        """
        extractor = ScoresExtractor(Detector([0, 2, 0, 2], tag_rows=True))
        stored = np.asarray(Embeddings(IMAGES[:4], extractor=extractor, batch_size=4))
        assert stored.shape == (4, 3)
        assert stored[:, 0].tolist() == [1.0, 1.0, 3.0, 3.0]
