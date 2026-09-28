"""from_metadata is what evaluate calls once it holds metadata, so the two agree on the same data."""

from typing import Any

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.bias import Balance, Diversity, Parity
from tests.conftest import MockICDataset

_FACTORS = {
    "var_cat": ["b", "b", "b", "b", "b", "a", "a", "b", "a", "b", "b", "a"],
    "var_float_cat": [1.1, 1.1, 0.1, 0.1, 1.1, 0.1, 1.1, 0.1, 0.1, 1.1, 1.1, 0.1],
}
_LABELS = [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]


def _dataset() -> Any:
    """A dataset of the factors above. Typed Any because the mock only approximates the dataset protocol."""
    images = np.zeros((len(_LABELS), 1, 16, 16))
    return MockICDataset(images, _LABELS, metadata=_FACTORS, classes=["cat", "dog"])  # type: ignore[arg-type]


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, pl.DataFrame):
        return isinstance(right, pl.DataFrame) and left.equals(right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    return left == right


_EVALUATORS = [Balance, Diversity, Parity]


@pytest.mark.required
class TestFromMetadata:
    @pytest.mark.parametrize("evaluator", _EVALUATORS, ids=lambda c: c.__name__)
    def test_it_agrees_with_evaluate_on_the_same_metadata(self, evaluator):
        metadata = Metadata(_dataset())
        assert _same(evaluator().from_metadata(metadata).data(), evaluator().evaluate(metadata).data())

    @pytest.mark.parametrize("evaluator", _EVALUATORS, ids=lambda c: c.__name__)
    def test_evaluate_on_a_dataset_builds_the_metadata_it_would_be_handed(self, evaluator):
        from_dataset = evaluator().evaluate(_dataset())
        from_metadata = evaluator().from_metadata(Metadata(_dataset()))
        assert _same(from_dataset.data(), from_metadata.data())

    @pytest.mark.parametrize("evaluator", _EVALUATORS, ids=lambda c: c.__name__)
    def test_each_call_records_its_own_name(self, evaluator):
        metadata = Metadata(_dataset())
        assert evaluator().from_metadata(metadata).meta().name.endswith(f"{evaluator.__name__}.from_metadata")
        assert evaluator().evaluate(metadata).meta().name.endswith(f"{evaluator.__name__}.evaluate")

    def test_the_metadata_is_kept_on_the_evaluator(self):
        metadata = Metadata(_dataset())
        balance = Balance()
        balance.from_metadata(metadata)
        assert balance.metadata is metadata
