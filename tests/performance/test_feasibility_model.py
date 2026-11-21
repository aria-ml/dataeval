"""Tests for Feasibility's ``model=`` input: the same determinations as ``predictions=``, in batches, class order."""

import re
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal

from dataeval.config import set_batch_size
from dataeval.performance import Feasibility, TargetSpec
from dataeval.performance._feasibility_model import check_class_order, model_scores, run_model
from tests.data.conftest import MockICDataset
from tests.performance.feasibility_helpers import make_population

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

AB = {0: "a", 1: "b"}


class _Indexed(MockICDataset):
    """A classification dataset whose image is its own index, so a fake model can look its scores up."""

    def __init__(self, labels: Any, index2label: Any = AB):
        super().__init__([int(v) for v in labels], index2label)

    def __getitem__(self, i: int) -> Any:
        _, target, meta = super().__getitem__(i)
        return np.array([i], dtype=np.float32), target, meta


class _Model:
    """A MAITE-shaped model that returns the given score row for each image, recording its batch sizes."""

    def __init__(self, scores, index2label: dict[int, str] | None = None, *, as_lists: bool = False):
        self.scores, self.as_lists, self.batches = scores, as_lists, []
        self.metadata: dict[str, Any] = {"id": "fake"} | ({} if index2label is None else {"index2label": index2label})

    def __call__(self, batch):
        self.batches.append(len(batch))
        rows = [self.scores[int(image[0])] for image in batch]
        return [row.tolist() for row in rows] if self.as_lists else rows


def _dataset(labels: Any) -> Any:
    """An indexed dataset typed ``Any``: the fakes do not satisfy the protocol's metadata type."""
    return _Indexed(labels)


def _gapped(labels: list[int | None]) -> Any:
    return _IndexedWithGaps(labels)


def _model(scores: Any, index2label: dict[int, str] | None = None, *, as_lists: bool = False) -> Any:
    return _Model(scores, index2label, as_lists=as_lists)


class _TorchModel:
    """A MAITE-shaped torch model: one row of float32 class scores per image, tensors that require grad."""

    def __init__(self, scores):
        self.table = torch.nn.Parameter(torch.as_tensor(scores, dtype=torch.float32))
        self.grad_enabled: list[bool] = []

    def __call__(self, batch):
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.table[torch.as_tensor([int(image[0]) for image in batch])]


def _torch_model(scores: Any) -> Any:
    return _TorchModel(scores)


def _wrong_count(scores: Any, sizes: list[int]) -> Any:
    return _WrongCount(scores, sizes)


def _wrapped(inner: Any, wrap: Any) -> Any:
    return _Wrapped(inner, wrap)


def _reusing_buffer(scores: Any) -> Any:
    return _ReusingBuffer(scores)


class _WrongCount(_Model):
    """A model whose k-th batch returns ``sizes[k]`` rows, whatever the batch holds."""

    def __init__(self, scores, sizes: list[int]):
        super().__init__(scores)
        self.sizes = iter(sizes)

    def __call__(self, batch):
        return (super().__call__(batch) * 2)[: next(self.sizes)]


class _Wrapped:
    """A model whose batch output is passed through ``wrap`` before it is returned."""

    def __init__(self, inner: Any, wrap: Any):
        self.inner, self.wrap = inner, wrap

    def __call__(self, batch):
        return self.wrap(self.inner(batch))


class _ByGetitem:
    """A sequence with only ``__getitem__`` and ``__len__``: it iterates through the legacy protocol."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class _ReusingBuffer:
    """A model that writes each batch into one float64 array it keeps and returns a view of it."""

    def __init__(self, scores):
        self.scores, self.buffer = np.asarray(scores, dtype=np.float64), np.zeros((3, scores.shape[1]))

    def __call__(self, batch):
        for k, image in enumerate(batch):
            self.buffer[k] = self.scores[int(image[0])]
        return self.buffer[: len(batch)]


def _population(n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    x, y, post = make_population("gauss2").sample(n, rng)
    return x, y, post


def _target() -> list[TargetSpec]:
    return [TargetSpec(metric="accuracy", threshold=0.6), TargetSpec(metric="ap", threshold=0.5, class_name="a")]


@pytest.mark.required
class TestModelInput:
    def test_a_model_alone_is_a_minimal_call(self):
        _, y, post = _population()
        out = Feasibility(seed=1).evaluate(_dataset(y), target=_target()[:1], model=_model(post))
        assert set(out.bounds["source"]) == {"user_model"}
        assert out.determinations["determination"].to_list() == ["demonstrated_feasible"]

    def test_model_matches_predictions_exactly(self):
        x, y, post = _population()
        by_model = Feasibility(seed=1).evaluate(_dataset(y), target=_target(), embeddings=x, model=_model(post))
        by_predictions = Feasibility(seed=1).evaluate(_dataset(y), target=_target(), embeddings=x, predictions=post)
        assert_frame_equal(by_model.determinations, by_predictions.determinations)
        assert_frame_equal(by_model.bounds, by_predictions.bounds)

    def test_model_keeps_float64_precision(self):
        x, y, _ = _population(n=1000)
        assert np.float32(0.5000000001) == np.float32(0.5)  # a float32 cast would make these scores a tie
        scores = np.tile(np.array([0.5, 0.5000000001]), (len(y), 1))  # the top class is always class 1
        spec = [TargetSpec(metric="accuracy", threshold=0.4)]
        by_model = Feasibility(seed=1).evaluate(_dataset(y), target=spec, embeddings=x, model=_model(scores))
        by_scores = Feasibility(seed=1).evaluate(_dataset(y), target=spec, embeddings=x, predictions=scores)
        always_class_1 = Feasibility(seed=1).evaluate(
            _dataset(y), target=spec, embeddings=x, predictions=np.ones(len(y), dtype=int)
        )
        assert_frame_equal(by_model.bounds, by_scores.bounds)
        user = lambda out: out.bounds.filter(pl.col("source") == "user_model").drop("assumptions")  # noqa: E731
        assert_frame_equal(user(by_model), user(always_class_1))  # class 1 won every item, not class 0 by a tie

    def test_model_float32_output_widens_exactly(self):
        x, y, post = _population()
        scores = post.astype(np.float32)
        by_model = Feasibility(seed=1).evaluate(_dataset(y), target=_target(), embeddings=x, model=_model(scores))
        by_scores = Feasibility(seed=1).evaluate(_dataset(y), target=_target(), embeddings=x, predictions=scores)
        assert_frame_equal(by_model.bounds, by_scores.bounds)

    def test_model_may_return_lists_of_scores(self):
        x, y, post = _population()
        listed = Feasibility(seed=1).evaluate(
            _dataset(y), target=_target(), embeddings=x, model=_model(post, as_lists=True)
        )
        arrays = Feasibility(seed=1).evaluate(_dataset(y), target=_target(), embeddings=x, model=_model(post))
        assert_frame_equal(listed.bounds, arrays.bounds)

    def test_a_torch_model_whose_outputs_require_grad_is_scored_without_autograd(self):
        x, y, post = _population(n=100)
        model = _torch_model(post)
        scores = model_scores(model, _dataset(y), 7)
        np.testing.assert_array_equal(scores, post.astype(np.float32).astype(np.float64))  # widened, never narrowed
        assert not any(model.grad_enabled)  # no autograd graph is built for the audit batches
        out = Feasibility(seed=1).evaluate(_dataset(y), target=_target()[:1], embeddings=x, model=_torch_model(post))
        assert out.determinations["determination"].to_list() == ["demonstrated_feasible"]

    def test_model_scores_run_in_batches_of_batch_size_and_score_every_item_once(self):
        x, y, post = _population(n=100)
        model = _model(post)
        Feasibility(seed=1, batch_size=3).evaluate(_dataset(y), target=_target()[:1], embeddings=x, model=model)
        assert max(model.batches) == 3
        assert sum(model.batches) == len(y)
        assert model_scores(_model(post), _dataset(y), 7).shape == post.shape

    def test_a_model_needs_a_batch_size_and_the_error_says_where_to_set_one(self):
        x, y, post = _population(n=100)
        set_batch_size(None)
        try:
            with pytest.raises(ValueError, match=r"Feasibility\(batch_size=.*set_batch_size"):
                Feasibility().evaluate(_dataset(y), target=_target()[:1], embeddings=x, model=_model(post))
            model = _model(post)
            Feasibility(batch_size=7).evaluate(_dataset(y), target=_target()[:1], embeddings=x, model=model)
            assert max(model.batches) == 7  # the constructor's batch size is all a model call needs
        finally:
            set_batch_size(16)  # the suite's batch size (tests/conftest.py)

    def test_model_and_predictions_together_raise(self):
        x, y, post = _population(n=100)
        with pytest.raises(ValueError, match="model or predictions, not both"):
            Feasibility().evaluate(
                _dataset(y), target=_target()[:1], embeddings=x, model=_model(post), predictions=post
            )

    def test_model_with_the_wrong_class_count_raises(self):
        x, y, _ = _population(n=100)
        three = np.full((100, 3), 1 / 3)
        with pytest.raises(ValueError, match=r"\(len\(dataset\), n_classes\) = \(100, 2\), got \(100, 3\)"):
            Feasibility().evaluate(_dataset(y), target=_target()[:1], embeddings=x, model=_model(three))

    def test_opposite_index2label_raises_value_error(self):
        x, y, post = _population(n=100)
        opposite = {0: "b", 1: "a"}
        with pytest.raises(ValueError, match="never remaps"):
            Feasibility().evaluate(_dataset(y), target=_target()[:1], embeddings=x, model=_model(post, opposite))

    def test_equal_index2label_passes_and_drops_the_assumption(self):
        x, y, post = _population()
        out = Feasibility(seed=1).evaluate(_dataset(y), target=_target(), embeddings=x, model=_model(post, dict(AB)))
        user = out.bounds.filter(pl.col("source") == "user_model")
        assert user.height > 0
        for assumptions in user["assumptions"].to_list():
            assert assumptions == ["model_independent_of_audit_items"]  # the class order was checked, not assumed

    @pytest.mark.parametrize("side", ["model", "dataset", "both"])
    def test_a_missing_mapping_on_either_side_lists_the_assumption(self, side):
        x, y, post = _population(n=300)
        model = _model(post, None if side in ("model", "both") else dict(AB))
        dataset = _dataset(y)
        if side in ("dataset", "both"):
            dataset.metadata.pop("index2label")
        spec = [TargetSpec(metric="accuracy", threshold=0.6)]
        out = Feasibility(seed=1).evaluate(dataset, target=spec, embeddings=x, model=model)
        for assumptions in out.bounds.filter(pl.col("source") == "user_model")["assumptions"].to_list():
            assert "class_order_matches_dataset" in assumptions

    def test_the_model_runs_on_the_whole_dataset_with_unlabeled_items(self):
        x, y, post = _population(n=200)
        labels: list[int | None] = [int(v) for v in y]
        labels[::10] = [None] * len(labels[::10])
        model = _model(post)
        out = Feasibility(seed=1).evaluate(_gapped(labels), target=_target()[:1], embeddings=x, model=model)
        assert sum(model.batches) == len(labels)  # every item, labeled or not, aligned with the embeddings
        assert set(out.bounds["population"]) == {"labeled_items"}

    def test_a_short_second_batch_is_refused_naming_its_first_image(self):
        _, y, post = _population(n=6)
        message = (
            "The model returned 2 outputs for the batch of 3 images starting at image 3. "
            "Return one output per image of the batch, in order."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model_scores(_wrong_count(post, [3, 2]), _dataset(y), 3)

    def test_a_long_first_batch_is_refused_at_image_0(self):
        _, y, post = _population(n=6)
        message = (
            "The model returned 4 outputs for the batch of 3 images starting at image 0. "
            "Return one output per image of the batch, in order."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model_scores(_wrong_count(post, [4]), _dataset(y), 3)

    @pytest.mark.parametrize(
        ("wrap", "name"), [(lambda rows: {"scores": rows}, "dict"), (lambda _: object(), "object")]
    )
    def test_a_mapping_or_non_iterable_batch_output_is_refused(self, wrap, name):
        _, y, post = _population(n=6)
        message = (
            f"The model returned a {name} for the batch of 3 images starting at image 0. "
            "Return one output per image of the batch, in order."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            model_scores(_wrapped(_model(post), wrap), _dataset(y), 3)

    def test_a_sequence_that_iterates_by_getitem_is_accepted(self):
        _, y, post = _population(n=20)
        by_getitem = model_scores(_wrapped(_model(post), _ByGetitem), _dataset(y), 7)
        np.testing.assert_array_equal(by_getitem, model_scores(_model(post), _dataset(y), 7))

    def test_a_reused_float64_output_buffer_does_not_overwrite_earlier_rows(self):
        _, y, post = _population(n=20)
        reused = model_scores(_reusing_buffer(post), _dataset(y), 3)
        np.testing.assert_array_equal(reused, model_scores(_model(post), _dataset(y), 3))

    def test_autograd_is_on_after_a_refused_batch(self):
        _, y, post = _population(n=20)
        one_row_short = _wrapped(_torch_model(post), lambda rows: rows[:-1])
        with pytest.raises(ValueError, match="returned 6 outputs for the batch of 7"):
            model_scores(one_row_short, _dataset(y), 7)
        assert torch.is_grad_enabled()

    def test_a_batch_size_above_the_dataset_runs_one_batch(self):
        _, y, post = _population(n=10)
        model = _model(post)
        assert model_scores(model, _dataset(y), 64).shape == post.shape
        assert model.batches == [10]

    def test_run_model_reads_each_output_with_its_dataset_index_and_without_autograd(self):
        _, y, post = _population(n=20)
        seen: list[tuple[int, bool]] = []

        def read(i: int, output: Any) -> int:
            seen.append((i, torch.is_grad_enabled()))
            return i

        assert run_model(_torch_model(post), _dataset(y), 7, read) == list(range(20))
        assert seen == [(i, False) for i in range(20)]

    def test_check_class_order_reports_what_it_could_verify(self):
        post = np.full((4, 2), 0.5)
        assert check_class_order(_model(post, dict(AB)), _dataset([0, 1])) is True
        assert check_class_order(_model(post), _dataset([0, 1])) is False
        with pytest.raises(ValueError, match="never remaps"):
            check_class_order(_model(post, {0: "b", 1: "a"}), _dataset([0, 1]))


class _IndexedWithGaps(_Indexed):
    """An indexed dataset in which a ``None`` label is an unlabeled item (an empty target)."""

    def __init__(self, labels: list[int | None]):
        super().__init__([0] * len(labels))
        self._gaps = labels

    def __getitem__(self, i: int) -> Any:
        image, target, meta = super().__getitem__(i)
        label = self._gaps[i]
        return image, np.zeros(0) if label is None else np.eye(2, dtype=np.float32)[label], meta
