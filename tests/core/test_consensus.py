import itertools

import numpy as np
import pytest

from dataeval.core import (
    consensus_max,
    cost_consensus,
    cost_consensus_max,
    disjoint_panels,
    kink_costs,
    zero_one_consensus,
)

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")


@pytest.mark.required
class TestConsensusValues:
    @pytest.mark.parametrize("n_classes", [2, 3, 5])
    @pytest.mark.parametrize("m", [2, 3, 5])
    def test_consensus_max_is_tight(self, n_classes, m):
        labels = np.array(list(itertools.product(range(n_classes), repeat=m)), dtype=np.intp)
        assert zero_one_consensus(labels, n_classes).max() == pytest.approx(consensus_max(n_classes, m))

    def test_cost_consensus_max_values(self):
        assert cost_consensus_max(8, 2) == 0.5
        assert cost_consensus_max(2, 3) == pytest.approx(2 / 3)
        assert cost_consensus_max(0, 4) == 0
        for m in (2, 3, 5):
            for c in kink_costs(m)[1:]:
                assert cost_consensus_max(c, m) == pytest.approx(c / (1 + c))

    def test_cost_consensus_matches_max(self):
        k = np.arange(6, dtype=np.intp)
        assert cost_consensus(k, 5, 1.5).max() == pytest.approx(cost_consensus_max(1.5, 5))

    def test_expected_binary_consensus_is_p_times_q(self):
        rng = np.random.default_rng(0)
        labels = (rng.random((200_000, 2)) < 0.3).astype(np.intp)
        assert zero_one_consensus(labels, 2).mean() == pytest.approx(0.21, abs=3e-3)

    def test_kink_costs(self):
        np.testing.assert_allclose(kink_costs(5), [0, 1 / 4, 2 / 3, 3 / 2, 4])
        np.testing.assert_array_equal(kink_costs(1), [0])

    @pytest.mark.parametrize(
        "call",
        [
            lambda: consensus_max(1, 2),
            lambda: consensus_max(2, 0),
            lambda: cost_consensus_max(-1, 2),
            lambda: cost_consensus_max(1, 0),
            lambda: kink_costs(0),
            lambda: zero_one_consensus(np.zeros(3, dtype=np.intp), 2),
            lambda: zero_one_consensus(np.array([[0, 2]]), 2),
            lambda: zero_one_consensus(np.array([[0, 1]]), 1),
            lambda: cost_consensus(np.array([3]), 2, 1.0),
            lambda: cost_consensus(np.array([1]), 2, -1.0),
        ],
    )
    def test_invalid_inputs(self, call):
        with pytest.raises(ValueError, match="must"):
            call()


@pytest.mark.required
class TestDisjointPanels:
    def test_panels_are_disjoint_and_within_cell(self):
        cells = np.repeat([0, 1, 2], [7, 10, 1])
        panels = disjoint_panels(cells, 3, np.random.default_rng(1))
        assert {h: p.shape for h, p in panels.items()} == {0: (2, 3), 1: (3, 3), 2: (0, 3)}
        used = np.concatenate([p.ravel() for p in panels.values()])
        assert len(used) == len(set(used.tolist()))
        for h, p in panels.items():
            assert np.all(cells[p] == h)

    def test_panels_depend_only_on_the_first_draw(self):
        cells = np.repeat([0, 1], 20)
        a, b = np.random.default_rng(7), np.random.default_rng(7)
        pa = disjoint_panels(cells, 2, a)
        pb = disjoint_panels(cells, 2, b)
        b.random(100)
        assert all(np.array_equal(pa[h], pb[h]) for h in pa)

    def test_invalid_inputs(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="m must"):
            disjoint_panels(np.zeros(4, dtype=np.intp), 1, rng)
        with pytest.raises(ValueError, match="1D integer"):
            disjoint_panels(np.zeros(4), 2, rng)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="1D integer"):
            disjoint_panels(np.zeros((2, 2), dtype=np.intp), 2, rng)
