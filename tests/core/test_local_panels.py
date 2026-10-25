import inspect
import math

import numpy as np
import pytest

from dataeval.core import fit_partition, local_panel_bound

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

# Spike toy (spike/feasibility-dfe:spike/feasibility/tests/test_routes.py, test_c3_bound_on_three_cell_toy):
# n = (600, 300, 100), N = 1000; cell 2 has no panel. Cell means .375, .5.
VALUES = {0: np.array([0.0, 0.5, 0.5, 0.5]), 1: np.array([0.5, 0.5])}
N_BY_CELL = np.array([600, 300, 100])
KW = {"value_max": 0.5, "delta_within": 0.025, "delta_between": 0.025}
S = math.sqrt(math.log(1 / 0.025) / (2 * 1000))  # between-cell Hoeffding scale at N = 1000


@pytest.mark.required
class TestLocalPanelBound:
    def test_three_cell_toy_matches_hand_value(self):
        point = 0.6 * 0.375 + 0.3 * 0.5
        r1 = 0.5 * math.sqrt(math.log(1 / 0.025) * (0.6**2 / 4 + 0.3**2 / 2) / 2)
        r = local_panel_bound(VALUES, N_BY_CELL, **KW)
        assert r["bound"] == pytest.approx(point - r1 - 0.5 * S, rel=1e-12)
        assert r["bound"] == pytest.approx(0.1040, abs=1e-4)
        assert (r["n_units"], r["n_panels"], r["panels_by_cell"]) == (1000, 6, {0: 4, 1: 2, 2: 0})
        assert r["value_max"] == 0.5

    def test_cells_without_panels_contribute_zero(self):
        empty = {**VALUES, 2: np.empty(0)}
        assert local_panel_bound(empty, N_BY_CELL, **KW) == local_panel_bound(VALUES, N_BY_CELL, **KW)
        assert local_panel_bound({}, N_BY_CELL, **KW)["bound"] == 0.0

    def test_beta_lowers_the_bound_by_the_weighted_mean_plus_range(self):
        base = local_panel_bound(VALUES, N_BY_CELL, **KW)["bound"]
        beta = np.array([0.01, 0.02, 0.03])
        # every cell pays its weighted beta, the panel-less cell 2 included
        shift = 0.6 * 0.01 + 0.3 * 0.02 + 0.1 * 0.03 + (0.03 - 0.01) * S
        assert local_panel_bound(VALUES, N_BY_CELL, beta=beta, **KW)["bound"] == pytest.approx(base - shift, rel=1e-12)
        assert local_panel_bound(VALUES, N_BY_CELL, beta=0.01, **KW)["bound"] == pytest.approx(base - 0.01, rel=1e-12)

    def test_ledger_is_two_halves(self):
        r = local_panel_bound(VALUES, N_BY_CELL, value_max=0.5, delta_within=0.02, delta_between=0.03)
        assert r["deltas"] == (0.02, 0.03)
        assert r["within_radius"] == pytest.approx(0.5 * math.sqrt(math.log(1 / 0.02) * 0.135 / 2), rel=1e-12)
        assert r["between_radius"] == pytest.approx(0.5 * math.sqrt(math.log(1 / 0.03) / 2000), rel=1e-12)

    def test_over_range_value_raises(self):
        with pytest.raises(ValueError, match="value_max"):
            local_panel_bound({0: np.array([0.5 * (1 + 1e-9)])}, np.array([10]), **KW)
        ulp = local_panel_bound({0: np.array([0.5 * (1 + 1e-14)])}, np.array([10]), **KW)
        assert ulp == local_panel_bound({0: np.array([0.5])}, np.array([10]), **KW)


@pytest.mark.required
def test_partition_ignores_labels():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 4))
    # labels have no way in: the fit sees features, cell count and seed only, so any labels
    # (original or shuffled) leave the same features and seed with the same cells
    assert list(inspect.signature(fit_partition).parameters) == ["features_fit", "n_cells", "seed"]
    cells = fit_partition(features, 5, 0)(features)
    np.testing.assert_array_equal(cells, fit_partition(features, 5, 0)(features))
    assert cells.dtype == np.intp
    assert cells.shape == (300,)
    assert set(cells) <= set(range(5))
