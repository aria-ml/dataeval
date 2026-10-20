import logging
import tracemalloc

import numpy as np
import pytest

from dataeval.core import _rao_scott
from dataeval.core._association import benjamini_hochberg, bias_corrected_cramers_v
from dataeval.core._rao_scott import _group_counts, _pooled_cell_variance, _sandwich, grouped_table

ALPHA = 0.05
K = 4  # classes in the generated scenarios


def bound(n_reps: int) -> float:
    """Nominal level plus three Monte Carlo standard errors."""
    return ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / n_reps)


def dense(codes: np.ndarray) -> np.ndarray:
    return np.unique(codes, return_inverse=True)[1]


def tertiles(x: np.ndarray) -> np.ndarray:
    return np.digitize(x, np.quantile(x, [1 / 3, 2 / 3]))


def mixed_classes(rng: np.random.Generator, sizes: np.ndarray, conc: float = 0.5) -> np.ndarray:
    """Class labels per group, each group drawing from its own Dirichlet class mix."""
    mix = rng.dirichlet(np.full(K, conc), sizes.size)
    return np.concatenate([rng.choice(K, m, p=p) for m, p in zip(sizes, mix, strict=True)])


def dense_sandwich(f: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The spec's definition, materialized: U_s = sum over rows in s of (x - p_f)(y - p_y)^T."""
    _, g = np.unique(g, return_inverse=True)
    n, s = f.size, g.max() + 1
    x_dev = np.eye(f.max() + 1)[f] - np.bincount(f) / n
    y_dev = np.eye(y.max() + 1)[y] - np.bincount(y) / n
    u = np.zeros((s, x_dev.shape[1] * y_dev.shape[1]))
    np.add.at(u, g, (x_dev[:, :, None] * y_dev[:, None, :]).reshape(n, -1))
    return s / (s - 1) * u.T @ u


# Null scenarios (spec validation row 3), ported from the evidence scripts. Each returns
# (factor_codes, class_codes, groups) under label-proportion homogeneity.


def reviewer_design(rng):  # script 21: 20 images x 20 detections vs 380 x 1, class constant per image
    sizes = np.r_[np.full(20, 20), np.ones(380, np.intp)]
    level = np.r_[np.zeros(20, np.intp), np.ones(380, np.intp)]
    g = np.repeat(np.arange(400), sizes)
    return level[g], rng.integers(0, 2, 400)[g], g


def od_sizes(rng, n_images, rare_share, size_rare, size_common):  # scripts 16/19
    level = (rng.random(n_images) < rare_share).astype(np.intp)
    level[:2] = [0, 1]
    sizes = np.maximum(1, np.where(level == 1, size_rare(n_images), size_common(n_images)).astype(np.intp))
    g = np.repeat(np.arange(n_images), sizes)
    return level[g], mixed_classes(rng, sizes), g


def od_rare_heavy(rng):  # a rare level (5%) holding heavy-tailed LN(3, 1.5) images
    return od_sizes(rng, 300, 0.05, lambda i: rng.lognormal(3, 1.5, i), lambda i: rng.lognormal(0.5, 0.5, i))


def seq_null(rng, n_seq, length=50):  # scripts 16/19: sequence-level factor with 3 levels
    level = rng.integers(0, 3, n_seq)
    level[:3] = [0, 1, 2]
    g = np.repeat(np.arange(n_seq), length)
    return level[g], mixed_classes(rng, np.full(n_seq, length)), g


def seq40(rng):
    return seq_null(rng, 40)


def seq10(rng):
    return seq_null(rng, 10)


def od_detection_factor(rng):  # script 06 A: detection-level factor with a per-image component
    sizes = rng.integers(3, 15, 200)
    g = np.repeat(np.arange(200), sizes)
    x = rng.normal(size=200)[g] + 0.5 * rng.normal(size=g.size)
    return tertiles(x), mixed_classes(rng, sizes), g


def mot_frame_factor(rng, n_seq=30, frames=40):  # script 06 B: drifting frame brightness, tracks fixed in class
    f, y, g = [], [], []
    for s in range(n_seq):
        mix = rng.dirichlet(np.full(K, 0.5))
        brightness = np.cumsum(0.15 * rng.normal(size=frames))
        for _ in range(6):
            cls = rng.choice(K, p=mix)
            start = rng.integers(0, frames - 5)
            length = rng.integers(5, frames - start + 1)
            f.append(brightness[start : start + length])
            y.append(np.full(length, cls))
            g.append(np.full(length, s))
    return tertiles(np.concatenate(f)), np.concatenate(y), np.concatenate(g)


def rare_class_bursty(rng, n_seq=60, holders=5):  # script 20: a rare class in a few big bursts
    level = rng.integers(0, 3, n_seq)
    level[:3] = [0, 1, 2]
    y = [rng.choice(K - 1, m) for m in rng.poisson(30, n_seq) + 1]
    for h in rng.choice(n_seq, holders, replace=False):
        y[h] = np.concatenate([y[h], np.full(int(rng.lognormal(4, 1)), K - 1)])
    g = np.repeat(np.arange(n_seq), [len(v) for v in y])
    return level[g], np.concatenate(y), g


def axis_group_level(rng, n_seq=60):  # class constant within groups, factor varying per row
    sizes = rng.poisson(20, n_seq) + 1
    g = np.repeat(np.arange(n_seq), sizes)
    x = rng.normal(size=n_seq)[g] + 0.5 * rng.normal(size=g.size)
    return tertiles(x), rng.integers(0, 3, n_seq)[g], g


def negative_icc(rng):  # script 21 s5A: pairs that usually hold one of each class
    pairs = rng.random(200) < 0.9
    y = np.concatenate([[0, 1] if p else rng.integers(0, 2, 2) for p in pairs])
    return np.repeat(rng.integers(0, 2, 200), 2), y, np.repeat(np.arange(200), 2)


def bursty_classes(rng, level, k, sizes, burst=0.5):  # Task 9b: each row is its group's own class w.p. `burst`
    g = np.repeat(np.arange(level.size), sizes)
    y = np.where(rng.random(g.size) < burst, rng.integers(0, k, level.size)[g], rng.integers(0, k, g.size))
    return level[g], y, g


def bursty_square(rng, r, n_groups):  # a group-level factor with r levels against r bursty classes
    return bursty_classes(rng, rng.integers(0, r, n_groups), r, 10)


def bursty_size_by_level(rng):  # 4 of 40 levels hold 40-row groups, the rest 4-row groups
    level = rng.integers(0, 40, 4000)
    return bursty_classes(rng, level, 40, np.where(level < 4, 40, 4))


def bursty_two_level_sizes(rng, burst):  # 10% of 4,000 groups hold 40 rows at level 0, the rest 4 rows
    level = (rng.random(4000) >= 0.1).astype(np.intp)
    return bursty_classes(rng, level, 100, np.where(level == 0, 40, 4), burst=burst)


def bursty_rare_clustered(rng):  # a rare binary factor on 90% of its groups' rows, against 100 bursty classes
    rare = (rng.random(4000) < 0.05).astype(np.intp)
    f, y, g = bursty_classes(rng, rare, 100, 10, burst=0.9)
    return np.where(rng.random(g.size) < 0.9, f, rng.random(g.size) < 0.05).astype(np.intp), y, g


def rare_bursty_class(rng, k, size=10):  # 3% of 4,000 groups hold a rare class in bursts; the other classes are iid
    g = np.repeat(np.arange(4000), size)
    burst = (rng.random(4000) < 0.03)[g] & (rng.random(g.size) < 0.9)
    return rng.integers(0, 40, 4000)[g], np.where(burst, k - 1, rng.integers(0, k - 1, g.size)), g


def level_class_bursts(rng, bursty, n_groups=2000):  # a row takes its group's class w.p. bursty[level, class]
    r, k = bursty.shape
    level = rng.integers(0, r, n_groups)
    g = np.repeat(np.arange(n_groups), 10)
    own, row_level = rng.integers(0, k, n_groups)[g], level[g]
    rest = np.cumsum((1 - bursty) / (1 - bursty).sum(1, keepdims=True), 1)  # keeps each level's class mix uniform
    y = (rng.random(g.size)[:, None] > rest[row_level]).sum(1)
    take = rng.random(g.size) < bursty[row_level, own]
    y[take] = own[take]
    return row_level, y, g


def level_split_bursts(rng, r=20, k=20):  # level 0 bursts in classes < k/2, the others in classes >= k/2
    return level_class_bursts(rng, np.where((np.arange(k) < k // 2) == (np.arange(r) == 0)[:, None], 0.9, 0.0))


def block_bursts(rng, r=20, b=2, n_groups=2000):  # each block of b levels bursts in its own b classes (scene context)
    blocks = np.arange(r) // b
    return level_class_bursts(rng, np.where(blocks[:, None] == blocks, 0.9, 0.0), n_groups)


def one_bursty_level(rng, r=10, hold=0.02):  # two classes; the rare one comes in bursts at level 0 only
    level = rng.integers(0, r, 1000)
    g = np.repeat(np.arange(1000), 10)
    at_zero, rate = level[g] == 0, hold * 0.9
    bursts = (rng.random(1000) < hold)[g] & (rng.random(g.size) < 0.9)
    return level[g], np.where(at_zero, bursts, rng.random(g.size) < rate).astype(np.intp), g


def seq_gap(rng, n_seq=60):  # scripts 10/20: class 0 x0.15 at level 2
    level = rng.integers(0, 3, n_seq)
    level[:3] = [0, 1, 2]
    mix = rng.dirichlet(np.ones(K), n_seq)
    mix[level == 2, 0] *= 0.15
    mix /= mix.sum(1, keepdims=True)
    y = np.concatenate([rng.choice(K, 50, p=p) for p in mix])
    g = np.repeat(np.arange(n_seq), 50)
    return level[g], y, g


@pytest.mark.required
def test_cell_variance_matches_permutation_variance():
    # Factor constant within 60 equal-size groups: under shuffling the factor across groups,
    # the mean of diag(V) equals the variance of O - E exactly, so they agree up to Monte Carlo error.
    rng = np.random.default_rng(1)
    n_seq, length, reps = 60, 20, 4000
    g = np.repeat(np.arange(n_seq), length)
    y = mixed_classes(rng, np.full(n_seq, length))
    levels = np.arange(n_seq) % 3
    p_y = np.bincount(y) / y.size
    diag_v, deviation = [], []
    for _ in range(reps):
        f = rng.permutation(levels)[g]
        p_f = np.bincount(f) / f.size
        diag_v.append(np.diag(_sandwich(f * K + y, g, n_seq, p_f, p_y)))
        deviation.append(np.bincount(f * K + y, minlength=3 * K) - f.size * np.outer(p_f, p_y).ravel())
    assert np.allclose(np.mean(diag_v, axis=0), np.var(deviation, axis=0), rtol=0.1)


@pytest.mark.required
@pytest.mark.parametrize("make", [od_rare_heavy, mot_frame_factor])
def test_pooled_cell_variance_matches_its_definition(make):
    # Σ_s (a_si - p_i)² m_s q_j (1 + (m_s - 1) Σ_i' a_si' ρ_i'j), ρ the largest of the class's correlation
    # pooled outside level i', its largest correlation at another single level, level i''s correlation pooled
    # over the other classes, and its largest correlation in another single class, each a clipped moment estimate.
    f, y, g = map(dense, make(np.random.default_rng(11)))
    n_groups, r, k = g.max() + 1, f.max() + 1, y.max() + 1
    m = np.bincount(g).astype(np.float64)
    p_f, p_y = np.bincount(f) / f.size, np.bincount(y) / y.size
    q, pairs = p_y * (1 - p_y), m * (m - 1)
    shares, counts = np.zeros((n_groups, r)), np.zeros((n_groups, k))
    np.add.at(shares, (g, f), 1.0)
    np.add.at(counts, (g, y), 1.0)
    shares /= m[:, None]
    h2 = (counts - m[:, None] * p_y) ** 2
    outside = (((1 - shares).T @ h2) - np.outer((1 - shares).T @ m, q)) / np.outer((1 - shares).T @ pairs, q)
    level_h2, q_other = shares.T @ h2, q.sum() - q
    other = ((level_h2.sum(1, keepdims=True) - level_h2) - np.outer(shares.T @ m, q_other)) / np.outer(
        shares.T @ pairs, q_other
    )
    per_level = np.clip((level_h2 - np.outer(shares.T @ m, q)) / np.outer(shares.T @ pairs, q), 0, 1)
    elsewhere = np.array([np.delete(per_level, i, 0).max(0) for i in range(r)])
    other_class = np.array([np.delete(per_level, j, 1).max(1) for j in range(k)]).T
    rho = np.maximum.reduce([np.clip(outside, 0, 1), elsewhere, np.clip(other, 0, 1), other_class])
    expected = ((shares - p_f) ** 2).T @ (q * (m[:, None] + pairs[:, None] * (shares @ rho)))
    pooled = _pooled_cell_variance(np.zeros((r, k)), _group_counts(f, g, r), _group_counts(y, g, k), m, p_y)
    np.testing.assert_allclose(pooled, expected, rtol=1e-10)


@pytest.mark.required
@pytest.mark.parametrize("make", [od_rare_heavy, axis_group_level, mot_frame_factor])
def test_sandwich_matches_dense_scores(make):
    f, y, g = map(dense, make(np.random.default_rng(2)))
    n, k = f.size, y.max() + 1
    v = _sandwich(f * k + y, g, g.max() + 1, np.bincount(f) / n, np.bincount(y) / n)
    expected = dense_sandwich(f, y, g)
    np.testing.assert_allclose(v, expected, rtol=1e-8, atol=1e-10 * np.abs(expected).max())


@pytest.mark.required
@pytest.mark.parametrize(
    "make",
    [
        reviewer_design,
        od_rare_heavy,
        seq40,
        seq10,
        od_detection_factor,
        mot_frame_factor,
        rare_class_bursty,
        axis_group_level,
    ],
)
def test_null_is_calibrated(make):
    rng = np.random.default_rng(3)
    reps = 300
    tables = []
    for _ in range(reps):
        f, y, g = make(rng)
        tables.append(grouped_table(dense(f), dense(y), g))
    cells = np.concatenate([t.cell_p_values.ravel() for t in tables])
    cells = cells[~np.isnan(cells)]
    assert np.mean([t.testable for t in tables]) >= 0.5
    assert sum(t.p_value < ALPHA for t in tables) / reps <= bound(reps)
    assert np.mean(cells < ALPHA) <= bound(reps)


@pytest.mark.required
@pytest.mark.parametrize(
    ("make", "reps"),
    [
        (lambda rng: bursty_square(rng, 20, 2000), 200),
        (lambda rng: bursty_square(rng, 40, 4000), 60),
        (bursty_size_by_level, 40),
        (bursty_rare_clustered, 200),
        (lambda rng: bursty_two_level_sizes(rng, 0.9), 60),
        (lambda rng: bursty_two_level_sizes(rng, 0.5), 60),
        (lambda rng: rare_bursty_class(rng, 2), 60),
        (lambda rng: rare_bursty_class(rng, 3, size=40), 40),
        (level_split_bursts, 40),
        (block_bursts, 40),
        (lambda rng: block_bursts(rng, r=40, n_groups=4000), 60),
    ],
    ids=[
        "20x20",
        "40x40",
        "size_by_level",
        "rare_clustered",
        "two_level_sizes",
        "two_level_sizes_half",
        "two_classes",
        "three_classes",
        "level_split",
        "blocks",
        "blocks_40x40",
    ],
)
def test_cells_hold_their_level_with_few_bursts_per_cell(make, reps):
    # A cell fed by a few class bursts has a sandwich variance that shrinks with its count;
    # on the sandwich alone BH flagged cells in 20-100% of these null tables. The 40x40 blocks
    # hold about 2.5 bursts per cell, so a cell's floor rests on its block's three other cells:
    # without the level's largest correlation in another class it flagged 13-24% of tables.
    rng = np.random.default_rng(12)
    flagged, cells = 0, []
    for _ in range(reps):
        f, y, g = make(rng)
        p = grouped_table(dense(f), dense(y), g).cell_p_values.ravel()
        p = p[~np.isnan(p)]
        flagged += bool((benjamini_hochberg(p) <= ALPHA).any())
        cells.append(p)
    assert flagged / reps <= bound(reps)
    assert np.mean(np.concatenate(cells) < ALPHA) <= bound(reps)


@pytest.mark.required
def test_a_rare_class_bursting_at_one_level_stays_at_its_measured_rate():
    # A known limit, not a calibration claim: the contraindication in Parity.md ("Contraindication: a class
    # that bursts at one level only") and the Parity docstrings. With two classes, a rare class that bursts at one level
    # only (about 2 bursty images there) leaves nothing outside the cell to estimate its variance from, so a
    # level that by chance holds no bursts reads as a shortage. Measured 11-17% of null tables flagged against
    # the nominal 5% (0.17 at this seed; 0.185 on the sandwich alone). The floor cannot lower this rate, so
    # the ceiling guards only against a change that makes the regime worse than no floor at all; a fix that
    # brings it within the band should tighten the ceiling to `bound(reps)`.
    rng = np.random.default_rng(12)
    reps, flagged = 200, 0
    for _ in range(reps):
        f, y, g = one_bursty_level(rng)
        p = grouped_table(dense(f), dense(y), g).cell_p_values.ravel()
        flagged += bool((benjamini_hochberg(p[~np.isnan(p)]) <= ALPHA).any())
    assert flagged / reps <= 0.25


@pytest.mark.required
def test_design_effect_below_one_is_reported_and_calibrated():
    rng = np.random.default_rng(4)
    f, y, g = negative_icc(rng)
    t = grouped_table(f, y, g)
    n = t.observed.sum()
    assert t.design_effect < 0.5
    assert n / t.design_effect > n
    assert t.design_effect == pytest.approx(np.diag(dense_sandwich(f, y, g)) @ (1 / t.expected.ravel()))  # df = 1
    corrected = bias_corrected_cramers_v(t.chi_squared, n, 2, 2, design_effect=t.design_effect)
    assert t.score == pytest.approx(float(corrected))
    reps = 400
    rejects = sum(grouped_table(*negative_icc(rng)).p_value < ALPHA for _ in range(reps))
    assert rejects / reps <= bound(reps)


@pytest.mark.required
def test_one_group_per_level_is_untestable():  # script 12
    rng = np.random.default_rng(5)
    t = grouped_table(np.arange(5).repeat(60), rng.integers(0, 4, 300), np.arange(5).repeat(60))
    assert not t.testable
    assert np.isnan(t.p_value)
    assert np.isnan(t.cell_p_values).all()
    assert np.isnan([t.chi_squared, t.design_effect, t.df_num, t.df_den, t.effective_groups, t.score]).all()
    assert t.cell_sparse.all()
    assert t.observed.sum() == 300


@pytest.mark.required
def test_zero_variance_cell_is_nan_not_inf():
    # Class 0 is exactly a quarter of every group, so its cells' scores are zero in every group.
    rng = np.random.default_rng(6)
    n_seq = 30
    y = np.concatenate([np.r_[0, 0, rng.integers(1, K, 6)] for _ in range(n_seq)])
    g = np.repeat(np.arange(n_seq), 8)
    t = grouped_table(g % 3, y, g)
    assert t.testable
    assert np.isnan(t.cell_p_values[:, 0]).all()
    assert np.isfinite(t.cell_p_values[:, 1:]).all()


@pytest.mark.required
def test_cell_degeneracy_tolerance_is_relative_and_inclusive(monkeypatch):
    # A cell is NaN when V_ii <= TOLERANCE * max diag(V): moving TOLERANCE across the
    # smallest-variance cell's ratio flips that cell, and only that cell.
    f, y, g = seq40(np.random.default_rng(10))
    n = f.size
    diag_v = np.diag(_sandwich(f * K + y, g, g.max() + 1, np.bincount(f) / n, np.bincount(y) / n))
    cell = np.argmin(diag_v)
    ratio = diag_v[cell] / diag_v.max()
    assert diag_v.max() > 2  # so an absolute rule would mask nothing at these tolerances
    for factor, masked in ((1 + 1e-6, True), (1 - 1e-6, False)):
        monkeypatch.setattr(_rao_scott, "TOLERANCE", ratio * factor)
        t = grouped_table(f, y, g)
        assert t.testable
        assert np.isnan(t.cell_p_values.flat[cell]) == masked
        assert np.isnan(t.cell_p_values).sum() == int(masked)
    # At TOLERANCE = 1 the largest-variance cell sits exactly on the boundary and is masked.
    monkeypatch.setattr(_rao_scott, "TOLERANCE", 1.0)
    t = grouped_table(f, y, g)
    assert t.testable
    assert np.isnan(t.cell_p_values).all()


@pytest.mark.required
def test_zero_trace_is_untestable(monkeypatch):
    # Every image holds one row of each class and the factor is image-level, so every
    # group's score is zero and tr(GV) = 0, while levels, classes and design df all pass.
    n_images = 20
    g = np.repeat(np.arange(n_images), K)
    sandwich, calls = _rao_scott._sandwich, []

    def spy(*args):
        calls.append(args)
        return sandwich(*args)

    monkeypatch.setattr(_rao_scott, "_sandwich", spy)
    t = grouped_table(g % 2, np.tile(np.arange(K), n_images), g)
    assert calls  # got past every earlier untestable return, so the trace rule decided
    assert not t.testable
    assert np.isnan(t.p_value)
    assert np.isnan(t.cell_p_values).all()


@pytest.mark.required
def test_class_in_one_group_has_untestable_cells():
    # A class seen in one group has no replication behind its cells (d_ij = 0), only sparsity.
    t = grouped_table(*rare_class_bursty(np.random.default_rng(9), holders=1))
    assert t.testable
    assert np.isnan(t.cell_p_values[:, K - 1]).all()
    assert t.cell_sparse[:, K - 1].all()
    assert np.isfinite(t.cell_p_values[:, : K - 1]).all()


@pytest.mark.required
def test_planted_gap_is_the_most_significant_deficit():
    rng = np.random.default_rng(7)
    reps, detected, located = 100, 0, 0
    for _ in range(reps):
        t = grouped_table(*seq_gap(rng))
        detected += t.p_value < ALPHA
        deficit_p = np.where(t.observed < t.expected, t.cell_p_values, np.inf)
        located += np.unravel_index(np.nanargmin(deficit_p), deficit_p.shape) == (2, 0)
    assert detected / reps >= 0.7
    assert located / reps >= 0.9


@pytest.mark.required
def test_axis_constant_within_groups_enters_design_df():
    # The factor varies per row across 40 groups, but only 3 groups hold class 2.
    rng = np.random.default_rng(8)
    n_seq = 40
    cls = np.r_[np.full(3, 2), rng.integers(0, 2, n_seq - 3)]
    g = np.repeat(np.arange(n_seq), 25)
    t = grouped_table(rng.integers(0, 3, g.size), cls[g], g)
    assert t.testable
    assert t.effective_groups <= 3
    assert t.df_den == pytest.approx(t.df_num * (t.effective_groups - 1))


@pytest.mark.required
def test_too_many_cells_is_untestable_without_allocating(caplog):
    f = np.arange(2001).repeat(2)
    y = np.arange(f.size) % 5
    tracemalloc.start()
    with caplog.at_level(logging.WARNING):
        t = grouped_table(f, y, np.arange(f.size) // 2)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert not t.testable
    assert np.isnan(t.p_value)
    assert "MAX_GROUPED_CELLS" in caplog.text
    assert peak < 50e6  # the (rK)^2 covariance would be 800 MB
