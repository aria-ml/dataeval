"""Unit tests for dataeval.core.label_parity: a symmetric two-sample homogeneity test."""

import inspect
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.stats import chi2_contingency

from dataeval._experimental import experimental
from dataeval.core import _label_parity as label_parity_module
from dataeval.core import label_parity
from dataeval.core._association import bias_corrected_cramers_v
from dataeval.core._parity import parity
from dataeval.exceptions import DeprecatedWarning, ExperimentalWarning

ALPHA = 0.05
KEYS = {"chi_squared", "p_value", "score", "insufficient", "df_num", "df_den", "test"}


def bound(replicates: int) -> float:
    """The largest rejection rate a calibrated test shows over `replicates`, within three standard errors."""
    return ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / replicates)


def table(x, y) -> np.ndarray:
    """The 2xK contingency table of two label samples over the classes that occur."""
    classes = np.unique(np.concatenate([x, y]))
    return np.array([[np.sum(s == c) for c in classes] for s in (x, y)], dtype=float)


def detections(rng, images, sizes, concentration: ArrayLike = 0.3, classes=4):
    """Detection labels and their image ids: each image draws its labels from its own Dirichlet class mix."""
    counts = sizes(images)
    mix = rng.dirichlet(np.ones(classes) * np.asarray(concentration, dtype=float), images)
    labels = np.concatenate([rng.choice(classes, m, p=p) for m, p in zip(counts, mix, strict=True)])
    return labels, np.repeat(np.arange(images), counts)


@pytest.mark.required
class TestLabelParity:
    def test_same_distribution_is_calibrated_at_unequal_sizes(self):
        """The old goodness-of-fit test rejected 87% of 200-vs-2000 draws of one distribution."""
        rng = np.random.default_rng(0)
        reps = 400
        rejected = sum(
            label_parity(
                rng.choice(4, 200, p=[0.4, 0.3, 0.2, 0.1]),
                rng.choice(4, 2000, p=[0.4, 0.3, 0.2, 0.1]),
                assume_independent=True,
            )["p_value"]
            < ALPHA
            for _ in range(reps)
        )
        assert rejected / reps <= bound(reps)

    def test_shifted_distribution_is_detected(self):
        rng = np.random.default_rng(1)
        result = label_parity(
            rng.choice(3, 300, p=[0.6, 0.3, 0.1]), rng.choice(3, 300, p=[0.2, 0.3, 0.5]), assume_independent=True
        )
        assert result["p_value"] < 1e-6
        assert result["score"] > 0.3

    def test_result_keys_and_types(self):
        result = label_parity([0, 1, 1, 0, 1], [1, 1, 0, 0, 0], assume_independent=True)
        assert set(result) == KEYS
        assert result["test"] == "pearson"
        assert result["df_num"] == 1.0
        assert result["df_den"] == np.inf
        assert isinstance(result["insufficient"], bool)
        assert all(isinstance(result[k], float) for k in ("chi_squared", "p_value", "score", "df_num", "df_den"))

    @pytest.mark.parametrize("grouped", [False, True])
    def test_symmetric(self, grouped):
        rng = np.random.default_rng(2)
        x, gx = detections(rng, 40, lambda n: rng.poisson(2, n) + 1)
        y, gy = detections(rng, 30, lambda n: rng.poisson(4, n) + 1)
        kw: dict[str, Any] = {"expected_groups": gx, "observed_groups": gy} if grouped else {"assume_independent": True}
        swapped: dict[str, Any] = {"expected_groups": gy, "observed_groups": gx} if grouped else kw
        a, b = label_parity(x, y, **kw), label_parity(y, x, **swapped)
        assert a["p_value"] == pytest.approx(b["p_value"])
        assert a["chi_squared"] == pytest.approx(b["chi_squared"])
        assert a["score"] == pytest.approx(b["score"])
        assert a["test"] == b["test"] == ("rao_scott_f" if grouped else "pearson")

    def test_matches_scipy_homogeneity(self):
        """Ungrouped, the statistic and p-value are Pearson's 2xK test without continuity correction."""
        rng = np.random.default_rng(3)
        x, y = rng.choice(5, 150, p=[0.3, 0.25, 0.2, 0.15, 0.1]), rng.choice(5, 90, p=[0.2, 0.2, 0.2, 0.2, 0.2])
        statistic, pvalue, dof, _ = chi2_contingency(table(x, y), correction=False)
        result = label_parity(x, y, assume_independent=True)
        assert result["chi_squared"] == pytest.approx(statistic)
        assert result["p_value"] == pytest.approx(pvalue)
        assert result["df_num"] == dof
        assert result["score"] == pytest.approx(float(bias_corrected_cramers_v(result["chi_squared"], 240, 2, 5)))

    def test_class_in_one_dataset_only_is_finite_and_symmetric(self):
        """The old test gave p=0 with chi-squared inf for a class missing from the reference."""
        x = np.array([0] * 30 + [2] * 20)
        y = np.array([0] * 25 + [2] * 15 + [7] * 10)
        a, b = label_parity(x, y, assume_independent=True), label_parity(y, x, assume_independent=True)
        assert np.isfinite([a["chi_squared"], a["p_value"], a["score"]]).all()
        assert a["p_value"] == pytest.approx(b["p_value"])
        assert a["chi_squared"] == pytest.approx(b["chi_squared"])
        assert a["p_value"] == pytest.approx(chi2_contingency(table(x, y), correction=False)[1])
        assert a["df_num"] == 2.0

    def test_unused_num_classes_is_finite_and_warns(self):
        """`num_classes` above the classes present gave NaN; it is now ignored, with a deprecation."""
        x, y = np.array([0, 1, 2, 3] * 10), np.array([0, 1, 2, 3, 3, 3, 1, 0] * 5)
        with pytest.warns(DeprecatedWarning, match="num_classes.*2.0"):
            result = label_parity(x, y, num_classes=5, assume_independent=True)
        assert np.isfinite([result["chi_squared"], result["p_value"]]).all()
        assert result == label_parity(x, y, assume_independent=True)

    def test_grouped_detections_are_calibrated(self):
        """Same class distribution, 500 vs 100 images with different image-size distributions (script 21)."""
        rng = np.random.default_rng(4)
        reps = 300
        grouped = rows = 0
        for _ in range(reps):
            x, gx = detections(rng, 500, lambda n: rng.poisson(2, n) + 1)
            y, gy = detections(rng, 100, lambda n: np.maximum(1, rng.lognormal(2, 1, n).astype(int)))
            grouped += label_parity(x, y, expected_groups=gx, observed_groups=gy)["p_value"] < ALPHA
            rows += label_parity(x, y, assume_independent=True)["p_value"] < ALPHA
        assert grouped / reps <= bound(reps)
        # Read as independent rows the same draws are rejected most of the time: the groups matter.
        assert rows / reps > 0.5

    def test_group_ids_are_distinct_between_the_datasets(self):
        """Both datasets number their images from 0, which must not merge image 0 of one with image 0 of the other."""
        rng = np.random.default_rng(5)
        x, gx = detections(rng, 30, lambda n: rng.poisson(2, n) + 1)
        y, gy = detections(rng, 20, lambda n: rng.poisson(3, n) + 1)
        shared = label_parity(x, y, expected_groups=gx, observed_groups=gy)
        disjoint = label_parity(x, y, expected_groups=gx + 1000, observed_groups=gy)
        negative = label_parity(x, y, expected_groups=gx, observed_groups=gy - 500)
        assert shared["p_value"] == pytest.approx(disjoint["p_value"])
        assert shared["p_value"] == pytest.approx(negative["p_value"])

    def test_grouped_shift_is_detected(self):
        rng = np.random.default_rng(6)
        x, gx = detections(rng, 60, lambda n: rng.poisson(2, n) + 1, concentration=[6, 4, 1, 1])
        y, gy = detections(rng, 60, lambda n: rng.poisson(2, n) + 1, concentration=[1, 1, 4, 6])
        result = label_parity(x, y, expected_groups=gx, observed_groups=gy)
        assert result["test"] == "rao_scott_f"
        assert result["p_value"] < 0.01

    def test_one_sided_groups_raise(self):
        labels, groups = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="both"):
            label_parity(labels, labels, expected_groups=groups)
        with pytest.raises(ValueError, match="both"):
            label_parity(labels, labels, observed_groups=groups)

    def test_groups_must_match_their_labels(self):
        labels = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="expected_groups"):
            label_parity(labels, labels, expected_groups=[0, 1], observed_groups=[0, 1, 2, 3])
        with pytest.raises(ValueError, match="observed_groups"):
            label_parity(labels, labels, expected_groups=[0, 1, 2, 3], observed_groups=[0, 1, 2])

    def test_groups_and_assume_independent_contradict(self):
        labels, groups = np.array([0, 1, 0, 1]), np.array([0, 1, 2, 3])
        with pytest.raises(ValueError, match="both"):
            label_parity(labels, labels, expected_groups=groups, observed_groups=groups, assume_independent=True)

    def test_declining_independence_without_groups_raises(self):
        with pytest.raises(ValueError, match="expected_groups"):
            label_parity([0, 1], [0, 1], assume_independent=False)

    def test_omitted_groups_warn_until_declared(self):
        x, y = [0, 1, 1, 0, 1], [1, 1, 0, 0, 0]
        with pytest.warns(DeprecatedWarning, match="2.0") as record:
            undeclared = label_parity(x, y)
        assert "assume_independent=True" in str(record[0].message)
        assert record[0].filename == __file__  # points at the caller

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            declared = label_parity(x, y, assume_independent=True)
            grouped = label_parity(x, y, expected_groups=[0, 0, 1, 1, 2], observed_groups=[0, 1, 1, 2, 2])
        assert undeclared == declared
        assert grouped["test"] == "rao_scott_f"

    def test_leaves_the_experimental_warning_of_parity_to_direct_callers(self, monkeypatch):
        """`label_parity` is stable, so it never warns that `parity` is experimental, nor uses up that notice."""
        # `experimental` warns once per process, so a fresh decorated copy makes the first call certain.
        fresh = experimental(inspect.unwrap(parity))
        monkeypatch.setattr(label_parity_module, "parity", fresh)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            label_parity([0, 1, 1, 0, 1], [1, 1, 0, 0, 0], assume_independent=True)
        with pytest.warns(ExperimentalWarning):  # a direct caller still gets the notice
            fresh(np.array([[0], [1]]), np.array([0, 1]), assume_independent=True)

    def test_single_class_is_no_shift_observed(self):
        """With one observed class the test cannot speak to unseen alternatives: no shift observed, not parity."""
        kws: list[dict[str, Any]] = [
            {"assume_independent": True},
            {"expected_groups": [0, 0, 1], "observed_groups": [0, 1, 1, 2]},
        ]
        for kw in kws:
            result = label_parity([3, 3, 3], [3, 3, 3, 3], **kw)
            assert (result["chi_squared"], result["p_value"], result["score"]) == (0.0, 1.0, 0.0)
            assert result["insufficient"] is True
            assert result["df_num"] == 0.0
            assert set(result) == KEYS

    @pytest.mark.parametrize(
        ("expected_groups", "observed_groups"),
        [([0, 0, 0, 0], [0, 0, 0, 0, 0]), ([0, 0, 1, 1], [0, 0, 0, 0, 0])],
        ids=["one image per dataset", "one image in one dataset"],
    )
    def test_grouped_but_untestable_is_nan_not_parity(self, expected_groups, observed_groups):
        """Too few effective groups to test: the statistic, p-value and score are NaN, not a verdict of parity."""
        result = label_parity(
            [0, 1, 1, 0], [1, 1, 0, 0, 0], expected_groups=expected_groups, observed_groups=observed_groups
        )
        assert np.isnan([result["chi_squared"], result["p_value"], result["score"]]).all()
        assert result["test"] == "rao_scott_f"
        assert set(result) == KEYS

    @pytest.mark.parametrize(("x", "y"), [([], [0, 1]), ([0, 1], []), ([], [])])
    def test_empty_input_raises(self, x, y):
        with pytest.raises(ValueError, match="No labels"):
            label_parity(np.array(x, dtype=np.intp), np.array(y, dtype=np.intp), assume_independent=True)

    def test_small_samples_are_flagged_insufficient(self):
        result = label_parity([0, 1], [1, 0], assume_independent=True)
        assert result["insufficient"] is True
        assert label_parity([0, 1] * 50, [0, 1] * 50, assume_independent=True)["insufficient"] is False
