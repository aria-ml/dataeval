import itertools
import json
import math

import numpy as np
import pytest

from dataeval.core import label_cooccurrence, label_stats
from dataeval.utils.scipy.stats import chi2_contingency

INDEX2LABEL = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"}


def _reference_bh(p):
    """Textbook step-up adjustment: ``scipy.stats.false_discovery_control`` needs scipy 1.11, we support 1.10."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


def _dense(result):
    """The matrices of a result computed with return_matrix=True, without their Optional."""
    classes = result["classes"]
    matrix = result["cooccurrence_matrix"]
    joint = result["cooccurrence_probability"]
    conditional = result["conditional_probability"]
    assert classes is not None
    assert matrix is not None
    assert joint is not None
    assert conditional is not None
    return classes, matrix, joint, conditional


def _random_dataset(seed: int, n_images: int = 60, n_classes: int = 6, empty_share: float = 0.2):
    rng = np.random.default_rng(seed)
    per_image = []
    for _ in range(n_images):
        if rng.random() < empty_share:
            per_image.append([])
        else:
            k = int(rng.integers(1, 4))
            # duplicates within an image are annotation-level repeats
            per_image.append(rng.integers(0, n_classes, k).tolist() + ([0] if rng.random() < 0.2 else []))
    labels = [c for classes in per_image for c in classes]
    items = [i for i, classes in enumerate(per_image) for _ in classes]
    return labels, items, per_image


def _oracle_v(chi: float, n: int) -> float:
    if n <= 1:
        return 0.0
    phi2 = max(0.0, chi / n - 1 / (n - 1))
    corr = 2 - 1 / (n - 1) - 1
    return math.sqrt(phi2 / corr) if corr > 0 else 0.0


def _oracle(per_image: list[list[int]], classes: list[int]):
    sets = [set(c) for c in per_image]
    labeled = [s for s in sets if s]
    n = len(labeled)
    out = {}
    for a, b in itertools.combinations(classes, 2):
        both = sum(a in s and b in s for s in labeled)
        if both == 0:
            continue
        only_a = sum(a in s and b not in s for s in labeled)
        only_b = sum(b in s and a not in s for s in labeled)
        neither = n - both - only_a - only_b
        table = np.array([[both, only_a], [only_b, neither]])
        try:
            chi, p, _, expected = chi2_contingency(table, correction=False)
        except ValueError:  # a zero margin
            chi, p, expected = 0.0, 1.0, np.full((2, 2), np.nan)
        sign = np.sign(both * neither - only_a * only_b)
        out[(a, b)] = {
            "counts": both,
            "lift": both * n / ((both + only_a) * (both + only_b)),
            "jaccard": both / (both + only_a + only_b),
            "score": sign * _oracle_v(chi, n),
            "p": p,
            "insufficient": bool(not np.all(expected >= 5)),
        }
    return out, n


@pytest.mark.parametrize("seed", range(8))
def test_pair_statistics_match_a_naive_oracle(seed):
    labels, items, per_image = _random_dataset(seed)
    result = label_cooccurrence(labels, items, INDEX2LABEL, image_count=len(per_image))
    expected, _ = _oracle(per_image, list(INDEX2LABEL))
    pairs = result["pairs"]

    keys = list(zip(pairs["class_a"].tolist(), pairs["class_b"].tolist(), strict=True))
    assert keys == sorted(expected)
    for i, key in enumerate(keys):
        ref = expected[key]
        assert pairs["counts"][i] == ref["counts"]
        assert pairs["lift"][i] == pytest.approx(ref["lift"])
        assert pairs["jaccard"][i] == pytest.approx(ref["jaccard"])
        assert pairs["scores"][i] == pytest.approx(ref["score"], abs=1e-12)
        assert pairs["p_values"][i] == pytest.approx(ref["p"], abs=1e-12)
        assert bool(pairs["insufficient"][i]) == ref["insufficient"]


@pytest.mark.parametrize("seed", range(4))
def test_adjusted_p_values_are_bh_over_all_present_class_pairs(seed):
    labels, items, per_image = _random_dataset(seed, n_images=200)
    result = label_cooccurrence(labels, items, INDEX2LABEL, image_count=len(per_image), return_matrix=True)
    pairs = result["pairs"]
    _, matrix, _, _ = _dense(result)
    k = int(np.count_nonzero(np.diag(matrix)))
    assert result["n_tests"] == k * (k - 1) // 2
    padded = np.concatenate([pairs["p_values"], np.ones(result["n_tests"] - len(pairs["p_values"]))])
    np.testing.assert_allclose(pairs["p_values_adjusted"], _reference_bh(padded)[: len(pairs["p_values"])])


def test_matrices_are_consistent_with_counts():
    labels, items, per_image = _random_dataset(3)
    result = label_cooccurrence(labels, items, INDEX2LABEL, image_count=len(per_image), return_matrix=True)
    sets = [set(c) for c in per_image]
    _, matrix, joint, conditional = _dense(result)
    for i, j in itertools.product(range(6), repeat=2):
        assert matrix[i, j] == sum(i in s and j in s for s in sets)
    np.testing.assert_allclose(joint, matrix / len(per_image))
    diag = np.diag(matrix)
    for i in range(6):
        if diag[i]:
            np.testing.assert_allclose(conditional[i], matrix[i] / diag[i])
        else:
            assert not conditional[i].any()


def test_empty_images_change_joint_probability_but_not_association():
    labels, items = [0, 1, 0, 1, 0, 2], [0, 0, 1, 1, 2, 3]
    base = label_cooccurrence(labels, items, INDEX2LABEL, image_count=4, return_matrix=True)
    padded = label_cooccurrence(labels, items, INDEX2LABEL, image_count=40, return_matrix=True)
    _, _, base_joint, base_conditional = _dense(base)
    _, _, padded_joint, padded_conditional = _dense(padded)
    for key in ("counts", "lift", "jaccard", "scores", "p_values", "p_b_given_a"):
        np.testing.assert_array_equal(base["pairs"][key], padded["pairs"][key])
    np.testing.assert_array_equal(base_conditional, padded_conditional)
    assert padded_joint[0, 1] == pytest.approx(base_joint[0, 1] / 10)


def test_score_sign_gives_direction():
    # a and b always together (positive); a and c never together except once (negative)
    per_image = [[0, 1]] * 10 + [[2]] * 10 + [[0, 2]] + [[1]] * 0 + [[3]] * 10
    labels = [c for s in per_image for c in s]
    items = [i for i, s in enumerate(per_image) for _ in s]
    pairs = label_cooccurrence(labels, items, INDEX2LABEL, image_count=len(per_image))["pairs"]
    scores = {(a, b): s for a, b, s in zip(pairs["class_a"], pairs["class_b"], pairs["scores"], strict=True)}
    assert scores[(0, 1)] > 0
    assert scores[(0, 2)] < 0


def test_partner_counts_and_isolated_classes():
    labels = [0, 1, 2, 3, 0]
    items = [0, 0, 0, 1, 2]
    result = label_cooccurrence(labels, items, INDEX2LABEL, image_count=3)
    assert dict(result["partner_counts"]) == {0: 2, 1: 2, 2: 2, 3: 0, 4: 0, 5: 0}
    assert result["isolated_classes"] == [3]  # absent classes 4, 5 are not isolated


@pytest.mark.parametrize("seed", range(3))
def test_stats_input_equals_raw_input(seed):
    labels, items, per_image = _random_dataset(seed)
    stats = label_stats(labels, items, INDEX2LABEL, image_count=len(per_image))
    raw = label_cooccurrence(labels, items, INDEX2LABEL, image_count=len(per_image), return_matrix=True)
    via = label_cooccurrence(stats, return_matrix=True)
    for key in ("cooccurrence_matrix", "cooccurrence_probability", "conditional_probability"):
        np.testing.assert_array_equal(raw[key], via[key])  # type: ignore[literal-required]
    for key in raw["pairs"]:
        np.testing.assert_array_equal(raw["pairs"][key], via["pairs"][key])  # type: ignore[literal-required]


def test_hand_built_stats_may_be_unsorted_and_repeat_classes():
    clean = {"classes_per_image": [[0, 1], [1, 2]], "index2label": {0: "a", 1: "b", 2: "c"}, "image_count": 2}
    messy = {**clean, "classes_per_image": [[1, 0, 1], [2, 1]]}
    a = label_cooccurrence(clean, return_matrix=True)  # type: ignore[arg-type]
    b = label_cooccurrence(messy, return_matrix=True)  # type: ignore[arg-type]
    np.testing.assert_array_equal(a["cooccurrence_matrix"], b["cooccurrence_matrix"])


def test_result_is_json_serialisable_after_tolist():
    labels, items, per_image = _random_dataset(0)
    stats = label_stats(labels, items, INDEX2LABEL, image_count=len(per_image))
    json.dumps({k: v for k, v in stats.items()}, default=lambda o: o.tolist())


class TestValidation:
    def test_stats_with_extra_arguments(self):
        stats = label_stats([0, 1], [0, 0], {0: "a", 1: "b"})
        with pytest.raises(ValueError, match="item_indices, image_count"):
            label_cooccurrence(stats, item_indices=[0], image_count=3)
        with pytest.raises(ValueError, match="index2label"):
            label_cooccurrence(stats, index2label={0: "a"})

    def test_stats_missing_keys(self):
        with pytest.raises(ValueError, match="missing the required key"):
            label_cooccurrence({"classes_per_image": [[0]]})  # type: ignore[arg-type]

    def test_stats_length_mismatch(self):
        bad = {"classes_per_image": [[0]], "index2label": {0: "a"}, "image_count": 2}
        with pytest.raises(ValueError, match="image_count"):
            label_cooccurrence(bad)  # type: ignore[arg-type]

    def test_class_missing_from_index2label(self):
        bad = {"classes_per_image": [[0, 7]], "index2label": {0: "a"}, "image_count": 1}
        with pytest.raises(ValueError, match=r"\[7\]"):
            label_cooccurrence(bad)  # type: ignore[arg-type]


class TestEdgeCases:
    def test_no_labels(self):
        result = label_cooccurrence([], [], {0: "a", 1: "b"}, image_count=3, return_matrix=True)
        _, matrix, _, _ = _dense(result)
        assert matrix.tolist() == [[0, 0], [0, 0]]
        assert result["pairs"]["counts"].size == 0
        assert result["isolated_classes"] == []
        assert result["n_tests"] == 0

    def test_zero_images(self):
        result = label_cooccurrence([], [], {0: "a"}, image_count=0, return_matrix=True)
        _, _, joint, _ = _dense(result)
        assert joint.tolist() == [[0.0]]

    def test_one_to_one_mapping_has_no_pairs(self):
        result = label_cooccurrence([0, 1, 1, 2], index2label={0: "a", 1: "b", 2: "c"})
        assert result["pairs"]["counts"].size == 0
        assert result["isolated_classes"] == [0, 1, 2]

    def test_class_in_every_image_has_zero_score_and_is_flagged(self):
        per_image = [[0, 1], [0], [0, 1], [0]]
        labels = [c for s in per_image for c in s]
        items = [i for i, s in enumerate(per_image) for _ in s]
        pairs = label_cooccurrence(labels, items, {0: "a", 1: "b"})["pairs"]
        assert pairs["scores"].tolist() == [0.0]
        assert pairs["p_values"].tolist() == [1.0]
        assert pairs["insufficient"].tolist() == [True]

    def test_numpy_input(self):
        result = label_cooccurrence(np.array([0, 1, 0]), np.array([0, 0, 1]), {0: "a", 1: "b"}, return_matrix=True)
        _, matrix, _, _ = _dense(result)
        assert int(matrix[0, 1]) == 1

    def test_classes_follow_index2label_order_sorted(self):
        result = label_cooccurrence([5, 2], [0, 0], {5: "x", 2: "y", 9: "z"}, return_matrix=True)
        classes, _, _, _ = _dense(result)
        assert classes == [2, 5, 9]
        assert list(result["partner_counts"]) == [2, 5, 9]
