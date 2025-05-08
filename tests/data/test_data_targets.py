from __future__ import annotations

import numpy as np
import pytest

from dataeval.data._targets import Targets


def make_target(targets: int | list[int], classes: int = 10):
    """
    Make an IC or OD target

    Parameters
    ----------
    targets : int | list[int]
        - `int` for number of classifications
        - `list[int]` for number of detections per image
    classes : int, default 10
        The number of classes
    """
    if isinstance(targets, int):
        return Targets(
            np.random.randint(0, classes, (targets,)),
            np.random.random((targets, classes)).astype(np.float32),
            None,
            None,
        )
    else:
        labels, scores, boxes, source = [], [], [], []
        for i, target in enumerate(targets):
            labels.extend(np.random.randint(0, classes, (target,)).tolist())
            boxes.extend(np.random.random((target, 4)).tolist())
            scores.extend(np.random.random((target,)).tolist())
            source.extend([i] * target)
        return Targets(np.asarray(labels), np.asarray(scores), np.asarray(boxes), np.asarray(source))


class TestTargets:
    def test_targets_ic_post_init(self):
        make_target(5)

    def test_targets_od_post_init(self):
        make_target([1, 1, 1, 1, 1])

    def test_targets_post_init_raise_boxes_no_source(self):
        with pytest.raises(ValueError):
            Targets(np.array([]), np.array([]), np.array([]), None)

    def test_targets_post_init_raise_source_no_boxes(self):
        with pytest.raises(ValueError):
            Targets(np.array([]), np.array([]), None, np.array([]))

    def test_targets_post_init_raise_invalid_boxes(self):
        with pytest.raises(ValueError):
            Targets(np.array([1]), np.array([1]), np.array([[1]]), np.array([1]))

    @pytest.mark.parametrize("pos, objects", [[0, True], [1, True], [2, True], [3, True], [0, False], [1, False]])
    def test_targets_post_init_raise_mismatched_length(self, pos, objects):
        args = [None if i > 1 and not objects else np.array([1] if i == pos else []) for i in range(4)]
        with pytest.raises(ValueError):
            Targets(*args)  # type: ignore

    @pytest.mark.parametrize("target_params, length", [(0, 0), (1, 1), ([0], 0), ([1], 1)])
    def test_targets_len(self, target_params, length):
        assert len(make_target(target_params)) == length

    def test_targets_ic_at(self):
        targets = Targets(
            np.array([0, 0, 1]),
            np.array([[0, 0], [0, 0], [0, 0]]),
            None,
            None,
        )
        np.array_equal(targets[0].labels, np.array([0]))
        np.array_equal(targets[1].labels, np.array([0]))

    def test_targets_od_at(self):
        targets = Targets(
            np.array([0, 0, 1]),
            np.array([[0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            np.array([0, 1, 1]),
        )
        np.array_equal(targets[0].labels, np.array([0]))
        np.array_equal(targets[1].labels, np.array([0, 1]))

    def test_targets_iter(self):
        targets = Targets(
            np.array([0, 0, 1]),
            np.array([[0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            np.array([0, 1, 1]),
        )
        for target in targets:
            assert isinstance(target, Targets)

    def test_targets_ic_size(self):
        ic_target = make_target(5)
        assert len(ic_target) == 5
        assert ic_target.size == 5

    def test_targets_od_size(self):
        od_target = make_target([2, 1, 2, 2])
        assert len(od_target) == 4
        assert od_target.size == 7
