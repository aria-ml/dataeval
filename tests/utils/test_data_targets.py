import numpy as np
import pytest

from dataeval.utils.data._targets import Targets


class TestTargets:
    def test_targets_ic_post_init(self):
        Targets(np.array([]), np.array([]), None, None)

    def test_targets_od_post_init(self):
        Targets(np.array([]), np.array([]), np.array([]), np.array([]))

    def test_targets_post_init_raise_mixed_boxes_source(self):
        with pytest.raises(ValueError):
            Targets(np.array([]), np.array([]), np.array([]), None)
        with pytest.raises(ValueError):
            Targets(np.array([]), np.array([]), None, np.array([]))

    def test_targets_post_init_raise_invalid_boxes(self):
        with pytest.raises(ValueError):
            Targets(np.array([]), np.array([]), np.array([[1]]), np.array([]))

    @pytest.mark.parametrize("pos, objects", [[0, True], [1, True], [2, True], [3, True], [0, False], [1, False]])
    def test_targets_post_init_raise_mismatched_length(self, pos, objects):
        args = [None if i > 1 and not objects else np.array([1] if i == pos else []) for i in range(4)]
        with pytest.raises(ValueError):
            Targets(*args)  # type: ignore

    def test_targets_len(self):
        assert len(Targets(np.array([]), np.array([]), None, None)) == 0
        assert len(Targets(np.array([]), np.array([]), np.array([]), np.array([]))) == 0
        assert len(Targets(np.array([1]), np.array([1]), None, None)) == 1
        assert len(Targets(np.array([1]), np.array([1]), np.array([[1, 2, 3, 4]]), np.array([1]))) == 1

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
