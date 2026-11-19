"""The tasks a dataset can be are declared once, in ``TASK_PROFILES``, and every dispatcher reads them there."""

from typing import Any, get_args

import numpy as np
import pytest

from dataeval.types._task import TASK_PROFILES, detect_task
from dataeval.utils import data as data_utils
from dataeval.utils.data import DatasetKind


class _Detection:
    boxes = np.zeros((2, 4))
    labels = np.zeros(2)
    scores = np.zeros(2)


class _Tracking:
    frame_tracks = ()


class _TrackingWithBoxes(_Tracking, _Detection):
    """Satisfies both target protocols, so only the check order decides it."""


@pytest.mark.required
class TestOneRegistry:
    def test_each_task_names_its_dataset_kind(self):
        """Filter and validate_dataset name a dataset by its kind; the profile is where the two names meet."""
        kinds = {task: profile.kind for task, profile in TASK_PROFILES.items()}
        assert kinds == {"MOT": "multiobject_tracking", "OD": "object_detection", "IC": "classification"}
        assert set(kinds.values()) <= set(get_args(DatasetKind))

    def test_validate_dataset_runs_the_profiles_own_checks(self):
        """A second copy of a task's check could drift from the one detect_task runs."""
        for profile in TASK_PROFILES.values():
            assert data_utils._TARGET_CHECKS[profile.kind] is profile.matches

    def test_the_checks_run_most_specific_first(self):
        assert list(TASK_PROFILES) == ["MOT", "OD", "IC"]
        assert list(data_utils._TARGET_CHECKS) == [
            "multiobject_tracking",
            "object_detection",
            "segmentation",
            "classification",
        ]

    @pytest.mark.parametrize(
        ("target", "task", "kind"),
        [
            (np.zeros(3), "IC", "classification"),
            (_Detection(), "OD", "object_detection"),
            (_Tracking(), "MOT", "multiobject_tracking"),
            (_TrackingWithBoxes(), "MOT", "multiobject_tracking"),
        ],
    )
    def test_detect_task_and_validate_dataset_agree(self, target: Any, task: str, kind: str):
        assert detect_task(target) == task
        assert data_utils.validate_dataset([(np.zeros((3, 4, 4)), target, {})], expected="any_target") == kind

    def test_a_target_no_task_recognizes(self):
        assert detect_task(object()) is None
