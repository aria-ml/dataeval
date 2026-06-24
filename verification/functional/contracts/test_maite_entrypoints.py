"""IR-1-H-3 / IR-1-R-2 — advertised MAITE task entrypoints.

DataEval is a MAITE task *provider* (``maite_interop_scope=provider``): its
evaluators are MAITE specialized tasks that accept MAITE AnnotatedDatasets /
MetadataLike. They are advertised under the ``maite.tasks`` entrypoint group in
``pyproject.toml`` (see IR-1-H-3). DataEval ships no concrete MAITE
``maite.protocols.*`` component classes, so only the ``maite.tasks`` group is
advertised and verified here.

This mirrors the program's ``validate_maite_entrypoints.py``, which verifies
``maite.tasks`` entrypoints by importing each target.
"""

from __future__ import annotations

from importlib.metadata import distribution

import pytest

DIST_NAME = "dataeval"
TASKS_PREFIX = "maite.tasks"


def _task_entrypoints():
    return [ep for ep in distribution(DIST_NAME).entry_points if ep.group.startswith(TASKS_PREFIX)]


def test_declares_maite_task_entrypoints() -> None:
    """provider scope requires advertised MAITE entrypoints (IR-1-H-3)."""
    eps = _task_entrypoints()
    assert eps, "no maite.tasks entrypoints advertised in installed metadata (IR-1-H-3)"


@pytest.mark.parametrize("ep", _task_entrypoints(), ids=lambda ep: ep.name)
def test_task_entrypoint_target_importable(ep) -> None:
    """Every advertised maite.tasks target must import (IR-1-H-3 / IR-1-R-2)."""
    obj = ep.load()
    assert obj is not None


def test_declares_maite_model_protocol_entrypoints() -> None:
    from importlib.metadata import distribution

    groups = {ep.group for ep in distribution("dataeval").entry_points}
    assert "maite.protocols.image_classification.Model" in groups
    assert "maite.protocols.object_detection.Model" in groups


def test_model_entrypoint_targets_importable() -> None:
    from importlib.metadata import distribution

    for ep in distribution("dataeval").entry_points:
        if ep.group.startswith("maite.protocols"):
            assert ep.load() is not None
