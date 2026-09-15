import json
from pathlib import Path

import pytest

from .video_generation_scripts.build_dataset import build_synthetic_dataset


@pytest.fixture(scope="session")
def video_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Builds synthetic videos once per test session in a temporary folder."""
    target_dir = tmp_path_factory.mktemp("synthetic_videos")
    return build_synthetic_dataset(target_dir)


@pytest.fixture(scope="session")
def manifest(video_dataset_dir: Path) -> dict:
    return json.loads((video_dataset_dir / "manifest.json").read_text())


@pytest.fixture(scope="session")
def video_path(video_dataset_dir: Path, manifest: dict):
    """video_path('clip_001') -> Path to that clip's video file."""

    def _get(clip_id: str) -> Path:
        return video_dataset_dir / "videos" / manifest["clips"][clip_id]["file"]

    return _get


@pytest.fixture(scope="session")
def track_annotations(video_dataset_dir: Path, manifest: dict):
    """track_annotations('clip_001', 'corrupted') -> list[dict] of track records."""

    def _get(clip_id: str, kind: str = "corrupted") -> list:
        info = manifest["clips"][clip_id].get("annotations")
        if info is None:
            raise KeyError(f"{clip_id} has no annotations (derivative clip?)")
        fname = info[f"{kind}_annotations"]
        return json.loads((video_dataset_dir / "annotations" / fname).read_text())

    return _get


@pytest.fixture(scope="session")
def planted_errors_for(manifest: dict):
    """planted_errors_for('clip_006') -> list of annotation-error descriptors."""

    def _get(clip_id: str) -> list:
        return manifest["clips"][clip_id]["annotations"]["planted_errors"]

    return _get


@pytest.fixture(scope="session")
def corruption_type(manifest: dict):
    """corruption_type('clip_006') -> type of file corruption."""

    def _get(clip_id: str) -> list:
        return manifest["clips"][clip_id]["corruption_type"]

    return _get


@pytest.fixture(scope="session")
def duplicate_pairs(manifest: dict):
    """All planted near-duplicate / re-encode / reframed relationships."""
    return manifest["duplicate_relationships"]


@pytest.fixture(scope="session")
def leakage_summary(manifest: dict):
    """Per-split leakage info, including exact containment_ratio."""
    return manifest["leakage_summary"]


@pytest.fixture(scope="session")
def control_clean_clip(manifest: dict):
    """The one clip guaranteed to have zero planted errors (false-positive check)."""
    return manifest["control_clips"]["clean_clip_no_planted_errors"]
