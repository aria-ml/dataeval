"""A datum's `value_range` is a claim about the pixels, so what it can get wrong is guarded. The key is experimental."""

import logging
import warnings
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

from dataeval import Metadata
from dataeval import _experimental as experimental_module
from dataeval.core import image_stats
from dataeval.data import TorchvisionTransform, View
from dataeval.exceptions import ExperimentalWarning
from dataeval.flags import ImageStats

pytestmark = pytest.mark.required

READ = ImageStats.VISUAL_BRIGHTNESS | ImageStats.PIXEL_ENTROPY | ImageStats.PIXEL_SATURATION


class Boxes:
    def __init__(self, n: int) -> None:
        self.boxes = np.array([[1, 1, 5, 5], [6, 6, 10, 10], [2, 8, 6, 12]], dtype=float)[:n]
        self.labels = np.arange(n)
        self.scores = np.ones(n)


class Items:
    """A dataset with one metadata mapping per item, and `n_boxes` detections per item when asked for them."""

    metadata = {"id": "items"}

    def __init__(self, images: list[np.ndarray], declared: Mapping[str, Any] | None = None, n_boxes: int = 0) -> None:
        self.images, self.declared, self.n_boxes = images, dict(declared or {}), n_boxes

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[np.ndarray, Any, dict[str, Any]]:
        target = Boxes(self.n_boxes) if self.n_boxes else np.array([1.0, 0.0])
        return self.images[i], target, {"id": i, **self.declared}


def u8(n: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (3, 16, 16)).astype(np.uint8) for _ in range(n)]


@pytest.fixture
def fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start the test with no experimental warning given yet, as in a new process."""
    monkeypatch.setattr(experimental_module, "_WARNED", set())


class TestMetadataLeavesTheKeyAlone:
    @pytest.mark.parametrize("n_boxes", [2, 3])
    def test_a_detection_dataset_gets_no_factor_from_it_whatever_the_box_count(self, n_boxes: int):
        metadata = Metadata(cast("Any", Items(u8(), {"value_range": (0, 4095)}, n_boxes=n_boxes)))
        assert "value_range" not in metadata.factor_names
        assert not [name for name in metadata.factor_names if name.startswith("value_range")]

    def test_the_dataset_keeps_what_it_wrote(self):
        metadata = Metadata(cast("Any", Items(u8(), {"value_range": (0, 4095)}, n_boxes=2)))
        assert metadata.raw[0]["value_range"] == (0, 4095)

    def test_other_keys_are_factors_as_before(self):
        metadata = Metadata(cast("Any", Items(u8(), {"value_range": (0, 4095), "camera": 1}, n_boxes=2)))
        assert metadata.factor_names == ["camera"]


class TestARescalingTransformDropsTheClaim:
    def view(self, transform: Any, declared: bool = True) -> View[Any]:
        data = Items(u8(), {"value_range": (0, 255)} if declared else None, n_boxes=1)
        return View(data, [TorchvisionTransform(transform)])  # pyright: ignore[reportArgumentType]

    def test_scaling_to_float_drops_the_declared_range_from_every_datum(self):
        v2 = pytest.importorskip("torchvision.transforms.v2")
        torch = pytest.importorskip("torch")
        view = self.view(v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
        with pytest.warns(UserWarning, match="rescaled"):
            assert "value_range" not in view[0][2]
        assert all("value_range" not in view[i][2] for i in range(len(view)))
        assert view[0][2]["id"] == 0

    def test_the_statistics_then_read_the_float_image_as_if_nothing_were_declared(self):
        v2 = pytest.importorskip("torchvision.transforms.v2")
        torch = pytest.importorskip("torch")
        scale = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = image_stats(self.view(scale), stats=ImageStats.VISUAL_BRIGHTNESS)["stats"]["brightness"]
            undeclared = image_stats(self.view(scale, declared=False), stats=ImageStats.VISUAL_BRIGHTNESS)
        np.testing.assert_array_equal(got, undeclared["stats"]["brightness"])

    def test_a_transform_that_keeps_the_values_keeps_the_claim(self):
        v2 = pytest.importorskip("torchvision.transforms.v2")
        view = self.view(v2.RandomHorizontalFlip(p=1.0))
        assert view[0][2]["value_range"] == (0, 255)


class TestAContradictedClaimIsSaid:
    MESSAGE = "outside the declared value range"

    def said(self, caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if self.MESSAGE in r.getMessage()]

    def run(self, data: Any, caplog: pytest.LogCaptureFixture, **kw: Any) -> None:
        with caplog.at_level(logging.WARNING, logger="dataeval"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image_stats(data, stats=kw.pop("stats", ImageStats.VISUAL_BRIGHTNESS), **kw)

    def test_a_datum_range_the_pixels_leave_is_said_for_the_images_that_do(self, caplog: pytest.LogCaptureFixture):
        full = [np.full((3, 8, 8), 60000, dtype=np.uint16), np.full((3, 8, 8), 3000, dtype=np.uint16)]
        self.run(Items(full, {"value_range": (0, 4095)}), caplog)
        (message,) = self.said(caplog)
        assert message.endswith("indices: 0")

    def test_a_call_range_the_pixels_leave_is_said_too(self, caplog: pytest.LogCaptureFixture):
        self.run([np.full((3, 8, 8), 60000, dtype=np.uint16)], caplog, value_range=(0, 4095))
        assert len(self.said(caplog)) == 1

    def test_a_range_the_pixels_stay_inside_is_not_said(self, caplog: pytest.LogCaptureFixture):
        self.run(Items([np.full((3, 8, 8), 3000, dtype=np.uint16)], {"value_range": (0, 4095)}), caplog)
        assert not self.said(caplog)

    def test_nothing_is_said_where_nothing_was_declared(self, caplog: pytest.LogCaptureFixture):
        self.run([np.full((3, 8, 8), 60000, dtype=np.uint16)], caplog)
        assert not self.said(caplog)

    def test_nothing_is_said_when_no_requested_statistic_reads_a_range(self, caplog: pytest.LogCaptureFixture):
        data = Items([np.full((3, 8, 8), 60000, dtype=np.uint16)], {"value_range": (0, 4095)})
        self.run(data, caplog, stats=ImageStats.HASH)
        assert not self.said(caplog)

    def test_nan_and_an_all_nan_image_are_not_a_contradiction(self, caplog: pytest.LogCaptureFixture):
        nans = [np.full((3, 8, 8), np.nan, dtype=np.float32)]
        self.run(nans, caplog, value_range=(0, 1))
        assert not self.said(caplog)


class TestTheKeyIsExperimental:
    def test_the_first_declaration_that_is_read_warns_once_and_names_the_key(self, fresh_process: None):
        data = Items(u8(), {"value_range": (0, 255)})
        with pytest.warns(ExperimentalWarning, match=r"value_range.*experimental") as first:
            image_stats(data, stats=READ)
        assert len(first) == 1
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExperimentalWarning)
            image_stats(data, stats=READ)

    def test_a_dataset_without_the_key_does_not_warn(self, fresh_process: None):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExperimentalWarning)
            image_stats(Items(u8()), stats=READ)

    def test_the_call_keyword_does_not_warn(self, fresh_process: None):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExperimentalWarning)
            image_stats(u8(), stats=READ, value_range=(0, 255))

    def test_a_declaration_the_call_overrides_is_never_read_so_it_does_not_warn(self, fresh_process: None):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExperimentalWarning)
            image_stats(Items(u8(), {"value_range": (0, 255)}), stats=READ, value_range=(0, 255))
