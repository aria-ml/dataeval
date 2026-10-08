"""Verify that the Sufficiency analysis class is importable and configurable.

Maps to meta repo test cases:
  - TC-8.1: Data sufficiency analysis
"""

import numpy as np
import pytest


class _CentroidModel:
    """Nearest-centroid classifier: the 'small model' for the end-to-end runs."""

    def __init__(self):
        self.centroids = {}


class _Train:
    def __init__(self):
        self.calls = 0

    def train(self, model, dataset, indices):
        self.calls += 1
        xs = np.stack([dataset[i][0] for i in indices])
        ys = np.array([dataset[i][1] for i in indices])
        model.centroids = {k: xs[ys == k].mean(0) for k in np.unique(ys)}


class _Evaluate:
    def __init__(self):
        self.calls = 0

    def evaluate(self, model, dataset):
        self.calls += 1
        xs = np.stack([dataset[i][0] for i in range(len(dataset))])
        ys = np.array([dataset[i][1] for i in range(len(dataset))])
        classes = sorted(model.centroids)
        distances = np.stack([np.linalg.norm(xs - model.centroids[k], axis=1) for k in classes], axis=1)
        return {"accuracy": float((np.array(classes)[distances.argmin(axis=1)] == ys).mean())}


class _Reset:
    def __init__(self):
        self.calls = 0

    def __call__(self, model):
        self.calls += 1
        return _CentroidModel()


def _noisy_two_class_data(n, rng):
    from verification.helpers import SimpleAnnotatedDataset

    y = rng.integers(0, 2, n)
    x = (rng.standard_normal((n, 4)) * 2.0 + y[:, None]).astype(np.float32)
    return SimpleAnnotatedDataset(x, y)


@pytest.fixture
def sufficiency_run():
    """A seeded 3-run Sufficiency evaluation over a Manual schedule, with call-counting strategies."""
    from dataeval.performance import Sufficiency

    rng = np.random.default_rng(0)
    np.random.seed(0)  # Sufficiency draws its training subsets from numpy's global RNG
    train, evaluate, reset = _Train(), _Evaluate(), _Reset()
    sufficiency = Sufficiency(_CentroidModel(), train, evaluate, reset, runs=3)
    schedule = [10, 40, 100, 400]
    output = sufficiency.evaluate(_noisy_two_class_data(400, rng), _noisy_two_class_data(400, rng), schedule=schedule)
    return output, train, evaluate, reset, schedule


class TestSufficiency:
    """Verify Sufficiency importability, configuration, and end-to-end runs."""

    def test_sufficiency_importable(self):
        from dataeval.performance import Sufficiency  # noqa: F401

    def test_sufficiency_output_importable(self):
        from dataeval.performance import SufficiencyOutput  # noqa: F401

    def test_sufficiency_config_exists(self):
        from dataeval.performance import Sufficiency

        assert hasattr(Sufficiency, "Config")

    def test_sufficiency_config_has_expected_fields(self):
        from dataeval.performance import Sufficiency

        config = Sufficiency.Config()
        assert hasattr(config, "runs")
        assert hasattr(config, "substeps")

    def test_sufficiency_end_to_end_with_both_schedules(self):
        from dataeval.performance import Sufficiency, SufficiencyOutput
        from dataeval.performance.schedules import GeometricSchedule, ManualSchedule

        rng = np.random.default_rng(0)
        np.random.seed(0)
        train, test = _noisy_two_class_data(400, rng), _noisy_two_class_data(400, rng)
        sufficiency = Sufficiency(_CentroidModel(), _Train(), _Evaluate(), _Reset(), runs=3)

        for schedule, expected_steps in (
            (GeometricSchedule(substeps=5), 5),
            (ManualSchedule([10, 40, 100, 400]), 4),
        ):
            output = sufficiency.evaluate(train, test, schedule=schedule)
            assert isinstance(output, SufficiencyOutput)
            assert len(output.steps) == expected_steps
            assert output.steps[-1] == 400
            assert output.measures["accuracy"].shape == (3, expected_steps)  # runs x steps
            assert output.averaged_measures["accuracy"].shape == (expected_steps,)
            np.testing.assert_allclose(output.averaged_measures["accuracy"], output.measures["accuracy"].mean(axis=0))

    def test_sufficiency_output_project(self, sufficiency_run):
        output = sufficiency_run[0]
        projected = output.project([800, 1600])
        assert projected["step"].to_list() == [800, 1600]
        assert "accuracy" in projected.columns
        values = projected["accuracy"].to_numpy()
        assert ((values >= 0) & (values <= 1)).all()  # unit-interval metric
        assert values[1] >= values[0]  # more data does not lower the fitted curve

    def test_sufficiency_output_inv_project(self, sufficiency_run):
        output = sufficiency_run[0]
        required = output.inv_project([0.6, 0.99])
        assert required["target"].to_list() == [0.6, 0.99]
        samples = required["accuracy"].to_list()
        assert 0 < samples[0] < 400  # an already-reached target needs a modest dataset size
        assert samples[1] == -1  # a target above the fitted asymptote is reported unachievable

    def test_sufficiency_calls_custom_strategies(self, sufficiency_run):
        _, train, evaluate, reset, schedule = sufficiency_run
        runs = 3
        assert reset.calls == runs
        assert train.calls == runs * len(schedule)
        assert evaluate.calls == runs * len(schedule)
