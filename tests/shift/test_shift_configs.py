"""Every shift detector's Config is a validated pydantic model, as every other evaluator's is."""

import dataclasses

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from dataeval.shift import (
    DriftDomainClassifier,
    DriftKNeighbors,
    DriftMMD,
    DriftReconstruction,
    DriftUnivariate,
    DriftWasserstein,
    OODDomainClassifier,
    OODKNeighbors,
    OODReconstruction,
)
from dataeval.types import EvaluatorConfig

_ALL = [
    DriftUnivariate,
    DriftMMD,
    DriftKNeighbors,
    DriftWasserstein,
    DriftDomainClassifier,
    DriftReconstruction,
    OODKNeighbors,
    OODDomainClassifier,
    OODReconstruction,
]

# The detectors that can be built without a model.
_BUILDABLE = [c for c in _ALL if c not in (DriftReconstruction, OODReconstruction)]

# One field per detector, and a value of the wrong type for it.
_WRONG_TYPE = {
    DriftUnivariate: ("p_val", "high"),
    DriftMMD: ("n_permutations", "many"),
    DriftKNeighbors: ("k", "ten"),
    DriftWasserstein: ("ratio_threshold", "big"),
    DriftDomainClassifier: ("n_folds", "five"),
    DriftReconstruction: ("epochs", "twenty"),
    OODKNeighbors: ("k", "ten"),
    OODDomainClassifier: ("n_repeats", "five"),
    OODReconstruction: ("epochs", "twenty"),
}


def _name(detector: type) -> str:
    return detector.__name__


@pytest.mark.required
class TestShiftConfigs:
    @pytest.mark.parametrize("detector", _ALL, ids=_name)
    def test_config_is_an_evaluator_config(self, detector):
        assert issubclass(detector.Config, EvaluatorConfig)
        assert not dataclasses.is_dataclass(detector.Config)

    @pytest.mark.parametrize("detector", _ALL, ids=_name)
    def test_config_refuses_a_wrong_type(self, detector):
        field, value = _WRONG_TYPE[detector]
        with pytest.raises(ValidationError, match=field):
            detector.Config(**{field: value})

    @pytest.mark.parametrize("detector", _BUILDABLE, ids=_name)
    def test_constructor_refuses_a_wrong_type(self, detector):
        field, value = _WRONG_TYPE[detector]
        with pytest.raises(ValidationError, match=field):
            detector(**{field: value})

    def test_literal_choice_is_checked(self):
        with pytest.raises(ValidationError, match="method"):
            DriftUnivariate(method="nope")  # type: ignore[arg-type]

    def test_model_copy_updates_a_field(self):
        config = DriftKNeighbors.Config(k=3).model_copy(update={"p_val": 0.01})
        assert (config.k, config.p_val) == (3, 0.01)

    def test_configs_compare_by_value(self):
        assert DriftKNeighbors.Config(k=3) == DriftKNeighbors.Config(k=3)
        assert DriftKNeighbors.Config(k=3) != DriftKNeighbors.Config(k=4)

    def test_object_fields_accept_real_objects(self):
        sigma = np.ones(3)
        config = DriftMMD.Config(sigma=sigma, device=torch.device("cpu"))
        assert config.sigma is sigma
        assert config.device == torch.device("cpu")

    def test_numpy_scalars_are_accepted(self):
        assert DriftKNeighbors(k=np.int64(3)).config.k == 3  # type: ignore[arg-type]


@pytest.mark.required
class TestShiftReprs:
    @pytest.mark.parametrize(
        ("detector", "expected"),
        [
            (
                DriftUnivariate(),
                (
                    "DriftUnivariate(method='ks', p_val=0.05, correction='bonferroni', alternative='two-sided', "
                    "n_features=None, update_strategy=None, extractor=None, fitted=False)"
                ),
            ),
            (
                DriftMMD(),
                (
                    "DriftMMD(p_val=0.05, sigma=None, n_permutations=100, permutation_batch_size='auto', device=None, "
                    "update_strategy=None, extractor=None, fitted=False)"
                ),
            ),
            (
                DriftKNeighbors(k=3),
                (
                    "DriftKNeighbors(k=3, distance_metric='euclidean', p_val=0.05, extractor=None, "
                    "update_strategy=None, fitted=False)"
                ),
            ),
            (
                DriftWasserstein(),
                (
                    "DriftWasserstein(ratio_threshold=1.4, n_features=None, update_strategy=None, extractor=None, "
                    "fitted=False)"
                ),
            ),
            (
                DriftDomainClassifier(),
                "DriftDomainClassifier(n_folds=5, threshold=0.55, extractor=None, update_strategy=None, fitted=False)",
            ),
            (
                OODKNeighbors(),
                "OODKNeighbors(k=10, distance_metric='cosine', threshold_perc=95.0, extractor=None, fitted=False)",
            ),
            (
                OODDomainClassifier(),
                (
                    "OODDomainClassifier(n_folds=5, n_repeats=5, n_std=2.0, threshold_perc=None, hyperparameters=None, "
                    "extractor=None, fitted=False)"
                ),
            ),
        ],
        ids=lambda value: type(value).__name__ if not isinstance(value, str) else "",
    )
    def test_repr_lists_the_config(self, detector, expected):
        assert repr(detector) == expected
