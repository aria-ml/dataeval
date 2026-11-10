"""Tests for Feasibility's inputs: typed requirements and allowances, and the arguments that no longer exist."""

import dataclasses
from typing import Any

import numpy as np
import pytest

from dataeval.performance import Allowances, Feasibility, TargetSpec
from dataeval.performance._feasibility_target import parse_targets
from tests.data.conftest import MockICDataset

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

AB = {0: "a", 1: "b"}


def _data(n: int = 200) -> tuple[Any, Any]:
    y = [i % 2 for i in range(n)]
    x = np.random.default_rng(0).normal(size=(n, 4)) + np.array(y)[:, None]
    return MockICDataset(y, AB), x


def _target() -> list[TargetSpec]:
    return [TargetSpec(metric="accuracy", threshold=0.6)]


@pytest.mark.required
class TestAllowances:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"beta": -0.1},
            {"beta": float("nan")},
            {"beta": float("inf")},
            {"beta": True},
            {"beta": [0.1, 0.2]},  # a per-cell beta names the cells of the internal partition, which are not exposed
            {"beta": 0.1, "gamma": -1.0},
            {"beta": 0.1, "gamma": float("nan")},
        ],
    )
    def test_reject_negative_and_non_finite_values(self, kwargs):
        with pytest.raises(ValueError, match="finite, non-negative"):
            Allowances(**kwargs)

    def test_are_frozen_and_comparable(self):
        allowances = Allowances(beta=0.2, gamma=0.1)
        assert allowances == Allowances(0.2, 0.1)
        assert hash(allowances) == hash(Allowances(0.2, 0.1))
        with pytest.raises(dataclasses.FrozenInstanceError):
            allowances.beta = 0.3  # type: ignore[misc]

    def test_gamma_is_optional_and_zero_beta_is_valid(self):
        assert Allowances(beta=0.0).gamma is None

    def test_evaluate_requires_an_allowances_object(self):
        dataset, x = _data()
        with pytest.raises(ValueError, match="Allowances"):
            Feasibility().evaluate(dataset, target=_target(), embeddings=x, allowances={"beta": 0.01})  # type: ignore[arg-type]


@pytest.mark.required
class TestTargetSpecDefaults:
    def test_task_defaults_to_classification(self):
        assert TargetSpec(metric="accuracy", threshold=0.9).task == "IC"

    def test_ap_defaults_to_the_integral_for_classification(self):
        assert TargetSpec(metric="ap", threshold=0.5).ap_functional == "population_ap_integral"

    def test_a_given_ap_functional_is_kept(self):
        assert TargetSpec(metric="ap", threshold=0.5, ap_functional="population_ap_101").ap_functional == (
            "population_ap_101"
        )

    def test_detection_ap_still_requires_a_functional(self):
        with pytest.raises(ValueError, match="ap_functional"):
            TargetSpec(task="OD", metric="ap", threshold=0.5, iou=0.5)


@pytest.mark.required
class TestTargets:
    @pytest.mark.parametrize("target", [{"accuracy": 0.9}, "accuracy", [("accuracy", 0.9)], None, 0.9])
    def test_only_a_sequence_of_target_specs_is_accepted(self, target):
        with pytest.raises(ValueError, match=r"sequence of TargetSpec, e\.g\. \[TargetSpec\(metric="):
            parse_targets(target, task="IC", class_names=["a", "b"], grouped=False)

    def test_a_single_target_spec_must_be_in_a_sequence(self):
        with pytest.raises(ValueError, match="sequence of TargetSpec"):
            parse_targets(TargetSpec(metric="accuracy", threshold=0.9), task="IC", class_names=None, grouped=False)  # type: ignore[arg-type]

    def test_duplicate_requirements_are_rejected_by_equality(self):
        twins = [TargetSpec(metric="accuracy", threshold=1), TargetSpec(metric="accuracy", threshold=1.0)]
        with pytest.raises(ValueError, match="duplicate"):
            parse_targets(twins, task="IC", class_names=None, grouped=False)


@pytest.mark.required
class TestRemovedArguments:
    @pytest.mark.parametrize("kwargs", [{"panel_size": 3}, {"n_cells": 3}, {"builtin_witnesses": False}])
    def test_constructor_knobs_are_gone(self, kwargs):
        with pytest.raises(TypeError):
            Feasibility(**kwargs)

    @pytest.mark.parametrize("kwargs", [{"provenance": {}}, {"sampling": None}, {"look": 1}])
    def test_evaluate_options_are_gone(self, kwargs):
        dataset, x = _data()
        with pytest.raises(TypeError):
            Feasibility().evaluate(dataset, target=_target(), embeddings=x, **kwargs)
