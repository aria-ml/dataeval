"""Tests for Feasibility's inputs: typed requirements and allowances, and the arguments that no longer exist."""

import dataclasses
import re
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

    def test_detection_ap_builds_without_a_functional(self):
        spec = TargetSpec(task="OD", metric="ap", threshold=0.5, iou=0.5, max_detections=20)
        assert spec.ap_functional == "population_ap_integral"


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


@pytest.mark.required
class TestMaxObjects:
    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (True, "max_objects must be an integer of at least 1, got True"),
            (1.0, "max_objects must be an integer of at least 1, got 1.0"),
            ("3", "max_objects must be an integer of at least 1, got '3'"),
            (0, "max_objects must be an integer of at least 1, got 0"),
            (-2, "max_objects must be an integer of at least 1, got -2"),
            ({0: 18}, "max_objects keys are class names (TargetSpec.class_name), e.g. {'0': 18}; got key 0"),
            ({"cat": 0}, "max_objects['cat'] must be an integer of at least 1, got 0"),
            ({"cat": True}, "max_objects['cat'] must be an integer of at least 1, got True"),
            ({"cat": 2.0}, "max_objects['cat'] must be an integer of at least 1, got 2.0"),
        ],
    )
    def test_max_objects_values(self, value, message):
        with pytest.raises(ValueError, match=re.escape(message)):
            Feasibility(max_objects=value)

    def test_integers_are_stored_as_python_ints(self):
        assert Feasibility().max_objects is None
        stored = Feasibility(max_objects=np.int64(3)).max_objects  # pyright: ignore[reportArgumentType]
        assert stored == 3
        assert type(stored) is int
        by_class = Feasibility(max_objects={"cat": np.int64(3), "dog": 5}).max_objects  # pyright: ignore[reportArgumentType]
        assert isinstance(by_class, dict)
        assert by_class == {"cat": 3, "dog": 5}
        assert all(type(v) is int for v in by_class.values())

    def test_a_mapping_is_copied(self):
        given = {"cat": 3}
        feasibility = Feasibility(max_objects=given)
        given["cat"] = 99
        assert feasibility.max_objects == {"cat": 3}

    def test_the_config_form_is_checked_the_same_way(self):
        assert Feasibility(config=Feasibility.Config(max_objects={"cat": 3})).max_objects == {"cat": 3}
        with pytest.raises(ValueError, match="keys are class names"):
            Feasibility.Config(max_objects={0: 18})  # pyright: ignore[reportArgumentType]

    def test_state_records_max_objects(self):
        dataset, x = _data()
        state = Feasibility().evaluate(dataset, target=_target(), embeddings=x).meta().state
        assert state["max_objects"] == "NoneType"
        assert state["max_sequence_length"] == "NoneType"
