"""An evaluator's constructor validates its arguments against its Config, as building the Config directly does."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from dataeval import Ontology
from dataeval.bias import Balance, Diversity, Parity
from dataeval.extractors import FlattenExtractor
from dataeval.performance import Sufficiency
from dataeval.quality import Duplicates, Outliers
from dataeval.scope import Coverage, Prioritize, Representation
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

_ONTOLOGY = Ontology.from_hierarchy({"animal": ["cat", "dog"]})

# Each evaluator, a builder for it, and one argument of the wrong type or an unknown choice.
_CASES = [
    ("Balance", lambda **kw: Balance(**kw), "num_neighbors", "abc"),
    ("Balance.factor_source", lambda **kw: Balance(**kw), "factor_source", "bogus"),
    ("Diversity", lambda **kw: Diversity(**kw), "method", "nope"),
    ("Parity", lambda **kw: Parity(**kw), "score_threshold", "high"),
    ("Duplicates", lambda **kw: Duplicates(**kw), "hash_radius", "wide"),
    ("Outliers", lambda **kw: Outliers(**kw), "cluster_algorithm", "nope"),
    ("Coverage", lambda **kw: Coverage(**kw), "method", "nope"),
    ("Prioritize", lambda **kw: Prioritize(**kw), "method", "nope"),
    ("Representation", lambda **kw: Representation(_ONTOLOGY, **kw), "expected", "lots"),
    ("Sufficiency", lambda **kw: Sufficiency(MagicMock(), **kw), "runs", "many"),
]


@pytest.mark.required
class TestConstructorsValidate:
    @pytest.mark.parametrize(("build", "field", "value"), [c[1:] for c in _CASES], ids=[c[0] for c in _CASES])
    def test_a_wrong_argument_raises(self, build, field, value):
        with pytest.raises(ValidationError, match=field):
            build(**{field: value})

    def test_numpy_scalars_are_accepted(self):
        assert Balance(num_neighbors=np.int64(7)).num_neighbors == 7  # type: ignore[arg-type]

    def test_an_object_argument_arrives_unchanged(self):
        extractor = FlattenExtractor()
        assert Coverage(extractor=extractor).config.extractor is extractor


@pytest.mark.required
class TestConstructorsKeepTheChosenFields:
    def test_arguments_are_the_chosen_fields(self):
        assert Balance(num_neighbors=7).config.model_fields_set == {"num_neighbors"}

    def test_a_config_and_arguments_combine(self):
        config = Balance(config=Balance.Config(label="weather"), num_neighbors=7).config
        assert config.model_fields_set == {"label", "num_neighbors"}
        assert (config.label, config.num_neighbors) == ("weather", 7)

    def test_an_argument_overrides_the_config(self):
        assert Balance(config=Balance.Config(num_neighbors=3), num_neighbors=7).num_neighbors == 7

    def test_defaults_are_not_chosen(self):
        assert Diversity().config.model_fields_set == set()


# Every constraint on a Config field: a value just inside it, and one just outside it.
_BOUNDS = [
    (Balance.Config, "num_neighbors", 1, 0),
    (Balance.Config, "class_imbalance_threshold", 1.0, 1.01),
    (Balance.Config, "class_imbalance_threshold", 0.0, -0.01),
    (Balance.Config, "factor_correlation_threshold", 1.0, 1.01),
    (Balance.Config, "factor_correlation_threshold", 0.0, -0.01),
    (Diversity.Config, "threshold", 1.0, 1.01),
    (Diversity.Config, "threshold", 0.0, -0.01),
    (Parity.Config, "score_threshold", 1.0, 1.01),
    (Parity.Config, "score_threshold", 0.0, -0.01),
    (Parity.Config, "p_value_threshold", 0.99, 1.0),
    (Parity.Config, "p_value_threshold", 0.01, 0.0),
    (Duplicates.Config, "n_clusters", 1, 0),
    (Duplicates.Config, "batch_size", 1, 0),
    (Duplicates.Config, "cluster_sensitivity", 0.01, 0.0),
    (Duplicates.Config, "hash_radius", 0, -1),
    (Duplicates.Config, "redundancy_radius", 0, -1),
    (Duplicates.Config, "max_segment_gap", 0, -1),
    (Duplicates.Config, "segment_offset_tolerance", 0, -1),
    (Duplicates.Config, "verify_alignment", 0, -1),
    (Duplicates.Config, "min_segment_frames", 1, 0),
    (Duplicates.Config, "min_track_frames", 1, 0),
    (Duplicates.Config, "frame_sample", 1, 0),
    (Duplicates.Config, "frame_sample", 0.5, 0.0),
    (Outliers.Config, "n_clusters", 1, 0),
    (Outliers.Config, "batch_size", 1, 0),
    (Coverage.Config, "num_observations", 1, 0),
    (Coverage.Config, "percent", 0.99, 1.0),
    (Coverage.Config, "percent", 0.01, 0.0),
    (Coverage.Config, "min_class_samples", 1, 0),
    (Coverage.Config, "isotropy_min_samples", 1, 0),
    (Coverage.Config, "near_duplicate_factor", 0.01, 0.0),
    (Coverage.Config, "batch_size", 1, 0),
    (Prioritize.Config, "k", 1, 0),
    (Prioritize.Config, "c", 1, 0),
    (Prioritize.Config, "max_cluster_size", 1, 0),
    (Prioritize.Config, "num_bins", 1, 0),
    (Prioritize.Config, "batch_size", 1, 0),
    (Representation.Config, "expected", {"cat": 1.0}, {"cat": 1.01}),
    (Representation.Config, "expected", {"cat": 0.0}, {"cat": -0.01}),
    (DriftUnivariate.Config, "p_val", 0.99, 1.0),
    (DriftUnivariate.Config, "p_val", 0.01, 0.0),
    (DriftUnivariate.Config, "n_features", 1, 0),
    (DriftMMD.Config, "p_val", 0.99, 1.0),
    (DriftMMD.Config, "n_permutations", 1, 0),
    (DriftMMD.Config, "permutation_batch_size", 1, 0),
    (DriftKNeighbors.Config, "k", 1, 0),
    (DriftKNeighbors.Config, "p_val", 0.01, 0.0),
    (DriftWasserstein.Config, "ratio_threshold", 0.01, 0.0),
    (DriftWasserstein.Config, "n_features", 1, 0),
    (DriftDomainClassifier.Config, "n_folds", 2, 1),
    (DriftReconstruction.Config, "p_val", 0.99, 1.0),
    (DriftReconstruction.Config, "epochs", 1, 0),
    (DriftReconstruction.Config, "batch_size", 1, 0),
    (DriftReconstruction.Config, "gmm_weight", 1.0, 1.01),
    (DriftReconstruction.Config, "gmm_weight", 0.0, -0.01),
    (OODKNeighbors.Config, "k", 1, 0),
    (OODKNeighbors.Config, "threshold_perc", 100.0, 100.01),
    (OODKNeighbors.Config, "threshold_perc", 0.0, -0.01),
    (OODDomainClassifier.Config, "n_folds", 2, 1),
    (OODDomainClassifier.Config, "n_repeats", 1, 0),
    (OODDomainClassifier.Config, "n_std", 0.01, 0.0),
    (OODDomainClassifier.Config, "threshold_perc", 100.0, 100.01),
    (OODReconstruction.Config, "epochs", 1, 0),
    (OODReconstruction.Config, "batch_size", 1, 0),
    (OODReconstruction.Config, "threshold_perc", 100.0, 100.01),
    (OODReconstruction.Config, "gmm_weight", 1.0, 1.01),
    (Sufficiency.Config, "runs", 1, 0),
    (Sufficiency.Config, "substeps", 1, 0),
]


def _bound_id(case: tuple) -> str:
    config, field, _, outside = case
    return f"{config.__qualname__.split('.')[0]}.{field}={outside}"


@pytest.mark.required
class TestConfigConstraints:
    @pytest.mark.parametrize(("config", "field", "inside", "outside"), _BOUNDS, ids=[_bound_id(c) for c in _BOUNDS])
    def test_the_boundary_is_enforced(self, config, field, inside, outside):
        assert getattr(config(**{field: inside}), field) == inside
        with pytest.raises(ValidationError, match=field):
            config(**{field: outside})

    def test_a_constraint_reaches_the_constructor(self):
        with pytest.raises(ValidationError, match="class_imbalance_threshold"):
            Balance(class_imbalance_threshold=7)
