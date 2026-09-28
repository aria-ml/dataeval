"""from_labels and from_embeddings are what evaluate calls once it holds labels or embeddings."""

from collections.abc import Mapping

import numpy as np
import pytest

from dataeval import Ontology
from dataeval.exceptions import ShapeMismatchError
from dataeval.scope import Coverage, Prioritize, Representation


class _Labels:
    """Class labels, and optionally their names: the least a LabelsLike needs."""

    def __init__(self, class_labels: np.ndarray, index2label: Mapping[int, str] | None = None) -> None:
        self.class_labels = np.asarray(class_labels, dtype=np.intp)
        if index2label is not None:
            self.index2label = index2label


_EMBEDDINGS = np.random.default_rng(0).random((60, 4))
_LABELS = np.repeat([0, 1, 2], 20)
_NAMES = {0: "cat", 1: "dog", 2: "owl"}
_ONTOLOGY = Ontology.from_hierarchy({"animal": ["cat", "dog", "owl"]})


@pytest.mark.required
class TestRepresentationFromLabels:
    def test_it_agrees_with_evaluate(self):
        labels = _Labels(_LABELS, _NAMES)
        assert (
            Representation(_ONTOLOGY)
            .from_labels(labels)
            .data()
            .equals(Representation(_ONTOLOGY).evaluate(labels).data())
        )

    def test_index2label_names_a_container_that_has_no_names(self):
        named = Representation(_ONTOLOGY).from_labels(_Labels(_LABELS), index2label=_NAMES)
        assert named.data().equals(Representation(_ONTOLOGY).evaluate(_Labels(_LABELS, _NAMES)).data())

    def test_it_records_its_own_name(self):
        output = Representation(_ONTOLOGY).from_labels(_Labels(_LABELS, _NAMES))
        assert output.meta().name.endswith("Representation.from_labels")


@pytest.mark.required
class TestCoverageFromEmbeddings:
    def test_it_agrees_with_evaluate(self):
        labels = _Labels(_LABELS, _NAMES)
        direct = Coverage().from_embeddings(_EMBEDDINGS, labels)
        through_evaluate = Coverage().evaluate(labels, embeddings=_EMBEDDINGS)
        assert direct.data().equals(through_evaluate.data())
        assert np.array_equal(direct.uncovered_indices, through_evaluate.uncovered_indices)

    def test_without_labels_it_runs_as_one_class(self):
        unlabeled = Coverage().from_embeddings(_EMBEDDINGS)
        all_zero = Coverage().evaluate(np.zeros(len(_EMBEDDINGS), dtype=np.intp), embeddings=_EMBEDDINGS)
        assert unlabeled.data().equals(all_zero.data())
        assert unlabeled.data()["class"].to_list() == ["0"]

    def test_a_count_mismatch_raises(self):
        with pytest.raises(ShapeMismatchError, match="60 embeddings for 59 labels"):
            Coverage().from_embeddings(_EMBEDDINGS, _Labels(_LABELS[:-1]))

    def test_it_records_its_own_name(self):
        assert Coverage().from_embeddings(_EMBEDDINGS).meta().name.endswith("Coverage.from_embeddings")


@pytest.mark.required
class TestPrioritizeFromEmbeddings:
    def test_it_agrees_with_evaluate(self):
        direct = Prioritize(k=5).from_embeddings(_EMBEDDINGS)
        through_evaluate = Prioritize(k=5).evaluate(_EMBEDDINGS)
        assert np.array_equal(direct.data(), through_evaluate.data())

    def test_labels_reach_a_class_balanced_ranking(self):
        direct = Prioritize(k=5, policy="class_balanced").from_embeddings(_EMBEDDINGS, _Labels(_LABELS))
        through_evaluate = Prioritize(k=5, policy="class_balanced").evaluate(_EMBEDDINGS, class_labels=_LABELS)
        assert np.array_equal(direct.data(), through_evaluate.data())

    def test_a_count_mismatch_raises(self):
        with pytest.raises(ShapeMismatchError, match="60 embeddings for 59 labels"):
            Prioritize(k=5).from_embeddings(_EMBEDDINGS, _Labels(_LABELS[:-1]))

    def test_it_records_its_own_name(self):
        assert Prioritize(k=5).from_embeddings(_EMBEDDINGS).meta().name.endswith("Prioritize.from_embeddings")
