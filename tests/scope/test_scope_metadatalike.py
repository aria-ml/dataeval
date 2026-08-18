"""Coverage and Representation must accept a generic MetadataLike, not only concrete Metadata.

A user with a custom lightweight metadata container (satisfying the MetadataLike
protocol) should be able to pass it directly, exactly like the bias evaluators allow.
"""

import numpy as np
import pytest

from dataeval import Ontology
from dataeval.protocols import MetadataLike
from dataeval.scope import Coverage, CoverageOutput, Representation, RepresentationOutput
from tests.conftest import MockMetadata


@pytest.fixture
def mock_metadata():
    n = 60
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=n).astype(np.intp)
    md = MockMetadata(
        class_labels=labels,
        factor_data=rng.integers(0, 4, size=(n, 2)).astype(np.int64),
        factor_names=["a", "b"],
        is_binned=[False, False],
        index2label={0: "cat", 1: "dog", 2: "bird"},
    )
    assert isinstance(md, MetadataLike)
    return md


def test_coverage_accepts_generic_metadatalike(mock_metadata):
    embeddings = np.random.default_rng(1).random((60, 8))
    out = Coverage().evaluate(mock_metadata, embeddings=embeddings)
    assert isinstance(out, CoverageOutput)


def test_representation_accepts_generic_metadatalike(mock_metadata):
    ontology = Ontology.from_hierarchy({"root": ["cat", "dog", "bird"]})
    out = Representation(ontology).evaluate(mock_metadata)
    assert isinstance(out, RepresentationOutput)
