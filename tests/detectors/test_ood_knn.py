import numpy as np
import pytest
from sklearn.datasets import load_digits

from dataeval.data import Embeddings
from dataeval.detectors.ood.knn import OOD_KNN

# Embedding dimensions for test
embedding_dim = 64


@pytest.fixture(scope="module")
def reference_embeddings() -> Embeddings:
    """Create reference embeddings for testing."""
    X, y = load_digits(return_X_y=True)
    assert isinstance(X, np.ndarray)
    X = (X.astype(np.float32)) / 255.0

    # Convert to embedding-like data (reduce dimensionality)
    # Simulate embeddings from first 500 samples
    n_ref = 500
    X_ref = X[:n_ref].reshape(n_ref, -1)[:, :embedding_dim]  # Take first 64 features as "embeddings"

    # Use from_array class method - much simpler!
    return Embeddings.from_array(X_ref)


@pytest.fixture(scope="module")
def query_embeddings() -> Embeddings:
    """Create query embeddings for testing (mix of ID and OOD-like)."""
    X, y = load_digits(return_X_y=True)
    assert isinstance(X, np.ndarray)
    X = (X.astype(np.float32)) / 255.0

    # Use different samples as queries (simulate some ID, some OOD)
    n_query = 200
    X_query = X[500 : 500 + n_query].reshape(n_query, -1)[:, :embedding_dim]

    # Add some artificial OOD samples (shifted distribution)
    X_ood = X_query[:50] + 2.0  # Make first 50 samples more "OOD-like"
    X_query[:50] = X_ood

    # Use from_array class method - much simpler!
    return Embeddings.from_array(X_query)


@pytest.mark.optional
@pytest.mark.parametrize("ood_type", ["instance"])  # KNN only supports instance-level
@pytest.mark.parametrize("k", [5, 10])
@pytest.mark.parametrize("distance_metric", ["cosine", "euclidean"])
def test_knn(ood_type, k, distance_metric, reference_embeddings, query_embeddings):
    # OOD_KNN parameters
    threshold_perc = 90.0

    # init OOD_KNN
    knn = OOD_KNN(k=k, distance_metric=distance_metric)

    # fit OOD_KNN, infer threshold and compute reference scores
    knn.fit_embeddings(reference_embeddings, threshold_perc=threshold_perc)

    # Check that reference scores were computed
    assert hasattr(knn, "_ref_score")
    assert knn._ref_score.instance_score is not None

    iscore = knn._ref_score.instance_score
    perc_score = 100 * (iscore < knn._threshold_score()).sum() / iscore.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions on query data
    query_array = query_embeddings.to_numpy()
    od_preds = knn.predict(query_array, ood_type=ood_type)
    scores = knn._threshold_score(ood_type)

    # Check instance-level predictions
    assert od_preds.is_ood.shape == (query_embeddings.to_numpy().shape[0],)
    assert od_preds.is_ood.sum() == (od_preds.instance_score > scores).sum()
    assert od_preds.instance_score.shape == (query_embeddings.to_numpy().shape[0],)

    # KNN doesn't provide feature-level scores
    assert od_preds.feature_score is None


@pytest.mark.required
def test_knn_fit_validation(reference_embeddings):
    """Test validation in fit_embeddings method."""
    knn = OOD_KNN(k=10)

    # Test with empty embeddings - create one with empty array
    empty_embeddings = Embeddings.from_array(np.array([]).reshape(0, embedding_dim))

    with pytest.raises(ValueError):  # Any ValueError is fine
        knn.fit_embeddings(empty_embeddings)

    # Test with k too large for non-empty embeddings
    small_embeddings = Embeddings.from_array(np.random.randn(5, embedding_dim))  # Only 5 embeddings
    knn_large_k = OOD_KNN(k=10)  # k=10 > 5 embeddings
    with pytest.raises(ValueError):  # Any ValueError is fine
        knn_large_k.fit_embeddings(small_embeddings)

    # Test with k equal to number of embeddings (should also fail)
    knn_equal_k = OOD_KNN(k=5)  # k=5 == 5 embeddings
    with pytest.raises(ValueError, match="k \\(5\\) must be less than number of reference embeddings \\(5\\)"):
        knn_equal_k.fit_embeddings(small_embeddings)


@pytest.mark.required
def test_knn_predict_validation(reference_embeddings, query_embeddings):
    """Test validation in predict method."""
    knn = OOD_KNN(k=10)

    # Test prediction before fitting
    query_array = query_embeddings.to_numpy()
    with pytest.raises(RuntimeError, match="Metric needs to be `fit` before method call"):
        knn.predict(query_array)

    # Fit the detector
    knn.fit_embeddings(reference_embeddings)

    # Test prediction after fitting
    od_preds = knn.predict(query_array)
    assert od_preds.is_ood.shape == (query_array.shape[0],)


@pytest.mark.required
def test_knn_score_computation(reference_embeddings, query_embeddings):
    """Test that KNN scores are computed correctly."""
    knn = OOD_KNN(k=5, distance_metric="cosine")
    knn.fit_embeddings(reference_embeddings)

    # Get scores
    query_array = query_embeddings.to_numpy()
    score_output = knn.score(query_array)

    # Check score properties
    assert score_output.instance_score is not None
    assert score_output.instance_score.shape == (query_array.shape[0],)
    assert np.all(score_output.instance_score >= 0)  # Distances should be non-negative
    assert score_output.feature_score is None  # KNN doesn't provide feature scores


@pytest.mark.required
def test_knn_different_distance_metrics(reference_embeddings, query_embeddings):
    """Test that different distance metrics produce different results."""
    query_array = query_embeddings.to_numpy()

    # Test cosine distance
    knn_cosine = OOD_KNN(k=10, distance_metric="cosine")
    knn_cosine.fit_embeddings(reference_embeddings)
    scores_cosine = knn_cosine.score(query_array)

    # Test euclidean distance
    knn_euclidean = OOD_KNN(k=10, distance_metric="euclidean")
    knn_euclidean.fit_embeddings(reference_embeddings)
    scores_euclidean = knn_euclidean.score(query_array)

    # Scores should be different (unless by extreme coincidence)
    assert not np.allclose(scores_cosine.instance_score, scores_euclidean.instance_score)
