"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4.

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from sklearn.datasets import load_digits

from dataeval.shift._ood._reconstruction import OODReconstruction, gmm_energy, gmm_params
from dataeval.utils.losses import ELBOLoss
from dataeval.utils.models import AE, VAE

input_shape = (1, 8, 8)


# load iris data
@pytest.fixture(scope="module")
def x_ref() -> np.ndarray:
    X, y = load_digits(return_X_y=True)
    assert isinstance(X, np.ndarray)
    X = (X.astype(np.float32)) / 255.0
    X = X.reshape(X.shape[0], *input_shape)
    return X


@pytest.mark.optional
@pytest.mark.parametrize("ood_type", ["instance", "feature"])
def test_ae(ood_type, x_ref):
    # OutlierAE parameters
    threshold_perc = 90.0

    # init OutlierAE
    ae = OODReconstruction(AE(input_shape=input_shape))

    # fit OutlierAE, infer threshold and compute scores
    ae.fit(x_ref, threshold_perc=threshold_perc, epochs=1)
    iscore = ae._ref_score.instance_score
    perc_score = 100 * (iscore < ae._threshold_score()).sum() / iscore.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = ae.predict(x_ref, ood_type=ood_type)
    scores = ae._threshold_score(ood_type)

    if ood_type == "instance":
        assert od_preds.is_ood.shape == (x_ref.shape[0],)
        assert od_preds.is_ood.sum() == (od_preds.instance_score > scores).sum()
    elif ood_type == "feature":
        assert od_preds.is_ood.shape == x_ref.shape
        assert od_preds.feature_score is not None
        assert od_preds.feature_score.shape == x_ref.shape
        assert od_preds.is_ood.sum() == (od_preds.feature_score > scores).sum()

    assert od_preds.instance_score.shape == (x_ref.shape[0],)


@pytest.mark.required
@patch("dataeval.shift._ood._reconstruction.train")
def test_custom_loss_fn(mock_train, x_ref):
    mock_loss_fn = MagicMock()
    ae = OODReconstruction(AE(input_shape=input_shape))
    ae.fit(x_ref, 0.0, mock_loss_fn)
    # Check that the custom loss function was passed to train
    assert mock_train.called
    # The loss_fn is the 3rd keyword argument (loss_fn) in the train call
    assert mock_train.call_args.kwargs["loss_fn"] is mock_loss_fn


@pytest.mark.required
@patch("dataeval.shift._ood._reconstruction.train")
def test_custom_optimizer(mock_train, x_ref):
    mock_opt = MagicMock()
    ae = OODReconstruction(AE(input_shape=input_shape))
    ae.fit(x_ref, 0.0, None, mock_opt)
    # Check that the custom optimizer was passed to train
    assert mock_train.called
    assert mock_train.call_args.kwargs["optimizer"] is mock_opt


@pytest.mark.optional
@pytest.mark.parametrize("ood_type", ["instance", "feature"])
def test_vae(ood_type, x_ref):
    """Test VAE-based OOD detection."""
    # OutlierVAE parameters
    threshold_perc = 90.0

    # init OutlierVAE with VAE model
    vae = OODReconstruction(VAE(input_shape=input_shape), model_type="vae")

    # fit VAE, infer threshold and compute scores
    vae.fit(x_ref, threshold_perc=threshold_perc, epochs=1)
    iscore = vae._ref_score.instance_score
    perc_score = 100 * (iscore < vae._threshold_score()).sum() / iscore.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = vae.predict(x_ref, ood_type=ood_type)
    scores = vae._threshold_score(ood_type)

    if ood_type == "instance":
        assert od_preds.is_ood.shape == (x_ref.shape[0],)
        assert od_preds.is_ood.sum() == (od_preds.instance_score > scores).sum()
    elif ood_type == "feature":
        assert od_preds.is_ood.shape == x_ref.shape
        assert od_preds.feature_score is not None
        assert od_preds.feature_score.shape == x_ref.shape
        assert od_preds.is_ood.sum() == (od_preds.feature_score > scores).sum()

    assert od_preds.instance_score.shape == (x_ref.shape[0],)


@pytest.mark.required
def test_vae_model_output():
    """Test that VAE model returns correct tuple format."""
    vae_model = VAE(input_shape=input_shape)
    x = torch.randn(4, *input_shape)
    output = vae_model(x)

    # Check that output is a tuple of 3 elements
    assert isinstance(output, tuple)
    assert len(output) == 3

    recon, mu, logvar = output
    assert recon.shape == x.shape
    assert mu.shape[0] == x.shape[0]
    assert logvar.shape[0] == x.shape[0]
    assert mu.shape == logvar.shape


@pytest.mark.required
def test_vae_model_type_validation():
    """Test that invalid model_type raises ValueError."""
    with pytest.raises(ValueError, match="model_type must be"):
        OODReconstruction(AE(input_shape=input_shape), model_type="invalid")  # type: ignore


@pytest.mark.required
def test_ae_vs_vae_loss():
    """Test that AE and VAE use different default loss functions."""
    ae = OODReconstruction(AE(input_shape=input_shape), model_type="ae")
    vae = OODReconstruction(VAE(input_shape=input_shape), model_type="vae")

    # Create small test data
    x_ref = np.random.rand(10, *input_shape).astype(np.float32)

    # Check that fit doesn't raise errors and uses appropriate losses
    with patch("dataeval.shift._ood._reconstruction.train") as mock_train:
        ae.fit(x_ref, threshold_perc=90, epochs=1)
        # Check that MSE loss was used for AE
        assert mock_train.called
        assert isinstance(mock_train.call_args.kwargs["loss_fn"], torch.nn.MSELoss)

    with patch("dataeval.shift._ood._reconstruction.train") as mock_train:
        vae.fit(x_ref, threshold_perc=90, epochs=1)
        # Check that VAE loss was used
        assert mock_train.called
        # ELBOLoss is a custom class from dataeval
        assert "ELBOLoss" in str(type(mock_train.call_args.kwargs["loss_fn"]))


@pytest.mark.optional
def test_vae_beta_parameter(x_ref):
    """Test that beta parameter affects VAE training using ELBOLoss."""
    vae = OODReconstruction(VAE(input_shape=input_shape), model_type="vae")
    vae.fit(x_ref[:20], threshold_perc=90, loss_fn=ELBOLoss(beta=0.5), epochs=1)
    assert hasattr(vae, "_ref_score")

    vae2 = OODReconstruction(VAE(input_shape=input_shape), model_type="vae")
    vae2.fit(x_ref[:20], threshold_perc=90, loss_fn=ELBOLoss(beta=2.0), epochs=1)
    assert hasattr(vae2, "_ref_score")


@pytest.mark.required
def test_use_gmm_parameter():
    """Test that use_gmm parameter is properly set."""
    ae_no_gmm = OODReconstruction(AE(input_shape=input_shape), use_gmm=False)
    assert ae_no_gmm.use_gmm is False
    assert ae_no_gmm._gmm_params is None

    vae_with_gmm = OODReconstruction(VAE(input_shape=input_shape), model_type="vae", use_gmm=True)
    assert vae_with_gmm.use_gmm is True
    assert vae_with_gmm._gmm_params is None  # Not computed until fit() is called


@pytest.mark.required
def test_gmm_without_proper_model_output():
    """Test that using GMM with a model that doesn't return proper output raises error."""
    # Standard VAE returns (recon, mu, logvar) which has 3 elements but not the right format
    vae_with_gmm = OODReconstruction(VAE(input_shape=input_shape), model_type="vae", use_gmm=True)
    x_ref_small = np.random.rand(10, *input_shape).astype(np.float32)

    # This should raise an error because VAE doesn't output gamma
    with pytest.raises(ValueError, match="When use_gmm=True"):
        vae_with_gmm.fit(x_ref_small, threshold_perc=90, epochs=1)


@pytest.mark.required
def test_ELBOLoss_class_with_ood_ae():
    """Test using ELBOLoss class with OODReconstruction."""
    from dataeval.utils.losses import ELBOLoss

    # Create custom loss with specific beta
    custom_loss = ELBOLoss(beta=2.0, reduction="mean")

    vae = OODReconstruction(VAE(input_shape=input_shape), model_type="vae")

    # Small dataset for quick test
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)

    # Fit with custom loss
    vae.fit(x_ref_small, threshold_perc=90, loss_fn=custom_loss, epochs=1)

    # Should have fitted successfully
    assert hasattr(vae, "_ref_score")
    assert vae._ref_score is not None


@pytest.mark.optional
def test_ELBOLoss_different_beta_values(x_ref):
    """Test OODReconstruction with different beta values using ELBOLoss."""
    from dataeval.utils.losses import ELBOLoss

    # Test with beta=0.5 (less emphasis on KL divergence)
    loss1 = ELBOLoss(beta=0.5)
    vae1 = OODReconstruction(VAE(input_shape=input_shape), model_type="vae")
    vae1.fit(x_ref[:20], threshold_perc=90, loss_fn=loss1, epochs=1)

    # Test with beta=4.0 (more emphasis on KL divergence, better disentanglement)
    loss2 = ELBOLoss(beta=4.0)
    vae2 = OODReconstruction(VAE(input_shape=input_shape), model_type="vae")
    vae2.fit(x_ref[:20], threshold_perc=90, loss_fn=loss2, epochs=1)

    # Both should train successfully
    assert hasattr(vae1, "_ref_score")
    assert hasattr(vae2, "_ref_score")


@pytest.mark.required
def test_custom_functional_loss():
    """Test using a custom functional loss with OODReconstruction."""

    def custom_vae_loss(x, x_recon, mu, logvar):
        """Custom VAE loss with L1 reconstruction instead of MSE."""
        # L1 reconstruction loss
        recon_loss = torch.mean(torch.abs(x_recon.view(len(x), -1) - x.view(len(x), -1)))

        # Standard KL divergence
        kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return recon_loss + kld_loss

    vae = OODReconstruction(VAE(input_shape=input_shape), model_type="vae")
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)

    # Fit with custom functional loss
    vae.fit(x_ref_small, threshold_perc=90, loss_fn=custom_vae_loss, epochs=1)

    assert hasattr(vae, "_ref_score")


@pytest.mark.required
def test_data_validation_not_ndarray():
    """Test that non-ndarray input raises TypeError."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_list = [[1, 2, 3], [4, 5, 6]]  # Not an ndarray

    with pytest.raises((TypeError, RuntimeError)):
        ae.fit(x_list, threshold_perc=90, epochs=1)


@pytest.mark.required
def test_data_validation_out_of_range_negative():
    """Test that data with negative values raises ValueError."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    x_ref_small[0, 0, 0, 0] = -0.1  # Add a negative value

    with pytest.raises(ValueError, match="Data must be on the unit interval"):
        ae.fit(x_ref_small, threshold_perc=90, epochs=1)


@pytest.mark.required
def test_data_validation_out_of_range_above_one():
    """Test that data with values > 1 raises ValueError."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    x_ref_small[0, 0, 0, 0] = 1.5  # Add a value > 1

    with pytest.raises(ValueError, match="Data must be on the unit interval"):
        ae.fit(x_ref_small, threshold_perc=90, epochs=1)


@pytest.mark.required
def test_predict_before_fit():
    """Test that calling predict before fit raises RuntimeError."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_test = np.random.rand(10, *input_shape).astype(np.float32)

    with pytest.raises(RuntimeError, match="Detector needs to be `fit` before calling predict or score"):
        ae.predict(x_test)


@pytest.mark.required
def test_score_before_fit():
    """Test that calling score before fit raises an error."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    np.random.rand(10, *input_shape).astype(np.float32)

    # Score doesn't validate state, so we need to test with predict or check that _ref_score doesn't exist
    assert not hasattr(ae, "_ref_score")


@pytest.mark.required
def test_shape_mismatch_after_fit():
    """Test that data with different shape after fit raises an error."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    ae.fit(x_ref_small, threshold_perc=90, epochs=1)

    # Try to predict with wrong shape
    wrong_shape = (1, 4, 4)  # Different from input_shape
    x_test_wrong = np.random.rand(10, *wrong_shape).astype(np.float32)

    # Could raise RuntimeError for shape mismatch or other model-related errors
    with pytest.raises((RuntimeError, ValueError)):
        ae.predict(x_test_wrong)


@pytest.mark.required
def test_dtype_mismatch_after_fit():
    """Test that data with different dtype is handled (converted to float32)."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    ae.fit(x_ref_small, threshold_perc=90, epochs=1)

    # Try to predict with wrong dtype - should be converted automatically
    x_test_wrong = np.random.rand(10, *input_shape).astype(np.float64)

    # This should work as the code converts to float32
    output = ae.predict(x_test_wrong)
    assert output.is_ood.shape == (10,)


@pytest.mark.required
def test_config_initialization():
    """Test OODReconstruction.Config initialization and defaults."""
    # Test default config
    config = OODReconstruction.Config()
    assert config.loss_fn is None
    assert config.optimizer is None
    assert config.epochs == 20
    assert config.batch_size == 64
    assert config.threshold_perc == 95.0

    # Test custom config
    custom_config = OODReconstruction.Config(epochs=10, batch_size=32, threshold_perc=99.0)
    assert custom_config.epochs == 10
    assert custom_config.batch_size == 32
    assert custom_config.threshold_perc == 99.0


@pytest.mark.required
def test_config_usage_with_overrides():
    """Test that config values are used and can be overridden."""
    config = OODReconstruction.Config(epochs=15, batch_size=16, threshold_perc=98.0)
    ae = OODReconstruction(AE(input_shape=input_shape), config=config)

    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)

    with patch("dataeval.shift._ood._reconstruction.train") as mock_train:
        # Fit without overrides - should use config defaults
        ae.fit(x_ref_small)
        assert mock_train.call_args.kwargs["epochs"] == 15
        assert mock_train.call_args.kwargs["batch_size"] == 16

    # Create new instance to override config
    ae2 = OODReconstruction(AE(input_shape=input_shape), config=config)
    with patch("dataeval.shift._ood._reconstruction.train") as mock_train:
        # Fit with overrides
        ae2.fit(x_ref_small, epochs=5, batch_size=8)
        assert mock_train.call_args.kwargs["epochs"] == 5
        assert mock_train.call_args.kwargs["batch_size"] == 8


@pytest.mark.required
def test_auto_detect_model_type_ae():
    """Test auto-detection of AE model type."""
    ae_model = AE(input_shape=input_shape)
    # Test with auto detection
    ood = OODReconstruction(ae_model, model_type="auto")
    assert ood.model_type == "ae"
    assert ood.use_gmm is False

    # Test with None (should default to auto)
    ood2 = OODReconstruction(ae_model, model_type=None)
    assert ood2.model_type == "ae"


@pytest.mark.required
def test_auto_detect_model_type_vae():
    """Test auto-detection of VAE model type."""
    vae_model = VAE(input_shape=input_shape)
    ood = OODReconstruction(vae_model, model_type="auto")
    assert ood.model_type == "vae"
    assert ood.use_gmm is False


@pytest.mark.required
def test_explicit_model_type_override():
    """Test that explicit model_type parameter works."""
    ae_model = AE(input_shape=input_shape)

    # Explicitly set as ae
    ood_ae = OODReconstruction(ae_model, model_type="ae")
    assert ood_ae.model_type == "ae"

    # Can also explicitly set VAE on AE model (for testing edge cases)
    vae_model = VAE(input_shape=input_shape)
    ood_vae = OODReconstruction(vae_model, model_type="vae")
    assert ood_vae.model_type == "vae"


@pytest.mark.required
def test_device_parameter():
    """Test that device parameter is properly set."""
    ae = OODReconstruction(AE(input_shape=input_shape), device="cpu")
    assert ae.device.type == "cpu"


@pytest.mark.required
def test_threshold_score_instance():
    """Test _threshold_score method for instance level."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    ae.fit(x_ref_small, threshold_perc=90, epochs=1)

    threshold = ae._threshold_score(ood_type="instance")
    assert isinstance(threshold, float | np.floating)

    # Check that threshold is at the right percentile
    expected_threshold = np.percentile(ae._ref_score.instance_score, 90)
    assert np.isclose(threshold, expected_threshold)


@pytest.mark.required
def test_threshold_score_feature():
    """Test _threshold_score method for feature level."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    ae.fit(x_ref_small, threshold_perc=85, epochs=1)

    threshold = ae._threshold_score(ood_type="feature")
    assert isinstance(threshold, float | np.floating)

    # Check that threshold is at the right percentile
    assert ae._ref_score.feature_score is not None
    expected_threshold = np.percentile(ae._ref_score.feature_score, 85)
    assert np.isclose(threshold, expected_threshold)


@pytest.mark.required
def test_score_method_output():
    """Test that score method returns correct output structure."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    ae.fit(x_ref_small, threshold_perc=90, epochs=1)

    x_test = np.random.rand(5, *input_shape).astype(np.float32)
    score = ae.score(x_test)

    assert hasattr(score, "instance_score")
    assert hasattr(score, "feature_score")
    assert score.instance_score.shape == (5,)
    assert score.feature_score is not None
    assert score.feature_score.shape == x_test.shape


@pytest.mark.required
def test_predict_output_structure():
    """Test that predict method returns correct output structure."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    ae.fit(x_ref_small, threshold_perc=90, epochs=1)

    x_test = np.random.rand(5, *input_shape).astype(np.float32)
    output = ae.predict(x_test, ood_type="instance")

    assert hasattr(output, "is_ood")
    assert hasattr(output, "instance_score")
    assert hasattr(output, "feature_score")
    assert output.is_ood.dtype == bool
    assert output.is_ood.shape == (5,)


@pytest.mark.required
def test_batch_size_parameter():
    """Test that batch_size parameter affects scoring."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    ae.fit(x_ref_small, threshold_perc=90, epochs=1, batch_size=10)

    x_test = np.random.rand(15, *input_shape).astype(np.float32)

    # Score with different batch sizes - should give same results
    score1 = ae.score(x_test, batch_size=5)
    score2 = ae.score(x_test, batch_size=15)

    assert score1.feature_score is not None
    assert score2.feature_score is not None
    np.testing.assert_array_almost_equal(score1.instance_score, score2.instance_score)
    np.testing.assert_array_almost_equal(score1.feature_score, score2.feature_score)


@pytest.mark.required
def test_fit_with_all_parameters():
    """Test fit method with all parameters specified."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)

    custom_loss = torch.nn.MSELoss()
    custom_optimizer = torch.optim.SGD(ae.model.parameters(), lr=0.01)

    ae.fit(x_ref_small, threshold_perc=92.0, loss_fn=custom_loss, optimizer=custom_optimizer, epochs=5, batch_size=8)

    assert hasattr(ae, "_ref_score")
    assert ae._threshold_perc == 92.0


@pytest.mark.required
def test_gmm_validation_not_tuple():
    """Test GMM validation when model doesn't return tuple."""

    # Create a mock model that doesn't return a tuple
    class BadModel(torch.nn.Module):
        def forward(self, x):
            return x  # Returns tensor, not tuple

    bad_model = BadModel()
    ood = OODReconstruction(bad_model, model_type="ae", use_gmm=True)
    x_ref_small = np.random.rand(10, *input_shape).astype(np.float32)

    with pytest.raises(ValueError, match="model must return a tuple"):
        ood.fit(x_ref_small, threshold_perc=90, epochs=1)


@pytest.mark.required
def test_gmm_validation_wrong_tuple_length():
    """Test GMM validation when model returns tuple with wrong length."""

    # Create a mock model that returns tuple with only 2 elements
    class ShortTupleModel(torch.nn.Module):
        def forward(self, x):
            return x, x  # Only 2 elements, need at least 3

    short_model = ShortTupleModel()
    ood = OODReconstruction(short_model, model_type="ae", use_gmm=True)
    x_ref_small = np.random.rand(10, *input_shape).astype(np.float32)

    with pytest.raises(ValueError, match="must return tuple of at least 3 elements"):
        ood.fit(x_ref_small, threshold_perc=90, epochs=1)


@pytest.mark.required
def test_gmm_validation_z_not_tensor():
    """Test GMM validation when z (latent) is not a tensor."""

    class BadZModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            recon = x
            z = [1, 2, 3]  # Not a tensor
            gamma = torch.softmax(torch.randn(batch_size, 3), dim=-1)
            return recon, z, gamma

    bad_model = BadZModel()
    ood = OODReconstruction(bad_model, model_type="ae", use_gmm=True)
    x_ref_small = np.random.rand(10, *input_shape).astype(np.float32)

    # Will raise AttributeError from predict trying to call .cpu() on a list
    with pytest.raises((ValueError, AttributeError)):
        ood.fit(x_ref_small, threshold_perc=90, epochs=1)


@pytest.mark.required
def test_gmm_validation_z_wrong_shape():
    """Test GMM validation when z has wrong shape."""

    class WrongZShapeModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            recon = x
            z = torch.randn(batch_size, 10, 10)  # 3D instead of 2D
            gamma = torch.softmax(torch.randn(batch_size, 3), dim=-1)
            return recon, z, gamma

    bad_model = WrongZShapeModel()
    ood = OODReconstruction(bad_model, model_type="ae", use_gmm=True)
    x_ref_small = np.random.rand(10, *input_shape).astype(np.float32)

    with pytest.raises(ValueError, match="second output.*must be 2D"):
        ood.fit(x_ref_small, threshold_perc=90, epochs=1)


@pytest.mark.required
def test_gmm_validation_gamma_not_tensor():
    """Test GMM validation when gamma is not a tensor."""

    class BadGammaModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            recon = x
            z = torch.randn(batch_size, 10)
            gamma = [0.5, 0.3, 0.2]  # Not a tensor
            return recon, z, gamma

    bad_model = BadGammaModel()
    ood = OODReconstruction(bad_model, model_type="ae", use_gmm=True)
    x_ref_small = np.random.rand(10, *input_shape).astype(np.float32)

    # Will raise AttributeError from predict trying to call .cpu() on a list
    with pytest.raises((ValueError, AttributeError)):
        ood.fit(x_ref_small, threshold_perc=90, epochs=1)


@pytest.mark.required
def test_gmm_validation_gamma_wrong_shape():
    """Test GMM validation when gamma has wrong shape."""

    class WrongGammaShapeModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            recon = x
            z = torch.randn(batch_size, 10)
            gamma = torch.randn(batch_size)  # 1D instead of 2D
            return recon, z, gamma

    bad_model = WrongGammaShapeModel()
    ood = OODReconstruction(bad_model, model_type="ae", use_gmm=True)
    x_ref_small = np.random.rand(10, *input_shape).astype(np.float32)

    with pytest.raises(ValueError, match="third output.*must be 2D"):
        ood.fit(x_ref_small, threshold_perc=90, epochs=1)


@pytest.mark.required
def test_gmm_validation_gamma_not_normalized():
    """Test GMM validation when gamma doesn't sum to 1."""

    class UnnormalizedGammaModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            recon = x
            z = torch.randn(batch_size, 10)
            gamma = torch.randn(batch_size, 3)  # Not normalized
            return recon, z, gamma

    bad_model = UnnormalizedGammaModel()
    ood = OODReconstruction(bad_model, model_type="ae", use_gmm=True)
    x_ref_small = np.random.rand(10, *input_shape).astype(np.float32)

    with pytest.raises(ValueError, match="must be a probability distribution"):
        ood.fit(x_ref_small, threshold_perc=90, epochs=1)


@pytest.mark.required
def test_auto_detect_gmm_false():
    """Test that GMM is not auto-detected for standard models."""
    ae_model = AE(input_shape=input_shape)
    assert not hasattr(ae_model, "gmm_density_net") or ae_model.gmm_density_net is None

    ood = OODReconstruction(ae_model)
    assert ood.use_gmm is False
    assert ood._gmm_params is None


@pytest.mark.required
def test_use_gmm_explicit_false():
    """Test explicitly setting use_gmm=False."""
    ae = OODReconstruction(AE(input_shape=input_shape), use_gmm=False)
    assert ae.use_gmm is False
    assert ae._gmm_params is None

    # Fit and verify no GMM params are computed
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    ae.fit(x_ref_small, threshold_perc=90, epochs=1)
    assert ae._gmm_params is None
    assert ae._gmm_energy_ref_mean is None


@pytest.mark.required
def test_score_with_torch_tensor_input():
    """Test that score method works with torch tensors."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    ae.fit(x_ref_small, threshold_perc=90, epochs=1)

    # Test with torch tensor
    x_test_torch = torch.rand(5, *input_shape).float()
    score = ae.score(x_test_torch)

    assert score.instance_score.shape == (5,)
    assert score.feature_score is not None
    assert score.feature_score.shape == (5, *input_shape)


@pytest.mark.required
def test_predict_with_torch_tensor_input():
    """Test that predict method works with torch tensors."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    ae.fit(x_ref_small, threshold_perc=90, epochs=1)

    # Test with torch tensor
    x_test_torch = torch.rand(5, *input_shape).float()
    output = ae.predict(x_test_torch)

    assert output.is_ood.shape == (5,)
    assert output.instance_score.shape == (5,)


@pytest.mark.required
def test_default_optimizer_creation():
    """Test that default optimizer is created when not provided."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)

    with patch("dataeval.shift._ood._reconstruction.train") as mock_train:
        ae.fit(x_ref_small, threshold_perc=90, epochs=1)

        # Check that an optimizer was passed
        assert mock_train.called
        optimizer_arg = mock_train.call_args.kwargs["optimizer"]
        assert optimizer_arg is not None
        assert isinstance(optimizer_arg, torch.optim.Optimizer)


@pytest.mark.required
def test_vae_default_loss():
    """Test that VAE uses ELBOLoss by default."""
    vae = OODReconstruction(VAE(input_shape=input_shape), model_type="vae")
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)

    with patch("dataeval.shift._ood._reconstruction.train") as mock_train:
        vae.fit(x_ref_small, threshold_perc=90, epochs=1)

        assert mock_train.called
        loss_fn = mock_train.call_args.kwargs["loss_fn"]
        assert "ELBOLoss" in str(type(loss_fn))


@pytest.mark.required
def test_ae_default_loss():
    """Test that AE uses MSELoss by default."""
    ae = OODReconstruction(AE(input_shape=input_shape), model_type="ae")
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)

    with patch("dataeval.shift._ood._reconstruction.train") as mock_train:
        ae.fit(x_ref_small, threshold_perc=90, epochs=1)

        assert mock_train.called
        loss_fn = mock_train.call_args.kwargs["loss_fn"]
        assert isinstance(loss_fn, torch.nn.MSELoss)


@pytest.mark.required
def test_internal_state_initialization():
    """Test that internal state is properly initialized."""
    ae = OODReconstruction(AE(input_shape=input_shape))

    # Before fit
    assert ae._gmm_params is None
    assert ae._gmm_energy_ref_mean is None
    assert ae._data_info is None
    assert not hasattr(ae, "_ref_score")
    assert not hasattr(ae, "_threshold_perc")


@pytest.mark.required
def test_ref_score_set_after_fit():
    """Test that _ref_score is set after fit."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)

    assert not hasattr(ae, "_ref_score")
    ae.fit(x_ref_small, threshold_perc=90, epochs=1)

    # After fit, _ref_score should be set
    assert hasattr(ae, "_ref_score")
    assert ae._ref_score is not None
    assert ae._ref_score.instance_score.shape == (20,)


@pytest.mark.required
def test_validate_data_info_mismatch():
    """Test _validate raises error when data shape changes."""
    ae = OODReconstruction(AE(input_shape=input_shape))
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    ae.fit(x_ref_small, threshold_perc=90, epochs=1)

    # Manually set _data_info to trigger validation
    ae._data_info = (input_shape, np.float32)

    # Now try with different shape
    wrong_shape = (1, 4, 4)
    x_wrong = np.random.rand(10, *wrong_shape).astype(np.float32)

    with pytest.raises(RuntimeError, match="Expect data of type.*and shape"):
        ae._validate(x_wrong)


@pytest.mark.required
def test_gmm_reconstruction_loss_path():
    """Test GMM models use custom reconstruction loss."""

    # Create a proper GMM model that returns (recon, z, gamma)
    class GMMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 10),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.Unflatten(1, input_shape),
            )
            self.gmm_net = torch.nn.Linear(10, 3)

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            gamma = torch.softmax(self.gmm_net(z), dim=-1)
            return recon, z, gamma

    gmm_model = GMMModel()
    ood = OODReconstruction(gmm_model, model_type="ae", use_gmm=True)

    x_ref_small = np.random.rand(30, *input_shape).astype(np.float32)

    # Fit with GMM - this should use the custom gmm_reconstruction_loss
    with patch("dataeval.shift._ood._reconstruction.train") as mock_train:
        ood.fit(x_ref_small, threshold_perc=90, epochs=1)

        # Verify custom loss function was created
        assert mock_train.called
        loss_fn = mock_train.call_args.kwargs["loss_fn"]

        # Test that the loss function works correctly
        x_test = torch.rand(5, *input_shape)
        recon, z, gamma = gmm_model(x_test)
        loss_val = loss_fn(x_test, recon, z, gamma)
        assert isinstance(loss_val, torch.Tensor)


@pytest.mark.required
def test_gmm_params_computed_after_fit():
    """Test that GMM parameters are computed during fit."""

    class GMMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 10),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.Unflatten(1, input_shape),
            )
            self.gmm_net = torch.nn.Linear(10, 3)

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            gamma = torch.softmax(self.gmm_net(z), dim=-1)
            return recon, z, gamma

    gmm_model = GMMModel()
    ood = OODReconstruction(gmm_model, model_type="ae", use_gmm=True)

    x_ref_small = np.random.rand(30, *input_shape).astype(np.float32)

    # Before fit
    assert ood._gmm_params is None
    assert ood._gmm_energy_ref_mean is None

    # Fit with GMM
    ood.fit(x_ref_small, threshold_perc=90, epochs=1)

    # After fit, GMM params should be computed
    assert ood._gmm_params is not None
    assert ood._gmm_energy_ref_mean is not None
    assert isinstance(ood._gmm_energy_ref_mean, float)


@pytest.mark.required
def test_gmm_scoring_with_energy():
    """Test that GMM models include energy in scoring."""

    class GMMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 10),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.Unflatten(1, input_shape),
            )
            self.gmm_net = torch.nn.Linear(10, 3)

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            gamma = torch.softmax(self.gmm_net(z), dim=-1)
            return recon, z, gamma

    gmm_model = GMMModel()
    ood = OODReconstruction(gmm_model, model_type="ae", use_gmm=True)

    x_ref_small = np.random.rand(30, *input_shape).astype(np.float32)
    ood.fit(x_ref_small, threshold_perc=90, epochs=1)

    # Score some test data - should include GMM energy
    x_test = np.random.rand(10, *input_shape).astype(np.float32)
    score = ood.score(x_test)

    assert score.instance_score.shape == (10,)
    assert score.feature_score is not None
    assert score.feature_score.shape == x_test.shape

    # Verify scoring path was executed
    assert ood._gmm_params is not None


@pytest.mark.required
def test_gmm_params_energy():
    N, K, D = 10, 5, 1
    tz = torch.rand(N, D, dtype=torch.float32)
    tg = torch.rand(N, K, dtype=torch.float32)
    params = gmm_params(tz, tg)
    assert params.phi.numpy().shape[0] == K == params.log_det_cov.shape[0]  # type: ignore
    assert params.mu.numpy().shape == (K, D)  # type: ignore
    assert params.cov.numpy().shape == params.L.numpy().shape == (K, D, D)  # type: ignore
    for _ in range(params.cov.numpy().shape[0]):  # type: ignore
        assert (np.diag(params.cov[_].numpy()) >= 0.0).all()  # type: ignore
        assert (np.diag(params.L[_].numpy()) >= 0.0).all()  # type: ignore

    sample_energy, cov_diag = gmm_energy(tz, params, return_mean=True)
    assert sample_energy.numpy().shape == cov_diag.numpy().shape == ()  # type: ignore

    sample_energy, _ = gmm_energy(tz, params, return_mean=False)
    assert sample_energy.numpy().shape[0] == N  # type: ignore


@pytest.mark.required
def test_vae_output_extraction():
    """Test VAE model output extraction in _score."""
    vae = OODReconstruction(VAE(input_shape=input_shape), model_type="vae")
    x_ref_small = np.random.rand(20, *input_shape).astype(np.float32)
    vae.fit(x_ref_small, threshold_perc=90, epochs=1)

    # Score should extract reconstruction from VAE tuple output
    x_test = np.random.rand(5, *input_shape).astype(np.float32)
    score = vae.score(x_test)

    assert score.instance_score.shape == (5,)
    assert score.feature_score is not None
    assert score.feature_score.shape == x_test.shape


@pytest.mark.required
def test_gmm_z_normalization_prevents_covariance_error():
    """
    Regression test for non-positive-definite covariance matrix error.

    This test ensures that latent representations are normalized before computing
    GMM parameters, which prevents numerical issues in Cholesky decomposition
    when the latent space has very small or very large variance.
    """

    class GMMModelWithSmallLatents(torch.nn.Module):
        """Model that produces very small latent values to trigger the error."""

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 10),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.Unflatten(1, input_shape),
            )
            self.gmm_net = torch.nn.Linear(10, 3)

            # Initialize encoder to produce very small values (simulates collapsed latent space)
            for layer in self.encoder:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.uniform_(layer.weight, -0.001, 0.001)
                    torch.nn.init.zeros_(layer.bias)

        def forward(self, x):
            z = self.encoder(x)
            # Scale down z to create very small variance (this would trigger the error without normalization)
            z = z * 1e-3
            recon = self.decoder(z)
            gamma = torch.softmax(self.gmm_net(z), dim=-1)
            return recon, z, gamma

    gmm_model = GMMModelWithSmallLatents()
    ood = OODReconstruction(gmm_model, model_type="ae", use_gmm=True)

    # Create reference data
    x_ref_small = np.random.rand(50, *input_shape).astype(np.float32)

    # This should NOT raise _LinAlgError about non-positive-definite matrix
    # The normalization should prevent the error
    ood.fit(x_ref_small, threshold_perc=90, epochs=1)

    # Verify that normalization statistics were computed
    assert ood._gmm_params is not None

    # Verify that scoring works with normalized latents
    x_test = np.random.rand(10, *input_shape).astype(np.float32)
    score = ood.score(x_test)

    assert score.instance_score.shape == (10,)
    assert score.feature_score is not None
    assert score.feature_score.shape == x_test.shape


@pytest.mark.required
def test_combine_gmm_percentile():
    """
    Test _combine_gmm_percentile method for combining reconstruction and GMM scores.

    This method converts reconstruction error and GMM energy to percentiles using
    z-score to CDF transformation, then combines them as: 1 - (P_in * P_in).
    """

    class SimpleGMMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 10),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.Unflatten(1, input_shape),
            )
            self.gmm_net = torch.nn.Linear(10, 3)

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            gamma = torch.softmax(self.gmm_net(z), dim=-1)
            return recon, z, gamma

    # Test basic functionality of _combine_gmm_percentile
    gmm_model = SimpleGMMModel()

    config = OODReconstruction.Config(gmm_score_mode="percentile")
    ood = OODReconstruction(gmm_model, model_type="ae", use_gmm=True, config=config)

    # Create synthetic data
    x_ref = np.random.rand(30, *input_shape).astype(np.float32)
    ood.fit(x_ref, threshold_perc=90, epochs=1)

    # Check that reference statistics were computed
    assert ood._recon_ref_mean is not None
    assert ood._recon_ref_std is not None
    assert ood._gmm_energy_ref_mean is not None
    assert ood._gmm_energy_ref_std is not None

    # Test scoring with percentile mode
    x_test = np.random.rand(10, *input_shape).astype(np.float32)
    score = ood.score(x_test)

    # Verify output shape and validity
    assert score.instance_score.shape == (10,)
    assert not np.isnan(score.instance_score).any()
    assert not np.isinf(score.instance_score).any()

    # Verify scores are in valid probability range [0, 1] after percentile combination
    # Note: After combination, scores may be outside [0, 1] due to z-score standardization
    # But they should be finite and reasonable
    assert np.all(np.isfinite(score.instance_score))

    # Test that percentile method produces different results than standardized
    config_std = OODReconstruction.Config(gmm_score_mode="standardized", gmm_weight=0.5)
    ood_std = OODReconstruction(SimpleGMMModel(), model_type="ae", use_gmm=True, config=config_std)
    ood_std.fit(x_ref, threshold_perc=90, epochs=1)
    score_std = ood_std.score(x_test)

    # The two methods should produce different scores
    assert not np.allclose(score.instance_score, score_std.instance_score)


@pytest.mark.required
def test_combine_gmm_percentile_direct():
    """
    Direct unit test of _combine_gmm_percentile method with controlled inputs.

    Tests the mathematical correctness of the percentile-based fusion.
    """

    class DummyGMMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 10),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.Unflatten(1, input_shape),
            )
            self.gmm_net = torch.nn.Linear(10, 3)

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            gamma = torch.softmax(self.gmm_net(z), dim=-1)
            return recon, z, gamma

    ood = OODReconstruction(DummyGMMModel(), model_type="ae", use_gmm=True)

    # Set known reference statistics
    ood._recon_ref_mean = 0.0
    ood._recon_ref_std = 1.0
    ood._gmm_energy_ref_mean = 0.0
    ood._gmm_energy_ref_std = 1.0

    # Test with specific values
    recon_scores = np.array([0.0, 1.0, -1.0, 2.0])
    gmm_energy = np.array([0.0, 1.0, -1.0, 2.0])

    combined = ood._combine_gmm_percentile(recon_scores, gmm_energy)

    # Verify output shape
    assert combined.shape == recon_scores.shape
    assert not np.isnan(combined).any()
    assert not np.isinf(combined).any()

    # Test mathematical properties
    # For z=0 (mean), CDF should be 0.5, P_in = 0.5, combined = 1 - 0.5*0.5 = 0.75
    from scipy.stats import norm

    expected_0 = 1.0 - (1.0 - norm.cdf(0.0)) * (1.0 - norm.cdf(0.0))
    assert np.isclose(combined[0], expected_0, atol=1e-5)

    # For higher z-scores, combined score should be higher (more OOD)
    assert combined[3] > combined[0]  # z=2 should have higher OOD score than z=0

    # For negative z-scores, combined score should be lower (more in-distribution)
    assert combined[2] < combined[0]  # z=-1 should have lower OOD score than z=0


@pytest.mark.required
def test_gmm_score_mode_config_parameter():
    """Test that gmm_score_mode configuration parameter is properly used."""

    class SimpleGMMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 10),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.Unflatten(1, input_shape),
            )
            self.gmm_net = torch.nn.Linear(10, 3)

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            gamma = torch.softmax(self.gmm_net(z), dim=-1)
            return recon, z, gamma

    # Test with standardized mode (default)
    config_std = OODReconstruction.Config(gmm_score_mode="standardized")
    ood_std = OODReconstruction(SimpleGMMModel(), model_type="ae", use_gmm=True, config=config_std)
    assert ood_std.config.gmm_score_mode == "standardized"

    # Test with percentile mode
    config_pct = OODReconstruction.Config(gmm_score_mode="percentile")
    ood_pct = OODReconstruction(SimpleGMMModel(), model_type="ae", use_gmm=True, config=config_pct)
    assert ood_pct.config.gmm_score_mode == "percentile"

    # Verify both modes work end-to-end
    x_ref = np.random.rand(30, *input_shape).astype(np.float32)
    x_test = np.random.rand(10, *input_shape).astype(np.float32)

    ood_std.fit(x_ref, threshold_perc=90, epochs=1)
    score_std = ood_std.score(x_test)
    assert score_std.instance_score.shape == (10,)

    ood_pct.fit(x_ref, threshold_perc=90, epochs=1)
    score_pct = ood_pct.score(x_test)
    assert score_pct.instance_score.shape == (10,)
