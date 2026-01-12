"""
Tests for PyTorch loss functions.

Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

import pytest
import torch

from dataeval.utils.losses import ELBOLoss


@pytest.mark.required
def test_vaeloss_class_initialization():
    """Test ELBOLoss class initialization."""
    # Default initialization
    loss_fn = ELBOLoss()
    assert loss_fn.beta == 1.0
    assert loss_fn.reduction == "mean"

    # Custom initialization
    loss_fn2 = ELBOLoss(beta=2.0, reduction="sum")
    assert loss_fn2.beta == 2.0
    assert loss_fn2.reduction == "sum"


@pytest.mark.required
def test_vaeloss_class_invalid_parameters():
    """Test that ELBOLoss raises errors for invalid parameters."""
    # Negative beta
    with pytest.raises(ValueError, match="beta must be non-negative"):
        ELBOLoss(beta=-1.0)

    # Invalid reduction
    with pytest.raises(ValueError, match="reduction must be"):
        ELBOLoss(reduction="invalid")


@pytest.mark.required
def test_vaeloss_class_call():
    """Test that ELBOLoss can be called like a function."""
    loss_fn = ELBOLoss(beta=1.0)

    batch_size = 8
    latent_dim = 16

    x = torch.randn(batch_size, 1, 28, 28)
    x_recon = torch.randn(batch_size, 1, 28, 28)
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)

    loss = loss_fn(x, x_recon, mu, logvar)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.required
def test_vaeloss_reduction_modes():
    """Test different reduction modes for ELBOLoss."""
    batch_size = 8
    latent_dim = 16

    x = torch.randn(batch_size, 1, 28, 28)
    x_recon = torch.randn(batch_size, 1, 28, 28)
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)

    loss_mean = ELBOLoss(reduction="mean")(x, x_recon, mu, logvar)
    loss_sum = ELBOLoss(reduction="sum")(x, x_recon, mu, logvar)

    # Sum should generally be larger than mean for batch_size > 1
    # (unless reconstruction error is very small)
    assert torch.isfinite(loss_mean)
    assert torch.isfinite(loss_sum)
    assert loss_mean.ndim == 0
    assert loss_sum.ndim == 0


@pytest.mark.required
def test_vaeloss_beta_zero():
    """Test ELBOLoss with beta=0 (no KL divergence)."""
    loss_fn = ELBOLoss(beta=0.0)

    batch_size = 4
    latent_dim = 8

    x = torch.randn(batch_size, 1, 28, 28)
    x_recon = torch.randn(batch_size, 1, 28, 28)
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)

    loss = loss_fn(x, x_recon, mu, logvar)

    # With beta=0, only reconstruction loss should be computed
    # Verify it's equivalent to MSE
    import torch.nn.functional as F

    expected_loss = F.mse_loss(x_recon.view(len(x), -1), x.view(len(x), -1))
    assert torch.allclose(loss, expected_loss, rtol=1e-5)


@pytest.mark.required
def test_vaeloss_gradient_flow():
    """Test that gradients flow through ELBOLoss."""
    loss_fn = ELBOLoss(beta=1.0)

    batch_size = 4
    latent_dim = 8

    x = torch.randn(batch_size, 1, 28, 28)
    x_recon = torch.randn(batch_size, 1, 28, 28, requires_grad=True)
    mu = torch.randn(batch_size, latent_dim, requires_grad=True)
    logvar = torch.randn(batch_size, latent_dim, requires_grad=True)

    loss = loss_fn(x, x_recon, mu, logvar)
    loss.backward()

    # Gradients should exist and be finite
    assert x_recon.grad is not None
    assert mu.grad is not None
    assert logvar.grad is not None
    assert torch.all(torch.isfinite(x_recon.grad))
    assert torch.all(torch.isfinite(mu.grad))
    assert torch.all(torch.isfinite(logvar.grad))


@pytest.mark.required
def test_vaeloss_repr():
    """Test ELBOLoss string representation."""
    loss_fn = ELBOLoss(beta=2.5, reduction="sum")
    repr_str = repr(loss_fn)

    assert "ELBOLoss" in repr_str
    assert "2.5" in repr_str
    assert "sum" in repr_str


@pytest.mark.optional
def test_vaeloss_different_input_shapes():
    """Test ELBOLoss with different input shapes."""
    loss_fn = ELBOLoss(beta=1.0)

    # Test with 1D latent (common case)
    batch_size = 4
    latent_dim = 16
    x = torch.randn(batch_size, 3, 32, 32)
    x_recon = torch.randn(batch_size, 3, 32, 32)
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)

    loss1 = loss_fn(x, x_recon, mu, logvar)
    assert torch.isfinite(loss1)

    # Test with smaller images
    x2 = torch.randn(batch_size, 1, 8, 8)
    x2_recon = torch.randn(batch_size, 1, 8, 8)
    mu2 = torch.randn(batch_size, 8)
    logvar2 = torch.randn(batch_size, 8)

    loss2 = loss_fn(x2, x2_recon, mu2, logvar2)
    assert torch.isfinite(loss2)


@pytest.mark.optional
def test_vaeloss_perfect_reconstruction():
    """Test ELBOLoss when reconstruction is perfect."""
    loss_fn = ELBOLoss(beta=1.0)

    batch_size = 4
    latent_dim = 8

    x = torch.randn(batch_size, 1, 28, 28)
    x_recon = x.clone()  # Perfect reconstruction
    mu = torch.zeros(batch_size, latent_dim)  # Standard normal prior
    logvar = torch.zeros(batch_size, latent_dim)  # Unit variance

    loss = loss_fn(x, x_recon, mu, logvar)

    # With perfect reconstruction and prior matching, loss should be close to zero
    # Reconstruction loss should be near 0
    # KL divergence for N(0,1) vs N(0,1) should be 0
    assert loss.item() < 1e-5


@pytest.mark.optional
def test_vaeloss_batch_size_one():
    """Test ELBOLoss with batch size of 1."""
    loss_fn = ELBOLoss(beta=1.0)

    x = torch.randn(1, 1, 28, 28)
    x_recon = torch.randn(1, 1, 28, 28)
    mu = torch.randn(1, 16)
    logvar = torch.randn(1, 16)

    loss = loss_fn(x, x_recon, mu, logvar)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.required
def test_vaeloss_protocol_compatibility():
    """Test that ELBOLoss is compatible with the VAELossFn protocol."""
    from dataeval.protocols import EvidenceLowerBoundLossFn

    loss_fn = ELBOLoss(beta=1.0)

    # Should be instance-checkable due to @runtime_checkable
    assert isinstance(loss_fn, EvidenceLowerBoundLossFn)

    # Should be callable with the right signature
    batch_size = 4
    latent_dim = 8
    x = torch.randn(batch_size, 1, 28, 28)
    x_recon = torch.randn(batch_size, 1, 28, 28)
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)

    # This should work without errors
    result = loss_fn(x, x_recon, mu, logvar)
    assert isinstance(result, torch.Tensor)
