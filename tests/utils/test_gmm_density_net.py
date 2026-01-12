"""Tests for GMMDensityNet model."""

import pytest
import torch
import torch.nn as nn

from dataeval.utils.models import VAE, GMMDensityNet


class TestGMMDensityNet:
    """Test suite for GMMDensityNet."""

    def test_instantiation(self):
        """Test basic instantiation of GMMDensityNet."""
        latent_dim = 32
        n_gmm = 3
        hidden_dim = 10

        gmm_net = GMMDensityNet(latent_dim, n_gmm, hidden_dim)

        assert gmm_net.latent_dim == latent_dim
        assert gmm_net.n_gmm == n_gmm
        assert gmm_net.hidden_dim == hidden_dim

    def test_forward_pass(self):
        """Test forward pass produces valid probability distributions."""
        latent_dim = 16
        n_gmm = 4
        batch_size = 8

        gmm_net = GMMDensityNet(latent_dim, n_gmm)
        z = torch.randn(batch_size, latent_dim)
        gamma = gmm_net(z)

        # Check shape
        assert gamma.shape == (batch_size, n_gmm)

        # Check probabilities sum to 1
        assert torch.allclose(gamma.sum(dim=-1), torch.ones(batch_size), atol=1e-6)

        # Check all probabilities are non-negative
        assert (gamma >= 0).all()

        # Check all probabilities are <= 1
        assert (gamma <= 1).all()

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="n_gmm must be at least 1"):
            GMMDensityNet(latent_dim=10, n_gmm=0)

        with pytest.raises(ValueError, match="latent_dim must be at least 1"):
            GMMDensityNet(latent_dim=0, n_gmm=2)

        with pytest.raises(ValueError, match="hidden_dim must be at least 1"):
            GMMDensityNet(latent_dim=10, n_gmm=2, hidden_dim=0)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        latent_dim = 8
        n_gmm = 2
        batch_size = 4

        gmm_net = GMMDensityNet(latent_dim, n_gmm)
        z = torch.randn(batch_size, latent_dim, requires_grad=True)
        gamma = gmm_net(z)

        # Create a target distribution and compute cross-entropy loss
        # This produces meaningful gradients unlike sum() which is constant due to softmax
        target = torch.zeros(batch_size, n_gmm)
        target[:, 0] = 1.0  # Target first component
        loss = -torch.sum(target * torch.log(gamma + 1e-8))
        loss.backward()

        # Check that gradients exist and are non-zero
        assert z.grad is not None
        # Use a looser tolerance since we expect actual gradients now
        assert (torch.abs(z.grad) > 1e-6).any()

    def test_integration_with_vae(self):
        """Test integration of GMMDensityNet with VAE model."""

        class VAE_GMM(nn.Module):
            """VAE with GMM density estimation."""

            def __init__(self, input_shape, latent_dim=None, n_gmm=2):
                super().__init__()
                self.vae = VAE(input_shape, latent_dim)
                self.gmm_density = GMMDensityNet(self.vae.latent_dim, n_gmm)

            def forward(self, x):
                recon, mu, logvar = self.vae(x)
                # Use mean (mu) as the latent representation for GMM
                z = mu
                gamma = self.gmm_density(z)
                return recon, z, gamma

        # Create model
        input_shape = (1, 16, 16)
        n_gmm = 3
        model = VAE_GMM(input_shape, latent_dim=32, n_gmm=n_gmm)

        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, *input_shape)
        recon, z, gamma = model(x)

        # Check outputs
        assert recon.shape == (batch_size, *input_shape)
        assert z.shape[0] == batch_size
        assert gamma.shape == (batch_size, n_gmm)
        assert torch.allclose(gamma.sum(dim=-1), torch.ones(batch_size), atol=1e-6)

    def test_different_batch_sizes(self):
        """Test that GMMDensityNet works with different batch sizes."""
        latent_dim = 20
        n_gmm = 5

        gmm_net = GMMDensityNet(latent_dim, n_gmm)

        for batch_size in [1, 5, 10, 100]:
            z = torch.randn(batch_size, latent_dim)
            gamma = gmm_net(z)

            assert gamma.shape == (batch_size, n_gmm)
            assert torch.allclose(gamma.sum(dim=-1), torch.ones(batch_size), atol=1e-6)

    def test_deterministic_output(self):
        """Test that same input produces same output (no randomness in forward pass)."""
        latent_dim = 12
        n_gmm = 3
        batch_size = 5

        gmm_net = GMMDensityNet(latent_dim, n_gmm)
        gmm_net.eval()  # Set to eval mode

        z = torch.randn(batch_size, latent_dim)
        gamma1 = gmm_net(z)
        gamma2 = gmm_net(z)

        assert torch.allclose(gamma1, gamma2)
