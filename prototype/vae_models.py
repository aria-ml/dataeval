
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import null_space as null

from metadata_utils import InstanceMNIST
from metadata_utils import collate_fn_2 as collate_fn

from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F

from torch import nn
import numpy as np


import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from dataeval.detectors.ood.vae import OOD_VAE
from dataeval.utils.torch.models import  AE
from dataeval.utils.data.datasets import MNIST
from dataeval.utils._array import as_numpy


# from dataeval.utils.metadata import preprocess
# from dataeval.utils.data.datasets import _base

from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Callable



device = "cuda" if torch.cuda.is_available() else "cpu"

# Loss function for MNIST
def vae_loss(x : torch.Tensor, recon_x : torch.Tensor, mu : torch.Tensor, logvar : torch.Tensor) -> torch.Tensor: # BCE for recon, plus KL
    bad = torch.any(recon_x > 1) or torch.any(recon_x < 0) or torch.any(x < 0) or torch.any(x > 1)
    if bad:
        print(f'bad: {bad}')
        
    recon_loss = F.binary_cross_entropy(recon_x.view(len(x), -1), x.view(len(x), -1), reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss


# Loss function for MNIST
def verbose_loss(x : torch.Tensor, recon_x : torch.Tensor, mu : torch.Tensor, logvar : torch.Tensor) -> torch.Tensor: # BCE for recon, plus KL
    recon_loss = F.binary_cross_entropy(recon_x, x.view(len(x), -1), reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print(f'recon: {recon_loss:.0f}, KL: {kld_loss:.0f}')
    return recon_loss + kld_loss


# try different Loss function for CIFAR-10
def vae_loss2(x : torch.Tensor, recon_x : torch.Tensor, mu : torch.Tensor, logvar : torch.Tensor) -> torch.Tensor:  # MSE for recon, plus KL
    recon_loss = F.mse_loss(recon_x.view(len(x),-1), x.view(len(x), -1), reduction='mean')
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def vae_loss_cf(x, recon, mean, var):
    KLD = -torch.mean(0.5 * torch.sum(1 + var - mean**2 - torch.exp(var),axis=1),axis=0) # sum over latent dimension, not batch
    recon_loss = F.mse_loss(recon.view(len(x),-1), x.view(len(x), -1), reduction='sum')
    return 5*KLD + recon_loss

def recon_loss_cf(x, recon, mean, var):
    return F.mse_loss(recon.reshape((len(x),-1)), x.reshape((len(x), -1)), reduction='sum')

def reg_loss_cf(x, recon, mean, var):
    return -torch.mean(0.5 * torch.sum(1 + var - mean**2 - torch.exp(var),axis=1),axis=0) # sum over latent dimension, not batch

def reg_loss_sphere(x, recon, latent, ignore):
    # These two terms differ in terms of which axis is summed over. 
    #
    # on unit sphere, i.e. magntiude of each latent should be 1.0
    off_sphere = torch.sum(torch.sqrt(torch.sum(latent**2, axis=1)) - 1.0)
    
    # spread uniformly, i.e. if the population sum of the latents is not zero, then the 
    #   distribution on the sphere is not uniform. 
    bunched = torch.sqrt(torch.sum(torch.sum(latent, axis=0)**2))

    brightness = torch.mean(x.reshape(len(x),-1), axis=1)
    brightness = 2*brightness - 1.0
    
    ordered = torch.abs(torch.sum(brightness - latent[:,0]))

    return off_sphere + bunched + ordered





def L2(x : torch.Tensor, y : torch.Tensor):
    xnp, ynp = x.detach().numpy().reshape((len(x), -1)), y.detach().numpy().reshape((len(y), -1)) 
    return np.sum((xnp - ynp)**2, axis=1)

def normdot(x : torch.Tensor, y : torch.Tensor) -> float: # just correlation except I forogt to remove means
    xnp, ynp = x.detach().cpu().numpy().reshape((len(x), -1)), y.detach().cpu().numpy().reshape((len(y), -1)) 
    xdenom = np.sqrt(np.sum(xnp*xnp, axis=1))
    ydenom = np.sqrt(np.sum(ynp*ynp, axis=1))

    return np.sum(xnp*ynp, axis=1)/(xdenom*ydenom)

def rho(x : torch.Tensor, y : torch.Tensor) -> float: # correlation for real
    xnp, ynp = x.detach().numpy().reshape((len(x), -1)), y.detach().numpy().reshape((len(y), -1)) 
    xnp -= np.mean(xnp, axis=1, keepdims=True)
    ynp -= np.mean(ynp, axis=1, keepdims=True)
    
    xdenom = np.sqrt(np.sum(xnp*xnp, axis=1))
    ydenom = np.sqrt(np.sum(ynp*ynp, axis=1))

    return np.sum(xnp*ynp, axis=1)/(xdenom*ydenom)

def ecdf(data):
    """Compute the empirical cumulative distribution function (ECDF) of a sample."""
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y

def ecdf_ratio(sample1, sample2):
    """Compute the ratio of the ECDFs of two samples."""
    x1, y1 = ecdf(sample1)
    x2, y2 = ecdf(sample2)

    x_all = np.sort(np.concatenate((x1, x2)))
    # Interpolate y values at x_all points
    y1_all = np.interp(x_all, x1, y1)
    y2_all = np.interp(x_all, x2, y2)

    # Compute the ratio of the ECDFs
    ratio = y1_all / y2_all

    return x_all, ratio


#-------------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, input_dim : int = 784, hidden_dim: int = 400, latent_dim : int = 20):
        super().__init__()

        self.latent_dim = latent_dim
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x : torch.Tensor):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def embed(self, x : torch.Tensor):
        mu, logvar = self.encode(x.view(len(x), -1))
        # z = self.reparameterize(mu, torch.zeros_like(mu))
        dz = torch.exp(0.5 * logvar)

        return mu, dz

    def reparameterize(self, mu : torch.Tensor, logvar : torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) 
            return mu + eps * std
        else:
            return mu

    def decode(self, z : torch.Tensor):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor] :
        mu, logvar = self.encode(x.view(len(x), -1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

#-------------------------------------------------------------------------------

class VAEmod(nn.Module):
    """
    "Modern" Variational Autoencoder (VAE) class.
    
    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layer.
        latent_dim (int): Dimensionality of the latent space.
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim, self.hidden_dim, self.latent_dim = input_dim, hidden_dim, latent_dim
                
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, 2 * latent_dim), # 2 for mean and variance.
        )
        self.softplus = nn.Softplus()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
        
    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.
        
        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.
        
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        
    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()
    
    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.
        
        Args:
            z (torch.Tensor): Data in the latent space.
        
        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)
    
    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.
        
        Returns:
            VAEOutput: VAE output dataclass.
        """
        x = x.reshape((-1, self.input_dim))
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)
        
        loss, loss_recon, loss_kl = None, None, None
        if compute_loss:
            loss_recon = F.binary_cross_entropy(recon_x, x, reduction='none').sum(-1).mean()
            std_normal = torch.distributions.MultivariateNormal(
                torch.zeros_like(z, device=z.device),
                scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
            )
            loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
                    
            loss = loss_recon + loss_kl
            
        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )
    
from dataclasses import dataclass

import torch

@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.
    
    Attributes:
        z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled value of the latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
        loss_recon (torch.Tensor): The reconstruction loss component of the VAE loss.
        loss_kl (torch.Tensor): The KL divergence component of the VAE loss.
    """
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    
    loss: torch.Tensor | None
    loss_recon: torch.Tensor | None
    loss_kl: torch.Tensor | None

#-------------------------------------------------------------------------------

class OutlierVAE(nn.Module):
    def __init__(self, sample_input,  latent_dim=None, image_channels=None, first_layer_channels=None):
        super().__init__()

        if len(sample_input.shape) != 4:
            print(f'Your sample has shape: {sample_input.shape}')
            print('Please provide an input sample with shape (batch_size, n_channels, height, width.')
            return None
        
        _, self.n_channels, self.height, self.width = sample_input.shape

        self.latent_dim = 1024 if latent_dim is None else latent_dim
        image_channels = 3 if image_channels is None else image_channels
        first_layer_channels = 64 if first_layer_channels is None else first_layer_channels
        flc = first_layer_channels
        flc2, flc4, flc8 = flc*2, flc*4, flc*8

        # conv2d sequence, boosting channels by 2, then by 4 (skips flc4)
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=flc, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=flc, out_channels=flc2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=flc2, out_channels=flc8, kernel_size=4, stride=2, padding=1)

        self.hidden_shape = self.conv3(self.conv2(self.conv1(sample_input[0:1]))).shape[1:] # check shape of data before Linear() into latent space
        self.n_hidden = np.prod(self.hidden_shape)

        # fully connected (fc) links to latent space, 
        self.fc_mu = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(in_features=self.n_hidden, out_features=self.latent_dim ))
        self.fc_logvar = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(in_features=self.n_hidden, out_features=self.latent_dim ))

        # _, _ = self.encode(sample_input) # sets self.hidden_shape
        self.reform = nn.Sequential(nn.Linear(self.latent_dim , self.n_hidden), nn.Unflatten(1, self.hidden_shape))

        # deconv (via transposed convolution), deflating by 2, then by 4 (skips flc2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=flc8, out_channels=flc4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=flc4, out_channels=flc, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=flc, out_channels=image_channels, kernel_size=4, stride=2, padding=1)
        
    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.reform(z)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))

        return x

        
    def embed(self, x : torch.Tensor): # deterministic even if self.training is True
        mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, torch.zeros_like(mu))
        std = torch.exp(0.5*logvar)
        return mu, std


    def reparameterize(self, mu : torch.Tensor, logvar : torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) 
            return mu + eps * std
        else:
            return mu

    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor] :
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# from torchsummary import summary

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
        
    def forward(self, x):
        return x.view(self.shape)
    
class Trim(nn.Module):
    def __init__(self):
        super(Trim, self).__init__()
    
    def forward(self, x):
        return x[:, :, :32, :32]
    
#-------------------------------------------------------------------------------
class VAEcf(nn.Module):
    def __init__(self, latent_dim = None):
        self.latent_dim = 100 if latent_dim is None else latent_dim
        
        # [(W−K+2P)/S]+1
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride = 2, padding = 1)
        self.flatten = nn.Flatten()
        
        self.linear_mean = nn.Linear(2048, self.latent_dim)
        self.linear_logvar = nn.Linear(2048, self.latent_dim)

        self.linear = nn.Linear(self.latent_dim, 2048)
        self.reshape = Reshape(-1, 128, 4, 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride = 2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride = 2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 3, stride = 2, padding=1)
        self.trim = Trim()
    
    def reparameterized(self, mean, logvar):
        if self.training:
            eps = torch.randn(mean.size(0), mean.size(1)).to(device)
            z = mean + eps * torch.exp(logvar / 2.)
            return z
        else:
            return mean

    def encode(self, x): # Using silu instead of relu here
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = self.conv3(x)
        x = self.flatten(x)
        mean = self.linear_mean(x)
        logvar = self.linear_logvar(x)
        z = self.reparameterized(mean, logvar)
        return mean, logvar, z

    def decode(self, z):
        z = self.linear(z.cuda())
        z = self.reshape(z)
        z = F.silu(self.deconv1(z))
        z = F.silu(self.deconv2(z))
        z = F.silu(self.deconv3(z))
        z = self.trim(z)
        # z = F.sigmoid(z)
        return z
    
    def forward(self, x):
        mean, var, z = self.encode(x)
        recon = self.decode(z)
        return recon, mean, var

    def embed(self, x : torch.Tensor):
        mu, logvar, _ = self.encode(x)
        std = torch.exp(0.5*logvar)
        return mu, std



#-------------------------------------------------------------------------------



def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels
    
    def forward(self, input):
        size = int((input.size(1) // self.n_channels) ** 0.5)
        return input.view(input.size(0), self.n_channels, size, size)


class ConvVAE(nn.Module):
    def __init__(self, device='cuda', img_channels=None, latent_dim=None, nx=None, ny=None, batch_size=None):
        super().__init__()
        self.batch_size = 64 if batch_size is None else batch_size
        self.device = device
        self.latent_dim = 20 if latent_dim is None else latent_dim
        self.img_channels = 3 if img_channels is None else img_channels
        # self.model = args.model
        img_sizex = 28 if nx is None else nx
        img_sizey = 28 if ny is None else ny
        filters_m = 32

        ## Build network
        self.encoder = self.get_encoder(self.img_channels, filters_m)

        # output size depends on input image size, compute the output size
        demo_input = torch.ones([1, self.img_channels, img_sizex, img_sizey])
        h_dim = self.encoder(demo_input).shape[1]
        print('h_dim', h_dim)
        
        # map to latent z
        self.fc11 = nn.Linear(h_dim, self.latent_dim)
        self.fc12 = nn.Linear(h_dim, self.latent_dim)

        # decoder
        self.fc2 = nn.Linear(self.latent_dim, h_dim)
        self.decoder = self.get_decoder(filters_m, self.img_channels)
        
        self.log_sigma = 0
        
    @staticmethod
    def get_encoder(img_channels, filters_m):
        return nn.Sequential(
            nn.Conv2d(img_channels, filters_m, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters_m, 2 * filters_m, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 4 * filters_m, (5, 5), stride=2, padding=2),
            nn.ReLU(),
            Flatten()
        )

    @staticmethod
    def get_decoder(filters_m, out_channels):
        return nn.Sequential(
            UnFlatten(4 * filters_m),
            nn.ConvTranspose2d(4 * filters_m, 2 * filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * filters_m, filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filters_m, out_channels, (5, 5), stride=1, padding=2),
            nn.Sigmoid(),
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc11(h), self.fc12(h)
    
    def embed(self, x : torch.Tensor):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5*logvar)
        return mu, std

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(self.fc2(z))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def sample(self, n):
        sample = torch.randn(n, self.latent_dim).to(self.device)
        return self.decode(sample)

    def reconstruction_loss(self, x_hat, x):
        """ Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 """
        
        log_sigma = ((x - x_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()
        self.log_sigma = log_sigma.item()

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)
        
        rec = gaussian_nll(x_hat, log_sigma, x).sum()
    
        return rec

    def loss_function(self, recon_x, x, mu, logvar):
        # Important: both reconstruction and KL divergence loss have to be summed over all element!
        # Here we also sum the over batch and divide by the number of elements in the data later

        rec = self.reconstruction_loss(recon_x, x)
        
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return rec, kl


def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)



#========================================================sigma optimal with pre-determined sigma
# First time I did this, Claude gave me the one where we train sigma. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np


class ResNet50Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet50Encoder, self).__init__()
        # Load pretrained ResNet-50 model
        # resnet = models.resnet50(pretrained=True)

        # Using pretrained weights:
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)        
        # Remove the final fully connected layer
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet50 features are 2048-dimensional
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)
        
        
    def forward(self, x):
        features = self.resnet_features(x)
        features = features.view(features.size(0), -1)  # Flatten
        mu, logvar = self.fc_mu(features), self.fc_logvar(features)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_channels=3, input_height=64, input_width=64):
        super(Decoder, self).__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        
        # Calculate output feature map dimensions after 4 upsampling operations
        self.initial_height = input_height // 16
        self.initial_width = input_width // 16
        initial_features = 512
        
        # Initial dense layer
        self.fc = nn.Linear(latent_dim, initial_features * self.initial_height * self.initial_width)
        
        # Transposed convolution layers
        self.deconv1 = nn.ConvTranspose2d(initial_features, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.deconv4 = nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, z):
        # Initial linear layer
        x = self.fc(z)
        x = x.view(-1, 512, self.initial_height, self.initial_width)
        
        # Transposed convolutions with ReLU and batch normalization
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        
        # Final transposed convolution with sigmoid activation
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction


class SigmaOptimalVAE(nn.Module):
    def __init__(self, latent_dim, input_channels=3, input_height=64, input_width=64, beta=1.0):
        super(SigmaOptimalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Initialize encoder and decoder
        self.encoder = ResNet50Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, input_channels, input_height, input_width)
        
        # Analytic optimal sigma (fixed, not learned)
        # Initialize to 1, will be set during forward pass based on reconstruction error
        self.register_buffer('optimal_sigma', torch.ones(1))
        self.log_sigma = 0.0 # OODdetector needs this. Is there a problem here? 
        
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # During training, add noise from a standard normal distribution NOPE, include logvar to be like other models. 
        if self.training:
            # Sample from standard normal
            # std = torch.ones_like(mu)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(mu)
            # Reparameterization trick
            z = mu + eps * std
            return z
        else:
            # During inference, just return the mean
            return mu
    
    def decode(self, z):
        return self.decoder(z)

    def embed(self, x : torch.Tensor):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5*logvar)
        return mu, std

    def compute_optimal_sigma(self, x, x_recon):
        """
        Compute the analytically optimal sigma based on reconstruction error
        """
        # MSE per datapoint, averaged over batch
        batch_size = x.size(0)
        mse = F.mse_loss(x_recon, x, reduction='none').view(batch_size, -1).mean(1)
        
        # Optimal sigma is the square root of the average MSE
        # We clip it to avoid numerical instability
        optimal_sigma = torch.sqrt(mse.mean()).clamp(min=1e-6)
        return optimal_sigma
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        
        # Update optimal sigma
        with torch.no_grad():
            self.optimal_sigma = self.compute_optimal_sigma(x, x_recon)
        
        return x_recon, mu, self.optimal_sigma
    
    # This loss function comes with this model, but I am using the standalone from ood_detector.py instead. 
    # def loss_function(self, x_recon, x, mu):
    #     """
    #     Sigma Optimal VAE loss function
    #     """
    #     batch_size = x.size(0)
        
    #     # Reconstruction loss (log likelihood under Gaussian observation model with optimal sigma)
    #     # Using equation: -0.5 * log(2π * sigma^2) - 0.5 * ||x - x_recon||^2 / sigma^2
    #     recon_error = F.mse_loss(x_recon, x, reduction='none').view(batch_size, -1).sum(1)
    #     log_likelihood = -0.5 * torch.log(2 * np.pi * self.optimal_sigma**2) - 0.5 * recon_error / self.optimal_sigma**2
    #     reconstruction_loss = -log_likelihood.mean()
        
    #     # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #     # Since we're using standard normal prior and standard normal for sampling,
    #     # the KL divergence simplifies to 0.5 * ||mu||^2
    #     kl_divergence = 0.5 * torch.sum(mu.pow(2), dim=1).mean()
        
    #     # Total loss is reconstruction loss + beta * KL divergence
    #     loss = reconstruction_loss + self.beta * kl_divergence
        
    #     return loss, reconstruction_loss, kl_divergence


# Training function
def train_sigma_optimal_vae(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, sigma = model(data)
        loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} '
                  f'Recon: {recon_loss.item():.6f} KL: {kl_loss.item():.6f} Sigma: {sigma.item():.6f}')
    
    avg_loss = train_loss / len(train_loader)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.6f}')
    return avg_loss


# Test function
def test_sigma_optimal_vae(model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            
            recon_batch, mu, sigma = model(data)
            loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu)
            
            test_loss += loss.item()
    
    avg_loss = test_loss / len(test_loader)
    print(f'====> Test set loss: {avg_loss:.6f}')
    return avg_loss


# # Example usage
# def main():
#     # Parameters
#     batch_size = 32
#     epochs = 10
#     latent_dim = 128
#     learning_rate = 1e-4
#     beta = 1.0
    
#     # Device configuration
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Load dataset
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor()
#     ])
    
#     train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
#     test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     # Initialize model, optimizer
#     model = SigmaOptimalVAE(latent_dim=latent_dim, input_channels=3, input_height=64, input_width=64, beta=beta)
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Training loop
#     for epoch in range(1, epochs + 1):
#         train_loss = train_sigma_optimal_vae(model, train_loader, optimizer, device, epoch)
#         test_loss = test_sigma_optimal_vae(model, test_loader, device, epoch)
        
#         # Optionally save model
#         if epoch % 5 == 0:
#             torch.save(model.state_dict(), f'sigma_optimal_vae_epoch_{epoch}.pt')
    
#     # Final save
#     torch.save(model.state_dict(), 'sigma_optimal_vae_final.pt')
    
#     print(f"Final optimal sigma: {model.optimal_sigma.item():.6f}")


# if __name__ == "__main__":
#     main()



#=======================================================================================
# The following is a perceptual loss called Normalized Laplacian Pyramid distance.
# https://gist.github.com/alper111/b9c6d80e2dba1ee0bfac15eb7dad09c8
# 
# MIT License
# 
# Copyright (c) 2024 Alper Ahmetoglu
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3, device=torch.device('cpu')):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
    
