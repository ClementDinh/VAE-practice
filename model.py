import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

#flatten and reshape
class VariationalAutoEncoder(nn.Module):
    def __init__(self,input_dim, h_dim, latent_dim, z_dim=10):
        super().__init__()

        self.img_hid = nn.Linear(input_dim, h_dim)
        self.hid_latent = nn.Linear(h_dim, latent_dim)
        self.latent_mu = nn.Linear(latent_dim, z_dim)
        self.latent_sigma = nn.Linear(latent_dim, z_dim)

        self.z_latent = nn.Linear(z_dim, latent_dim)
        self.latent_hid = nn.Linear(latent_dim, h_dim)
        self.hid_img = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()
    def encode(self,x):
        h = self.relu(self.img_hid(x))
        latent = self.relu(self.hid_latent(h))
        mu, sigma = self.latent_mu(latent), self.latent_sigma(latent)
        return mu, sigma

    def decode(self, z):
        latent = self.relu(self.z_latent(z))
        h = self.relu(self.latent_hid(latent))
        return torch.sigmoid(self.hid_img(h))

    def forward(self, x):
        mu,sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparameterized)
        return x_reconstructed, mu, sigma

