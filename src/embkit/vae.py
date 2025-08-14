import os
import json

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_device():

    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return  torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Metal GPU")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")
    

def vae_loss(recon_x, x, mu, logvar):
    bce_per_sample = F.binary_cross_entropy(recon_x, x, reduction="none").mean(dim=1)
    reconstruction_loss = x.size(1) * bce_per_sample
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return (reconstruction_loss + kl_loss).mean(), reconstruction_loss.mean(), kl_loss.mean()


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, features=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log-variance
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        self.features = list(features) if features is not None else None

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), os.path.join(path, "vae_model.pth"))
        if self.features is not None:
            with open(os.path.join(path, "index"), "wt", encoding="ascii") as handle:
                handle.write(json.dumps(list(self.features)))


    def fit(self, X, epochs=20):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        logger.debug(self)

        device = get_device()

        x_tensor = torch.from_numpy(X.values.astype(np.float32))
        x_tensor.to(device)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            recon_x, mu, logvar = self(x_tensor)
            total_loss, recon_loss, kl_loss = vae_loss(recon_x, x_tensor, mu, logvar)

            total_loss.backward()
            optimizer.step()

            logger.debug(f"Epoch {epoch+1} | Loss: {total_loss.item():.4f} | "
                f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")    
