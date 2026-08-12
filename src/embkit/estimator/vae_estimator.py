from typing import Optional
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import pandas as pd

from ..models.vae.vae import VAE
from ..models.vae.base_vae import BaseVAE
from ..losses import BCEWithLogitsLoss

class VAEEstimator(BaseEstimator):
    """
    Scikit-learn style wrapper for quick experiments.
    """

    def __init__(self, latent_dim: int = 100, learning_rate: float = 1e-4,
                 batch_size: int = 30, epochs: int = 130, beta: float = 0.0,
                 device: Optional[str] = None):
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta
        self.history = None
        self.model: Optional[VAE] = None
        self.device = device

    def fit(self, X: pd.DataFrame) -> "VAEEstimator":
        """
        Fit the VAE model to the input data.

        Parameters:
            X (pd.DataFrame): The input data with shape (number_of_samples, number_of_features).

        Returns:
            None
        """
        feature_dim = X.shape[1]
        features = list(X.columns)

        vae = VAE(features=features, latent_dim=self.latent_dim)

        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        vae.to(device)

        x = torch.tensor(X.values, dtype=torch.float32, device=device)
        loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=True)
        opt = Adam(vae.parameters(), lr=self.learning_rate)
        loss_fn = BCEWithLogitsLoss(beta=self.beta)

        loss_hist = []
        for _ in range(self.epochs):
            vae.train()
            tot = 0.0
            n = 0
            for (batch_x,) in loader:
                opt.zero_grad()
                recon, mu, logvar, _ = vae(batch_x)
                total, _, _ = loss_fn(recon, batch_x, mu, logvar)
                total.backward()
                opt.step()
                tot += float(total.item())
                n += 1
            loss_hist.append(tot / max(1, n))

        self.history = {"loss": loss_hist}
        self.model = vae
        return self

    def score(self, X: pd.DataFrame) -> float:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(X.values, dtype=torch.float32, device=next(self.model.parameters()).device)
            mu, _, _ = self.model.encoder(x)
            recon = self.model.decoder(mu).cpu().numpy()
        return -mean_squared_error(X.values, recon)

