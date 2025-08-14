import logging
from typing import Dict, List, Optional
import numpy as np
import torch
import pandas as pd
from torch.optim import Adam
from .base_vae import BaseVAE
from ...losses import vae_loss

# If Encoder/Decoder kwargs need constraints etc., they’ll be passed through.

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        logger.info("Using Metal GPU")
        return torch.device("mps")
    logger.info("Using CPU")
    return torch.device("cpu")


class VAE(BaseVAE):
    """
    Concrete VAE that composes the modular Encoder/Decoder from BaseVAE
    and provides a simple fit() loop.

    BaseVAE.forward(x) returns: recon, mu, logvar, z
    """

    def __init__(
            self,
            features: List[str],
            *,
            latent_dim: Optional[int] = None,
            encoder: Optional[torch.nn.Module] = None,
            decoder: Optional[torch.nn.Module] = None,
            encoder_layers: Optional[List[Dict]] = None,
            constraint=None,
            batch_norm: bool = False,
            activation: str = "relu",
            lr: float = 1e-3,
    ):
        """
        Args:
            features: list[str] feature names (len(features) == input_dim)
            latent_dim: size of latent space
            encoder_layers: optional list of layer configs for Encoder
            constraint, batch_norm, activation: forwarded to Encoder builder
            hidden_dim_ignored: kept only to mirror old API (not used)
            lr: default learning rate for fit()
        """
        super().__init__(features=features)
        self.lr = lr

        feature_dim = len(features)

        if encoder is not None and decoder is not None:
            # Loaded path: use provided modules (e.g., open_model)
            self.encoder = encoder
            self.decoder = decoder
        else:
            # Fresh build path: need latent_dim
            if latent_dim is None:
                raise ValueError(
                    "latent_dim is required when encoder/decoder are not provided."
                )
            self.encoder = self.build_encoder(
                feature_dim=feature_dim,
                latent_dim=latent_dim,
                layers=encoder_layers,
                constraint=constraint,
                batch_norm=batch_norm,
                activation=activation,
            )
            self.decoder = self.build_decoder(
                feature_dim=feature_dim,
                latent_dim=latent_dim,
            )

        # A place to record simple history if you want
        self.history: Dict[str, list] = {"loss": []}
        self.latent_index = None
        self.latent_groups = None
        self.normal_stats = None

    def fit(self, X: pd.DataFrame, y=None, *, epochs: int = 20, lr: Optional[float] = None,
            device: Optional[torch.device] = None, progress: bool = True):
        """
        Full-batch training loop using vae_loss(recon, x, mu, logvar).

        X: pandas.DataFrame with float features, columns must match `self.features`.
        """
        if lr is None:
            lr = self.lr

        if device is None:
            device = get_device()

        # Safety check for feature alignment
        if hasattr(X, "columns") and self.features is not None:
            if list(X.columns) != list(self.features):
                raise ValueError(
                    "Input DataFrame columns do not match model features.\n"
                    f"Data columns: {list(X.columns)[:5]}... (n={len(X.columns)})\n"
                    f"Model features: {self.features[:5]}... (n={len(self.features)})"
                )

        self.to(device)
        self.train()

        # IMPORTANT: assign the result of .to(device)
        x_tensor = torch.from_numpy(X.values.astype(np.float32)).to(device)

        optimizer = Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            recon, mu, logvar, _z = super().forward(x_tensor)
            total_loss, recon_loss, kl_loss = vae_loss(recon, x_tensor, mu, logvar)

            total_loss.backward()
            optimizer.step()

            self.history["loss"].append(float(total_loss.detach().cpu()))
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss.item():.4f} | "
                  f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")
            if progress:
                logger.debug(
                    f"Epoch {epoch + 1} | Loss: {total_loss.item():.4f} | "
                    f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}"
                )


if __name__ == "__main__":
    # Example usage
    N = 100
    df = pd.DataFrame({
        "feat1": np.random.rand(N),
        "feat2": np.random.rand(N),
    })

    vae: VAE = VAE(features=list(df.columns), latent_dim=2)
    vae.fit(df, epochs=10, lr=0.01)

    # Save the model if needed
    vae.save("vae_model")

    vae: VAE = VAE.open_model(path="vae_model", model_cls=VAE, device="cpu")
    print("Model loaded with features:", vae.features)
