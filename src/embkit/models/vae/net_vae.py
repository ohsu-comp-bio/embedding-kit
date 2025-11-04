"""
NetVAE implementation
"""
import logging

from typing import Dict, List, Optional, Callable, Union
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from .base_vae import BaseVAE
from .encoder import Encoder
from .decoder import Decoder
from ...losses import net_vae_loss
from ...constraints import NetworkConstraint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# NetVae (training with optional alternating constraint)
# ---------------------------------------------------------

class NetVAE(BaseVAE):
    """
    NetVAE

    A VAE model with group based constraint. Designed to work with 
    transcription factor network groups. All elements controlled by a common
    transcription factor a pooled into a single embedding variable. All other connections
    in from the input layer are forced to be zero
    """

    def __init__(self, features: List[str], encoder: Optional[Encoder] = None, decoder: Optional[Decoder] = None):
        super().__init__(features=features, encoder=encoder, decoder=decoder)
        self.latent_groups: Optional[Dict[str, List[str]]] = None
        self.latent_index: Optional[List[str]] = None
        self.history: Optional[Dict[str, List[float]]] = None
        self.normal_stats: Optional[pd.DataFrame] = None

    def fit(
            self,
            X: Union[pd.DataFrame, torch.Tensor],
            *,
            latent_dim: Optional[int] = None,
            latent_index: Optional[List[str]] = None,
            latent_groups: Optional[Dict[str, List[str]]] = None,
            learning_rate: float = 1e-3,
            batch_size: int = 128,
            epochs: int = 80,
            phases: Optional[List[int]] = None,  # e.g. [warmup, constrained, finetune]
            device: Optional[str] = None,
            grouping_fn: Optional[Callable[[np.ndarray, List[str]], Dict[str, List[str]]]] = None,
    ) -> None:
        """
        Train the model on X. Builds encoder/decoder if missing.
        Supply either latent_dim or latent_index.
        """
        # --- inputs ---
        if isinstance(X, torch.Tensor):
            if not self.features:
                raise ValueError("Tensor input requires self.features to be defined.")
            df = pd.DataFrame(X.detach().cpu().numpy(), columns=self.features)
        else:
            df = X

        if latent_index is None:
            if latent_dim is None:
                raise ValueError("Provide latent_dim or latent_index.")
            latent_index = [f"z{i}" for i in range(latent_dim)]

        # --- device ---
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device(device)

        # --- constraint + (re)build modules if needed ---
        constraint = NetworkConstraint(list(df.columns), latent_index, latent_groups)
        if self.encoder is None or self.decoder is None:
            feature_dim = len(df.columns)
            self.encoder = BaseVAE.build_encoder(feature_dim=feature_dim, latent_dim=len(latent_index))
            self.decoder = BaseVAE.build_decoder(feature_dim=feature_dim, latent_dim=len(latent_index))
        
        # Attach the constraint to the encoder
        self.encoder.constraint = constraint
        
        self.latent_index = list(latent_index)

        # --- data/optim ---
        self.to(device)
        x = torch.tensor(df.values, dtype=torch.float32, device=device)
        loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.history = {"loss": [], "reconstruction_loss": [], "kl_loss": []}

        # --- helpers ---
        def refresh_mask():
            self.encoder.refresh_mask(device)

        def start_constrained_phase():
            if grouping_fn is not None:
                with torch.no_grad():
                    # example access; adjust to your encoder’s structure
                    w = self.encoder.pathway.linear.weight.detach().cpu().numpy()  # [latent, features]
                    new_groups = grouping_fn(w, list(df.columns))
                    constraint.update_membership(new_groups)
            constraint.set_active(True)
            refresh_mask()

        def start_unconstrained_phase():
            constraint.set_active(False)
            refresh_mask()

        # initial (usually unconstrained)
        constraint.set_active(False)
        refresh_mask()

        total_epochs = sum(phases) if phases else epochs
        boundaries = np.cumsum(phases).tolist() if phases else []

        # --- training loop ---
        for epoch in range(total_epochs):

            if boundaries and epoch in boundaries:
                idx = boundaries.index(epoch)
                (start_constrained_phase if idx % 2 == 0 else start_unconstrained_phase)()

            self.train()
            epoch_tot = epoch_rec = epoch_kl = 0.0
            n_batches = 0

            for (batch_x,) in loader:
                opt.zero_grad()
                total, recon, kl = net_vae_loss(self, batch_x)  # calls self.forward(batch_x)
                total.backward()
                opt.step()
                epoch_tot += float(total.item())
                epoch_rec += float(recon.item())
                epoch_kl += float(kl.item())
                n_batches += 1

            self.history["loss"].append(epoch_tot / max(1, n_batches))
            self.history["reconstruction_loss"].append(epoch_rec / max(1, n_batches))
            self.history["kl_loss"].append(epoch_kl / max(1, n_batches))

            # 👇 show progress
            if epoch % 2 == 0:
                print(f"Epoch {epoch + 1}/{total_epochs} | ")
                print(f"loss={self.history['loss'][-1]:.4f} | "
                      f"recon={self.history['reconstruction_loss'][-1]:.4f} | "
                      f"kl={self.history['kl_loss'][-1]:.4f}")
                logger.info(
                    "Epoch %d/%d | loss=%.4f | recon=%.4f | kl=%.4f",
                    epoch + 1, total_epochs,
                    self.history["loss"][-1],
                    self.history["reconstruction_loss"][-1],
                    self.history["kl_loss"][-1],
                )

        # --- artifacts ---
        self.latent_groups = constraint.latent_membership

        self.eval()
        with torch.no_grad():
            mu, _, _ = self.encoder(x)
            recon = self.decoder(mu).cpu().numpy()
        normal_pred = pd.DataFrame(recon, index=df.index, columns=df.columns)
        resid = normal_pred - df
        self.normal_stats = pd.DataFrame({"mean": resid.mean(), "std": resid.std(ddof=0)})


if __name__ == "__main__":
    # Make a simple 2-feature dataset with 1-D columns
    N = 100
    df = pd.DataFrame({
        "feat1": np.random.rand(N),
        "feat2": np.random.rand(N),
    })

    # Setup and train NetVae (this builds encoder/decoder internally)
    net = NetVAE(features=list(df.columns))
    net.encoder = BaseVAE.build_encoder(feature_dim=len(df.columns), latent_dim=2)
    net.decoder = BaseVAE.build_decoder(feature_dim=len(df.columns), latent_dim=2)
    net.fit(df, latent_dim=2, epochs=10, learning_rate=0.01, batch_size=16)
    # Save artifacts
    net.save("net_vae_model")

    model: NetVAE = BaseVAE.open_model(path="net_vae_model", model_cls=NetVAE, device="cpu")
    print("Model loaded with features:", model.features)
    print(model.decoder)
