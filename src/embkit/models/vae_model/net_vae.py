from typing import Dict, List, Optional, Callable, Union
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
import logging
from .base_vae import VAE
from .encoder import Encoder
from .decoder import Decoder
from ...losses import net_vae_loss
from src.embkit.constraints import NetworkConstraint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# NetVae (training with optional alternating constraint)
# ---------------------------------------------------------

class NetVae(VAE):
    """
    """

    def __init__(self, features: List[str], encoder: Optional[Encoder] = None, decoder: Optional[Decoder] = None):
        super().__init__(features=features, encoder=encoder, decoder=decoder)
        self.latent_groups: Optional[Dict[str, List[str]]] = None
        self.latent_index: Optional[List[str]] = None
        self.history: Optional[Dict[str, List[float]]] = None
        self.normal_stats: Optional[pd.DataFrame] = None

    def fit(self, X: Union[pd.DataFrame | torch.Tensor], **kwargs):
        """
        Override train() to ensure we use the custom run_train() method.
        """
        for epoch in range(100):
            net.run_train(
                df=df,
                latent_index=["latent1", "latent2"],  # latent_dim = 2
                learning_rate=1e-3,
                batch_size=32,
                epochs=10,
                device="cpu",
                phases=None,
            )
            # print model training stats
            print(
                f"Epoch {epoch + 1}: Loss={net.history['loss'][-1]:.4f}, Recon Loss={net.history['reconstruction_loss'][-1]:.4f}, KL Loss={net.history['kl_loss'][-1]:.4f}")

    def run_train(self, df: pd.DataFrame, latent_index: List[str],
                  latent_groups: Optional[Dict[str, List[str]]] = None,
                  learning_rate: float = 1e-3, batch_size: int = 128,
                  phases: Optional[List[int]] = None, epochs: int = 80,
                  device: Optional[str] = None,
                  grouping_fn: Optional[Callable[[np.ndarray, List[str]], Dict[str, List[str]]]] = None):

        feature_dim = len(df.columns)
        latent_dim = len(latent_index)
        self.latent_index = list(latent_index)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        # Constraint + models
        constraint = NetworkConstraint(list(df.columns), latent_index, latent_groups)
        constraint.set_active(False)

        # Build and attach to self (no second VAE instance)
        encoder = VAE.build_encoder(feature_dim=feature_dim, latent_dim=latent_dim, constraint=constraint)
        decoder = VAE.build_decoder(feature_dim, latent_dim)
        self.encoder = encoder
        self.decoder = decoder

        # Move whole module to device
        self.to(device)

        # Data
        x = torch.tensor(df.values, dtype=torch.float32, device=device)
        loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)

        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # History
        total_hist: List[float] = []
        recon_hist: List[float] = []
        kl_hist: List[float] = []

        # Determine total epochs if phases provided
        if phases is not None:
            total_epochs = sum(phases)
            boundaries = np.cumsum(phases).tolist()
        else:
            total_epochs = epochs
            boundaries = []

        # Helper: apply constraint mask to encoder
        def refresh_mask():
            # encoder is attached to self
            self.encoder.refresh_mask(device)

        # Phase control
        def start_constrained_phase():
            if grouping_fn is not None:
                with torch.no_grad():
                    w = self.encoder.pathway.linear.weight.detach().cpu().numpy()  # [latent, features]
                    new_groups = grouping_fn(w, list(df.columns))
                    constraint.update_membership(new_groups)
            constraint.set_active(True)
            refresh_mask()

        def start_unconstrained_phase():
            constraint.set_active(False)
            refresh_mask()

        # Initial mask
        refresh_mask()

        # Training loop
        for epoch in range(total_epochs):
            if boundaries and epoch in boundaries:
                idx = boundaries.index(epoch)
                if idx % 2 == 0:
                    start_constrained_phase()
                else:
                    start_unconstrained_phase()

            self.train()
            epoch_tot = epoch_rec = epoch_kl = 0.0
            n_batches = 0

            for (batch_x,) in loader:
                opt.zero_grad()
                # assumes vae_loss_from_model(model, x) calls model.forward(x)
                total, recon, kl = net_vae_loss(self, batch_x)
                total.backward()
                opt.step()
                epoch_tot += float(total.item())
                epoch_rec += float(recon.item())
                epoch_kl += float(kl.item())
                n_batches += 1

            total_hist.append(epoch_tot / max(1, n_batches))
            recon_hist.append(epoch_rec / max(1, n_batches))
            kl_hist.append(epoch_kl / max(1, n_batches))

        # Store artifacts
        self.latent_groups = constraint.latent_membership
        self.history = {
            "loss": total_hist,
            "reconstruction_loss": recon_hist,
            "kl_loss": kl_hist,
        }

        # normal_stats using deterministic recon from μ
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
    net = NetVae(features=list(df.columns))
    net.train()
    # Save artifacts
    net.save("vae_model")

    model: NetVae = VAE.open_model(path="vae_model", model_cls=NetVae, device="cpu")
    print("Model loaded with features:", model.features)
    print(model.decoder)
