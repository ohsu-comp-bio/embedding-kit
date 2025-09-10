import logging
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import pandas as pd
from tqdm.autonotebook import tqdm
from torch.optim import Adam
from ...layers import LayerInfo
from .base_vae import BaseVAE
from ...losses import bce_with_logits
from ... import get_device, dataframe_loader

# If Encoder/Decoder kwargs need constraints etc., they’ll be passed through.

logger = logging.getLogger(__name__)


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
            encoder_layers: Optional[List[LayerInfo]] = None,
            decoder_layers: Optional[List[LayerInfo]] = None,
            constraint=None,
            batch_norm: bool = False,
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
            )
            self.decoder = self.build_decoder(
                feature_dim=feature_dim,
                latent_dim=latent_dim,
                layers=decoder_layers
            )

        # A place to record simple history if you want
        self.history: Dict[str, list] = {"loss": [], "recon": [], "kl": []}
        self.latent_index = None
        self.latent_groups = None
        self.normal_stats = None

    def fit(self, X: Union[pd.DataFrame, torch.utils.data.DataLoader],
            y=None,
            *,
            epochs: int = 20,
            lr: Optional[float] = None,
            beta: float = 1.0,
            optimizer: Optional[torch.optim.Optimizer] = None,
            reset_optimizer: bool = False,
            device: Optional[torch.device] = None,
            progress: bool = True,
            beta_schedule: Optional[List[tuple]] = None):
        """
        Training loop using vae_loss(recon, x, mu, logvar).

        X: pandas.DataFrame with float features, columns must match `self.features`.
        If `beta_schedule` is provided as a list of (beta, epochs) pairs, it overrides
        the single-phase (beta, epochs) arguments and runs multiple phases while
        reusing the same optimizer/momentum.
        """
        # --- setup ---
        if lr is None:
            lr = self.lr
        if device is None:
            device = get_device()

        # Column alignment safety check if a DataFrame is passed
        if hasattr(X, "columns") and self.features is not None:
            if list(X.columns) != list(self.features):
                raise ValueError(
                    "Input DataFrame columns do not match model features.\n"
                    f"Data columns: {list(X.columns)[:5]}... (n={len(X.columns)})\n"
                    f"Model features: {self.features[:5]}... (n={len(self.features)})"
                )

        self.to(device)
        self.train()

        # Build dataloader once
        if isinstance(X, pd.DataFrame):
            data_loader = dataframe_loader(X, device=device)  # ensure it shuffles in training mode
        else:
            data_loader = X

        # --- persistent optimizer (reuse momentum/velocity across phases) ---
        if optimizer is not None:
            self._optimizer = optimizer
        elif reset_optimizer or not hasattr(self, "_optimizer") or self._optimizer is None:
            self._optimizer = Adam(self.parameters(), lr=lr)
        else:
            # Reuse existing optimizer but refresh LR if changed
            for g in self._optimizer.param_groups:
                g["lr"] = lr
        opt = self._optimizer

        # --- one simple epoch runner (PyTorch-idiomatic) ---
        def run_epochs(n_epochs: int, beta_value: float) -> float:
            running_loss = 0.0
            last_loss = 0.0

            epoch_iter = tqdm(range(n_epochs)) if progress else range(n_epochs)
            for epoch_idx in epoch_iter:
                epoch_loss_sum = 0.0
                epoch_recon_sum = 0.0
                epoch_kl_sum = 0.0
                epoch_batches = 0

                for i, (x_tensor,) in enumerate(data_loader):
                    # Zero your gradients for every batch
                    opt.zero_grad(set_to_none=True)

                    # Move/ensure dtype
                    x_tensor = x_tensor.to(device).float()

                    # Forward: recon, mu, logvar, z
                    recon, mu, logvar, _ = self(x_tensor)

                    # Compute loss components
                    total_loss, recon_loss, kl_loss = bce_with_logits(recon, x_tensor, mu, logvar,
                                                                              beta=beta_value)

                    # Backward + step
                    total_loss.backward()
                    opt.step()

                    # Intra-epoch reporting-style accumulation
                    running_loss += float(total_loss.detach().cpu())
                    if (i + 1) % 100 == 0:
                        last_loss = running_loss / 100.0
                        if progress:
                            epoch_iter.set_description(
                                f"Epoch {epoch_idx + 1} | Loss {last_loss:.4f} | β={beta_value}"
                            )
                        running_loss = 0.0

                    # Epoch means
                    epoch_loss_sum += float(total_loss.detach().cpu())
                    epoch_recon_sum += float(recon_loss.detach().cpu())
                    epoch_kl_sum += float(kl_loss.detach().cpu())
                    epoch_batches += 1

                # Store epoch-mean stats in history
                if epoch_batches > 0:
                    self.history["loss"].append(epoch_loss_sum / epoch_batches)
                    self.history["recon"].append(epoch_recon_sum / epoch_batches)
                    self.history["kl"].append(epoch_kl_sum / epoch_batches)

            return last_loss

        # --- single phase or multi-phase (beta schedule) ---
        if beta_schedule is None:
            return run_epochs(epochs, beta)
        else:
            last = 0.0
            for beta_value, n_epochs in beta_schedule:
                last = run_epochs(n_epochs, beta_value)
            return last


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
