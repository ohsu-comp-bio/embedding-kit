import logging
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import pandas as pd
from tqdm.autonotebook import tqdm
from torch.optim import Adam
from ...layers import LayerInfo
from .base_vae import BaseVAE
from collections.abc import Callable
from ... import get_device, dataframe_loader
from torch import nn

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
            encoder_layers: Optional[List[LayerInfo]] = None,
            decoder_layers: Optional[List[LayerInfo]] = None,
            batch_norm: bool = False,
            lr: float = 1e-3,
            encoder: Optional[nn.Module] = None,
            decoder: Optional[nn.Module] = None,
            device= None
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
        self._encoder_layers_cfg = list(encoder_layers or [])
        self._decoder_layers_cfg = list(decoder_layers or [])
        self._batch_norm = batch_norm

        feature_dim = len(features)

        if encoder is not None and decoder is not None:
            # Loaded path (from BaseVAE.open_model): modules are already built & weight-loaded
            self.encoder = encoder
            self.decoder = decoder
        else:
            # Fresh build path
            if latent_dim is None:
                raise ValueError("latent_dim is required when encoder/decoder are not provided.")
            self.encoder = self.build_encoder(
                feature_dim=feature_dim,
                latent_dim=latent_dim,
                layers=encoder_layers,
                batch_norm=batch_norm,
                device=device,
            )
            self.decoder = self.build_decoder(
                feature_dim=feature_dim,
                latent_dim=latent_dim,
                layers=decoder_layers,
                device=device,
            )

        # A place to record simple history if you want
        self.history: Dict[str, list] = {"loss": [], "recon": [], "kl": []}
        self.latent_index = None
        self.latent_groups = None
        self.normal_stats = None

    def fit(self, X: Union[pd.DataFrame, torch.Tensor, torch.utils.data.DataLoader], **kwargs):
        """
        Training loop using vae_loss(recon, x, mu, logvar).

        X: pandas.DataFrame with float features, columns must match `self.features`.
        If `beta_schedule` is provided as a list of (beta, epochs) pairs, it overrides
        the single-phase (beta, epochs) arguments and runs multiple phases while
        reusing the same optimizer/momentum.
        """

        epochs: int = int(kwargs.pop("epochs", 20))
        lr: Optional[float] = kwargs.pop("lr", None)
        beta: float = float(kwargs.pop("beta", 1.0))
        optimizer: Optional[torch.optim.Optimizer] = kwargs.pop("optimizer", None)
        loss: Optional[Callable] = kwargs.pop("loss", None)
        reset_optimizer: bool = bool(kwargs.pop("reset_optimizer", False))
        device: Optional[torch.device] = kwargs.pop("device", None)
        progress: bool = bool(kwargs.pop("progress", True))
        beta_schedule = kwargs.pop("beta_schedule", None)
        y = kwargs.pop("y", None)  # if you need it, fetch it from kwargs

        if loss is None:
            raise ValueError("loss function is required (e.g., from embkit.losses.vae_loss)")

        # --- setup ---
        if lr is None:
            lr = self.lr
        if device is None:
            device = get_device()

        if beta_schedule is not None:
            logger.info(f"Using beta_schedule: {beta_schedule}")

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

        # --- epoch runner (epoch-only progress) ---
        def run_epochs(n_epochs: int, beta_value: float) -> float:
            last_loss = 0.0
            epoch_bar = tqdm(range(n_epochs), disable=not progress, desc=f"β={beta_value:.2f}")
            for epoch_idx in epoch_bar:
                epoch_loss_sum = 0.0
                epoch_recon_sum = 0.0
                epoch_kl_sum = 0.0
                epoch_batches = 0

                for (x_tensor,) in data_loader:
                    opt.zero_grad(set_to_none=True)
                    x_tensor = x_tensor.to(device).float()

                    # Forward
                    recon, mu, logvar, _ = self(x_tensor)

                    total_loss, recon_loss, kl_loss = loss(recon, x_tensor, mu, logvar, beta=beta_value)

                    # Backprop
                    total_loss.backward()
                    opt.step()

                    # Accumulate epoch stats
                    tl = float(total_loss.detach().cpu())
                    epoch_loss_sum += tl
                    epoch_recon_sum += float(recon_loss.detach().cpu())
                    epoch_kl_sum += float(kl_loss.detach().cpu())
                    epoch_batches += 1
                    last_loss = tl

                # Compute epoch means
                if epoch_batches > 0:
                    ep_loss = epoch_loss_sum / epoch_batches
                    ep_recon = epoch_recon_sum / epoch_batches
                    ep_kl = epoch_kl_sum / epoch_batches
                    self.history["loss"].append(ep_loss)
                    self.history["recon"].append(ep_recon)
                    self.history["kl"].append(ep_kl)

                    # Update the epoch progress bar once per epoch (no jitter)
                    if progress:
                        epoch_bar.set_postfix(loss=f"{ep_loss:.3f}",
                                              recon=f"{ep_recon:.3f}",
                                              kl=f"{ep_kl:.3f}")

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
