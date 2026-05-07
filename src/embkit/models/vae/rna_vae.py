"""
RNAVAE - RNA-specific Variational Autoencoder

Integrated version using BaseVAE infrastructure while preserving exact
TensorFlow architecture with BatchNorm and ReLU on latent heads.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any
import torch
import pandas as pd
from torch.optim import Adam
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from .base_vae import BaseVAE
from .encoder import Encoder
from ...factory.layers import Layer, LayerList
from ... import get_device
from ...losses import bce_kl_weighted
from ... import factory

logger = logging.getLogger(__name__)


class RNAEncoder(Encoder):
    """
    Extended Encoder for RNA VAE that adds BatchNorm + ReLU to latent heads.
    
    WHY THIS EXISTS:
    The standard Encoder produces latent heads as: mu = Linear(h), logvar = Linear(h)
    This allows mu and logvar to be any real number (standard VAE practice).
    
    Your TensorFlow RNA VAE uses: mu = ReLU(BatchNorm(Linear(h)))
    This constrains mu and logvar to be non-negative (≥ 0), fundamentally changing
    the latent space behavior. Without this custom encoder, the PyTorch model would
    produce mathematically different embeddings than your TensorFlow model.
    
    Architecture:
    - z_mean: Linear -> BatchNorm -> ReLU (NOT standard VAE)
    - z_log_var: Linear -> BatchNorm -> ReLU (NOT standard VAE)
    """
    
    def __init__(self, feature_dim: int, latent_dim: int, 
                 layers: Optional[List[Layer]] = None,
                 batch_norm: bool = False):
        # Initialize parent without making latent heads
        super().__init__(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=layers,
            batch_norm=batch_norm,
            make_latent_heads=False  # We'll build custom ones
        )
        
        # Build custom latent heads with BatchNorm + ReLU
        # Linear -> BatchNorm -> ReLU (matching TensorFlow)
        self.z_mean_linear = nn.Linear(self._final_width, latent_dim)
        self.z_mean_bn = nn.BatchNorm1d(latent_dim)
        
        self.z_log_var_linear = nn.Linear(self._final_width, latent_dim)
        self.z_log_var_bn = nn.BatchNorm1d(latent_dim)
        
        # Xavier/Glorot uniform initialization (TensorFlow default)
        nn.init.xavier_uniform_(self.z_mean_linear.weight)
        nn.init.zeros_(self.z_mean_linear.bias)
        nn.init.xavier_uniform_(self.z_log_var_linear.weight)
        nn.init.zeros_(self.z_log_var_linear.bias)
    
    def forward(self, x: torch.Tensor):
        # Pass through main network
        h = x
        for layer in self.net:
            h = layer(h)
        
        # z_mean: Linear -> BatchNorm -> ReLU
        mu = self.z_mean_linear(h)
        mu = self.z_mean_bn(mu)
        mu = torch.relu(mu)
        
        # z_log_var: Linear -> BatchNorm -> ReLU  
        logvar = self.z_log_var_linear(h)
        logvar = self.z_log_var_bn(logvar)
        logvar = torch.relu(logvar)
        
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return mu, logvar, z


@factory.nn_module
class RNAVAE(BaseVAE):
    """    
    Architecture:
    - Encoder: feature_dim -> feature_dim//2 -> feature_dim//3 -> latent_dim
      - Latent heads: Linear -> BatchNorm -> ReLU
    - Decoder: latent_dim -> feature_dim (sigmoid)
    - Loss: feature_dim * BCE + 5 * beta * KL
    - Beta warmup: 0 -> 1 (kappa rate per epoch)
    """

    def __init__(
            self,
            features: List[str],
            latent_dim: int = 768,
            lr: float = 0.0005,
    ):
        super().__init__(features=features)
        self.lr = lr
        self.latent_dim = latent_dim

        feature_dim = len(features)

        # Build encoder: feature_dim -> feature_dim//2 -> feature_dim//3
        enc_layers = [
            Layer(units=feature_dim // 2, activation="relu"),
            Layer(units=feature_dim // 3, activation="relu"),
        ]
        
        # Use custom RNAEncoder with BatchNorm + ReLU on latent heads
        self.encoder = RNAEncoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=LayerList(enc_layers),
            batch_norm=False  # We add BN to latent heads specifically
        )

        # Build decoder: latent_dim -> feature_dim with sigmoid
        dec_layers = [
            Layer(units=feature_dim, activation="sigmoid"),
        ]
        
        self.decoder = self.build_decoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=LayerList(dec_layers),
        )
        
        # Initialize weights with Xavier/Glorot (TensorFlow default)
        self._initialize_weights()

        # History tracking
        self.history: Dict[str, list] = {"loss": [], "recon": [], "kl": [], "beta": []}

    def _initialize_weights(self):
        """Initialize weights with glorot_uniform like TensorFlow"""
        for m in self.modules():
            if isinstance(m, nn.Linear) and not hasattr(m, '_initialized'):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                m._initialized = True

    def forward(self, x: torch.Tensor):
        """Standard VAE forward pass"""
        mu, logvar, z = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def fit(
            self,
            X: Union[pd.DataFrame, torch.Tensor],
            epochs: int = 100,
            batch_size: int = 512,
            kappa: float = 1.0,
            early_stopping_patience: int = 3,
            device: Optional[torch.device] = None,
            progress: bool = True,
    ):
        """
        Train the RNA VAE with beta warmup.
        
        Args:
            X: Input data (DataFrame or Tensor)
            epochs: Number of training epochs
            batch_size: Batch size for training
            kappa: Beta warmup rate (beta increases by kappa each epoch)
            early_stopping_patience: Stop if loss doesn't improve for this many epochs
            device: Device to use ('cuda', 'mps', or 'cpu')
            progress: Show progress bar
        """
        # Setup device
        if device is None:
            device = get_device()
        
        self.to(device)
        self.train()

        # Column alignment safety check
        if hasattr(X, "columns") and self.features is not None:
            if list(X.columns) != list(self.features):
                raise ValueError(
                    f"Input DataFrame columns do not match model features.\n"
                    f"Data columns: {list(X.columns)[:5]}... (n={len(X.columns)})\n"
                    f"Model features: {self.features[:5]}... (n={len(self.features)})"
                )

        # Convert to tensor
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.tensor(X.to_numpy(dtype="float32", copy=True), dtype=torch.float32, device=device)
        else:
            X_tensor = X.to(device)

        # Build dataloader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = Adam(self.parameters(), lr=self.lr)

        # Beta warmup and early stopping
        beta = 0.0
        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            # Beta warmup
            beta = min(beta + kappa, 1.0)
            
            # Train epoch
            epoch_loss_sum = 0.0
            epoch_recon_sum = 0.0
            epoch_kl_sum = 0.0
            epoch_batches = 0

            for (batch_x,) in dataloader:
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                recon, mu, logvar, z = self(batch_x)

                # Compute loss with current beta and kl_weight=5.0 (RNA VAE specific)
                total_loss, recon_loss, kl_loss = bce_kl_weighted(
                    recon, batch_x, mu, logvar, beta=beta, kl_weight=5.0
                )

                # Backprop
                total_loss.backward()
                optimizer.step()

                # Accumulate stats
                epoch_loss_sum += float(total_loss.detach().cpu())
                epoch_recon_sum += float(recon_loss.detach().cpu())
                epoch_kl_sum += float(kl_loss.detach().cpu())
                epoch_batches += 1

            # Compute epoch means
            ep_loss = epoch_loss_sum / epoch_batches
            ep_recon = epoch_recon_sum / epoch_batches
            ep_kl = epoch_kl_sum / epoch_batches
            
            self.history["loss"].append(ep_loss)
            self.history["recon"].append(ep_recon)
            self.history["kl"].append(ep_kl)
            self.history["beta"].append(beta)

            logger.info("Epoch %d/%d - loss: %.4f - beta: %.4f", epoch + 1, epochs, ep_loss, beta)

            # Early stopping check
            if ep_loss < best_loss:
                best_loss = ep_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    if best_state is not None:
                        self.load_state_dict(best_state)
                    break

        training_time = time.time() - start_time
        logger.info("Training completed in %.2f seconds", training_time)

        return self.history

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Specific check for RNAVAE to ensure the BatchNorm+ReLU latent heads
        are producing strictly non-negative mu and logvar.
        """
        report = super().verify_integrity()

        self.eval()
        device = next(self.parameters()).device
        dummy_input = torch.randn(100, len(self.features), device=device)
        with torch.no_grad():
            mu, logvar, _ = self.encoder(dummy_input)

            min_mu = float(torch.min(mu))
            min_logvar = float(torch.min(logvar))

            # Strict numerical tolerance for integrity mode
            is_non_negative = (min_mu >= -1e-9 and min_logvar >= -1e-9)

            report["rna_diagnostics"] = {
                "min_mu": min_mu,
                "min_logvar": min_logvar,
                "is_non_negative": is_non_negative
            }

            if not is_non_negative:
                report["healthy"] = False
                report["issues"].append(
                    f"Negative values detected in RNAVAE latent heads "
                    f"(mu_min={min_mu:.2e}, logvar_min={min_logvar:.2e}). "
                    f"Architectural constraints violated (ReLU/BatchNorm bypass)."
                )

        return report

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "latent_dim": self.latent_dim,
            "lr": self.lr,
            "history": getattr(self, "history", {}) or {}
        }

    @classmethod
    def from_dict(cls, d):
        model = RNAVAE(
            features=d["features"],
            latent_dim=d.get("latent_dim", 768),
            lr=d.get("lr", 0.0005),
        )
        model.history = d.get("history") or {}
        return model
