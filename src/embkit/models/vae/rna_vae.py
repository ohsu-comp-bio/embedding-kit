"""
RNAVAE - RNA-specific Variational Autoencoder

Integrated version using BaseVAE infrastructure while preserving exact
TensorFlow architecture with BatchNorm and ReLU on latent heads.
"""

import logging
import time
from typing import Dict, List, Optional, Union
import torch
import pandas as pd
from torch.optim import Adam
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from .base_vae import BaseVAE
from .encoder import Encoder
from ...layers import LayerInfo
from ... import get_device
from ...losses import bce_kl_weighted

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
                 layers: Optional[List[LayerInfo]] = None,
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
            LayerInfo(units=feature_dim // 2, activation="relu"),
            LayerInfo(units=feature_dim // 3, activation="relu"),
        ]
        
        # Use custom RNAEncoder with BatchNorm + ReLU on latent heads
        self.encoder = RNAEncoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=enc_layers,
            batch_norm=False  # We add BN to latent heads specifically
        )

        # Build decoder: latent_dim -> feature_dim with sigmoid
        dec_layers = [
            LayerInfo(units=feature_dim, activation="sigmoid"),
        ]
        
        self.decoder = self.build_decoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=dec_layers,
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
            X_tensor = torch.FloatTensor(X.values).to(device)
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

            # Print like Keras verbose=1
            print(f'Epoch {epoch+1}/{epochs} - loss: {ep_loss:.4f} - beta: {beta:.4f}')

            # Early stopping check
            if ep_loss < best_loss:
                best_loss = ep_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                    if best_state is not None:
                        self.load_state_dict(best_state)
                    break

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return self.history
