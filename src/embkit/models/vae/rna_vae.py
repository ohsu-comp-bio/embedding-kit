"""
RNAVAE - RNA-specific Variational Autoencoder

Converted from TensorFlow/Keras implementation to PyTorch using BaseVAE.
Features beta warmup for KL divergence annealing.
"""

import logging
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import pandas as pd
from tqdm.autonotebook import tqdm
from torch.optim import Adam
from torch import nn

from .base_vae import BaseVAE
from ...layers import LayerInfo
from ... import get_device, dataframe_loader

logger = logging.getLogger(__name__)


class RNAVAE(BaseVAE):
    """
    RNA-specific VAE with beta warmup schedule.
    
    Architecture matches the original TensorFlow implementation:
    - Encoder: feature_dim -> feature_dim//2 -> feature_dim//3 -> latent_dim
    - Decoder: latent_dim -> feature_dim
    - Beta warmup: Gradually increase KL weight from 0 to 1
    """

    def __init__(
            self,
            features: List[str],
            latent_dim: int = 768,
            lr: float = 0.0005,
            batch_norm: bool = True,
    ):
        """
        Args:
            features: list[str] feature names (len(features) == input_dim)
            latent_dim: size of latent space (default: 768 from original)
            lr: learning rate (default: 0.0005 from original)
            batch_norm: use batch normalization (default: True from original)
        """
        super().__init__(features=features)
        self.lr = lr
        self.latent_dim = latent_dim
        self._batch_norm = batch_norm

        feature_dim = len(features)

        # Build encoder: feature_dim -> feature_dim//2 -> feature_dim//3 -> latent_dim
        enc_layers = [
            LayerInfo(units=feature_dim // 2, activation="relu"),
            LayerInfo(units=feature_dim // 3, activation="relu"),
            LayerInfo(units=latent_dim, activation="relu", batch_norm=batch_norm),
        ]
        
        self.encoder = self.build_encoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=enc_layers,
            batch_norm=False,  # We add BN in the layers themselves
        )

        # Build decoder: latent_dim -> feature_dim with sigmoid activation
        dec_layers = [
            LayerInfo(units=feature_dim, activation="sigmoid"),
        ]
        
        self.decoder = self.build_decoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            layers=dec_layers,
        )

        # History tracking
        self.history: Dict[str, list] = {"loss": [], "recon": [], "kl": [], "beta": []}

    def fit(
            self,
            X: Union[pd.DataFrame, torch.Tensor, torch.utils.data.DataLoader],
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
                    "Input DataFrame columns do not match model features.\n"
                    f"Data columns: {list(X.columns)[:5]}... (n={len(X.columns)})\n"
                    f"Model features: {self.features[:5]}... (n={len(self.features)})"
                )

        # Build dataloader
        if isinstance(X, pd.DataFrame):
            data_loader = dataframe_loader(X, batch_size=batch_size, device=device, shuffle=True)
        else:
            data_loader = X

        # Optimizer
        optimizer = Adam(self.parameters(), lr=self.lr)

        # Beta warmup setup
        beta = 0.0

        # Early stopping setup
        best_loss = float('inf')
        patience_counter = 0

        # Training loop
        epoch_bar = tqdm(range(epochs), disable=not progress, desc="Training RNAVAE")
        
        for epoch in epoch_bar:
            epoch_loss_sum = 0.0
            epoch_recon_sum = 0.0
            epoch_kl_sum = 0.0
            epoch_batches = 0

            for (x_tensor,) in data_loader:
                optimizer.zero_grad(set_to_none=True)
                x_tensor = x_tensor.to(device).float()

                # Forward pass
                recon, mu, logvar, z = self(x_tensor)

                # Compute loss with current beta
                total_loss, recon_loss, kl_loss = self._rna_vae_loss(
                    recon, x_tensor, mu, logvar, beta=beta
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
            if epoch_batches > 0:
                ep_loss = epoch_loss_sum / epoch_batches
                ep_recon = epoch_recon_sum / epoch_batches
                ep_kl = epoch_kl_sum / epoch_batches
                
                self.history["loss"].append(ep_loss)
                self.history["recon"].append(ep_recon)
                self.history["kl"].append(ep_kl)
                self.history["beta"].append(beta)

                # Update progress bar
                if progress:
                    epoch_bar.set_postfix(
                        loss=f"{ep_loss:.3f}",
                        recon=f"{ep_recon:.3f}",
                        kl=f"{ep_kl:.3f}",
                        beta=f"{beta:.3f}"
                    )

                # Early stopping check
                if ep_loss < best_loss:
                    best_loss = ep_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

            # Beta warmup: increase beta each epoch until it reaches 1
            beta = min(beta + kappa, 1.0)

        return self.history

    def _rna_vae_loss(
            self,
            recon: torch.Tensor,
            x: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            beta: float = 1.0
    ):
        """
        RNA VAE loss matching the original TensorFlow implementation.
        
        Loss = feature_dim * BCE + 5 * beta * KL
        
        Args:
            recon: Reconstructed data
            x: Original data
            mu: Latent mean
            logvar: Latent log variance
            beta: KL weight (warmup parameter)
        
        Returns:
            (total_loss, reconstruction_loss, kl_loss)
        """
        feature_dim = x.size(1)
        
        # Binary cross-entropy per sample
        bce_per_sample = nn.functional.binary_cross_entropy(
            recon, x, reduction='none'
        ).mean(dim=1)
        
        # Scale by feature dimension (matches original TensorFlow implementation)
        reconstruction_loss = feature_dim * bce_per_sample
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        )
        
        # Total loss with beta warmup and scaling factor of 5 (from original)
        total_loss = (reconstruction_loss + 5 * beta * kl_loss).mean()
        
        return total_loss, reconstruction_loss.mean(), kl_loss.mean()


if __name__ == "__main__":
    # Example usage matching the original script
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    
    # Create sample data
    N = 1000
    feature_dim = 100
    
    df = pd.DataFrame(
        np.random.rand(N, feature_dim),
        columns=[f"gene_{i}" for i in range(feature_dim)]
    )
    
    # Scale data (as in original)
    df_scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(df),
        columns=df.columns
    )
    
    # Create and train model
    rna_vae = RNAVAE(
        features=df_scaled.columns.tolist(),
        latent_dim=768,
        lr=0.0005
    )
    
    # Train with beta warmup
    history = rna_vae.fit(
        df_scaled,
        epochs=50,
        batch_size=512,
        kappa=1.0,  # Beta warmup rate
        early_stopping_patience=3,
        progress=True
    )
    
    # Generate embeddings
    rna_vae.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(df_scaled.values, dtype=torch.float32)
        mu, logvar, z = rna_vae.encoder(X_tensor)
        embeddings = pd.DataFrame(mu.cpu().numpy())
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Final loss: {history['loss'][-1]:.4f}")
