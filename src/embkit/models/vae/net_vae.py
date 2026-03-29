"""
NetVAE implementation
"""
import logging

from typing import Dict, List, Optional, Callable, Union
import numpy as np
import pandas as pd
import torch

from .base_vae import BaseVAE
from .encoder import Encoder
from .decoder import Decoder
from ... import factory
from ...optimize import fit_net_vae

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# NetVae (training with optional alternating constraint)
# ---------------------------------------------------------

@factory.nn_module
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

    def to_dict(self):
        return {
            "features": self.features,
            "encoder": self.encoder.to_dict() if self.encoder else None,
            "decoder": self.decoder.to_dict() if self.decoder else None,
            "latent_index": self.latent_index,
            "latent_groups": self.latent_groups,
        }

    @classmethod
    def from_dict(cls, d):
        model = NetVAE(
            features=d["features"],
            encoder=Encoder.from_dict(d["encoder"]) if d.get("encoder") else None,
            decoder=Decoder.from_dict(d["decoder"]) if d.get("decoder") else None,
        )
        model.latent_index = d.get("latent_index")
        model.latent_groups = d.get("latent_groups")
        return model

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
        fit_net_vae(
            model=self,
            X=X,
            latent_dim=latent_dim,
            latent_index=latent_index,
            latent_groups=latent_groups,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            phases=phases,
            device=device,
            grouping_fn=grouping_fn,
        )


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
    from ...factory import save, load

    save(net, "net_vae_model")

    model: NetVAE = load("net_vae_model", device="cpu")
    print("Model loaded with features:", model.features)
    print(model.decoder)
