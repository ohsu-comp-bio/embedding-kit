from torch import nn
import torch

class Decoder(nn.Module):
    """
    Mirrors build_decoder (Dense with relu), but returns probabilities in [0,1] to
    match the binary cross-entropy use in the original loss pipeline.
    """

    def __init__(self, latent_dim: int, feature_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, feature_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Keep it simple: return sigmoid outputs for BCE
        return torch.sigmoid(self.fc(z))

