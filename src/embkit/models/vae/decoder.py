"""
Decoder for the VAE.

The decoder is the inverse of the encoder: it maps a latent vector back to the
input feature space, producing the model's reconstruction of a sample. Requiring
the decoder to faithfully reconstruct the input is what forces the encoder's
latent representation to actually carry the information needed to describe each
sample — without it, the latent space would collapse to nothing useful.
"""
from typing import List, Optional
import torch
from torch import nn

from ... import factory
from ...modules import MaskedLinear
from ...factory.layers import Layer, LayerList
import logging

logger = logging.getLogger(__name__)


def _module_out_features(module: nn.Module) -> Optional[int]:
    """
    Report a module's output width, if it has one (see the same helper in
    :mod:`embkit.models.vae.encoder`).
    """
    if isinstance(module, MaskedLinear):
        return int(module.linear.out_features)
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    return None


@factory.nn_module
class Decoder(nn.Module):
    """
    Fully-connected decoder that maps a latent vector back to input features.

    Pipeline: ``z`` -> a stack of linear layers -> reconstruction of the input
    features. Built from a :class:`~embkit.factory.layers.LayerList`; if the
    stack does not already end at ``feature_dim``, a final projection is added
    so the output always has the input feature width.

    The output of :meth:`forward` is the reconstruction tensor, which the
    training loss compares against the original input to drive learning.
    """

    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        layers: Optional[LayerList] = None,
        batch_norm: bool = False,
        device=None, dtype=None
    ):
        """
        Build the decoder body, ensuring the output width equals ``feature_dim``.

        Parameters:
            latent_dim: Width of the input latent space.
            feature_dim: Width of the reconstructed output (input feature count).
            layers: Optional layer stack describing the decoder body; if the
                stack does not end at ``feature_dim`` a final projection is
                appended.
            batch_norm: Whether batch normalization is enabled for the decoder.
            device, dtype: Device/dtype for module construction.
        """
        super().__init__()
        self.latent_dim = int(latent_dim)  # <- help BaseVAE.save()
        self.feature_dim = int(feature_dim)
        self._global_bn = batch_norm
        self._layers_cfg = layers
        self.net = nn.ModuleList()

        in_features = latent_dim

        if layers:
            logger.info("Building decoder with %d layers", len(layers))
            dec_net = layers.build( latent_dim, feature_dim, device=device, dtype=dtype )
            self.net.extend(dec_net)
            in_features = self.feature_dim
            for module in reversed(dec_net):
                width = _module_out_features(module)
                if width is not None:
                    in_features = width
                    break
            logger.debug("Decoder net built with %d modules", len(dec_net))
            
        # Final projection to feature_dim if not already there
        if in_features != self.feature_dim or not layers:
            logger.info("Adding final projection layer to %d units", self.feature_dim)
            self.out = nn.Linear(in_features, self.feature_dim, device=device, dtype=dtype)
            self.net.append(self.out)
        else:
            # If the last layer matches feature_dim, identify it as self.out
            # assuming the last layer in self.net that is a Linear/MaskedLinear is 'out'
            # But the tests seem to expect a dedicated .out attribute
            # Let's find the last Linear-like layer if it exists
            linear_layers = [
                m for m in self.net if isinstance(m, (nn.Linear, MaskedLinear))
            ]
            if linear_layers:
                self.out = linear_layers[-1]
            else:
                # Fallback: create identity or just set it
                self.out = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent vector back to the input feature space.

        Parameters:
            z: Latent vector(s) of shape ``(batch_size, latent_dim)``.

        Returns:
            The reconstruction tensor of shape ``(batch_size, feature_dim)``.
        """
        h = z
        for layer in self.net:
            h = layer(h)
        return h

    def to_dict(self):
        """
        Serialize the decoder's configuration (architecture, not weights) so
        that :meth:`from_dict` can rebuild an equivalent decoder.
        """
        return {
            "latent_dim": self.latent_dim,
            "feature_dim": self.feature_dim,
            "batch_norm": self._global_bn,
            "layers": [li.to_dict() for li in self._layers_cfg] if self._layers_cfg else []
        }

    @classmethod
    def from_dict(cls, d):
        """
        Rebuild a :class:`Decoder` from its :meth:`to_dict` description.
        """
        layers = [Layer.from_dict(ld) for ld in d.get("layers", [])]
        return Decoder(
            latent_dim=d["latent_dim"],
            feature_dim=d["feature_dim"],
            batch_norm=d.get("batch_norm", False),
            layers=LayerList(layers) if layers else None
        )
