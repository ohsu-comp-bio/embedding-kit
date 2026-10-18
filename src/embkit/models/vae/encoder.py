"""
Encoder components for the VAE.

The encoder is the half of a VAE that compresses a sample's input features into
the latent space: it produces the mean (``mu``) and log-variance (``logvar``)
of the latent distribution, and a sampled latent vector ``z``. A good encoder is
what makes the downstream embedding meaningful — it decides which structure in
the input is preserved in the compact representation.

This module provides two encoders:

- :class:`VAEEncoder` — a general wrapper that takes an arbitrary backbone
  network, adds the latent mean/log-variance heads, and computes per-unit KL
  terms.
- :class:`Encoder` — the concrete fully-connected encoder used by the standard
  embedding VAE (:class:`~embkit.models.vae.vae.BaseVAE`), built from a
  :class:`~embkit.factory.layers.LayerList` and supporting optional
  connection constraints (used by
  :class:`~embkit.models.vae.net_vae.NetVAE`).
"""
from typing import Optional, List, Union, NamedTuple
from torch import nn
import torch



from ... import factory
from ...modules import MaskedLinear
from ...factory.core import build
from ...factory.layers import Layer, LayerList
from ...factory.mapping import get_activation
from ...factory.layers import ConstraintInfo

import logging

logger = logging.getLogger(__name__)


def _module_out_features(module: nn.Module) -> Optional[int]:
    """
    Report a module's output width, if it has one.

    Helper for validating that an encoder's final hidden width matches the
    declared ``latent_dim``. Understands both :class:`~embkit.modules.MaskedLinear`
    (whose width lives on the inner ``linear``) and plain ``nn.Linear``; returns
    ``None`` for modules without an output width (activations, batch-norm, etc.).

    Parameters:
        module: An ``nn.Module``.

    Returns:
        The module's output feature count, or ``None`` if it doesn't have one.
    """
    if isinstance(module, MaskedLinear):
        return int(module.linear.out_features)
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    return None

class EncoderOutput(NamedTuple):
    """Named output of :meth:`Encoder.forward`."""
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


import torch
import torch.nn as nn
from typing import Tuple

@factory.nn_module
class VAEEncoder(nn.Module):
    """
    VAE Encoder wrapper that takes a backbone network (e.g., FFN, CNN), 
    projects features to latent mean and log-variance, and calculates KL divergence.
    """
    def __init__(self, backbone: nn.Module, feature_dim: int, latent_dim: int, device=None, dtype=None):
        """
        Args:
            backbone (nn.Module): Feature extractor module outputting a tensor of shape (batch_size, feature_dim).
            feature_dim (int): Output feature dimension of the backbone.
            latent_dim (int): Dimension of the latent space z.
        """
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Linear projections for mu and log-variance
        self.fc_mu = nn.Linear(feature_dim, latent_dim, device=device, dtype=dtype)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim, device=device, dtype=dtype)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps
        During evaluation (eval mode), returns mu deterministically.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def compute_kl_elements(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Computes the element-wise KL divergence between q(z|x) ~ N(mu, sigma^2) 
        and the standard Gaussian prior p(z) ~ N(0, I):
        
        D_KL = -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        """
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE Encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            z (torch.Tensor): Sampled latent vectors of shape (batch_size, latent_dim).
            kl_element (torch.Tensor): Element-wise KL divergence of shape (batch_size, latent_dim).
            mu (torch.Tensor): Mean vector of shape (batch_size, latent_dim).
            logvar (torch.Tensor): Log-variance vector of shape (batch_size, latent_dim).
        """
        # 1. Pass through feature extractor backbone
        h = self.backbone(x)
        
        # Flatten if backbone outputs spatial dimensions (e.g., unflattened CNN)
        if h.dim() > 2:
            h = torch.flatten(h, start_dim=1)
            
        # 2. Predict distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 3. Sample latent code z using reparameterization
        z = self.reparameterize(mu, logvar)
        
        # 4. Compute per-element KL divergence
        # kl_element = self.compute_kl_elements(mu, logvar)
        
        return EncoderOutput(mu=mu, logvar=logvar, z=z)

    def to_dict(self):
        """
        Serialize the encoder configuration (backbone description + dimensions).

        The backbone is captured as its own description so it can be resolved
        through the factory on rebuild.
        """
        return {
            "backbone": self.backbone.to_dict(),
            "latent_dim": self.latent_dim,
            "feature_dim": self.feature_dim
        }

    @classmethod
    def from_dict(cls, d):
        """
        Rebuild a :class:`VAEEncoder` from its :meth:`to_dict` description,
        resolving the backbone through the factory build.
        """
        return VAEEncoder(
            feature_dim=d["feature_dim"],
            latent_dim=d["latent_dim"],
            backbone=build(d["backbone"])
        )

@factory.nn_module
class Encoder(nn.Module):
    """
    Fully-connected encoder that maps input features to the latent distribution.

    Pipeline: input -> [optional global batch-norm] -> a stack of linear layers
    -> two heads producing the latent mean (``mu``) and log-variance
    (``logvar``) -> a sampled latent vector ``z``.

    Two construction modes:

    - If ``layers`` is provided, the encoder body is built from that
      :class:`~embkit.factory.layers.LayerList`, which must terminate at a width
      equal to ``latent_dim`` (the latent heads are then added on top).
    - If ``layers`` is ``None``/empty, a single linear projection to
      ``latent_dim`` is inserted automatically (optionally masked by a
      constraint, e.g. for :class:`~embkit.models.vae.net_vae.NetVAE`), with an
      optional activation and batch-norm, before the latent heads.

    The :meth:`forward` output is an :class:`EncoderOutput` of
    ``(mu, logvar, z)``. During training ``z`` is sampled from the latent
    distribution (reparameterization); during evaluation ``z == mu`` so the
    embedding is deterministic.
    """

    def __init__(self,
                 feature_dim: int,
                 latent_dim: int,
                 layers: Optional[LayerList] = None,
                 batch_norm: bool = False,
                 default_activation: Union[str, None] = "relu",
                 sampling : bool = True,
                 constraint: Optional[ConstraintInfo] = None,
                 device=None, dtype=None):
        """
        Build the encoder body and the latent mean/log-variance heads.

        Parameters:
            feature_dim: Number of input features (matrix columns).
            latent_dim: Width of the latent space the sample is encoded into.
            layers: Optional layer stack describing the encoder body; if omitted
                a single projection to ``latent_dim`` is created.
            batch_norm: Whether to apply batch normalization (on the input and,
                when auto-projecting, after the projection).
            default_activation: Activation used after the auto-projection.
            sampling: Whether ``forward`` samples the latent vector during
                training (True) or returns ``mu`` deterministically (False).
            constraint: Optional connection constraint used to mask the
                auto-projection (e.g. pathway groups in NetVAE).
            device, dtype: Device/dtype for module construction.

        Raises:
            ValueError: If a ``layers`` stack is given whose final width does not
                equal ``latent_dim``.
        """
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.default_activation = default_activation
        self.sampling = sampling
        self.constraint = constraint
        self.layers = layers
        self.batch_norm = batch_norm

        self.net = nn.ModuleList()
        in_features = feature_dim

        # Optional global BN on input
        if batch_norm:
            self.net.append(nn.BatchNorm1d(in_features, device=device, dtype=dtype))

        if layers:
            logger.info("Building encoder with %d layers", len(layers))
            enc_net = layers.build( input_dim=in_features, output_dim=self.latent_dim, device=device, dtype=dtype)
            self.net.extend(enc_net)
            in_features = self.latent_dim
            for module in reversed(enc_net):
                width = _module_out_features(module)
                if width is not None:
                    in_features = width
                    break

            # Latent heads requirement
            self.z_mean = None
            self.z_log_var = None
            self.z_mean = nn.Linear(self.latent_dim, self.latent_dim, device=device, dtype=dtype)
            self.z_log_var = nn.Linear(self.latent_dim, self.latent_dim, device=device, dtype=dtype)

        else:
            logger.info("No encoder layers provided; inserting auto-projection to latent_dim=%d", self.latent_dim)

            # Auto projection to latent size (masked when constraint is provided)
            if self.constraint is not None:
                proj = MaskedLinear(in_features, self.latent_dim, bias=True, device=device, dtype=dtype)
                self.net.append(proj)
                m = self.constraint.gen_mask(in_features, self.latent_dim)
                proj.set_mask(torch.as_tensor(m, dtype=proj.mask.dtype, device=proj.mask.device))
                setattr(proj, "constraint_info", self.constraint)
            else:
                proj = nn.Linear(in_features, self.latent_dim, bias=True, device=device, dtype=dtype)
                self.net.append(proj)

            # Optional default activation after the auto-projection
            act = get_activation(self.default_activation)
            if act is not None:
                self.net.append(act())

            # Optional BN after the auto-projection
            if batch_norm:
                self.net.append(nn.BatchNorm1d(self.latent_dim, device=device, dtype=dtype))

            in_features = self.latent_dim

            # Latent heads
            self.z_mean = None
            self.z_log_var = None
            self.z_mean = nn.Linear(self.latent_dim, self.latent_dim, device=device, dtype=dtype)
            self.z_log_var = nn.Linear(self.latent_dim, self.latent_dim, device=device, dtype=dtype)

        self._final_width = in_features

    def forward(self, x: torch.Tensor):
        """
        Encode a batch of samples into latent statistics and a latent sample.

        Pushes the input through the encoder body, then the two latent heads.
        Returns an :class:`EncoderOutput` of ``(mu, logvar, z)`` where ``z`` is
        the sampled latent vector (via reparameterization during training, and
        equal to ``mu`` during evaluation).

        Parameters:
            x: Input batch tensor of shape ``(batch_size, feature_dim)``.

        Returns:
            EncoderOutput: Tuple ``(mu, logvar, z)``.
        """
        h = x
        for layer in self.net:
            h = layer(h)
        mu = self.z_mean(h)
        logvar = self.z_log_var(h)
        if self.sampling and self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return EncoderOutput(mu=mu, logvar=logvar, z=z)
    
    def to_dict(self):
        """
        Serialize the encoder's configuration (architecture, not weights).

        Captures dimensions, the layer stack, batch-norm/activation/sampling
        flags, and any connection constraint so that :meth:`from_dict` can
        rebuild an equivalent encoder. Learned weights are restored separately
        from the state dict during load.
        """
        return {
            "feature_dim": self.feature_dim,
            "latent_dim": self.latent_dim,
            "batch_norm": self.batch_norm,
            "default_activation": self.default_activation,
            "sampling": self.sampling,
            "constraint": self.constraint.to_dict() if self.constraint else None,
            "layers": self.layers.to_dict() if self.layers else None,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Rebuild an :class:`Encoder` from its :meth:`to_dict` description.

        Resolves any constraint and layer stack through the factory
        :func:`~embkit.factory.core.build` and reconstructs an equivalent
        encoder. Weight tensors are loaded separately from the state dict.
        """
        constraint = build(d["constraint"]) if d.get("constraint") else None
        layers = build(d["layers"]) if d.get("layers") else None
        return Encoder(
            feature_dim=d["feature_dim"],
            latent_dim=d["latent_dim"],
            layers=layers,
            batch_norm=d.get("batch_norm", False),
            default_activation=d.get("default_activation", "relu"),
            sampling=d.get("sampling", False),
            constraint=constraint
        )

    def refresh_mask(self, device: torch.device) -> None:
        """
        Update masks in all MaskedLinear layers using the constraint.
        This is a no-op if there's no constraint.

        Args:
            device: The device to move the mask tensor to
        """
        fallback_constraint = self.constraint

        for module in self.net:
            if isinstance(module, MaskedLinear):
                constraint_info = getattr(module, "constraint_info", None)
                if constraint_info is not None:
                    m = constraint_info.gen_mask(module.linear.in_features, module.linear.out_features)
                    module.set_mask(torch.as_tensor(m, dtype=module.mask.dtype, device=module.mask.device))
                elif fallback_constraint is not None:
                    m = fallback_constraint.gen_mask(module.linear.in_features, module.linear.out_features)
                    module.set_mask(torch.as_tensor(m, dtype=module.mask.dtype, device=module.mask.device))
