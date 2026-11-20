"""
Convolutional encoder backbone and decoder for grid-shaped inputs.

Image tiles (H&E, bright-field, IF) and ordered 1-D signals (coverage tracks,
spectra) have local structure that a fully-connected encoder ignores. These
modules let an embkit VAE use convolutions while the data keeps flowing through
the rest of the kit as a flat ``(samples, features)`` matrix in channel-first
order (``features = C * H * W`` for images, ``C * L`` for 1-D signals). Both
modules reshape internally, so loaders, losses and ``save``/``load`` need no
special cases.

- :class:`ConvBackbone` — strided residual conv stack: flat grid -> feature
  vector. Use it as the ``backbone`` of
  :class:`~embkit.models.vae.encoder.VAEEncoder`.
- :class:`ConvDecoder` — upsampling residual conv stack: latent -> flat grid.

Each entry of ``channels`` is one stage that halves the resolution, so every
spatial size must be divisible by ``2 ** len(channels)``.

Example::

    backbone = ConvBackbone(in_channels=3, spatial_shape=(32, 32), channels=(32, 64))
    encoder = VAEEncoder(backbone, feature_dim=backbone.feature_dim, latent_dim=16)
    decoder = ConvDecoder(latent_dim=16, in_channels=3, spatial_shape=(32, 32),
                          channels=(32, 64), output_activation="sigmoid")
    model = VAE(encoder=encoder, decoder=decoder)

Ported from UniVII (``univii.models.conv``).
"""
import math
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from .. import factory

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "leakyrelu": lambda: nn.LeakyReLU(0.2)}


def _validate_conv_args(in_channels: int, spatial_shape: Sequence[int], channels: Sequence[int],
                        kernel_size: int, norm: str, activation: str, dropout: float,
                        augment: Sequence[str] = ()) -> None:
    """Check a conv configuration up front, so errors name the bad argument."""
    shape = tuple(int(v) for v in spatial_shape)
    if len(shape) not in (1, 2):
        raise ValueError(f"spatial_shape must have 1 or 2 entries, got {tuple(spatial_shape)!r}")
    if int(in_channels) < 1 or any(v < 1 for v in shape):
        raise ValueError("in_channels and spatial_shape must be positive")
    chans = [int(c) for c in channels]
    if not chans or any(c < 1 for c in chans):
        raise ValueError("channels must be a non-empty list of positive ints")
    factor = 2 ** len(chans)
    if any(v % factor for v in shape):
        raise ValueError(f"spatial_shape {shape} must be divisible by {factor} (2 ** number of stages)")
    if int(kernel_size) < 1 or int(kernel_size) % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd int, got {kernel_size}")
    if norm not in ("group", "batch", "none"):
        raise ValueError(f"norm must be 'group', 'batch' or 'none', got {norm!r}")
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {tuple(_ACTIVATIONS)}, got {activation!r}")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    bad = [a for a in augment if a not in ("flip", "rot90")]
    if bad:
        raise ValueError(f"augment entries must be 'flip' or 'rot90', got {bad}")
    if augment and len(shape) != 2:
        raise ValueError("augment is only supported for 2-D inputs")


def _conv(ndim: int):
    return nn.Conv1d if ndim == 1 else nn.Conv2d


def _conv_t(ndim: int):
    return nn.ConvTranspose1d if ndim == 1 else nn.ConvTranspose2d


def _norm(kind: str, channels: int, groups: int, ndim: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "batch":
        return (nn.BatchNorm1d if ndim == 1 else nn.BatchNorm2d)(channels)
    g = math.gcd(int(groups), int(channels))  # largest group count that divides the channels
    return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    """Pre-activation residual block: ``x + conv(act(norm(conv(act(norm(x))))))``."""

    def __init__(self, ch: int, ndim: int, kernel_size: int, norm: str, groups: int, activation: str):
        super().__init__()
        k = int(kernel_size)
        self.body = nn.Sequential(
            _norm(norm, ch, groups, ndim), _ACTIVATIONS[activation](), _conv(ndim)(ch, ch, k, padding=k // 2),
            _norm(norm, ch, groups, ndim), _ACTIVATIONS[activation](), _conv(ndim)(ch, ch, k, padding=k // 2),
        )

    def forward(self, x):
        return x + self.body(x)


def _stage_tail(ch: int, ndim: int, kernel_size: int, norm: str, groups: int, activation: str,
                blocks_per_stage: int, dropout: float) -> nn.Sequential:
    layers = [_norm(norm, ch, groups, ndim), _ACTIVATIONS[activation]()]
    layers += [_ResBlock(ch, ndim, kernel_size, norm, groups, activation) for _ in range(int(blocks_per_stage))]
    if float(dropout) > 0:
        layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


def augment_images(x: torch.Tensor, augment: Sequence[str]) -> torch.Tensor:
    """
    Apply one random flip and/or quarter turn to a whole batch ``(B, C, H, W)``.

    The same transform is used for every image in the batch; batches differ.
    Tile orientation carries no meaning in most microscopy, so this teaches the
    encoder to ignore it.
    """
    if "flip" in augment and bool(torch.rand(()) < 0.5):
        x = torch.flip(x, dims=(-1,))
    if "rot90" in augment:
        x = torch.rot90(x, int(torch.randint(0, 4, ())), dims=(-2, -1))
    return x


@factory.nn_module
class ConvBackbone(nn.Module):
    """
    Strided residual convolutional feature extractor: flat grid -> feature vector.

    Accepts either a flat ``(B, C * prod(spatial_shape))`` matrix or an already
    shaped ``(B, C, *spatial_shape)`` tensor and returns ``(B, feature_dim)``.
    Designed to be the ``backbone`` of
    :class:`~embkit.models.vae.encoder.VAEEncoder`, which adds the latent
    mean/log-variance heads.

    Args:
        in_channels: Number of input channels ``C``.
        spatial_shape: ``(H, W)`` for images or ``(L,)`` for 1-D signals.
        channels: Width of each downsampling stage.
        kernel_size: Odd convolution kernel size.
        blocks_per_stage: Residual blocks after each downsampling conv.
        norm: ``"group"``, ``"batch"`` or ``"none"``.
        groups: Group count for group norm (reduced to divide the channels).
        activation: ``"relu"``, ``"gelu"``, ``"silu"`` or ``"leakyrelu"``.
        dropout: Dropout after each stage.
        pool: ``"flatten"`` keeps the final feature map's layout; ``"mean"``
            averages over positions (translation invariant, smaller).
        hidden_dim: Width of a dense layer after pooling; ``None`` returns the
            pooled features directly.
        augment: Train-time augmentation for 2-D inputs: any of ``"flip"``, ``"rot90"``.
        device: Device for module construction.
        dtype: Dtype for module construction.
    """

    def __init__(self, in_channels: int = 3, spatial_shape: Sequence[int] = (32, 32),
                 channels: Sequence[int] = (32, 64, 128), kernel_size: int = 3, blocks_per_stage: int = 1,
                 norm: str = "group", groups: int = 8, activation: str = "silu", dropout: float = 0.0,
                 pool: str = "flatten", hidden_dim: Optional[int] = 256, augment: Sequence[str] = (),
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        _validate_conv_args(in_channels, spatial_shape, channels, kernel_size, norm, activation, dropout, augment)
        if pool not in ("flatten", "mean"):
            raise ValueError(f"pool must be 'flatten' or 'mean', got {pool!r}")
        self.in_channels = int(in_channels)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.channels = tuple(int(c) for c in channels)
        self.kernel_size, self.blocks_per_stage = int(kernel_size), int(blocks_per_stage)
        self.norm, self.groups, self.activation = norm, int(groups), activation
        self.dropout, self.pool = float(dropout), pool
        self.hidden_dim = None if hidden_dim is None else int(hidden_dim)
        self.augment = tuple(augment)

        nd, k = len(self.spatial_shape), self.kernel_size
        layers = [_conv(nd)(self.in_channels, self.channels[0], k, padding=k // 2)]
        prev = self.channels[0]
        for ch in self.channels:
            layers += [_conv(nd)(prev, ch, k, stride=2, padding=k // 2),
                       _stage_tail(ch, nd, k, norm, self.groups, activation, self.blocks_per_stage, self.dropout)]
            prev = ch
        self.features_net = nn.Sequential(*layers)
        self.out_shape = tuple(v // 2 ** len(self.channels) for v in self.spatial_shape)
        self.pooled_dim = prev if pool == "mean" else prev * int(math.prod(self.out_shape))
        if self.hidden_dim is None:
            self.head = nn.Identity()
            self.feature_dim = self.pooled_dim
        else:
            self.head = nn.Sequential(nn.Linear(self.pooled_dim, self.hidden_dim), _ACTIVATIONS[activation]())
            self.feature_dim = self.hidden_dim
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    @property
    def flat_dim(self) -> int:
        """Number of input features in the flat layout, ``C * prod(spatial_shape)``."""
        return self.in_channels * int(math.prod(self.spatial_shape))

    def _as_grid(self, x: torch.Tensor) -> torch.Tensor:
        shape = (self.in_channels, *self.spatial_shape)
        if x.ndim == 2:
            if x.shape[1] != self.flat_dim:
                raise ValueError(f"expected {self.flat_dim} features (C * prod(spatial_shape)), got {x.shape[1]}")
            return x.reshape(x.shape[0], *shape)
        if tuple(x.shape[1:]) != shape:
            raise ValueError(f"expected input of shape (B, {', '.join(map(str, shape))}), got {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of grids into feature vectors.

        Parameters:
            x: ``(B, C * prod(spatial_shape))`` or ``(B, C, *spatial_shape)``.

        Returns:
            Feature tensor of shape ``(B, feature_dim)``.
        """
        grid = self._as_grid(x)
        if self.training and self.augment:
            grid = augment_images(grid, self.augment)
        h = self.features_net(grid)
        h = h.flatten(2).mean(-1) if self.pool == "mean" else h.flatten(1)
        return self.head(h)

    def to_dict(self):
        return {
            "in_channels": self.in_channels,
            "spatial_shape": list(self.spatial_shape),
            "channels": list(self.channels),
            "kernel_size": self.kernel_size,
            "blocks_per_stage": self.blocks_per_stage,
            "norm": self.norm,
            "groups": self.groups,
            "activation": self.activation,
            "dropout": self.dropout,
            "pool": self.pool,
            "hidden_dim": self.hidden_dim,
            "augment": list(self.augment),
        }

    @classmethod
    def from_dict(cls, params):
        return ConvBackbone(
            in_channels=params["in_channels"],
            spatial_shape=params["spatial_shape"],
            channels=params["channels"],
            kernel_size=params.get("kernel_size", 3),
            blocks_per_stage=params.get("blocks_per_stage", 1),
            norm=params.get("norm", "group"),
            groups=params.get("groups", 8),
            activation=params.get("activation", "silu"),
            dropout=params.get("dropout", 0.0),
            pool=params.get("pool", "flatten"),
            hidden_dim=params.get("hidden_dim", 256),
            augment=params.get("augment", ()),
        )


@factory.nn_module
class ConvDecoder(nn.Module):
    """
    Upsampling residual convolutional decoder: latent vector -> flat grid.

    The mirror of :class:`ConvBackbone`. Returns the reconstruction as a flat
    ``(B, C * prod(spatial_shape))`` tensor, the same layout as the input, so
    it plugs into :class:`~embkit.models.vae.vae.VAE` and the standard losses.

    Args:
        latent_dim: Width of the latent input.
        in_channels: Number of channels ``C`` to reconstruct.
        spatial_shape: ``(H, W)`` or ``(L,)`` of the reconstruction.
        channels: Stage widths, listed in encoder order (the decoder reverses them).
        kernel_size: Odd convolution kernel size.
        blocks_per_stage: Residual blocks after each upsampling step.
        norm: ``"group"``, ``"batch"`` or ``"none"``.
        groups: Group count for group norm.
        activation: ``"relu"``, ``"gelu"``, ``"silu"`` or ``"leakyrelu"``.
        dropout: Dropout after each stage.
        hidden_dim: Width of the dense layer between the latent and the first feature map.
        upsample: ``"nearest"`` (upsample + conv) or ``"transpose"`` (transposed conv).
        output_activation: ``"none"`` (e.g. with MSE / logits losses) or
            ``"sigmoid"`` (data in ``[0, 1]``).
        device: Device for module construction.
        dtype: Dtype for module construction.
    """

    def __init__(self, latent_dim: int, in_channels: int = 3, spatial_shape: Sequence[int] = (32, 32),
                 channels: Sequence[int] = (32, 64, 128), kernel_size: int = 3, blocks_per_stage: int = 1,
                 norm: str = "group", groups: int = 8, activation: str = "silu", dropout: float = 0.0,
                 hidden_dim: int = 256, upsample: str = "nearest", output_activation: str = "none",
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        _validate_conv_args(in_channels, spatial_shape, channels, kernel_size, norm, activation, dropout)
        if upsample not in ("nearest", "transpose"):
            raise ValueError(f"upsample must be 'nearest' or 'transpose', got {upsample!r}")
        if output_activation not in ("none", "sigmoid"):
            raise ValueError(f"output_activation must be 'none' or 'sigmoid', got {output_activation!r}")
        self.latent_dim = int(latent_dim)
        self.in_channels = int(in_channels)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.channels = tuple(int(c) for c in channels)
        self.kernel_size, self.blocks_per_stage = int(kernel_size), int(blocks_per_stage)
        self.norm, self.groups, self.activation = norm, int(groups), activation
        self.dropout, self.hidden_dim = float(dropout), int(hidden_dim)
        self.upsample, self.output_activation = upsample, output_activation
        self.feature_dim = self.in_channels * int(math.prod(self.spatial_shape))

        nd, k = len(self.spatial_shape), self.kernel_size
        self.base_shape: Tuple[int, ...] = tuple(v // 2 ** len(self.channels) for v in self.spatial_shape)
        self.base_ch = self.channels[-1]
        act = _ACTIVATIONS[activation]
        self.fc = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim), act(),
                                nn.Linear(self.hidden_dim, self.base_ch * int(math.prod(self.base_shape))), act())
        layers, prev = [], self.base_ch
        for ch in list(reversed(self.channels[:-1])) + [self.channels[0]]:
            if upsample == "transpose":
                layers.append(_conv_t(nd)(prev, ch, 4, stride=2, padding=1))
            else:
                layers += [nn.Upsample(scale_factor=2, mode="nearest"), _conv(nd)(prev, ch, k, padding=k // 2)]
            layers.append(_stage_tail(ch, nd, k, norm, self.groups, activation, self.blocks_per_stage, self.dropout))
            prev = ch
        layers.append(_conv(nd)(prev, self.in_channels, k, padding=k // 2))
        self.net = nn.Sequential(*layers)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to flat grids.

        Parameters:
            z: Latent tensor of shape ``(B, latent_dim)``.

        Returns:
            Reconstruction of shape ``(B, C * prod(spatial_shape))``.
        """
        h = self.fc(z).reshape(z.shape[0], self.base_ch, *self.base_shape)
        out = self.net(h)
        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        return out.flatten(1)

    def to_dict(self):
        return {
            "latent_dim": self.latent_dim,
            "in_channels": self.in_channels,
            "spatial_shape": list(self.spatial_shape),
            "channels": list(self.channels),
            "kernel_size": self.kernel_size,
            "blocks_per_stage": self.blocks_per_stage,
            "norm": self.norm,
            "groups": self.groups,
            "activation": self.activation,
            "dropout": self.dropout,
            "hidden_dim": self.hidden_dim,
            "upsample": self.upsample,
            "output_activation": self.output_activation,
        }

    @classmethod
    def from_dict(cls, params):
        return ConvDecoder(
            latent_dim=params["latent_dim"],
            in_channels=params["in_channels"],
            spatial_shape=params["spatial_shape"],
            channels=params["channels"],
            kernel_size=params.get("kernel_size", 3),
            blocks_per_stage=params.get("blocks_per_stage", 1),
            norm=params.get("norm", "group"),
            groups=params.get("groups", 8),
            activation=params.get("activation", "silu"),
            dropout=params.get("dropout", 0.0),
            hidden_dim=params.get("hidden_dim", 256),
            upsample=params.get("upsample", "nearest"),
            output_activation=params.get("output_activation", "none"),
        )
