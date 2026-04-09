"""
NetVAE implementation
"""
import logging

from typing import Dict, List, Optional
import pandas as pd
import torch

from .base_vae import BaseVAE
from ... import factory
from ...pathway import build_feature_map_indices, PathwayConstraintInfo

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

    def __init__(
            self,
            features: List[str],
            latent_groups: Dict[str, List[str]],
            latent_index: Optional[List[str]] = None,
            group_layer_size: Optional[List[int]] = None,
            group_layer_scaling: Optional[List[int]] = None,
            batch_norm: bool = False,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        if not latent_groups:
            raise ValueError("latent_groups cannot be empty for NetVAE.")

        # Canonical field: group_layer_size. Keep group_layer_scaling as a deprecated alias.
        if group_layer_size is not None and group_layer_scaling is not None and list(group_layer_size) != list(group_layer_scaling):
            raise ValueError(
                "group_layer_size and group_layer_scaling disagree; "
                "use group_layer_size as the canonical setting."
            )
        if group_layer_size is None:
            group_layer_size = group_layer_scaling
        if group_layer_size is None:
            group_layer_size = [1, 1]
        group_layer_size = [int(v) for v in group_layer_size]
        if any(v <= 0 for v in group_layer_size):
            raise ValueError(f"group_layer_size must contain positive integers, got {group_layer_size}.")

        if latent_index is None:
            _, group_idx = build_feature_map_indices(latent_groups)
            latent_index = list(group_idx)
        else:
            latent_index = list(latent_index)
        if len(latent_index) == 0:
            raise ValueError("latent_index cannot be empty.")

        feature_list = list(features)
        latent_size = len(latent_index)

        # Encoder: features -> groups*s0 -> groups*s1 -> ... -> groups*sN
        enc_layers = [
            factory.Layer(
                units=latent_size * group_layer_size[0],
                op="masked_linear",
                constraint=PathwayConstraintInfo(
                    "features-to-group",
                    feature_map=latent_groups,
                    feature_index=feature_list,
                    group_index=latent_index,
                    out_group_scaling=group_layer_size[0],
                ),
            )
        ]
        for in_scale, out_scale in zip(group_layer_size[:-1], group_layer_size[1:]):
            enc_layers.append(
                factory.Layer(
                    units=latent_size * out_scale,
                    op="masked_linear",
                    constraint=PathwayConstraintInfo(
                        "group-to-group",
                        feature_map=latent_groups,
                        feature_index=feature_list,
                        group_index=latent_index,
                        in_group_scaling=in_scale,
                        out_group_scaling=out_scale,
                    ),
                )
            )

        # Decoder: groups*sN -> ... -> groups*s1 -> groups*s0 -> features
        dec_layers = []
        for in_scale, out_scale in zip(reversed(group_layer_size[1:]), reversed(group_layer_size[:-1])):
            dec_layers.append(
                factory.Layer(
                    units=latent_size * out_scale,
                    op="masked_linear",
                    constraint=PathwayConstraintInfo(
                        "group-to-group",
                        feature_map=latent_groups,
                        feature_index=feature_list,
                        group_index=latent_index,
                        in_group_scaling=in_scale,
                        out_group_scaling=out_scale,
                    ),
                )
            )
        dec_layers.append(
            factory.Layer(
                units=len(feature_list),
                op="masked_linear",
                constraint=PathwayConstraintInfo(
                    "group-to-features",
                    feature_map=latent_groups,
                    feature_index=feature_list,
                    group_index=latent_index,
                    in_group_scaling=group_layer_size[0],
                ),
                activation="none",
            )
        )

        encoder = self.build_encoder(
            feature_dim=len(feature_list),
            latent_dim=latent_size,
            layers=factory.LayerList(enc_layers),
            batch_norm=batch_norm,
            device=device,
            dtype=dtype,
        )
        decoder = self.build_decoder(
            feature_dim=len(feature_list),
            latent_dim=latent_size,
            layers=factory.LayerList(dec_layers),
            device=device,
            dtype=dtype,
        )

        super().__init__(features=feature_list, encoder=encoder, decoder=decoder)
        self.latent_groups: Dict[str, List[str]] = latent_groups
        self.latent_index: List[str] = latent_index
        self.group_layer_size: List[int] = list(group_layer_size)
        # Keep both names during transition for compatibility with older configs.
        self.group_layer_scaling: List[int] = list(group_layer_size)
        self.history: Optional[Dict[str, List[float]]] = None
        self.normal_stats: Optional[pd.DataFrame] = None

    def to_dict(self):
        return {
            "features": self.features,
            "latent_groups": self.latent_groups,
            "latent_index": self.latent_index,
            "group_layer_size": self.group_layer_size,
            "group_layer_scaling": self.group_layer_scaling,
        }

    @classmethod
    def from_dict(cls, d):
        features = d.get("features")
        if features is None:
            fmap = d.get("latent_groups") or {}
            feature_set = set()
            for members in fmap.values():
                feature_set.update(members)
            features = sorted(feature_set)

        model = NetVAE(
            features=features,
            latent_groups=d.get("latent_groups"),
            latent_index=d.get("latent_index"),
            group_layer_size=d.get("group_layer_size"),
            group_layer_scaling=d.get("group_layer_scaling"),
        )
        return model
