"""
NetVAE implementation
"""
import logging

from typing import Dict, List, Optional
import pandas as pd
import torch

from .base_vae import BaseVAE
from .encoder import Encoder
from ... import factory
from ...pathway import build_feature_map_indices, PathwayConstraintInfo
from ...constraints import NetworkConstraint

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
        if group_layer_size is None:
            group_layer_size = group_layer_scaling
        if group_layer_size is None:
            group_layer_size = [1, 1]

        if latent_index is None:
            _, group_idx = build_feature_map_indices(latent_groups)
            latent_index = list(group_idx)
        else:
            latent_index = list(latent_index)

        feature_list = list(features)
        latent_size = len(latent_index)
        
        enc_layers = [
            factory.Layer(units=latent_size*group_layer_scaling[0], op="masked_linear",
                  constraint=PathwayConstraintInfo("features-to-group", feature_map=latent_groups,
                                                   feature_index=feature_list, group_index=latent_index,
                                                      out_group_scaling=group_layer_scaling[0]))
        ]
        for i in range(1, len(group_layer_scaling)):
            enc_layers.append(
                factory.Layer(latent_size*i, op="masked_linear", 
                      constraint=PathwayConstraintInfo("group-to-group", feature_map=latent_groups,
                                                       feature_index=feature_list, group_index=latent_index)
            ))


        dec_layers = [
            factory.Layer(latent_size*group_layer_scaling[-1], op="masked_linear", 
                  constraint=PathwayConstraintInfo("group-to-group", feature_map=latent_groups,
                                                   feature_index=feature_list, group_index=latent_index))
        ] 
        for i in reversed(range(len(group_layer_scaling))):
            dec_layers.append(
                factory.Layer(len(features), op="masked_linear",
                    constraint=PathwayConstraintInfo("group-to-features", feature_map=latent_groups,
                                                     feature_index=feature_list, group_index=latent_index),
                                                        activation="none")
            )


        encoder = self.build_encoder(feature_dim=len(features), latent_dim=latent_size, layers=factory.LayerList( enc_layers) )
        decoder = self.build_decoder(feature_dim=len(features), latent_dim=latent_size, layers=factory.LayerList( dec_layers) )

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

