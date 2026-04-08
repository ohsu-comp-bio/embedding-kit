"""
NetVAE implementation
"""
import logging

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ...factory.layers import LayerList
from ...factory.layers import Layer

from .base_vae import BaseVAE
from ... import factory
from ...pathway import PathwayControlConstraint, build_feature_map_indices

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

    def __init__(self, latent_groups: Dict[str, List[str]], group_layer_scaling: Optional[List[int]] = None):

        if group_layer_scaling is None:
            group_layer_scaling = [1,1]
        
        feature_idx, group_idx = build_feature_map_indices(latent_groups)

        latent_size = len(latent_index)
        
        enc_layers = [
            Layer(units=latent_size*group_layer_scaling[0], op="masked_linear",
                  constraint=PathwayControlConstraint("features-to-group", latent_groups,
                                                      out_group_scaling=group_layer_scaling[0]))
        ]
        for i in range(1, len(group_layer_scaling)):
            enc_layers.append(
                Layer(latent_size*i, op="masked_linear", 
                      constraint=PathwayControlConstraint("group-to-group", latent_groups,
                                                          in_group_count=group_layer_size[i-1],
                                                          out_group_count=group_layer_size[i]))
            )


        dec_layers = [
            Layer(latent_size*group_layer_scaling[-1], op="masked_linear", 
                  constraint=PathwayControlConstraint("group-to-group", latent_groups,
                                                      in_group_count=group_layer_size[-2],
                                                      out_group_count=group_layer_size[-1]))
        ] 
        for i in reversed(range(len(group_layer_scaling))):
            dec_layers.append(
                Layer(len(features), op="masked_linear",
                    constraint=PathwayControlConstraint("group-to-features", latent_groups,
                                                          in_group_count=group_layer_size[i],
                                                          out_group_count=group_layer_size[i-1]),
                    activation="none")
            )


        encoder = self.build_encoder(feature_dim=len(features), latent_dim=latent_size, layers=LayerList( enc_layers) )
        decoder = self.build_decoder(feature_dim=len(features), latent_dim=latent_size, layers=LayerList( dec_layers) )

        super().__init__(features=features, encoder=encoder, decoder=decoder)
        self.latent_groups: Dict[str, List[str]] = latent_groups
        self.latent_index: Optional[List[str]] = latent_index
        self.group_layer_size = group_layer_size
        self.history: Optional[Dict[str, List[float]]] = None
        self.normal_stats: Optional[pd.DataFrame] = None

    def to_dict(self):
        return {
            "latent_groups": self.latent_groups,
            "group_layer_scaling": self.group_layer_scaling,
        }

    @classmethod
    def from_dict(cls, d):
        model = NetVAE(
            latent_groups=d.get("latent_groups"),
            group_layer_scaling=d.get("group_layer_scaling")
        )
        return model
