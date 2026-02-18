"""
LayerInfo - Layer Build description
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Literal, Dict, Any
import torch
from torch import nn
from ..modules import MaskedLinear
from .mapping import Linear, Sequential, get_activation
from ..pathway import FeatureGroups


ConstraintOP = Literal["features-to-group", "group-to-features", "group-to-group"]

class ConstraintInfo:
    def __init__(self, op: ConstraintOP, groups: Optional[FeatureGroups] = None, in_group_count=1, out_group_count=1):
        self.op = op
        self.groups = groups
        self.in_group_count = in_group_count
        self.out_group_count = out_group_count

    def gen_mask(self):

        if self.op == "features-to-group":
            feature_idx, group_idx = self.groups.to_indices()
            return build_features_to_group_mask(self.groups.map, feature_idx, group_idx, group_node_count=self.out_group_count)
        elif self.op == "group-to-features":
            feature_idx, group_idx = self.groups.to_indices()
            return build_features_to_group_mask(self.groups.map, feature_idx, group_idx, group_node_count=self.in_group_count, forward=False)
        elif self.op == "group-to-group":
            return build_group_to_group_mask(len(self.groups.map), self.in_group_count, self.out_group_count)
        raise ValueError(f"Unknown ConstraintInfo.op '{self.op}'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "in_group_count": int(self.in_group_count),
            "out_group_count": int(self.out_group_count),
            "groups": (self.groups.to_dict() if hasattr(self.groups, "to_dict")
                       else {"map": getattr(self.groups, "map", None)} if self.groups is not None
            else None),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ConstraintInfo":
        g = d.get("groups")
        groups = None
        if g is not None:
            if hasattr(FeatureGroups, "from_dict"):
                groups = FeatureGroups.from_dict(g)
            else:
                # Fallback if you just have a mapping
                groups = FeatureGroups(map=g.get("map", {}))
        return ConstraintInfo(
            op=d["op"],
            groups=groups,
            in_group_count=int(d.get("in_group_count", 1)),
            out_group_count=int(d.get("out_group_count", 1)),
        )


def idx_to_list(x):
    """
    idx_to_list: takes an index map ( name -> position ) to a list of names
    ordered by position
    """
    out = [None] * len(x)
    for k, v in x.items():
        out[v] = k
    return out


def build_features_to_group_mask(feature_map, feature_idx, group_idx, group_node_count=1, forward=True):
    """
    Build a masked linear layer based on connecting all features to a 
    single group node and forcing all other connections to be zero
    """
    features = idx_to_list(feature_idx)
    groups = idx_to_list(group_idx)

    in_dim = len(features)
    out_dim = len(groups) * group_node_count

    if forward:
        mask = np.zeros((out_dim, in_dim), dtype=np.float32)
    else:
        mask = np.zeros((in_dim, out_dim), dtype=np.float32)

    fi = pd.Index(features)
    for gnum, group in enumerate(groups):
        for f in feature_map[group]:
            if f in fi:
                floc = fi.get_loc(f)
                # print(gnum, group_node_count)
                # print(list(range(gnum*group_node_count, (gnum+1)*(group_node_count))))
                for pos in range(gnum * group_node_count, (gnum + 1) * (group_node_count)):
                    if forward:
                        mask[pos, floc] = 1.0
                    else:
                        mask[floc, pos] = 1.0
    return mask


def build_group_to_group_mask(group_count: int, in_group_node_count, out_group_node_count):
    """
    build_group_to_group
    Build a mask that constricts connections between 2 group layer nodes
    """
    in_dim = group_count * in_group_node_count
    out_dim = group_count * out_group_node_count

    mask = np.zeros((out_dim, in_dim), dtype=np.float32)
    for g in range(group_count):
        for i in range(g * in_group_node_count, (g + 1) * in_group_node_count):
            for j in range(g * out_group_node_count, (g + 1) * out_group_node_count):
                mask[j, i] = 1.0
    return mask



class Layer:
    """
    Layer information for building a neural network layer.
    Holds configuration details for a layer, including the number of units,
    the type of operation (e.g., linear), the activation function, and whether
    to apply batch normalization.
    """

    def __init__(self, units: int, *, op: str = "linear",
                 activation: Optional[str] = "relu", 
                 constraint: Optional[ConstraintInfo] = None,
                 batch_norm: bool = False, bias: bool = True):
        """
        Initialize LayerInfo with specified parameters.
        Args:
            units (int): Number of units in the layer.
            op (str): Type of operation, default is "linear".
            activation (Optional[str]): Activation function to use, default is "relu".
            batch_norm (bool): Whether to apply batch normalization, default is False.
            bias (bool): Whether to include a bias term in the layer, default is True.

        Raises:
            ValueError: If the specified operation is not supported.
        """
        self.units = units
        self.op = op
        self.activation = activation
        self.batch_norm = batch_norm
        self.constraint = constraint
        self.bias = bias
    
    def gen_layer(self, in_features: int, device=None, dtype=None) -> List[nn.Module]:
        out_features = self.units
        layers = []
        if self.op == "masked_linear":
            init_mask = None
            if self.constraint is not None:
                m = self.constraint.gen_mask()
                # Expect (out_features, in_features)
                if m.shape != (out_features, in_features):
                    raise ValueError(
                        f"Constraint mask shape {m.shape} does not match "
                        f"(units, in_features)=({out_features}, {in_features})."
                    )
                init_mask = torch.as_tensor(m, dtype=torch.float32, device=device)
            layers.append(MaskedLinear(in_features, out_features, bias=self.bias, mask=init_mask, device=device, dtype=dtype))
        elif self.op ==  "linear":
            layers.append(Linear(in_features, out_features, bias=self.bias, device=device, dtype=dtype))
        else:
            raise ValueError(f"Unknown LayerInfo.op '{self.op}'")
        
        if self.activation is not None:
            act = get_activation(self.activation)
            if act is not None:
                layers.append(act())
        if self.batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        return layers

    def to_dict(self) -> dict:
        return {
            "units": self.units,
            "op": self.op,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "bias": self.bias,
            "constraint": (self.constraint.to_dict() if self.constraint else None),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Layer":
        c = d.get("constraint", None)
        constraint = ConstraintInfo.from_dict(c) if c is not None else None
        return Layer(
            units=int(d.get("units", d.get("size"))),  # tolerate old files that used "size"
            op=d.get("op", "linear"),
            activation=d.get("activation", "relu"),
            batch_norm=bool(d.get("batch_norm", False)),
            bias=bool(d.get("bias", True)),
            constraint=constraint,
        )

class LayerList:
    def __init__(self, layers: Optional[List[Layer]] = None, activation="relu", end_activation="relu"):
        new_layers = []
        if layers is not None:
            for i, l in enumerate(layers):
                if isinstance(l, int):
                    if i < len(layers)-1:
                        new_layers.append( Layer(l, activation=activation) )
                    else:
                        new_layers.append( Layer(l, activation=end_activation))
                else:
                    new_layers.append(l)
        self.layers = new_layers

    def build(self, input_dim:int, output_dim:int, device=None, dtype=None) -> nn.Module:
        if not self.layers:
            return Linear(in_features=input_dim, out_features=output_dim, device=device, dtype=dtype)
        
        cur_dim = input_dim
        layers = []
        for layer in self.layers:
            if isinstance(layer, Layer):
                layers.extend(layer.gen_layer(cur_dim, device=device, dtype=dtype))
                cur_dim = layer.units
            elif isinstance(layer, int):
                # Fallback handling in case raw integers are present in self.layers.
                # Treat the integer as the number of units for a Linear layer.
                layers.append(Linear(in_features=cur_dim, out_features=layer, device=device, dtype=dtype))
                cur_dim = layer
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

        layers.append(Linear(in_features=cur_dim, out_features=output_dim, device=device, dtype=dtype))
        return Sequential(*layers)

    def __len__(self):
        return len(self.layers)
    
    def __iter__(self):
        return iter(self.layers)