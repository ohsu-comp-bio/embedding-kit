from typing import Optional, Any, Literal, Dict
from ..pathway import FeatureGroups
import numpy as np
import pandas as pd

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
