"""
Methods for opening and processing Pathway files
"""
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Tuple, Iterable, Optional, Literal

import numpy as np
import pandas as pd

from .factory.layers import ConstraintInfo


# ---------- SIF parsing ----------

def extract_sif_interactions(
        sif_path: str,
        relation: str = "controls-expression-of",
) -> Dict[str, List[str]]:
    """
    Extracts pathway info from a SIF-like TSV with columns: from, relation, to.
    - drops rows with CHEBI* nodes
    - filters by `relation`
    - ensures each source maps to a list that includes itself as the first element
      (stable, no duplicates)
    """
    pc = pd.read_csv(
        sif_path,
        sep="\t",
        header=None,
        names=["from", "relation", "to"],
        dtype=str,
    ).fillna("")

    # Fast, null-safe CHEBI filter
    mask_chebi = pc["from"].str.contains("CHEBI", na=False) | pc["to"].str.contains("CHEBI", na=False)
    pc = pc.loc[~mask_chebi]

    # Relation filter
    if relation is not None:
        pc = pc.loc[pc["relation"] == relation]

    fmap: Dict[str, List[str]] = {}
    # Use defaultdict only internally to dedup, then materialize deterministic lists
    tmp: Dict[str, "OrderedDict[str, None]"] = defaultdict(OrderedDict)
    for src, _, dst in pc[["from", "relation", "to"]].itertuples(index=False, name=None):
        # keep self as the first item
        if src not in tmp:
            tmp[src][src] = None
        tmp[src][dst] = None

    # materialize as lists with stable (insertion) order
    for src, od in tmp.items():
        fmap[src] = list(od.keys())

    return fmap


def build_mask(
        feature_map: Dict[str, List[str]],
        src_index: pd.Index,
        dst_index: pd.Index,
        min_group_size: int = 2,
) -> np.ndarray:
    """
    Build a binary mask from a SIF file using explicit source/destination indices.

    Because this is a controller mapping (ie the src controls the dst) the inner layer
    dimension corresponds to the source (controller) and the outer dimension corresponds
    to the destination (target). This way the mask can be directly applied to a weight
    matrix of shape (out_features, in_features) where out_features are the targets and
    in_features are the controllers.

    The returned mask has shape (len(dst_index), len(src_index)) and is compatible
    with weight matrices shaped (out_features, in_features) for MaskedLinear layers.
    Only edges that map from a known source to a known destination are set to 1.

    Parameters
    ----------
    feature_map : Dict[str, List[str]]
        Mapping from source node to list of destination nodes (e.g. TF to target genes).
    src_index : pd.Index
        Index of source nodes.
    dst_index : pd.Index
        Index of destination nodes.

    Returns
    -------
    np.ndarray
        Binary mask with shape (len(dst_index), len(src_index)).
    """
    mask = np.zeros((len(dst_index), len(src_index)), dtype=np.float32)
    for src, members in feature_map.items():
        if src in src_index:
            if dst_index.intersection(members).size >= min_group_size:
                for dst in members:
                    if dst in dst_index:
                        mask[dst_index.get_loc(dst), src_index.get_loc(src)] = 1.0

    return mask


# ---------- map intersection/subsetting ----------

def feature_map_intersect(
        feature_map: Dict[str, List[str]],
        features: Iterable[str],
        min_group_size: int = 2,
        include_self: bool = True,
) -> Dict[str, List[str]]:
    """
    Subset `feature_map` to nodes present in `features`. Keeps the group's self-node
    if it is in `features`. Members are deduplicated. if include_self is True, the 
    source node will be included in the members list if it is present in the features.     

    Returns:
        subset_map: Dict[str, List[str]]: subset of feature_map with only features in `features`
    """
    features_list = list(features)
    out_map = {}
    for src, members in feature_map.items():
        filtered_members = [m for m in set(members) if m in features_list]
        if include_self and src in features_list:
            filtered_members = [src] + filtered_members
        if len(filtered_members) >= min_group_size:
            out_map[src] = filtered_members
    return out_map

def build_feature_map_indices(
        feature_map: Dict[str, List[str]]) -> Tuple[pd.Index, pd.Index]:
    """
    Create feature index from a feature map
    """
    feature_set = set()
    group_set = sorted(feature_map.keys())
    for group in group_set:
        feature_set.update(feature_map[group])
    feature_idx = pd.Index( sorted(feature_set) )
    group_idx = pd.Index(group_set)
    return feature_idx, group_idx




ConstraintOP = Literal["features-to-group", "group-to-features", "group-to-group"]

class PathwayControlConstraint(ConstraintInfo):
    """
    ConstraintInfo for pathway-based masking. Supports three types of constraints:
        - "features-to-group": connects features to group nodes (e.g. TF to target genes)
        - "group-to-features": connects group nodes to features (e.g. TF to target genes, reverse direction)
        - "group-to-group": connects group nodes to group nodes (e.g. between latent layers)
    """
    def __init__(self, op: ConstraintOP, feature_map : Dict[str, List[str]],
                 feature_index: pd.Index, group_index: pd.Index,
                 in_group_scaling: int = 1, 
                 out_group_scaling: int = 1):
        self.op = op
        self.feature_map = feature_map
        self.in_group_scaling = in_group_scaling
        self.out_group_scaling = out_group_scaling

    def gen_mask(self, in_features: Optional[int] = None, out_features: Optional[int] = None):

        feature_index, group_index = build_feature_map_indices(self.feature_map)

        if self.op == "features-to-group":
            return build_features_to_group_mask(self.feature_map, feature_index, group_index, group_node_count=out_features)
        elif self.op == "group-to-features":
            return build_features_to_group_mask(self.feature_map, feature_index, group_index, group_node_count=in_features, forward=False)
        elif self.op == "group-to-group":
            return build_group_to_group_mask(len(self.feature_map), in_features, out_features)
        raise ValueError(f"Unknown ConstraintInfo.op '{self.op}'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "feature_map": self.feature_map,
            "in_group_scaling": self.in_group_scaling,
            "out_group_scaling": self.out_group_scaling,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PathwayControlConstraint":
        return PathwayControlConstraint(
            op=d["op"],
            feature_map=d["feature_map"],
            feature_index=pd.Index(d.get("feature_index")),
            group_index=pd.Index(d.get("group_index")),
            in_group_scaling=d.get("in_group_scaling", 1),
            out_group_scaling=d.get("out_group_scaling", 1)
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



