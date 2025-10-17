"""
Methods for opening and processing Pathway files
"""
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd


# ---------- SIF parsing ----------

def extract_pathway_interactions(
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


# ---------- map intersection/subsetting ----------

def feature_map_intersect(
        feature_map: Dict[str, List[str]],
        features: Iterable[str],
        *,
        keep_lonely_groups: bool = False,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Subset `feature_map` to nodes present in `features`. Keeps the group's self-node
    if it is in `features`. Members are deduplicated and ordered by the order
    they appear in the original mapping (stable).

    Returns:
        (subset_map, intersect_list_in_input_order)
    """
    features_list = list(features)
    feature_set = set(features_list)

    # Build a stable list of all nodes (sources + members)
    all_nodes: "OrderedDict[str, None]" = OrderedDict()
    for src, members in feature_map.items():
        all_nodes[src] = None
        for m in members:
            all_nodes[m] = None

    # Intersection, respecting the order of features_list (caller’s column order)
    isect = [f for f in features_list if f in all_nodes]

    out: Dict[str, List[str]] = {}
    for src, members in feature_map.items():
        # intersect members (including possible self if it was in the original list)
        filtered = [m for m in members if m in feature_set]
        # Also include the group key itself if present in features even when not in members
        if src in feature_set and src not in filtered:
            filtered = [src] + filtered

        if filtered or (keep_lonely_groups and src in feature_set):
            # Keep group only if it retains any member in the intersection,
            # unless keep_lonely_groups=True and the group itself is in features
            out[src] = filtered

    return out, isect


# ---------- FeatureGroups ----------

class FeatureGroups:
    """
    Thin wrapper that preserves insertion order of groups and members,
    supports (de)serialization, and builds deterministic index maps.
    """

    def __init__(self, map: Dict[str, List[str]]):
        od = OrderedDict()
        for k, v in map.items():
            # enforce list and preserve order while removing duplicates
            seen = set()
            ordered_members = []
            for m in v:
                if m not in seen:
                    seen.add(m)
                    ordered_members.append(m)
            od[k] = ordered_members
        self.map: "OrderedDict[str, List[str]]" = od

    def to_indices(
            self,
            *,
            feature_order: Optional[Iterable[str]] = None,
            group_order: Optional[Iterable[str]] = None,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Create index maps for features and groups.

        If `feature_order` is provided, features are indexed by that order (and
        filtered to those present in the groups). Otherwise, insertion order
        (group-by-group) is used.

        If `group_order` is provided, groups are indexed by that order; otherwise
        insertion order is used.
        """
        # group index
        groups_iter = list(self.map.keys()) if group_order is None else [g for g in group_order if g in self.map]
        group_idx = {g: i for i, g in enumerate(groups_iter)}

        # features seen in groups (preserve insertion order across groups)
        seen = OrderedDict()
        for g in groups_iter:
            for m in self.map[g]:
                seen[m] = None

        if feature_order is None:
            feat_iter = list(seen.keys())
        else:
            feat_iter = [f for f in feature_order if f in seen]

        feature_idx = {f: i for i, f in enumerate(feat_iter)}
        return feature_idx, group_idx

    # convenience
    def __len__(self) -> int:
        return len(self.map)

    def items(self):
        return self.map.items()

    def features(self) -> List[str]:
        out = OrderedDict()
        for _, members in self.map.items():
            for m in members:
                out[m] = None
        return list(out.keys())

    # --- serialization helpers ---

    def to_dict(self) -> Dict[str, List[str]]:
        # plain JSON-serializable dict
        return {k: list(v) for k, v in self.map.items()}

    @staticmethod
    def from_dict(d: Dict[str, List[str]]) -> "FeatureGroups":
        return FeatureGroups(d)
