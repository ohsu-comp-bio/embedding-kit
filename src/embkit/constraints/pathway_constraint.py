from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ..factory.layers import ConstraintInfo
from ..pathway import (
    _normalize_index,
    build_feature_map_indices,
    build_features_to_group_mask,
    build_group_to_group_mask,
)

ConstraintOP = Literal["features-to-group", "group-to-features", "group-to-group"]


class PathwayConstraintInfo(ConstraintInfo):
    """
    ConstraintInfo for pathway-based masking.
    """

    def __init__(
        self,
        op: ConstraintOP,
        feature_map: Dict[str, List[str]],
        feature_index: Optional[Any] = None,
        group_index: Optional[Any] = None,
        in_group_scaling: int = 1,
        out_group_scaling: int = 1,
    ):
        self.op = op
        self.feature_map = feature_map
        self.feature_index = _normalize_index(feature_index) if feature_index is not None else None
        self.group_index = _normalize_index(group_index) if group_index is not None else None
        self.in_group_scaling = in_group_scaling
        self.out_group_scaling = out_group_scaling
        self.active = True

    def set_active(self, active: bool) -> None:
        self.active = bool(active)

    def update_membership(self, feature_map: Dict[str, List[str]]) -> None:
        self.feature_map = feature_map

    def gen_mask(self, in_features: Optional[int] = None, out_features: Optional[int] = None):
        feature_index = self.feature_index
        group_index = self.group_index
        if feature_index is None or group_index is None:
            feature_index, group_index = build_feature_map_indices(self.feature_map)
        group_count = len(group_index)
        if group_count == 0:
            raise ValueError("Cannot generate pathway mask with zero groups.")
        if in_features is not None and out_features is not None and not self.active:
            return np.ones((out_features, in_features), dtype=np.float32)

        if self.op == "features-to-group":
            if out_features is None or out_features % group_count != 0:
                raise ValueError(
                    f"features-to-group expects out_features divisible by group count ({group_count}); got {out_features}."
                )
            group_node_count = out_features // group_count
            return build_features_to_group_mask(
                self.feature_map, feature_index, group_index, group_node_count=group_node_count
            )
        if self.op == "group-to-features":
            if in_features is None or in_features % group_count != 0:
                raise ValueError(
                    f"group-to-features expects in_features divisible by group count ({group_count}); got {in_features}."
                )
            group_node_count = in_features // group_count
            return build_features_to_group_mask(
                self.feature_map, feature_index, group_index, group_node_count=group_node_count, forward=False
            )
        if self.op == "group-to-group":
            if in_features is None or out_features is None:
                raise ValueError("group-to-group requires in_features and out_features.")
            if in_features % group_count != 0 or out_features % group_count != 0:
                raise ValueError(
                    f"group-to-group expects both dimensions divisible by group count ({group_count}); "
                    f"got in_features={in_features}, out_features={out_features}."
                )
            in_group_nodes = in_features // group_count
            out_group_nodes = out_features // group_count
            return build_group_to_group_mask(group_count, in_group_nodes, out_group_nodes)
        raise ValueError(f"Unknown ConstraintInfo.op '{self.op}'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "feature_map": self.feature_map,
            "feature_index": (list(self.feature_index) if self.feature_index is not None else None),
            "group_index": (list(self.group_index) if self.group_index is not None else None),
            "in_group_scaling": self.in_group_scaling,
            "out_group_scaling": self.out_group_scaling,
            "active": self.active,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PathwayConstraintInfo":
        obj = PathwayConstraintInfo(
            op=d["op"],
            feature_map=d["feature_map"],
            feature_index=d.get("feature_index"),
            group_index=d.get("group_index"),
            in_group_scaling=d.get("in_group_scaling", 1),
            out_group_scaling=d.get("out_group_scaling", 1),
        )
        obj.set_active(d.get("active", True))
        return obj

__all__ = ["PathwayConstraintInfo"]
