
"""
Methods for opening and processing Pathway files
"""

import pandas as pd
from collections import OrderedDict
from typing import Dict, List

def extract_pathway_interactions(sif_path, relation='controls-expression-of'):
    """
    extract_pathway_interactions 

    Extracts pathway information from a SIF file. Ignores chemical compounds 
    and filters for 'controls-expression-of' relationship
    """
    pc = pd.read_csv(sif_path, sep="\t", 
                     header=None, names=["from", "relation", "to"])
    pc = pc[ ~(pc['from'].map(lambda x:"CHEBI" in x) | pc['to'].map(lambda x:"CHEBI" in x)) ]
    df = pc[ pc["relation"] == relation ]

    feature_map = {}
    for _, row in df.iterrows():
        feature_map[row['from']] = feature_map.get(row['from'], [row['from']]) + [ row['to'] ]
    return feature_map

def feature_map_intersect(feature_map, features):
    """
    given a list of features (such as the columns in a dataframe) subset the feature map
    with the intersection between the two elements
    """

    feature_map_features = set()
    for k, v in feature_map.items():
        feature_map_features.add(k)
        for i in v:
            feature_map_features.add(i)
    isect = feature_map_features.intersection(features)
    out = {}
    for k, v in feature_map.items():
        z = list(set(v).intersection(isect))
        if len(z) > 0:
            out[k] = z
    return out, list(isect)


class FeatureGroups:
    def __init__(self, fmap:Dict[str,List[str]]):
        self.map = OrderedDict()
        for k, v in fmap.items():
            self.map[k] = v
    
    def to_indices(self):
        """
        take a feature map, which is map of feature groups organized by group name,
        and convert them into indices of features and groups
        """
        feature_idx={}
        group_idx={}
        for k, members in self.map.items():
            group_idx[k] = group_idx.get(k, len(group_idx))
            for m in members:
                feature_idx[m] = feature_idx.get(m, len(feature_idx))
        return feature_idx, group_idx

    def __len__(self):
        return len(self.map)

    def items(self):
        return self.map.items()