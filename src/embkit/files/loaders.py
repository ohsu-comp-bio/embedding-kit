"""
Methods to load alternative matrix formats
"""

import pandas as pd
from pathlib import Path
from os import PathLike

def load_gct(path: Path | PathLike[str], nrows: int = None):
    """
    gct files start with a special header:
        #1.2
        59033   19616
        Name    Description [sample 1]  [samples 2] .....
        GeneID  GeneName    0.0  0.01
    
        This function opens and transposes a GCT file
    """
    if nrows is not None:
        df = pd.read_csv(path, sep="\t", skiprows=2, index_col=0, nrows=nrows)
    else:
        df = pd.read_csv(path, sep="\t", skiprows=2, index_col=0)
    return df.drop(columns=["Description"]).transpose()


def load_raw_hugo(path: Path | PathLike[str]):
    """
    Load the raw hugo file as a dataframe
    """
    return pd.read_csv(path, sep="\t", index_col=0)

def load_gtex_hugo(path: Path | PathLike[str], nrows: int = None):
    """
    Load a hugo gtex converted file as a dataframe
    """
    if nrows is not None:
        df = pd.read_csv(path, sep="\t", index_col=0, nrows=nrows)
    else:
        df = pd.read_csv(path, sep="\t", index_col=0)
    return df