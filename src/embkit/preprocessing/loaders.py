"""
Methods to load alternative matrix formats
"""

import pandas as pd

def load_gct(path):
    """
    gct files start with a special header:
        #1.2
        59033   19616
        Name    Description [sample 1]  [samples 2] .....
        GeneID  GeneName    0.0  0.01
    
        This function opens and transposes a GCT file
    """
    df = pd.read_csv(path, sep="\t", skiprows=2, index_col=0)
    return df.drop(columns=["Description"]).transpose()