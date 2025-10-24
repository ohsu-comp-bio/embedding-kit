import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from hopcroftkarp import HopcroftKarp

def calc_rmsd(array1, array2):
    """
    Calculates the Root Mean Square Deviation (RMSD) between two arrays.
    Parameters:
        array1 (numpy.ndarray): The first array.
        array2 (numpy.ndarray): The second array, must be of the same length as array1.

    Returns:
        float: The RMSD between the two arrays.
    """
    if len(array1) != len(array2):
        raise ValueError("Arrays must be of the same length")
    diff = array1 - array2
    squared_diff = diff ** 2
    mean_squared_diff = np.mean(squared_diff)
    rmsd = np.sqrt(mean_squared_diff)

    return rmsd


def matrix_spearman_alignment_linear(a,b,cuttoff=0.0):
    """
    matrix_spearman_alignment_linear

    use the linear_sum_assignment algorithm to 
    create mapping between two matrices based on optimizing
    spearmanr scores
    """
    # identify overlapping columns
    isect = a.columns.intersection( b.columns )
    # run spearman calculation on common features
    o = spearmanr( a[isect], b[isect], axis=1 )
    # use the corner of the correlation matrix that is the intersection of the 2 matrices
    a_count=a.shape[0]
    sdf = pd.DataFrame( o.correlation[a_count:,:a_count], index=b.index, columns=a.index)
    # the matrix represents the edge weights in a bi-partite graph, use linear sum assignment
    # to obtain the maximal set
    mapping = linear_sum_assignment( sdf, True )
    # remap output to dict
    out_a = []
    out_b = []
    out_score = []
    for k, v in zip(mapping[0], mapping[1]):
        c = sdf.iloc[ k, v ]
        if c >= cuttoff:
            out_a.append( a.index[v] )
            out_b.append( b.index[k] )
            out_score.append( c )
    return out_a, out_b, out_score

def matrix_spearman_alignment_hopkraft(a,b,cuttoff=0.0):
    """
    matrix_spearman_alignment_hopkraft

    use the HopcroftKarp maximum matching algorithm to 
    create mapping between two matrices based on optimizing
    spearmanr scores
    """

    # identify overlapping columns
    isect = a.columns.intersection( b.columns )
    # run spearman calculation on common features
    o = spearmanr( a[isect], b[isect], axis=1 )
    # use the corner of the correlation matrix that is the intersection of the 2 matrices
    a_count=a.shape[0]
    sdf = pd.DataFrame( o.correlation[a_count:,:a_count], index=b.index, columns=a.index)

    #create a bi-partite graph connecting 10 best correlated edges from each TCGA sample
    m = {}
    for k, row in sdf.apply(lambda x:x.nlargest(10).index, axis=0).transpose().iterrows():
        m[k] = row.tolist()

    id_map = HopcroftKarp(m).maximum_matching(keys_only=True)
    out_a = []
    out_b = []
    out_score = []
    for k, v in id_map.items():
        #c = sdf.loc[ k, v ]
        c = sdf.loc[ v, k ]
        if c >= cuttoff:
            out_a.append(k)
            out_b.append(v)
            out_score.append(c)
    return out_a, out_b, out_score


def procrustes(X, Y):
    """
    Computes the Procrustes transformation between two matrices.

    Args:
        X: The first matrix.
        Y: The second matrix.

    Returns:
        Rotation matrix
    """
    # Ensure matrices have the same dimensions
    assert X.shape == Y.shape
    # Compute the singular value decomposition (SVD)
    U, S, Vt = np.linalg.svd(np.dot(X.T, Y))
    # Determine the rotation matrix
    R = np.dot(Vt.T, U.T)
    return R.T
