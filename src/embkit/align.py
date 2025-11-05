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


def matrix_spearman_alignment_linear(a, b, cutoff=0.0):
    """
    matrix_spearman_alignment_linear

    Uses the linear_sum_assignment algorithm to create an optimal
    mapping between two matrices by maximizing the total Spearman
    correlation score.
    """
    # Identify overlapping columns
    isect = a.columns.intersection(b.columns)

    combined_df = pd.concat([a[isect], b[isect]], axis=0)
    o = spearmanr(combined_df, axis=1)

    a_count = a.shape[0]
    sdf = pd.DataFrame(o.correlation[a_count:, :a_count], index=b.index, columns=a.index)

    rows, cols = linear_sum_assignment(-sdf)

    out_a = []
    out_b = []
    out_score = []

    for r, c in zip(rows, cols):
        score = sdf.iloc[r, c]
        if score >= cutoff:
            out_a.append(a.index[c])
            out_b.append(b.index[r])
            out_score.append(score)

    return out_a, out_b, out_score


def matrix_spearman_alignment_hopkraft(a, b, cutoff=0.0):
    """
    matrix_spearman_alignment_hopkraft

    Uses the HopcroftKarp maximum *cardinality* matching algorithm.
    It finds the LARGEST NUMBER of matches, not the BEST (highest score)
    match.

    Note: This is fundamentally different from linear_sum_assignment.
    """
    isect = a.columns.intersection(b.columns)

    combined_df = pd.concat([a[isect], b[isect]], axis=0)
    o = spearmanr(combined_df, axis=1)

    a_count = a.shape[0]
    sdf = pd.DataFrame(o.correlation[a_count:, :a_count], index=b.index, columns=a.index)

    m = {}
    s_result = sdf.apply(lambda x: x.nlargest(10).index, axis=0)
    for k, row in s_result.items():
        m[k] = row.tolist()

    id_map = HopcroftKarp(m).maximum_matching(keys_only=True)

    out_a = []
    out_b = []
    out_score = []

    for k, v in id_map.items():
        c = sdf.loc[v, k]
        if c >= cutoff:
            out_a.append(k)
            out_b.append(v)
            out_score.append(c)

    return out_a, out_b, out_score


def procrustes(X, Y):
    """
    Computes the Procrustes transformation (optimal PURE rotation)
    to align X to Y, correcting for reflections.

    Args:
        X: The first matrix (N_points, N_dims).
        Y: The second matrix (N_points, N_dims).

    Returns:
        R: The optimal rotation matrix (guaranteed det(R) = +1).
    """
    assert X.shape == Y.shape

    M = np.dot(X.T, Y)

    U, S, Vt = np.linalg.svd(M)

    R = np.dot(U, Vt)

    if np.linalg.det(R) < 0:
        Vt_corrected = Vt.copy()
        Vt_corrected[-1, :] *= -1
        R = np.dot(U, Vt_corrected)

    return R


def procrustes_scale(X, Y):
    """
    Compute the procrustes transformation (R), then compute a factor (k) to rescale the src matrix. 
    
    To apply transformtion:
    src.dot(R) * k

    
    Args:
        X: The first matrix (N_points, N_dims).
        Y: The second matrix (N_points, N_dims).

    Returns:
        R: The optimal rotation matrix (guaranteed det(R) = +1).
        k: Scaling factor

    """
    R = procrustes(X,Y)
    A = np.array(X.dot(R))
    B = np.array(Y)
    numerators = np.sum(A * B, axis=0)
    denominators = np.sum(A * A, axis=0)
    k = np.divide(numerators, denominators, out=np.zeros_like(numerators), where=denominators!=0)
    return R, k