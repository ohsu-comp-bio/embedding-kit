import numpy as np
import statistics
from scipy.spatial import distance

def eucpair(v1, v2) -> float:
    """
    Euclidean distance between two lists of vectors.
    example: (list(1,3,4,5), list(6,7,8,9)) indicates euclidean distance between each pair of values (order matters)
    Args:
        v1: list of np.arrays
        v2: list of np.arrays

    Returns:
        float: mean euclidean distance between each pair of vectors in the lists

    """
    assert len(v1) == len(v2), 'two lists must be same length'
    euc_list = []
    for i in range(0, len(v1)):
        euc = np.linalg.norm(v1[i] - v2[i])
        euc_list.append(euc)
    mean_euc = statistics.mean(euc_list) 
    return round(mean_euc, 5)


def mhat(v1, v2) -> float:
    """
    Manhattan distance between two arrays.
    Example: (np.array([1,2,3]), np.array([4,5,6])) indicates manhattan distance between the two arrays
    Args:
        v1: np.array
        v2: np.array
    Returns:
        float: manhattan distance between the two arrays
    """
    return distance.cityblock(v1, v2)
