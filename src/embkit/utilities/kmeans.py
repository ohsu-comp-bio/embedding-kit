from pathlib import Path
from ..file_readers import LargeCsvReader
import numpy as np
import faiss
import pandas as pd

def run_kmeans(input_file: Path | str, chunk_size: int = 1000, n_iter=10, sample_size: int = 1000000, k: int = 10, verbose: bool = False, max_points_per_centroid: int = 100000, output_file: Path | str | None = None) -> np.ndarray:
    """
    Runs KMeans clustering on the data from the input file and optionally saves the cluster assignments.
    
    Args:
        input_file (Path | str): Path to the input CSV file.
        chunk_size (int): Number of rows to read at a time.
        n_iter (int): Number of iterations for KMeans.
        sample_size (int): Number of samples to use for training KMeans.
        k (int): Number of clusters.
        verbose (bool): Whether to print progress messages.
        max_points_per_centroid (int): Maximum number of points per centroid.
        output_file (Path | str | None): Optional path to save the cluster assignments as a CSV file.
        
    Returns:
        np.ndarray: Array of cluster assignments for each data point.
    """

    csvfile: LargeCsvReader = LargeCsvReader(input_file, sep="\t", index_column=0, skip_header=True, cache_size=128)
    d = csvfile.shape[1] - 1

    data = np.array(list(csvfile.read(show_progress=verbose)))

    keys = list(csvfile._index.keys())

    names = []
    with csvfile:
        for k, _ in csvfile:
            names.append(k)


    kmeans = faiss.Kmeans(data.shape[1], k, niter=n_iter, verbose=verbose, max_points_per_centroid=max_points_per_centroid )
    kmeans.train(data)

    D, I = kmeans.index.search(data, 1)

    if output_file is not None:
        pd.Series(I[:,0], index=names).to_csv(output_file, sep="\t", header=None)
        
    return I[:,0]