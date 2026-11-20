from pathlib import Path
from ..files import LargeCsvReader
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


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

    csvfile = LargeCsvReader(input_file, sep="\t", skip_header=True)
    data = np.array(list(csvfile.read(show_progress=verbose)))

    names = [row[0] for row in csvfile]

    training_data = data
    max_training_points = k * max_points_per_centroid
    if len(data) > max_training_points:
        sample_indices = np.random.choice(len(data), max_training_points, replace=False)
        training_data = data[sample_indices]

    kmeans = KMeans(
        n_clusters=k,
        max_iter=n_iter,
        n_init=10,
        verbose=int(verbose),
    )
    kmeans.fit(training_data)
    labels = kmeans.predict(data)

    if output_file is not None:
        pd.Series(labels, index=names).to_csv(output_file, sep="\t", header=None)
        
    return labels
