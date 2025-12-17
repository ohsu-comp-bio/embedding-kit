from pathlib import Path
from ..file_readers import LargeCsvReader
import numpy as np
import faiss
import pandas as pd

def run_pca(input_file: Path | str, pca_size: int, output_file: Path | str | None = None) -> pd.DataFrame:
    """
    Runs PCA on the data from the input file and optionally saves the result.

    Args:
        input_file (Path | str): Path to the input CSV file.
        pca_size (int): Number of principal components to keep.
        output_file (Path | str | None): Optional path to save the PCA result as a CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the PCA-transformed data.
    """
    csvfile: LargeCsvReader = LargeCsvReader(input_file, sep="\t",
                             index_column=0, skip_header=True,
                             cache_size=128)

    data = np.array(list(csvfile.read(show_progress=True)))
    names = []
    with csvfile:
        for k, _ in csvfile:
            names.append(k)

    print("calculating PCA")
    mat = faiss.PCAMatrix(data.shape[1], pca_size)
    mat.train(data)

    print("build pca matrix")
    data_pca = mat.apply(data)

    df = pd.DataFrame(data_pca, index=names)
    if output_file is not None:
        df.to_csv(output_file, sep="\t")
    return df