from pathlib import Path
import logging
from ..files import LargeCsvReader
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

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
    csvfile = LargeCsvReader(input_file, sep="\t", skip_header=False)

    data = np.array(list(csvfile.read(show_progress=True)))
    names = [row[0] for row in csvfile]

    logger.info("Calculating PCA")
    data_pca = PCA(n_components=pca_size).fit_transform(data)

    df = pd.DataFrame(data_pca, index=names)
    if output_file is not None:
        df.to_csv(output_file, sep="\t")
    return df
