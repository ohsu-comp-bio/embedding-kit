from ..preprocessing.csv import LargeCsvReader
import numpy as np
import faiss


def run_pca(args):
    """
    Docstring for run_pca

    :param args: Description
    """
    csvfile = LargeCsvReader(args.input_file, sep="\t",
                             index_column=0, skip_header=True,
                             cache_size=128)

    data = np.array(list(csv_file_reader(csvfile, show_progress=True)))

    names = []
    with csvfile:
        for k, _ in csvfile:
            names.append(k)

    print("calculating PCA")
    mat = faiss.PCAMatrix(data.shape[1], args.pca_size)
    mat.train(data)

    print("build pca matrix")
    data_pca = mat.apply(data)

    pd.DataFrame(data_pca, index=names).to_csv(args.output_file, sep="\t", header=None)