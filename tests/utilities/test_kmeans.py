import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from embkit.utilities.kmeans import run_kmeans
from embkit.utilities.pca import run_pca


class TestUtilities(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.tmp.name, "data.tsv")
        with open(self.input_path, "w", encoding="utf-8") as f:
            f.write("id\tx\ty\n")
            f.write("a\t0\t0\n")
            f.write("b\t0\t1\n")
            f.write("c\t10\t10\n")
            f.write("d\t10\t11\n")
        self.kmeans_input_path = os.path.join(self.tmp.name, "kmeans.tsv")
        with open(self.kmeans_input_path, "w", encoding="utf-8") as f:
            f.write("a\t0\t0\n")
            f.write("b\t0\t1\n")
            f.write("c\t10\t10\n")
            f.write("d\t10\t11\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_kmeans_returns_and_writes_cluster_assignments(self):
        output_path = os.path.join(self.tmp.name, "clusters.tsv")

        labels = run_kmeans(
            self.kmeans_input_path,
            k=2,
            n_iter=100,
            output_file=output_path,
        )

        self.assertEqual(labels.shape, (4,))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        written = pd.read_csv(output_path, sep="\t", header=None, index_col=0)
        np.testing.assert_array_equal(written.iloc[:, 0].to_numpy(), labels)
        self.assertEqual(list(written.index), ["a", "b", "c", "d"])

    def test_run_pca_returns_requested_components(self):
        result = run_pca(self.input_path, pca_size=1)

        self.assertEqual(result.shape, (4, 1))
        self.assertEqual(list(result.index), ["a", "b", "c", "d"])


if __name__ == "__main__":
    unittest.main()
