import unittest
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from embkit.__main__ import cli_main


class TestMatrixCommands(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.df = pd.DataFrame(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            index=["s1", "s2", "s3", "s4", "s5"],
            columns=["G1", "G2", "G3", "G4"],
        )

    def test_pca_smoke(self):
        with self.runner.isolated_filesystem():
            self.df.to_csv("toy.tsv", sep="\t")
            result = self.runner.invoke(
                cli_main,
                [
                    "matrix",
                    "pca",
                    "toy.tsv",
                    "--pca-size",
                    "2",
                    "--out",
                    "toy.pca.tsv",
                ],
            )

            self.assertEqual(result.exit_code, 0, msg=result.output)
            out_df = pd.read_csv("toy.pca.tsv", sep="\t", index_col=0)
            self.assertEqual(out_df.shape, (5, 2))

    def test_pca_default_out_and_bad_size(self):
        with self.runner.isolated_filesystem():
            self.df.to_csv("toy.tsv", sep="\t")

            bad = self.runner.invoke(
                cli_main,
                ["matrix", "pca", "toy.tsv", "--pca-size", "0"],
            )
            self.assertNotEqual(bad.exit_code, 0)
            self.assertIn("--pca-size must be a positive integer", bad.output)

            ok = self.runner.invoke(
                cli_main,
                ["matrix", "pca", "toy.tsv", "--pca-size", "2"],
            )
            self.assertEqual(ok.exit_code, 0, msg=ok.output)
            self.assertIn("No output path provided, using default naming", ok.output)
            self.assertTrue(Path("toy.pca.tsv").exists())

    def test_normalize_smoke_and_features_subset(self):
        df2 = self.df * 2.0
        with self.runner.isolated_filesystem():
            self.df.to_csv("a.tsv", sep="\t")
            df2.to_csv("b.tsv", sep="\t")
            with open("features.txt", "w", encoding="ascii") as f:
                f.write("G1\nG3\n")

            result = self.runner.invoke(
                cli_main,
                [
                    "matrix",
                    "normalize",
                    "a.tsv",
                    "b.tsv",
                    "--out",
                    "norm.tsv",
                    "--features",
                    "features.txt",
                    "--precision",
                    "4",
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            out_df = pd.read_csv("norm.tsv", sep="\t", index_col=0)
            self.assertEqual(list(out_df.columns), ["G1", "G3"])
            self.assertEqual(out_df.shape[0], 10)
            self.assertTrue((out_df.values >= 0.0).all())
            self.assertTrue((out_df.values <= 1.0).all())

    def test_normalize_col_quantile_and_empty_sources(self):
        with self.runner.isolated_filesystem():
            empty = self.runner.invoke(
                cli_main,
                ["matrix", "normalize", "--out", "norm.tsv"],
            )
            self.assertEqual(empty.exit_code, 0, msg=empty.output)
            self.assertIn("No matrices defined", empty.output)

            self.df.to_csv("a.tsv", sep="\t")
            colq = self.runner.invoke(
                cli_main,
                [
                    "matrix",
                    "normalize",
                    "a.tsv",
                    "--out",
                    "colq.tsv",
                    "--col-quantile",
                    "--quantile-max",
                    "0.8",
                ],
            )
            self.assertEqual(colq.exit_code, 0, msg=colq.output)
            out_df = pd.read_csv("colq.tsv", sep="\t", index_col=0)
            self.assertEqual(out_df.shape, self.df.shape)


if __name__ == "__main__":
    unittest.main()
