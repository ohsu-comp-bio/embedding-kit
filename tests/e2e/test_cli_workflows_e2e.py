import unittest
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from embkit.__main__ import cli_main
from embkit.files import H5Writer


class TestCLIWorkflowsE2E(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_train_vae_then_encode_tsv_e2e(self):
        df = pd.DataFrame(
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.20, 0.30, 0.40, 0.50],
                [0.30, 0.40, 0.50, 0.60],
                [0.40, 0.50, 0.60, 0.70],
                [0.50, 0.60, 0.70, 0.80],
                [0.60, 0.70, 0.80, 0.90],
            ],
            index=[f"s{i}" for i in range(1, 7)],
            columns=["G1", "G2", "G3", "G4"],
        )

        with self.runner.isolated_filesystem():
            df.to_csv("toy.tsv", sep="\t")

            train_result = self.runner.invoke(
                cli_main,
                [
                    "model",
                    "train-vae",
                    "toy.tsv",
                    "--latent",
                    "2",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--out",
                    "toy_vae.model",
                    "--save-stats",
                ],
            )
            self.assertEqual(train_result.exit_code, 0, msg=train_result.output)
            self.assertTrue(Path("toy_vae.model").exists())
            self.assertTrue(Path("toy_vae.model.stats.tsv").exists())

            encode_result = self.runner.invoke(
                cli_main,
                ["model", "encode", "toy.tsv", "toy_vae.model", "--out", "toy_embed.tsv"],
            )
            self.assertEqual(encode_result.exit_code, 0, msg=encode_result.output)
            out_df = pd.read_csv("toy_embed.tsv", sep="\t", index_col=0)
            self.assertEqual(out_df.shape, (6, 2))

    def test_train_vae_h5_normalize_guard_and_success_e2e(self):
        with self.runner.isolated_filesystem():
            writer = H5Writer("toy.h5", "rna", index=["s1", "s2", "s3"], columns=["G1", "G2"])
            writer.set_irow(0, [1.0, 2.0])
            writer.set_irow(1, [2.0, 3.0])
            writer.set_irow(2, [3.0, 4.0])
            writer.close()

            bad_result = self.runner.invoke(
                cli_main,
                [
                    "model",
                    "train-vae",
                    "toy.h5",
                    "--group",
                    "rna",
                    "--normalize",
                    "expMinMax",
                    "--epochs",
                    "1",
                ],
            )
            self.assertNotEqual(bad_result.exit_code, 0)
            self.assertIn("Normalization for HDF5 input is not supported in train-vae", bad_result.output)

            ok_result = self.runner.invoke(
                cli_main,
                [
                    "model",
                    "train-vae",
                    "toy.h5",
                    "--group",
                    "rna",
                    "--normalize",
                    "none",
                    "--latent",
                    "2",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--out",
                    "toy_h5.model",
                ],
            )
            self.assertEqual(ok_result.exit_code, 0, msg=ok_result.output)
            self.assertTrue(Path("toy_h5.model").exists())

    def test_train_netvae_no_overlap_fails_loudly_e2e(self):
        df = pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            index=["s1", "s2", "s3"],
            columns=["G1", "G2"],
        )

        with self.runner.isolated_filesystem():
            df.to_csv("rna.tsv", sep="\t")
            with open("pathway.sif", "w", encoding="utf-8") as f:
                f.write("TF1\tcontrols-expression-of\tX1\n")
                f.write("TF2\tcontrols-expression-of\tX2\n")

            result = self.runner.invoke(
                cli_main,
                [
                    "model",
                    "train-netvae",
                    "rna.tsv",
                    "pathway.sif",
                    "--epochs",
                    "1",
                    "--out",
                    "bad.model",
                ],
            )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIsInstance(result.exception, ValueError)
            self.assertIn("latent_groups cannot be empty", str(result.exception))

    def test_align_pair_cli_e2e(self):
        a = pd.DataFrame(
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
            index=["A1", "A2"],
            columns=["F1", "F2", "F3"],
        )
        b = pd.DataFrame(
            [[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]],
            index=["B1", "B2"],
            columns=["F1", "F2", "F3"],
        )

        with self.runner.isolated_filesystem():
            a.to_csv("a.tsv", sep="\t")
            b.to_csv("b.tsv", sep="\t")

            result = self.runner.invoke(
                cli_main,
                ["align", "pair", "a.tsv", "b.tsv", "--cutoff", "0.9"],
            )

            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("A1\tB1", result.output)
            self.assertIn("A2\tB2", result.output)


if __name__ == "__main__":
    unittest.main()
