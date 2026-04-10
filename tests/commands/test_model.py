import unittest
from unittest.mock import MagicMock, patch
import importlib

from click.testing import CliRunner

from embkit.__main__ import cli_main
from embkit.files import H5Writer

model_cmd = importlib.import_module("embkit.commands.model")


class TestModelCommands(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch.object(model_cmd, "save")
    @patch.object(model_cmd, "fit_vae")
    @patch.object(model_cmd, "dataframe_loader", return_value="loader")
    @patch.object(model_cmd, "NetVAE")
    def test_train_netvae_smoke(self, netvae_cls, loader_mock, fit_mock, save_mock):
        dummy_model = MagicMock(name="netvae")
        netvae_cls.return_value = dummy_model

        with self.runner.isolated_filesystem():
            with open("rna.tsv", "w", encoding="utf-8") as f:
                f.write(
                    "sample\tG1\tG2\tG3\tG4\n"
                    "s1\t1\t2\t3\t4\n"
                    "s2\t4\t3\t2\t1\n"
                )
            with open("pathway.sif", "w", encoding="utf-8") as f:
                f.write(
                    "TF1\tcontrols-expression-of\tG1\n"
                    "TF1\tcontrols-expression-of\tG2\n"
                    "TF2\tcontrols-expression-of\tG3\n"
                    "TF2\tcontrols-expression-of\tG4\n"
                )

            result = self.runner.invoke(
                cli_main,
                [
                    "model",
                    "train-netvae",
                    "rna.tsv",
                    "pathway.sif",
                    "--epochs",
                    "1",
                    "--group-layer-size",
                    "4,2,1",
                    "--out",
                    "netvae.model",
                ],
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        netvae_cls.assert_called_once()
        netvae_args = netvae_cls.call_args.args
        netvae_kwargs = netvae_cls.call_args.kwargs
        self.assertEqual(netvae_args[0], ["G1", "G2", "G3", "G4"])
        self.assertEqual(netvae_kwargs["group_layer_size"], [4, 2, 1])
        self.assertEqual(set(netvae_kwargs["latent_groups"].keys()), {"TF1", "TF2"})

        loader_mock.assert_called_once()
        fit_mock.assert_called_once()
        self.assertEqual(fit_mock.call_args.kwargs["X"], "loader")
        save_mock.assert_called_once_with(dummy_model, "netvae.model")

    def test_train_vae_h5_rejects_normalization(self):
        with self.runner.isolated_filesystem():
            writer = H5Writer("matrix.h5", "rna", index=["s1", "s2"], columns=["G1", "G2"])
            writer.set_irow(0, [1.0, 2.0])
            writer.set_irow(1, [3.0, 4.0])
            writer.close()

            result = self.runner.invoke(
                cli_main,
                [
                    "model",
                    "train-vae",
                    "matrix.h5",
                    "--group",
                    "rna",
                    "--normalize",
                    "expMinMax",
                    "--epochs",
                    "1",
                ],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Normalization for HDF5 input is not supported in train-vae", result.output)


if __name__ == "__main__":
    unittest.main()
