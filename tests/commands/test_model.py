import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from embkit.__main__ import cli_main


class TestModelCommands(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("embkit.commands.model.save")
    @patch("embkit.commands.model.fit_vae")
    @patch("embkit.commands.model.dataframe_loader", return_value="loader")
    @patch("embkit.commands.model.NetVAE")
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
                    "--out",
                    "netvae.model",
                ],
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        netvae_cls.assert_called_once()
        netvae_args = netvae_cls.call_args.args
        netvae_kwargs = netvae_cls.call_args.kwargs
        self.assertEqual(netvae_args[0], ["G1", "G2", "G3", "G4"])
        self.assertEqual(netvae_kwargs["group_layer_size"], [5, 2, 1])
        self.assertEqual(set(netvae_kwargs["latent_groups"].keys()), {"TF1", "TF2"})

        loader_mock.assert_called_once()
        fit_mock.assert_called_once()
        self.assertEqual(fit_mock.call_args.kwargs["X"], "loader")
        save_mock.assert_called_once_with(dummy_model, "netvae.model")


if __name__ == "__main__":
    unittest.main()
