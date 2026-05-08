import importlib
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from embkit.__main__ import cli_main

resources_cmd = importlib.import_module("embkit.commands.resources")


class TestResourcesCommands(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_gtex_rejects_unknown_dataset(self):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli_main,
                ["resources", "gtex", "-t", "does_not_exist", "-f", "out"],
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("not recognized", result.output)

    @patch.object(resources_cmd, "GTEx")
    def test_gtex_download_invokes_resource(self, gtex_cls):
        gtex_cls.NAMES = {"gene_tpm": "dummy"}

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli_main,
                ["resources", "gtex", "-t", "gene_tpm", "-f", "out"],
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        gtex_cls.assert_called_once_with(data_type="gene_tpm", save_path="out")

    @patch.object(resources_cmd, "SIF")
    def test_sif_download_invokes_resource(self, sif_cls):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli_main,
                ["resources", "sif", "-f", "out"],
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        sif_cls.assert_called_once_with(save_path="out")

    @patch.object(resources_cmd, "Hugo")
    def test_hugo_download_without_conversion(self, hugo_cls):
        hugo_cls.return_value = SimpleNamespace(save_path="out")

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli_main,
                ["resources", "hugo", "-f", "out"],
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Downloading hugo dataset into 'out'", result.output)
        hugo_cls.assert_called_once_with(save_path="out")

    @patch.object(resources_cmd, "load_raw_hugo")
    @patch.object(resources_cmd, "load_gct")
    @patch.object(resources_cmd, "GTEx")
    @patch.object(resources_cmd, "Hugo")
    def test_hugo_gtex_conversion_writes_output(self, hugo_cls, gtex_cls, load_gct_mock, load_raw_hugo_mock):
        hugo_cls.return_value = SimpleNamespace(save_path="out")
        gtex_cls.return_value = "gtex.gct"
        load_gct_mock.return_value = pd.DataFrame(
            {
                "ENSG000001.1": [1.0, 2.0],
                "ENSG000099.9": [3.0, 4.0],
            },
            index=["s1", "s2"],
        )
        load_raw_hugo_mock.return_value = pd.DataFrame(
            {
                "locus_group": ["protein-coding gene", "RNA, transfer"],
                "symbol": ["TP53", "TRNA1"],
                "ensembl_gene_id": ["ENSG000001", "ENSG000099"],
            }
        )

        with self.runner.isolated_filesystem():
            Path("out").mkdir(parents=True, exist_ok=True)
            result = self.runner.invoke(
                cli_main,
                ["resources", "hugo", "-f", "out", "-gtex"],
            )

            out_path = Path("out/gtex.hugo.tsv")
            self.assertTrue(out_path.exists())
            converted = pd.read_csv(out_path, sep="\t", index_col=0)
            self.assertIn("TP53", converted.columns)
            self.assertNotIn("ENSG000099.9", converted.columns)

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Converting hugo dataset using GTEx dataset", result.output)
        self.assertIn("Done.", result.output)


if __name__ == "__main__":
    unittest.main()
