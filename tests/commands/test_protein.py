import unittest
from unittest.mock import MagicMock, patch
import importlib

import torch
from click.testing import CliRunner

from embkit.__main__ import cli_main

protein_cmd = importlib.import_module("embkit.commands.protein")


class TestProteinCommands(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_fasta_reader_and_stringify(self):
        with self.runner.isolated_filesystem():
            with open("toy.fasta", "w", encoding="utf-8") as f:
                f.write(">A1\nMKT\n>B2\nAAA\n")

            rows = list(protein_cmd.fasta_reader("toy.fasta", filter="^A"))
            self.assertEqual(rows, [("A1", "MKT")])

            out = protein_cmd.stringify([1.23456, 2.0], trim=2)
            self.assertEqual(out, ["1.23", "2"])

    @patch.object(protein_cmd, "get_device", return_value=torch.device("cpu"))
    @patch.object(protein_cmd, "ProteinEncoder")
    def test_encode_mean_stdout(self, enc_cls, _dev):
        dummy = MagicMock()
        dummy.encode.return_value = [("seq1", torch.tensor([1.2, 3.4]))]
        enc_cls.return_value = dummy

        with self.runner.isolated_filesystem():
            with open("toy.fasta", "w", encoding="utf-8") as f:
                f.write(">seq1\nMKT\n")

            res = self.runner.invoke(
                cli_main,
                ["protein", "encode", "toy.fasta", "--pool", "mean", "--model", "t6"],
            )

        self.assertEqual(res.exit_code, 0, msg=res.output)
        self.assertIn("seq1\t1.200000\t3.400000", res.output)
        dummy.to.assert_called_once()

    @patch.object(protein_cmd, "get_device", return_value=torch.device("cpu"))
    @patch.object(protein_cmd, "ProteinEncoder")
    def test_encode_vector_output_file(self, enc_cls, _dev):
        dummy = MagicMock()
        dummy.encode.return_value = [("seq1", torch.tensor([[1.23456, 2.34567]]))]
        enc_cls.return_value = dummy

        with self.runner.isolated_filesystem():
            with open("toy.fasta", "w", encoding="utf-8") as f:
                f.write(">seq1\nMKT\n")

            res = self.runner.invoke(
                cli_main,
                [
                    "protein",
                    "encode",
                    "toy.fasta",
                    "--pool",
                    "none",
                    "--trim",
                    "2",
                    "--output",
                    "vec.jsonl",
                    "--model",
                    "t6",
                    "--fix-len",
                    "8",
                ],
            )

            self.assertEqual(res.exit_code, 0, msg=res.output)
            with open("vec.jsonl", "r", encoding="utf-8") as fh:
                text = fh.read().strip()
            self.assertTrue(text.startswith("seq1\t"))
            self.assertIn("[[1.23, 2.35]]", text)


if __name__ == "__main__":
    unittest.main()
