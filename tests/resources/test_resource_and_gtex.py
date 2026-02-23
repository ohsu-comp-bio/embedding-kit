"""Tests for Dataset base class and GTEx downloader behavior."""

import unittest
import tempfile
import os
from pathlib import Path
from unittest import mock

from embkit.resources.resource import Resource, SingleFileDownloader
from embkit.resources.gtex import GTEx


class DummyDataset(Resource):
    def download(self):
        # No actual download
        pass


class TestDatasetBase(unittest.TestCase):
    def test_default_save_path_created(self):
        ds = DummyDataset(name="test", save_path=None, download=False)
        # Should create a .embkit directory in home
        default_dir = Path(Path.home(), ".embkit")
        self.assertTrue(ds.save_path == default_dir)
        self.assertTrue(ds.save_path.is_dir())

    def test_custom_save_path_created(self):
        with tempfile.TemporaryDirectory() as td:
            ds = DummyDataset(name="test", save_path=td, download=False)
            self.assertEqual(ds.save_path, Path(td))
            self.assertTrue(ds.save_path.is_dir())

    def test_str_representation(self):
        ds = DummyDataset(name="test", save_path=None, download=False)
        s = str(ds)
        self.assertIn(str(ds.save_path), s)
        self.assertIn("Unpacked file", s)


class TestGTExDownloader(unittest.TestCase):
    @mock.patch("requests.get")
    def test_download_successful(self, mock_get):
        # Mock a tiny response
        mock_resp = mock.Mock()
        mock_resp.iter_content = lambda chunk_size: [b"data"]
        mock_resp.headers = {"Content-Length": "4"}
        mock_resp.raise_for_status = lambda: None
        mock_get.return_value = mock_resp

        with tempfile.TemporaryDirectory() as td:
            gtex = GTEx(data_type="gene_tpm", save_path=td, download=False)
            # Ensure download writes file and returns bytes
            data = gtex.download()
            target = Path(td, GTEx.NAMES["gene_tpm"]).read_bytes()
            self.assertEqual(data, target)
            self.assertEqual(data, b"data")

    @mock.patch("requests.get")
    def test_download_skips_if_exists(self, mock_get):
        with tempfile.TemporaryDirectory() as td:
            target_path = Path(td, GTEx.NAMES["gene_tpm"])
            target_path.write_bytes(b"already")
            gtex = GTEx(data_type="gene_tpm", save_path=td, download=False)
            data = gtex.download()
            # Should return empty bytes and not call requests.get
            self.assertEqual(data, b"")
            mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
