"""Tests for Dataset base class and SingleFileDownloader."""

import unittest
import tempfile
from pathlib import Path
from unittest import mock

from embkit.resources.resource import Resource, SingleFileDownloader


class DummyDataset(Resource):
    def download(self):
        # No-op for base‑class test
        pass


class TestDatasetBase(unittest.TestCase):
    def test_default_save_path_created(self):
        # When save_path is None, a default .embkit dir under HOME should be created
        ds = DummyDataset(name="tmp", save_path=None, download=False)
        self.assertTrue(ds.save_path.exists())
        # Ensure it is a directory
        self.assertTrue(ds.save_path.is_dir())

    def test_custom_save_path(self):
        with tempfile.TemporaryDirectory() as td:
            ds = DummyDataset(name="tmp", save_path=td, download=False)
            self.assertEqual(str(ds.save_path), td)
            self.assertTrue(Path(td).exists())


class TestSingleFileDownloader(unittest.TestCase):
    @mock.patch("requests.get")
    def test_download_success_and_return_bytes(self, mock_get):
        # Mock a small response (1 byte)
        mock_resp = mock.Mock()
        mock_resp.iter_content = lambda chunk_size: [b"x"]
        mock_resp.headers = {"Content-Length": "1"}
        mock_resp.raise_for_status = lambda: None
        mock_get.return_value = mock_resp

        with tempfile.TemporaryDirectory() as td:
            dl = SingleFileDownloader(
                url="http://example.com/file",
                name="myfile",
                save_path=td,
                download=False,
            )
            data = dl.download()
            # Should return the bytes that were written
            self.assertEqual(data, b"x")
            # The file should exist on disk with correct content
            target = Path(td, "myfile")
            self.assertTrue(target.exists())
            self.assertEqual(target.read_bytes(), b"x")

    @mock.patch("requests.get")
    def test_download_already_exists_warns(self, mock_get):
        # Simulate an existing file – download should emit a warning and return empty bytes
        with tempfile.TemporaryDirectory() as td:
            target = Path(td, "myfile")
            target.write_bytes(b"already")
            dl = SingleFileDownloader(
                url="http://example.com/file",
                name="myfile",
                save_path=td,
                download=False,
            )
            # First call should detect existing file and log a warning, returning b""
            data = dl.download()
            self.assertEqual(data, b"")
            # Ensure the original file content is untouched
            self.assertEqual(target.read_bytes(), b"already")
            # requests.get should never be called because the file exists
            mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
