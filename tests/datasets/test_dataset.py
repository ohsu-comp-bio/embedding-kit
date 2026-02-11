import unittest
import tempfile
import logging
from pathlib import Path
from embkit.datasets.dataset import Dataset, REPO_DIR

class DummyDataset(Dataset):
    def download(self) -> None:
        self._was_downloaded = True


class FailingDataset(Dataset):
    def download(self) -> None:
        raise RuntimeError("Failed to download")


class TestDataset(unittest.TestCase):

    def test_creates_directory_and_downloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = DummyDataset(name="dummy", save_path=tmpdir, download=True)
            self.assertTrue(Path(tmpdir).exists())
            self.assertTrue(dataset._download_called_from_init)
            self.assertTrue(hasattr(dataset, '_was_downloaded'))

    def test_skips_download_when_flag_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = DummyDataset(name="dummy", save_path=tmpdir, download=False)
            self.assertFalse(dataset._download_called_from_init)
            self.assertFalse(hasattr(dataset, '_was_downloaded'))

    def test_creates_nested_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "subdir1" / "subdir2"
            self.assertFalse(nested_path.exists())
            dataset = DummyDataset(name="dummy", save_path=nested_path, download=False)
            self.assertTrue(nested_path.exists())

    def test_download_exception_logged(self):
        with self.assertLogs("embkit.datasets.dataset", level="ERROR") as log:
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset = FailingDataset(name="dummy", save_path=tmpdir, download=True)
                self.assertIn("Failed to download", "".join(log.output))
                self.assertFalse(dataset._download_called_from_init)

    def test_none_save_path_with_download_skips_creation(self):
        dataset = DummyDataset(name="dummy", save_path=None, download=False)
        expected_path = Path.home() / REPO_DIR
        self.assertEqual(Path(dataset.save_path).resolve(), expected_path.resolve())


if __name__ == '__main__':
    unittest.main()