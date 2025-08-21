import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
from requests.exceptions import RequestException
import tarfile
import shutil

from embkit.datasets import CBIOPortal


class TestCBIOPortal(unittest.TestCase):

    def setUp(self):
        self.study_name = "test_study"
        self.test_data = b"Fake .tar.gz content"

    @patch("embkit.datasets.c_bio_portal.requests.get")
    def test_download_success(self, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_response = MagicMock()
            mock_response.iter_content = lambda chunk_size: [self.test_data]
            mock_response.headers = {"Content-Length": str(len(self.test_data))}
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            dataset = CBIOPortal(
                study_name=self.study_name,
                save_path=tmpdir,
                download=False
            )
            data = dataset.download()

            expected_file = Path(tmpdir) / f"{self.study_name}.tar.gz"
            self.assertTrue(expected_file.exists())
            self.assertEqual(data, self.test_data)

    @patch("embkit.datasets.c_bio_portal.requests.get")
    def test_download_warns_on_repeat(self, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = CBIOPortal(
                study_name=self.study_name,
                save_path=tmpdir,
                download=True
            )
            with self.assertWarns(UserWarning):
                dataset.download()

    @patch("embkit.datasets.c_bio_portal.requests.get")
    def test_download_failure_raises(self, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get.side_effect = RequestException("Boom!")
            dataset = CBIOPortal(study_name=self.study_name, save_path=tmpdir, download=False)
            with self.assertRaises(RuntimeError):
                dataset.download()

    @patch("embkit.datasets.c_bio_portal.tarfile.open")
    def test_unpack_success(self, mock_tar_open):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / f"{self.study_name}.tar.gz"
            tar_path.write_bytes(b"fake tar file")

            dataset = CBIOPortal(
                study_name=self.study_name,
                save_path=tmpdir,
                download=False
            )

            mock_tar = MagicMock()
            mock_tar_open.return_value.__enter__.return_value = mock_tar

            dataset.unpack()
            args, kwargs = mock_tar.extractall.call_args
            self.assertEqual(kwargs["path"].resolve(), Path(tmpdir).resolve())
            self.assertEqual(dataset.unpacked_file_path.resolve(), Path(tmpdir, self.study_name).resolve())

    def test_unpack_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = CBIOPortal(
                study_name=self.study_name,
                save_path=tmpdir,
                download=False
            )
            with self.assertRaises(FileNotFoundError):
                dataset.unpack()

    @patch("embkit.datasets.c_bio_portal.tarfile.open")
    def test_unpack_tarfile_error(self, mock_tar_open):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / f"{self.study_name}.tar.gz"
            tar_path.write_bytes(b"bad tar")

            dataset = CBIOPortal(
                study_name=self.study_name,
                save_path=tmpdir,
                download=False
            )

            mock_tar_open.side_effect = tarfile.TarError("broken tar")
            with self.assertRaises(RuntimeError):
                dataset.unpack()

    def test_unpack_skips_if_folder_has_extra_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / f"{self.study_name}.tar.gz"
            tar_path.write_bytes(b"fake")
            study_folder = Path(tmpdir) / self.study_name
            study_folder.mkdir()
            (study_folder / "some_file.txt").write_text("extra")

            dataset = CBIOPortal(
                study_name=self.study_name,
                save_path=tmpdir,
                download=False
            )
            dataset.unpack()
            self.assertEqual(dataset.unpacked_file_path.resolve(), study_folder.resolve())

    @patch("embkit.datasets.c_bio_portal.requests.get")
    def test_download_warns_if_already_called_from_init(self, mock_get):
        mock_response = MagicMock()
        mock_response.iter_content = lambda chunk_size: [b"x" * 10]
        mock_response.headers = {"Content-Length": "10"}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertWarns(UserWarning):
                dataset = CBIOPortal(study_name=self.study_name, save_path=tmpdir, download=True)
                dataset.download()  # triggers the warning path

    def test_default_save_path_creation(self):
        default_path = Path.home() / "embkit"
        if default_path.exists():
            shutil.rmtree(default_path)
        self.assertFalse(default_path.exists())

        dataset = CBIOPortal(study_name=self.study_name, save_path=None, download=False)
        self.assertTrue(Path(dataset.save_path).exists())
        self.assertEqual(Path(dataset.save_path).resolve(), default_path.resolve())

    # ✅ Covers lines 69–70 (target_file and unpacked_folder path logic)
    @patch("embkit.datasets.c_bio_portal.requests.get")
    def test_download_resolves_expected_paths(self, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_response = MagicMock()
            mock_response.iter_content = lambda chunk_size: [b"x" * 10]
            mock_response.headers = {"Content-Length": "10"}
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            dataset = CBIOPortal(study_name=self.study_name, save_path=tmpdir, download=False)
            dataset.download()

            target_file = Path(tmpdir) / f"{self.study_name}.tar.gz"
            unpack_folder = Path(tmpdir) / self.study_name

            self.assertTrue(target_file.exists())
            self.assertEqual(target_file.name, f"{self.study_name}.tar.gz")
            self.assertEqual(unpack_folder.name, self.study_name)

    @patch("embkit.datasets.c_bio_portal.requests.get")
    def test_default_embkit_path_created_if_missing(self, mock_get):
        default_path = Path.home() / "embkit"

        # Clean up before test
        if default_path.exists():
            shutil.rmtree(default_path)
        self.assertFalse(default_path.exists())

        # Setup mock for download
        mock_response = MagicMock()
        mock_response.iter_content = lambda chunk_size: [b"x" * 10]
        mock_response.headers = {"Content-Length": "10"}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Run
        dataset = CBIOPortal(study_name=self.study_name, save_path=None, download=False)
        self.assertEqual(Path(dataset.save_path).resolve(), default_path.resolve())

        # Should be created in download
        dataset.download()
        self.assertTrue(default_path.exists())
        self.assertTrue((default_path / f"{self.study_name}.tar.gz").exists())

    @patch("embkit.datasets.c_bio_portal.requests.get")
    @patch("embkit.datasets.c_bio_portal.logger")
    def test_skips_download_if_tar_gz_exists(self, mock_logger, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            tar_file = save_path / "test_study.tar.gz"
            unpacked_folder = save_path / "test_study"

            # Simulate tar file exists, unpacked folder does not
            tar_file.write_text("fake")
            dataset = CBIOPortal("test_study", save_path=save_path, download=False)

            result = dataset.download()

            self.assertEqual(result, b"")
            mock_logger.info.assert_called_with(f"File {tar_file} already exists. Skipping download.")

    @patch("embkit.datasets.c_bio_portal.requests.get")
    @patch("embkit.datasets.c_bio_portal.logger")
    def test_skips_download_if_unpacked_folder_exists(self, mock_logger, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            unpacked_folder = save_path / "test_study"
            unpacked_folder.mkdir(parents=True)

            dataset = CBIOPortal("test_study", save_path=save_path, download=False)

            result = dataset.download()

            self.assertEqual(result, b"")
            mock_logger.info.assert_called_with(f"Unpacked folder {unpacked_folder} already exists. Skipping download.")


if __name__ == "__main__":
    unittest.main()