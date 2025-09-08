"""
Dataset base classes
"""

from abc import ABC, abstractmethod
from pathlib import Path
import logging
import warnings
import tempfile
import requests

from tqdm import tqdm

logger = logging.getLogger(__name__)

REPO_DIR = ".embkit"

class Dataset(ABC):
    def __init__(self, save_path: Path | str | None, download: bool = True) -> None:
        """
        Initialize the TMP dataset handler.
        :param save_path: Path to save the dataset.
        :param download: Whether to download the dataset immediately. If False, you must call download() manually.
        :raises ValueError: If save_path is not provided.
        :raises FileNotFoundError: If the specified save path does not exist.
        """
        self._download_called_from_init = False

        if save_path is None:
            self.save_path: Path = Path(Path.home(), REPO_DIR)
            if not self.save_path.exists():
                self.save_path.mkdir(parents=True, exist_ok=True)

        else:
            self.save_path: Path = Path(save_path)
            if not self.save_path.exists():
                self.save_path.mkdir(parents=True, exist_ok=True)

        if download:
            try:
                self.download()
                self._download_called_from_init = True
            except Exception as e:
                logger.error(e)

    @abstractmethod
    def download(self) -> None:
        """
        Download the dataset to the specified path.
        If no path is provided, it will use the default save path.
        """
        pass # pragma: no cover

class SingleFileDownloader(Dataset):
    def __init__(self, save_path = None, download = True):
        """
        :param save_path: Path to save the dataset
        :param download: Whether to immediately download
        """
        if not hasattr(self, 'URL'):
            raise NotImplementedError("Subclass must define the 'URL' attribute.")
        if not hasattr(self, 'NAME'):
            raise NotImplementedError("Subclass must define the 'NAME' attribute.")
        self.__unpacked_file_path: Path = Path()
        super().__init__(save_path=save_path, download=download)

    @property
    def unpacked_file_path(self) -> Path:
        """
        Returns the name of the unpacked file.
        This is set after unpacking the downloaded tar.gz file.
        """
        return self.__unpacked_file_path

    def download(self) -> bytes:
        """
        download data file
        """
        if getattr(self, "_download_called_from_init", False):
            warnings.warn(
                "Download was already triggered during initialization. "
                "Calling 'download()' again manually is redundant and may be unintended.",
                stacklevel=2
            )
            return b''

        # Create specific study save path if default path is used
        if Path(self.save_path).expanduser().resolve() == (Path.home() / "embkit").resolve():
            save_path = Path(self.save_path)
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True) # pragma: no cover
        else:
            save_path = self.save_path

        target_file: Path = Path(save_path, self.NAME)

        # Check if already downloaded or unpacked
        if target_file.exists():
            self.__unpacked_file_path = target_file
            logger.info(f"File {target_file} already exists. Skipping download.")
            return b''

        try:
            profiles_url = self.URL
            logger.info(f"Downloading {self.NAME} study data from {self.URL}")
            response = requests.get(profiles_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0))

            # Use a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                with tqdm(
                        desc=f"Downloading {self.NAME}",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                            bar.update(len(chunk))

            # Move to final destination after successful download
            tmp_path.replace(target_file)
            self.__unpacked_file_path = target_file
            logger.info(f"Data downloaded and saved to {target_file}")
            return target_file.read_bytes()

        except requests.RequestException as e:
            logger.error(f"Failed to download {self.NAME}: {e}")
            raise RuntimeError(f"Failed to download {self.NAME} data: {e}")


    