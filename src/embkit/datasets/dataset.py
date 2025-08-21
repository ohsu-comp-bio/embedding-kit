from abc import ABC, abstractmethod
from pathlib import Path
import logging

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
