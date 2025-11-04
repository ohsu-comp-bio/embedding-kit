from pathlib import Path
import requests
from .dataset import Dataset
import logging
import tempfile
from tqdm import tqdm
import tarfile
import warnings

logger = logging.getLogger(__name__)


class CBIOPortal(Dataset):
    BASE_URL = "https://cbioportal-datahub.s3.amazonaws.com/"

    def __init__(
            self,
            study_id: str,
            save_path: Path | str | None = None,
            download: bool = True,
            **kwargs
    ) -> None:
        """
        Initialize the CBIOPortal dataset handler.
        :param dataset_name: cBioPortal study ID, e.g., 'brca_tcga'
        :param save_path: Path to save the dataset
        :param download: Whether to immediately download
        """
        self.study_id = study_id
        self.__unpacked_file_path: Path = Path()
        super().__init__(name=study_id, save_path=save_path, download=download)

    @property
    def unpacked_file_path(self) -> Path:
        """
        Returns the name of the unpacked file.
        This is set after unpacking the downloaded tar.gz file.
        """
        return self.__unpacked_file_path

    def download(self) -> bytes:
        """
        Download all molecular profile data for all samples in the given study.
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

        target_file: Path = Path(save_path, f"{self.study_id}.tar.gz")
        unpacked_folder: Path = Path(save_path, self.study_id)

        # Check if already downloaded or unpacked
        if target_file.exists():
            logger.info(f"File {target_file} already exists. Skipping download.")
            return b''
        if unpacked_folder.exists():
            logger.info(f"Unpacked folder {unpacked_folder} already exists. Skipping download.")
            return b''

        try:
            profiles_url = f"{self.BASE_URL}/{self.study_id}.tar.gz"
            logger.info(f"Downloading cBioPortal study data from {profiles_url}")
            response = requests.get(profiles_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0))

            # Use a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                with tqdm(
                        desc=f"Downloading {self.study_id}",
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
            logger.info(f"Data downloaded and saved to {target_file}")
            return target_file.read_bytes()

        except requests.RequestException as e:
            logger.error(f"Failed to download cBioPortal study data: {e}")
            raise RuntimeError(f"Failed to download cBioPortal study data: {e}")

    def unpack(self) -> Path:
        """
        Unpack the downloaded tar.gz file into the save path.
        Skips unpacking if the unpacked folder contains files other than the tar.gz itself.
        :return: Path to the unpacked folder
        :raises FileNotFoundError: If the tar.gz file does not exist
        """
        save_path = Path(self.save_path).expanduser().resolve()
        target_file = Path(save_path, f"{self.study_id}.tar.gz")
        unpack_folder = Path(save_path, self.study_id)

        if unpack_folder.exists():
            # List all contents other than the tar.gz
            contents = list(unpack_folder.iterdir())
            other_files = [p for p in contents if p.name != f"{self.study_id}.tar.gz"]
            if other_files:
                logger.info(f"Unpacked folder {unpack_folder} already contains files. Skipping unpack.")
                self.__unpacked_file_path = unpack_folder
                return unpack_folder

        if not target_file.exists():
            raise FileNotFoundError(f"File {target_file} does not exist.")

        try:
            with tarfile.open(target_file, "r:gz") as tar:
                tar.extractall(path=save_path)
            logger.info(f"Unpacked {target_file} into {save_path}")
            self.__unpacked_file_path = unpack_folder
            return unpack_folder
        except tarfile.TarError as e:
            logger.error(f"Failed to unpack tar.gz file: {e}")
            raise RuntimeError(f"Failed to unpack tar.gz file: {e}")