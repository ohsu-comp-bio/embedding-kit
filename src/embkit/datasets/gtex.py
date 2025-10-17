"""
GTEx data downloader
"""
import logging
import tempfile
import warnings
from pathlib import Path
import requests

from tqdm import tqdm

from .dataset import Dataset

logger = logging.getLogger(__name__)


class GTEx(Dataset):
    # pragma: no cover
    BASE_URLS = {
        "gene_tpm": "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz",
        "transcript_tpm": "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/GTEx_Analysis_v10_RSEMv1.3.3_transcripts_tpm.txt.gz"
    }
    NAMES = {
        "gene_tpm": "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz",
        "transcript_tpm": "GTEx_Analysis_v10_RSEMv1.3.3_transcripts_tpm.txt.gz"
    }

    def __init__(
            self,
            data_type: str = "gene_tpm",
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
        self.data_type = data_type
        super().__init__(save_path=save_path, download=download, name=self.NAMES[data_type])

    @property
    def unpacked_file_path(self) -> Path:
        """
        Returns the name of the unpacked file.
        This is set after unpacking the downloaded tar.gz file.
        """
        return self._unpacked_file_path

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

        if Path(self.save_path).expanduser().resolve() == (Path.home() / ".embkit").resolve():
            save_path = Path(self.save_path)
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)  # pragma: no cover
        else:
            save_path = self.save_path

        target_file: Path = Path(save_path, self.NAMES[self.data_type])

        # Check if already downloaded or unpacked
        if target_file.exists():
            self._unpacked_file_path = target_file
            logger.info(f"File {target_file} already exists. Skipping download.")
            return b''

        try:
            profiles_url = self.BASE_URLS[self.data_type]
            logger.info(f"Downloading GTEx study data from {profiles_url}")
            response = requests.get(profiles_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0))

            # Use a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                with tqdm(
                        desc=f"Downloading {self.data_type}",
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
            self._unpacked_file_path = target_file
            logger.info(f"Data downloaded and saved to {target_file}")
            return target_file.read_bytes()

        except requests.RequestException as e:
            logger.error(f"Failed to download GTEx data: {e}")
            raise RuntimeError(f"Failed to download GTEx data: {e}")
