"""
Human Genome Organisation (HUGO)
"""

from .dataset import SingleFileDownloader
from pathlib import Path


class Hugo(SingleFileDownloader):
    """
    HUGO definition file downloader
    """

    def __init__(self, save_path: Path | str = None, download: bool = True):
        url: str = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
        name: str = "hgnc_complete_set.txt"
        super().__init__(url=url, name=name, save_path=save_path, download=download)
