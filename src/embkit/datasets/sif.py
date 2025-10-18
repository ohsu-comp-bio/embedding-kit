"""
Pathway Commons SIF file downloader
"""

from .dataset import SingleFileDownloader
from pathlib import Path


class SIF(SingleFileDownloader):
    """
    Pathway commons definition file downloader
    """

    def __init__(self, save_path: Path | str = None, download: bool = True):
        url: str = "https://download.baderlab.org/PathwayCommons/PC2/v14/pc-hgnc.sif.gz"
        name: str = "pc-hgnc.sif.gz"
        super().__init__(url=url, name=name, save_path=save_path, download=download)
