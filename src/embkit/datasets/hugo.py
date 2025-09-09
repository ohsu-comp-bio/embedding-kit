"""
Human Genome Organisation (HUGO)
"""

from .dataset import SingleFileDownloader


class Hugo(SingleFileDownloader):
    """
    HUGO definition file downloader
    """
    URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
    NAME = "hgnc_complete_set.txt"