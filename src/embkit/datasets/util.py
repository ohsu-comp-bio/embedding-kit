

import tempfile
from pathlib import Path
import logging

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

def download_file(url, destination_path):
    """
    Downloads a file from a given URL and saves it to the specified path.

    Args:
        url (str): The URL of the file to download.
        destination_path (str): The local path where the file will be saved.
    """
    # Use a temporary file
    response = requests.get(url, stream=True, timeout=60)  # Use stream=True for large files
    response.raise_for_status()  # Raise an exception for bad status codes
    total_size = int(response.headers.get("Content-Length", 0))

    with tqdm(
            desc=f"Downloading {url}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as sbar:
                        
        try:
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        sbar.update(len(chunk))

            tmp_path.replace(destination_path)

            logger.info("Data downloaded and saved to %s", destination_path)
        except requests.exceptions.RequestException as e:
            logger.error("Error downloading file: %s", e)


             