import io
import logging
import pathlib
import zipfile

import requests

logger = logging.getLogger(__name__)


def download_url(url: str, path: pathlib.Path, *, unzip: bool = False) -> None:
    """Download the given URL.

    Args:
        url: The URL to download.
        path: The file or directory path.
        unzip: Whether to unzip the downloaded data.
    """
    logger.debug(f"Download {url}")
    response = requests.get(url)
    if not unzip:
        with open(path, "wb") as file:
            file.write(response.content)
    else:
        zip_data = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_data, "r") as zip_ref:
            zip_ref.extractall(path)
