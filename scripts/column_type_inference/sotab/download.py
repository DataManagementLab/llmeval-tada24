import logging

import hydra
from omegaconf import DictConfig

from lib.data import get_download_dir
from lib.downloading import download_url

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../config/column_type_inference", config_name="config.yaml")
def download(cfg: DictConfig) -> None:
    assert cfg.dataset.dataset_name == "sotab", "This script is dataset-specific."
    download_dir = get_download_dir(cfg.task_name, cfg.dataset.dataset_name, clear=True)

    logger.info(f"Download SOTAB dataset.")
    download_url(url=cfg.dataset.train_url, path=download_dir, unzip=True)
    download_url(url=cfg.dataset.val_url, path=download_dir, unzip=True)
    download_url(url=cfg.dataset.test_url, path=download_dir, unzip=True)


if __name__ == "__main__":
    download()
