import json
import logging
import os
import pathlib
import statistics

import attrs
import hydra
from hydra.core.config_store import ConfigStore

from lib.data import get_data_path, load_json

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    file_path: pathlib.Path = "data/column_type_inference/sapdata/stats.json"


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    stats = load_json(cfg.file_path)

    def extract_rows(x: str) -> int:
        return json.loads(x)["COUNT(*)"]["0"]

    stats["num_rows"] = [extract_rows(x) for x in stats["num_rows"]]
    characteristics = {
        "number of tables": len(stats["num_cols"]),
        "tables per database": None,
        "mean columns per table": sum(stats["num_cols"]) / len(stats["num_cols"]),
        "median columns per table": statistics.median(stats["num_cols"]),
        "95th columns per table": statistics.quantiles(stats["num_cols"], n=20)[-1],
        "mean rows per table": sum(stats["num_rows"]) / len(stats["num_rows"]),
        "median rows per table": statistics.median(stats["num_rows"]),
        "95th rows per table": statistics.quantiles(stats["num_rows"], n=20)[-1],
        "sparsity": sum(stats["sparsity"]) / len(stats["sparsity"]),
        "non-numerical columns": stats["num_non_numerical_cols"] / (
                stats["num_non_numerical_cols"] + stats["num_numerical_cols"]),
        "numerical columns": stats["num_numerical_cols"] / (
                stats["num_non_numerical_cols"] + stats["num_numerical_cols"])
    }

    path = get_data_path() / "analyze_datasets"
    os.makedirs(path, exist_ok=True)
    with open(path / "sapdata.json", "w", encoding="utf-8") as file:
        json.dump(characteristics, file)
    logger.info("Done!")


if __name__ == "__main__":
    main()
